import hashlib
import hmac
import logging
import os
import re
import secrets
import smtplib
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import List, Literal, Optional

import asyncpg
import cloudinary
import cloudinary.uploader
import cloudinary.utils
import jwt
import uvicorn
from argon2 import PasswordHasher
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel, EmailStr, Field

load_dotenv()
log = logging.getLogger("audio")

# ───────────────────────── CONFIG ─────────────────────────
DATABASE_URL = os.environ["DATABASE_URL"]
JWT_SECRET = os.environ["JWT_SECRET"]
ACCESS_MIN = int(os.getenv("ACCESS_TOKEN_MINUTES", "30"))
REFRESH_DAYS = int(os.getenv("REFRESH_TOKEN_DAYS", "30"))

# CORS: env se padho, comma-separated. Fallback to localhost:5173 + 3000.
_raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLOUD_NAME = os.environ["CLOUDINARY_CLOUD_NAME"]
CLOUD_KEY = os.environ["CLOUDINARY_API_KEY"]
CLOUD_SECRET = os.environ["CLOUDINARY_API_SECRET"]
cloudinary.config(cloud_name=CLOUD_NAME, api_key=CLOUD_KEY, api_secret=CLOUD_SECRET, secure=True)

CREATOR_ROLES = ("creator", "admin")
pool: asyncpg.Pool = None  # set in lifespan
ph = PasswordHasher()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=int(os.getenv("DB_POOL_MIN", "2")),
        max_size=int(os.getenv("DB_POOL_MAX", "20")),
    )
    print("CORS origins loaded:", CORS_ORIGINS)
    yield
    await pool.close()


app = FastAPI(title="Audio App API", lifespan=lifespan, default_response_class=ORJSONResponse)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ───────────────────────── HELPERS ─────────────────────────
USER_COLS = "id, email, username, display_name, avatar_url, bio, role, created_at"
SONG_COLS = """s.id, s.title, s.duration::float8 AS duration, s.url, s.thumbnail, s.genre, s.genres,
    s.description, s.is_public, s.play_count, s.like_count, s.created_at, s.view_key,
    s.creator_id, u.display_name AS creator_name, u.username AS creator_username,
    u.avatar_url AS creator_avatar"""
SONG_FROM = "FROM songs s LEFT JOIN users u ON u.id = s.creator_id"


def now():
    return datetime.now(timezone.utc)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def otp_hash(otp: str) -> str:
    return hmac.new(JWT_SECRET.encode(), otp.encode(), hashlib.sha256).hexdigest()


def rows(records):
    return [dict(r) for r in records]


def changes(model: BaseModel, nullable=()):
    """Only fields the client actually sent (None allowed only for nullable columns)."""
    return {k: v for k, v in model.model_dump(exclude_unset=True).items() if v is not None or k in nullable}


def build_update(fields: dict):
    cols = list(fields)
    return ", ".join(f"{c}=${i + 1}" for i, c in enumerate(cols)), [fields[c] for c in cols]


def normalize_genres(raw) -> List[str]:
    """Trim, lowercase-dedupe, drop empties. Preserves original casing of first occurrence."""
    out, seen = [], set()
    for g in raw or []:
        g2 = (g or "").strip()
        if not g2:
            continue
        key = g2.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(g2)
    return out


async def hash_pw(p: str) -> str:
    return await run_in_threadpool(ph.hash, p)


async def check_pw(h: str, p: str) -> bool:
    try:
        return await run_in_threadpool(ph.verify, h, p)
    except Exception:
        return False


_hits: dict = {}


def rate_limit(limit: int, window: int = 60):
    async def dep(request: Request):
        key = (request.client.host if request.client else "?", request.url.path)
        t = time.monotonic()
        recent = [x for x in _hits.get(key, []) if t - x < window]
        if len(recent) >= limit:
            raise HTTPException(429, "Too many requests, try again later")
        recent.append(t)
        _hits[key] = recent

    return dep


auth_limit = Depends(rate_limit(10, 60))

# ───────────────────────── TOKENS / AUTH ─────────────────────────
bearer = HTTPBearer(auto_error=False)


async def issue_tokens(user) -> dict:
    access = jwt.encode(
        {"sub": str(user["id"]), "role": user["role"], "type": "access",
         "exp": now() + timedelta(minutes=ACCESS_MIN)},
        JWT_SECRET, algorithm="HS256",
    )
    refresh = secrets.token_urlsafe(48)
    await pool.execute(
        "INSERT INTO refresh_tokens(user_id, token_hash, expires_at) VALUES($1,$2,$3)",
        user["id"], sha(refresh), now() + timedelta(days=REFRESH_DAYS),
    )
    return {"access_token": access, "refresh_token": refresh,
            "token_type": "bearer", "expires_in": ACCESS_MIN * 60}


async def _user_from_creds(cred: Optional[HTTPAuthorizationCredentials]):
    if not cred:
        return None
    try:
        data = jwt.decode(cred.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token")
    if data.get("type") != "access":
        raise HTTPException(401, "Invalid token type")
    user = await pool.fetchrow(f"SELECT {USER_COLS} FROM users WHERE id=$1 AND is_active", int(data["sub"]))
    if not user:
        raise HTTPException(401, "User not found")
    return user


async def current_user(cred=Depends(bearer)):
    user = await _user_from_creds(cred)
    if not user:
        raise HTTPException(401, "Not authenticated")
    return user


async def optional_user(cred=Depends(bearer)):
    return await _user_from_creds(cred)


async def require_creator(user=Depends(current_user)):
    if user["role"] not in CREATOR_ROLES:
        raise HTTPException(403, "Creator account required")
    return user


def _send_email(to: str, subject: str, body: str):
    host = os.getenv("SMTP_HOST")
    if not host:
        log.warning("SMTP not configured. Email to %s | %s | %s", to, subject, body)
        print(f"[DEV EMAIL] to={to} | {subject} | {body}")
        return
    msg = EmailMessage()
    msg["From"] = os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "no-reply@example.com"))
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=15) as s:
        s.starttls()
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        s.send_message(msg)


def check_media(url: str, public_id: str, sub: str, uid: int):
    """Make sure the media really lives in this user's Cloudinary folder."""
    folder = f"audioapp/{sub}/{uid}/"
    if not (url.startswith(f"https://res.cloudinary.com/{CLOUD_NAME}/")
            and public_id.startswith(folder) and public_id in url):
        raise HTTPException(400, "Invalid media reference")


def _destroy_assets(*items):
    for public_id, rtype in items:
        if public_id:
            try:
                cloudinary.uploader.destroy(public_id, resource_type=rtype, invalidate=True)
            except Exception:
                log.exception("Cloudinary delete failed for %s", public_id)


# ───────────────────────── SCHEMAS ─────────────────────────
class RegisterIn(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=30, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(min_length=8, max_length=128)
    display_name: Optional[str] = Field(None, max_length=60)


class LoginIn(BaseModel):
    identifier: str
    password: str


class GoogleIn(BaseModel):
    id_token: str


class RefreshIn(BaseModel):
    refresh_token: str


class ForgotIn(BaseModel):
    email: EmailStr


class ResetIn(BaseModel):
    email: EmailStr
    otp: str = Field(min_length=6, max_length=6)
    new_password: str = Field(min_length=8, max_length=128)


class ChangePwIn(BaseModel):
    current_password: Optional[str] = None
    new_password: str = Field(min_length=8, max_length=128)


class ProfileUpdate(BaseModel):
    display_name: Optional[str] = Field(None, max_length=60)
    username: Optional[str] = Field(None, min_length=3, max_length=30, pattern=r"^[a-zA-Z0-9_]+$")
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = None
    avatar_public_id: Optional[str] = None


class SignIn(BaseModel):
    kind: Literal["audio", "thumbnail", "avatar"]


class SongIn(BaseModel):
    title: str = Field(min_length=1, max_length=150)
    url: str
    audio_public_id: str
    thumbnail: str
    thumbnail_public_id: str
    duration: Optional[float] = Field(None, ge=0)
    genres: List[str] = Field(default_factory=list)   # multiple genres
    description: Optional[str] = Field(None, max_length=2000)
    is_public: bool = True


class SongUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=150)
    genres: Optional[List[str]] = None
    description: Optional[str] = Field(None, max_length=2000)
    is_public: Optional[bool] = None


class PlaylistIn(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    cover_url: Optional[str] = None
    cover_public_id: Optional[str] = None
    is_public: bool = True


class PlaylistUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    cover_url: Optional[str] = None
    cover_public_id: Optional[str] = None
    is_public: Optional[bool] = None


class PlaylistSongIn(BaseModel):
    song_id: int


class ReorderIn(BaseModel):
    song_ids: List[int]


# ───────────────────────── AUTH ROUTES ─────────────────────────
@app.get("/health")
async def health():
    await pool.fetchval("SELECT 1")
    return {"ok": True}


@app.post("/auth/register", status_code=201, dependencies=[auth_limit])
async def register(data: RegisterIn):
    pw = await hash_pw(data.password)
    try:
        user = await pool.fetchrow(
            f"INSERT INTO users(email, username, password_hash, display_name) VALUES($1,$2,$3,$4) RETURNING {USER_COLS}",
            data.email.lower(), data.username.lower(), pw, data.display_name or data.username,
        )
    except asyncpg.UniqueViolationError:
        raise HTTPException(409, "Email or username already taken")
    return {"success": True, "user": dict(user), **await issue_tokens(user)}


@app.post("/auth/login", dependencies=[auth_limit])
async def login(data: LoginIn):
    ident = data.identifier.strip().lower()
    row = await pool.fetchrow(
        f"SELECT {USER_COLS}, password_hash FROM users WHERE (email=$1 OR username=$1) AND is_active", ident)
    if not row or not row["password_hash"] or not await check_pw(row["password_hash"], data.password):
        raise HTTPException(401, "Invalid credentials")
    user = dict(row)
    user.pop("password_hash")
    return {"success": True, "user": user, **await issue_tokens(user)}


@app.post("/auth/google", dependencies=[auth_limit])
async def google_login(data: GoogleIn):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(501, "Google login not configured")
    try:
        info = await run_in_threadpool(
            google_id_token.verify_oauth2_token, data.id_token, google_requests.Request(), GOOGLE_CLIENT_ID)
    except ValueError:
        raise HTTPException(401, "Invalid Google token")
    if not info.get("email_verified"):
        raise HTTPException(400, "Google email not verified")

    gid, email = info["sub"], info["email"].lower()
    q = "SELECT id, is_active FROM users WHERE "
    existing = await pool.fetchrow(q + "google_id=$1", gid) or await pool.fetchrow(q + "email=$1", email)
    if existing:
        if not existing["is_active"]:
            raise HTTPException(403, "Account disabled")
        user = await pool.fetchrow(
            f"""UPDATE users SET google_id=COALESCE(google_id,$2), avatar_url=COALESCE(avatar_url,$3),
                updated_at=now() WHERE id=$1 RETURNING {USER_COLS}""",
            existing["id"], gid, info.get("picture"))
    else:
        base = re.sub(r"[^a-z0-9_]", "", email.split("@")[0])[:20] or "user"
        username = base
        while await pool.fetchval("SELECT 1 FROM users WHERE username=$1", username):
            username = f"{base}{secrets.randbelow(10000):04d}"
        user = await pool.fetchrow(
            f"""INSERT INTO users(email, username, display_name, avatar_url, google_id)
                VALUES($1,$2,$3,$4,$5) RETURNING {USER_COLS}""",
            email, username, info.get("name") or username, info.get("picture"), gid)
    return {"success": True, "user": dict(user), **await issue_tokens(user)}


@app.post("/auth/refresh", dependencies=[auth_limit])
async def refresh(data: RefreshIn):
    row = await pool.fetchrow(
        """UPDATE refresh_tokens SET revoked_at=now()
           WHERE token_hash=$1 AND revoked_at IS NULL AND expires_at>now() RETURNING user_id""",
        sha(data.refresh_token))
    if not row:
        raise HTTPException(401, "Invalid refresh token")
    user = await pool.fetchrow(f"SELECT {USER_COLS} FROM users WHERE id=$1 AND is_active", row["user_id"])
    if not user:
        raise HTTPException(401, "User not found")
    return {"success": True, **await issue_tokens(user)}


@app.post("/auth/logout")
async def logout(data: RefreshIn):
    await pool.execute("UPDATE refresh_tokens SET revoked_at=now() WHERE token_hash=$1 AND revoked_at IS NULL",
                       sha(data.refresh_token))
    return {"success": True}


@app.post("/auth/forgot-password", dependencies=[Depends(rate_limit(5, 60))])
async def forgot_password(data: ForgotIn, bg: BackgroundTasks):
    user = await pool.fetchrow("SELECT id, email FROM users WHERE email=$1 AND is_active", data.email.lower())
    if user:
        otp = f"{secrets.randbelow(10**6):06d}"
        await pool.execute(
            """INSERT INTO password_resets(user_id, otp_hash, expires_at) VALUES($1,$2,$3)
               ON CONFLICT (user_id) DO UPDATE SET otp_hash=$2, expires_at=$3, attempts=0, created_at=now()""",
            user["id"], otp_hash(otp), now() + timedelta(minutes=10))
        bg.add_task(_send_email, user["email"], "Your password reset code",
                    f"Your reset code is {otp}. It expires in 10 minutes.")
    return {"success": True, "message": "If that email exists, a reset code has been sent"}


@app.post("/auth/reset-password", dependencies=[Depends(rate_limit(10, 60))])
async def reset_password(data: ResetIn):
    bad = HTTPException(400, "Invalid or expired code")
    user = await pool.fetchrow("SELECT id FROM users WHERE email=$1 AND is_active", data.email.lower())
    if not user:
        raise bad
    pr = await pool.fetchrow("SELECT otp_hash, attempts, expires_at FROM password_resets WHERE user_id=$1", user["id"])
    if not pr or pr["expires_at"] < now() or pr["attempts"] >= 5:
        raise bad
    if not hmac.compare_digest(pr["otp_hash"], otp_hash(data.otp)):
        await pool.execute("UPDATE password_resets SET attempts=attempts+1 WHERE user_id=$1", user["id"])
        raise bad
    pw = await hash_pw(data.new_password)
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("UPDATE users SET password_hash=$2, updated_at=now() WHERE id=$1", user["id"], pw)
        await conn.execute("DELETE FROM password_resets WHERE user_id=$1", user["id"])
        await conn.execute("UPDATE refresh_tokens SET revoked_at=now() WHERE user_id=$1 AND revoked_at IS NULL", user["id"])
    return {"success": True, "message": "Password updated. Please log in again."}


@app.post("/auth/change-password")
async def change_password(data: ChangePwIn, user=Depends(current_user)):
    current = await pool.fetchval("SELECT password_hash FROM users WHERE id=$1", user["id"])
    if current and not (data.current_password and await check_pw(current, data.current_password)):
        raise HTTPException(400, "Current password is incorrect")
    pw = await hash_pw(data.new_password)
    await pool.execute("UPDATE users SET password_hash=$2, updated_at=now() WHERE id=$1", user["id"], pw)
    await pool.execute("UPDATE refresh_tokens SET revoked_at=now() WHERE user_id=$1 AND revoked_at IS NULL", user["id"])
    return {"success": True}


# ───────────────────────── PROFILE ─────────────────────────
@app.get("/users/me")
async def me(user=Depends(current_user)):
    return {"success": True, "user": dict(user)}


@app.patch("/users/me")
async def update_me(data: ProfileUpdate, user=Depends(current_user)):
    fields = changes(data, nullable=("bio", "avatar_url"))
    if "avatar_url" in fields:
        if fields["avatar_url"]:
            check_media(fields["avatar_url"], fields.get("avatar_public_id") or "", "avatars", user["id"])
        else:
            fields["avatar_public_id"] = None
    if "username" in fields:
        fields["username"] = fields["username"].lower()
    if not fields:
        raise HTTPException(400, "Nothing to update")
    sets, vals = build_update(fields)
    try:
        row = await pool.fetchrow(
            f"UPDATE users SET {sets}, updated_at=now() WHERE id=${len(vals) + 1} RETURNING {USER_COLS}",
            *vals, user["id"])
    except asyncpg.UniqueViolationError:
        raise HTTPException(409, "Username already taken")
    return {"success": True, "user": dict(row)}


@app.post("/users/me/become-creator")
async def become_creator(user=Depends(current_user)):
    await pool.execute("UPDATE users SET role='creator', updated_at=now() WHERE id=$1 AND role='listener'", user["id"])
    return {"success": True, "role": "creator" if user["role"] == "listener" else user["role"]}


@app.get("/users/{username}")
async def public_profile(username: str):
    row = await pool.fetchrow(
        """SELECT id, username, display_name, avatar_url, bio, role, created_at,
                  (SELECT COUNT(*) FROM songs WHERE creator_id=users.id AND is_public) AS song_count
           FROM users WHERE username=$1 AND is_active""", username.lower())
    if not row:
        raise HTTPException(404, "User not found")
    return {"success": True, "user": dict(row)}


@app.get("/me/likes")
async def my_likes(user=Depends(current_user), limit: int = Query(30, ge=1, le=100), offset: int = Query(0, ge=0)):
    r = await pool.fetch(
        f"""SELECT {SONG_COLS} {SONG_FROM} JOIN song_likes l ON l.song_id=s.id
            WHERE l.user_id=$1 AND (s.is_public OR s.creator_id=$1)
            ORDER BY l.created_at DESC LIMIT $2 OFFSET $3""", user["id"], limit, offset)
    return {"success": True, "songs": rows(r)}


@app.get("/me/history")
async def my_history(user=Depends(current_user)):
    r = await pool.fetch(
        f"""SELECT {SONG_COLS}, h.played_at {SONG_FROM} JOIN listening_history h ON h.song_id=s.id
            WHERE h.user_id=$1 AND (s.is_public OR s.creator_id=$1)
            ORDER BY h.played_at DESC LIMIT 50""", user["id"])
    return {"success": True, "songs": rows(r)}


# ───────────────────────── CLOUDINARY DIRECT UPLOAD ─────────────────────────
UPLOAD_KINDS = {"audio": ("songs", "video"), "thumbnail": ("thumbnails", "image"), "avatar": ("avatars", "image")}


@app.post("/uploads/signature", dependencies=[Depends(rate_limit(30, 60))])
async def upload_signature(data: SignIn, user=Depends(current_user)):
    if data.kind != "avatar" and user["role"] not in CREATOR_ROLES:
        raise HTTPException(403, "Creator account required")
    sub, rtype = UPLOAD_KINDS[data.kind]
    params = {"folder": f"audioapp/{sub}/{user['id']}", "timestamp": int(time.time())}
    signature = cloudinary.utils.api_sign_request(params, CLOUD_SECRET)
    return {
        "success": True,
        "upload_url": f"https://api.cloudinary.com/v1_1/{CLOUD_NAME}/{rtype}/upload",
        "api_key": CLOUD_KEY, "signature": signature, **params,
    }


# ───────────────────────── SONGS ─────────────────────────
async def fetch_song(song_id: int, viewer_id: Optional[int]):
    r = await pool.fetchrow(
        f"""SELECT {SONG_COLS}, EXISTS(SELECT 1 FROM song_likes l WHERE l.song_id=s.id AND l.user_id=$2) AS is_liked
            {SONG_FROM} WHERE s.id=$1""", song_id, viewer_id)
    if not r or (not r["is_public"] and r["creator_id"] != viewer_id):
        return None
    return dict(r)


async def owned_song(song_id: int, user):
    s = await pool.fetchrow(
        "SELECT id, creator_id, audio_public_id, thumbnail_public_id FROM songs WHERE id=$1", song_id)
    if not s:
        raise HTTPException(404, "Song not found")
    if s["creator_id"] != user["id"] and user["role"] != "admin":
        raise HTTPException(403, "Not your song")
    return s


@app.get("/songs")
async def list_songs(
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    q: Optional[str] = Query(None),
    genre: Optional[str] = Query(None),
    creator_id: Optional[int] = Query(None),
    sort: Literal["newest", "popular", "liked", "title", "random"] = Query("newest"),
):
    where, args = ["s.is_public"], []

    if q:
        safe = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pat = f"%{safe}%"
        args.append(pat)
        i = len(args)
        where.append(
            f"(s.title ILIKE ${i} "
            f"OR EXISTS (SELECT 1 FROM unnest(s.genres) g WHERE g ILIKE ${i}) "
            f"OR s.genre ILIKE ${i} "
            f"OR u.display_name ILIKE ${i} "
            f"OR u.username ILIKE ${i})"
        )

    if genre:
        safe = genre.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        args.append(f"%{safe}%")
        where.append(
            f"(EXISTS (SELECT 1 FROM unnest(s.genres) g WHERE g ILIKE ${len(args)}) "
            f"OR s.genre ILIKE ${len(args)})"
        )

    if creator_id:
        args.append(creator_id)
        where.append(f"s.creator_id=${len(args)}")

    where_sql = " AND ".join(where)

    total = await pool.fetchval(
        f"SELECT COUNT(*) {SONG_FROM} WHERE {where_sql}",
        *args,
    )

    order = {
        "newest":  "s.created_at DESC, s.id DESC",
        "popular": "s.play_count DESC, s.created_at DESC",
        "liked":   "s.like_count DESC, s.created_at DESC",
        "title":   "s.title ASC",
        "random":  "RANDOM()",
    }[sort]

    args += [limit, offset]
    r = await pool.fetch(
        f"""SELECT {SONG_COLS} {SONG_FROM}
            WHERE {where_sql}
            ORDER BY {order}
            LIMIT ${len(args) - 1} OFFSET ${len(args)}""",
        *args,
    )

    return {
        "success": True,
        "query": {"q": q, "genre": genre, "creator_id": creator_id, "sort": sort},
        "songs": rows(r),
        "limit": limit,
        "offset": offset,
        "total": int(total or 0),
        "has_more": offset + len(r) < int(total or 0),
    }


@app.get("/genres")
async def list_genres():
    """All distinct genres across every song (array + legacy single-genre column)."""
    r = await pool.fetch(
        """
        WITH all_genres AS (
            SELECT btrim(g) AS genre
            FROM songs, unnest(genres) AS g
            WHERE is_public
              AND btrim(g) <> ''
            UNION
            SELECT btrim(genre) AS genre
            FROM songs
            WHERE is_public
              AND genre IS NOT NULL
              AND btrim(genre) <> ''
        )
        SELECT genre, COUNT(*)::int AS song_count
        FROM all_genres
        GROUP BY genre
        ORDER BY (lower(genre) = 'trending') DESC,
                 song_count DESC,
                 genre ASC
        LIMIT 30
        """
    )
    return {"success": True, "genres": rows(r)}


@app.get("/songs/{song_id}")
async def get_song(song_id: int, user=Depends(optional_user)):
    song = await fetch_song(song_id, user["id"] if user else None)
    if not song:
        raise HTTPException(404, "Song not found")
    return {"success": True, "song": song}


@app.post("/songs", status_code=201)
async def create_song(data: SongIn, user=Depends(require_creator)):
    check_media(data.url, data.audio_public_id, "songs", user["id"])
    check_media(data.thumbnail, data.thumbnail_public_id, "thumbnails", user["id"])

    genres = normalize_genres(data.genres)
    primary_genre = genres[0] if genres else None

    song_id = await pool.fetchval(
        """INSERT INTO songs(title, duration, url, thumbnail, view_key, genre, genres,
                             description, is_public, creator_id, audio_public_id, thumbnail_public_id)
           VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) RETURNING id""",
        data.title, data.duration, data.url, data.thumbnail, secrets.token_urlsafe(8),
        primary_genre, genres,
        data.description, data.is_public, user["id"],
        data.audio_public_id, data.thumbnail_public_id)
    return {"success": True, "song": await fetch_song(song_id, user["id"])}


@app.patch("/songs/{song_id}")
async def update_song(song_id: int, data: SongUpdate, user=Depends(current_user)):
    await owned_song(song_id, user)
    fields = changes(data, nullable=("description",))

    if "genres" in fields:
        genres = normalize_genres(fields.pop("genres"))
        fields["genres"] = genres
        fields["genre"] = genres[0] if genres else None

    if not fields:
        raise HTTPException(400, "Nothing to update")
    sets, vals = build_update(fields)
    await pool.execute(f"UPDATE songs SET {sets}, updated_at=now() WHERE id=${len(vals) + 1}", *vals, song_id)
    return {"success": True, "song": await fetch_song(song_id, user["id"])}


@app.delete("/songs/{song_id}")
async def delete_song(song_id: int, bg: BackgroundTasks, user=Depends(current_user)):
    s = await owned_song(song_id, user)
    await pool.execute("DELETE FROM songs WHERE id=$1", song_id)
    bg.add_task(_destroy_assets, (s["audio_public_id"], "video"), (s["thumbnail_public_id"], "image"))
    return {"success": True}


@app.post("/songs/{song_id}/play")
async def register_play(song_id: int, user=Depends(optional_user)):
    uid = user["id"] if user else None
    n = await pool.fetchval(
        "UPDATE songs SET play_count=play_count+1 WHERE id=$1 AND (is_public OR creator_id=$2) RETURNING play_count",
        song_id, uid)
    if n is None:
        raise HTTPException(404, "Song not found")
    if uid:
        await pool.execute("INSERT INTO listening_history(user_id, song_id) VALUES($1,$2)", uid, song_id)
    return {"success": True, "play_count": n}


@app.post("/songs/{song_id}/like")
async def like_song(song_id: int, user=Depends(current_user)):
    async with pool.acquire() as conn, conn.transaction():
        res = await conn.execute(
            """INSERT INTO song_likes(user_id, song_id)
               SELECT $1, id FROM songs WHERE id=$2 AND (is_public OR creator_id=$1) ON CONFLICT DO NOTHING""",
            user["id"], song_id)
        if res.endswith(" 1"):
            await conn.execute("UPDATE songs SET like_count=like_count+1 WHERE id=$1", song_id)
        count = await conn.fetchval("SELECT like_count FROM songs WHERE id=$1", song_id)
    if count is None:
        raise HTTPException(404, "Song not found")
    return {"success": True, "liked": True, "like_count": count}


@app.delete("/songs/{song_id}/like")
async def unlike_song(song_id: int, user=Depends(current_user)):
    async with pool.acquire() as conn, conn.transaction():
        res = await conn.execute("DELETE FROM song_likes WHERE user_id=$1 AND song_id=$2", user["id"], song_id)
        if res.endswith(" 1"):
            await conn.execute("UPDATE songs SET like_count=GREATEST(like_count-1,0) WHERE id=$1", song_id)
        count = await conn.fetchval("SELECT like_count FROM songs WHERE id=$1", song_id)
    if count is None:
        raise HTTPException(404, "Song not found")
    return {"success": True, "liked": False, "like_count": count}


# ───────────────────────── CREATOR CENTER ─────────────────────────
@app.get("/creator/songs")
async def creator_songs(user=Depends(require_creator), limit: int = Query(30, ge=1, le=200), offset: int = Query(0, ge=0)):
    r = await pool.fetch(
        f"SELECT {SONG_COLS} {SONG_FROM} WHERE s.creator_id=$1 ORDER BY s.created_at DESC LIMIT $2 OFFSET $3",
        user["id"], limit, offset)
    return {"success": True, "songs": rows(r)}


@app.get("/creator/stats")
async def creator_stats(user=Depends(require_creator)):
    r = await pool.fetchrow(
        """SELECT COUNT(*) AS songs,
                  COALESCE(SUM(play_count),0)::bigint AS plays,
                  COALESCE(SUM(like_count),0)::bigint AS likes,
                  (SELECT COUNT(*) FROM playlists WHERE owner_id=$1) AS playlists
           FROM songs WHERE creator_id=$1""", user["id"])
    return {"success": True, "stats": dict(r)}


# ───────────────────────── PLAYLISTS ─────────────────────────
PL_COLS = """p.id, p.owner_id, p.title, p.description, p.cover_url, p.is_public, p.created_at, p.updated_at,
    u.display_name AS owner_name, u.username AS owner_username,
    (SELECT COUNT(*) FROM playlist_songs ps WHERE ps.playlist_id=p.id) AS song_count"""
PL_FROM = "FROM playlists p JOIN users u ON u.id=p.owner_id"


async def owned_playlist(pid: int, user):
    p = await pool.fetchrow("SELECT id, owner_id FROM playlists WHERE id=$1", pid)
    if not p:
        raise HTTPException(404, "Playlist not found")
    if p["owner_id"] != user["id"] and user["role"] != "admin":
        raise HTTPException(403, "Not your playlist")
    return p


@app.get("/playlists")
async def public_playlists(limit: int = Query(20, ge=1, le=50), offset: int = Query(0, ge=0)):
    r = await pool.fetch(
        f"SELECT {PL_COLS} {PL_FROM} WHERE p.is_public ORDER BY p.created_at DESC LIMIT $1 OFFSET $2", limit, offset)
    return {"success": True, "playlists": rows(r)}


@app.get("/playlists/mine")
async def my_playlists(user=Depends(require_creator)):
    r = await pool.fetch(f"SELECT {PL_COLS} {PL_FROM} WHERE p.owner_id=$1 ORDER BY p.created_at DESC", user["id"])
    return {"success": True, "playlists": rows(r)}


@app.get("/playlists/{pid}")
async def get_playlist(pid: int, user=Depends(optional_user)):
    uid = user["id"] if user else None
    p = await pool.fetchrow(f"SELECT {PL_COLS} {PL_FROM} WHERE p.id=$1", pid)
    if not p or (not p["is_public"] and p["owner_id"] != uid):
        raise HTTPException(404, "Playlist not found")
    songs = await pool.fetch(
        f"""SELECT {SONG_COLS}, ps.position {SONG_FROM} JOIN playlist_songs ps ON ps.song_id=s.id
            WHERE ps.playlist_id=$1 AND (s.is_public OR s.creator_id=$2) ORDER BY ps.position""", pid, uid)
    return {"success": True, "playlist": dict(p), "songs": rows(songs)}


@app.post("/playlists", status_code=201)
async def create_playlist(data: PlaylistIn, user=Depends(require_creator)):
    if data.cover_url:
        check_media(data.cover_url, data.cover_public_id or "", "thumbnails", user["id"])
    pid = await pool.fetchval(
        """INSERT INTO playlists(owner_id, title, description, cover_url, cover_public_id, is_public)
           VALUES($1,$2,$3,$4,$5,$6) RETURNING id""",
        user["id"], data.title, data.description, data.cover_url, data.cover_public_id, data.is_public)
    return {"success": True, "id": pid}


@app.patch("/playlists/{pid}")
async def update_playlist(pid: int, data: PlaylistUpdate, user=Depends(require_creator)):
    await owned_playlist(pid, user)
    fields = changes(data, nullable=("description", "cover_url"))
    if "cover_url" in fields:
        if fields["cover_url"]:
            check_media(fields["cover_url"], fields.get("cover_public_id") or "", "thumbnails", user["id"])
        else:
            fields["cover_public_id"] = None
    if not fields:
        raise HTTPException(400, "Nothing to update")
    sets, vals = build_update(fields)
    await pool.execute(f"UPDATE playlists SET {sets}, updated_at=now() WHERE id=${len(vals) + 1}", *vals, pid)
    return {"success": True}


@app.delete("/playlists/{pid}")
async def delete_playlist(pid: int, user=Depends(require_creator)):
    await owned_playlist(pid, user)
    await pool.execute("DELETE FROM playlists WHERE id=$1", pid)
    return {"success": True}


@app.post("/playlists/{pid}/songs", status_code=201)
async def add_song_to_playlist(pid: int, data: PlaylistSongIn, user=Depends(require_creator)):
    await owned_playlist(pid, user)
    res = await pool.execute(
        """INSERT INTO playlist_songs(playlist_id, song_id, position)
           SELECT $1, s.id, COALESCE((SELECT MAX(position)+1 FROM playlist_songs WHERE playlist_id=$1), 0)
           FROM songs s WHERE s.id=$2 AND (s.is_public OR s.creator_id=$3) ON CONFLICT DO NOTHING""",
        pid, data.song_id, user["id"])
    if res.endswith(" 0"):
        raise HTTPException(409, "Song not found or already in playlist")
    await pool.execute("UPDATE playlists SET updated_at=now() WHERE id=$1", pid)
    return {"success": True}


@app.delete("/playlists/{pid}/songs/{song_id}")
async def remove_song_from_playlist(pid: int, song_id: int, user=Depends(require_creator)):
    await owned_playlist(pid, user)
    await pool.execute("DELETE FROM playlist_songs WHERE playlist_id=$1 AND song_id=$2", pid, song_id)
    await pool.execute("UPDATE playlists SET updated_at=now() WHERE id=$1", pid)
    return {"success": True}


@app.put("/playlists/{pid}/order")
async def reorder_playlist(pid: int, data: ReorderIn, user=Depends(require_creator)):
    await owned_playlist(pid, user)
    async with pool.acquire() as conn, conn.transaction():
        current = {r["song_id"] for r in await conn.fetch(
            "SELECT song_id FROM playlist_songs WHERE playlist_id=$1", pid)}
        if len(data.song_ids) != len(current) or set(data.song_ids) != current:
            raise HTTPException(400, "song_ids must contain exactly the songs in the playlist")
        await conn.executemany(
            "UPDATE playlist_songs SET position=$3 WHERE playlist_id=$1 AND song_id=$2",
            [(pid, sid, i) for i, sid in enumerate(data.song_ids)])
        await conn.execute("UPDATE playlists SET updated_at=now() WHERE id=$1", pid)
    return {"success": True}


if __name__ == "__main__":
    uvicorn.run("server:app", port=8080, host="0.0.0.0", reload=True)