import psycopg 
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def get_connection(self):
        try:
            # Get the database URL from environment variables
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                raise ValueError("DATABASE_URL is not set in environment variables.")
            
            # Establish a connection to the PostgreSQL database
            connection = psycopg.connect(database_url)
            cursor = connection.cursor(row_factory=psycopg.rows.dict_row)
            return connection, cursor
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            return None , None