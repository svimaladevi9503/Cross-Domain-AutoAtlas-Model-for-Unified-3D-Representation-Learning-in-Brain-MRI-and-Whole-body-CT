import sqlite3
import os
import uuid
from contextlib import contextmanager

# Define DB_PATH relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "autoatlas.db")

@contextmanager
def get_db_connection():
    """Context manager for SQLite database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        # Enable foreign key support
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise
    finally:
        conn.close()

def init_db():
    """Initialize the database tables if they don't exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                filename TEXT,
                modality TEXT,
                file_path TEXT,
                uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id INTEGER,
                translated_path TEXT,
                segmented_path TEXT,
                atlas_used TEXT,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (upload_id) REFERENCES uploads(id)
            )
        ''')
        conn.commit()

def create_session():
    """Create a new session and return the session_id."""
    session_id = str(uuid.uuid4())
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO sessions (session_id) VALUES (?)', (session_id,))
        conn.commit()
    return session_id

def save_upload(session_id, filename, modality, file_path):
    """Save upload metadata and return the upload_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO uploads (session_id, filename, modality, file_path) 
            VALUES (?, ?, ?, ?)
        ''', (session_id, filename, modality, file_path))
        upload_id = cursor.lastrowid
        conn.commit()
    return upload_id

def save_result(upload_id, translated_path, segmented_path, atlas_used):
    """Save processing results."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO results (upload_id, translated_path, segmented_path, atlas_used) 
            VALUES (?, ?, ?, ?)
        ''', (upload_id, translated_path, segmented_path, atlas_used))
        conn.commit()

def get_session_history(session_id):
    """Retrieve history for a specific session."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT u.id as upload_id, u.filename, u.modality, u.uploaded_at,
                   r.translated_path, r.segmented_path, r.atlas_used, r.processed_at
            FROM uploads u
            LEFT JOIN results r ON u.id = r.upload_id
            WHERE u.session_id = ?
            ORDER BY u.uploaded_at DESC
        ''', (session_id,))
        rows = cursor.fetchall()
    return [dict(row) for row in rows]
