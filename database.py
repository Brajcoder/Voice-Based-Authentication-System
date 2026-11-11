"""
Database management module for SQLite in Voice-Based Authentication.
Handles connection lifecycles, user storage, and voice embedding persistence.
"""

import sqlite3
import os
from flask import g, current_app

DB_NAME = "voice_auth.db"


def get_db_path() -> str:
    """Get absolute path to SQLite database."""
    base_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_dir, DB_NAME)


def get_db() -> sqlite3.Connection:
    """
    Get a database connection for the current request context.
    If called outside request context, opens a standalone connection.
    """
    # Check if inside Flask application context
    try:
        if "db" not in g:
            db = sqlite3.connect(get_db_path(), detect_types=sqlite3.PARSE_DECLTYPES)
            db.row_factory = sqlite3.Row
            g.db = db
        return g.db
    except RuntimeError:
        # Outside Flask application context
        db = sqlite3.connect(get_db_path(), detect_types=sqlite3.PARSE_DECLTYPES)
        db.row_factory = sqlite3.Row
        return db


def close_db(e=None):
    """Close the database connection at the end of the request."""
    try:
        db = g.pop("db", None)
        if db is not None:
            db.close()
    except RuntimeError:
        pass


def init_db(app=None):
    """Initialize database tables."""
    db = sqlite3.connect(get_db_path())
    cursor = db.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            voice_embedding BLOB,
            is_enrolled INTEGER DEFAULT 0,
            enrolled_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    db.commit()
    db.close()

    if app is not None:
        app.teardown_appcontext(close_db)


def create_user(username: str, password_hash: str) -> int | None:
    """
    Create a new user with hashed password.
    Returns the user id if successful, or None if username already exists.
    """
    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username.strip(), password_hash),
        )
        db.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None


def get_user_by_username(username: str) -> sqlite3.Row | None:
    """Retrieve user record by username."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
    return cursor.fetchone()


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    """Retrieve user record by ID."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()


def save_voice_embedding(user_id: int, embedding_bytes: bytes) -> bool:
    """
    Save the serialized voice embedding and update enrollment status.
    """
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """
        UPDATE users
        SET voice_embedding = ?, is_enrolled = 1, enrolled_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (embedding_bytes, user_id),
    )
    db.commit()
    return cursor.rowcount > 0


def delete_voice_embedding(user_id: int) -> bool:
    """
    Remove voice enrollment data for a user.
    """
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """
        UPDATE users
        SET voice_embedding = NULL, is_enrolled = 0, enrolled_at = NULL
        WHERE id = ?
        """,
        (user_id,),
    )
    db.commit()
    return cursor.rowcount > 0

