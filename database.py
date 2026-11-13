"""
Database management module for SQLite in Voice-Based Authentication.
Handles connection lifecycles, user storage, and voice embedding persistence.
"""

import sqlite3
import os
import io
import pickle
import math
from datetime import datetime, timezone, timedelta
import numpy as np
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
    """Initialize database tables and run schema migrations."""
    db = sqlite3.connect(get_db_path())
    cursor = db.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            voice_embedding BLOB,
            voice_embeddings_list BLOB,
            voice_fail_count INTEGER DEFAULT 0,
            voice_locked_until TIMESTAMP,
            is_enrolled INTEGER DEFAULT 0,
            enrolled_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    # Check if new columns exist (for migration of existing databases)
    cursor.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cursor.fetchall()]
    if "voice_embeddings_list" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN voice_embeddings_list BLOB")
    if "voice_fail_count" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN voice_fail_count INTEGER DEFAULT 0")
    if "voice_locked_until" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN voice_locked_until TIMESTAMP")

    db.commit()
    db.close()

    if app is not None:
        app.teardown_appcontext(close_db)


def embeddings_list_to_bytes(embeddings: list) -> bytes:
    """Serialize a list of numpy arrays to binary bytes for SQLite BLOB storage."""
    clean_list = [np.asarray(e, dtype=np.float32) for e in embeddings]
    return pickle.dumps(clean_list)


def bytes_to_embeddings_list(data: bytes | None) -> list:
    """
    Deserialize binary bytes from SQLite BLOB to a list of numpy embedding arrays.
    Supports pickle and np.load (2D, 1D, or object array) for robust compatibility.
    """
    if not data:
        return []

    # Try pickle first
    try:
        obj = pickle.loads(data)
        if isinstance(obj, list):
            return [np.asarray(x, dtype=np.float32) for x in obj]
        elif isinstance(obj, np.ndarray):
            if obj.ndim == 2:
                return [row.astype(np.float32) for row in obj]
            elif obj.ndim == 1:
                return [obj.astype(np.float32)]
    except Exception:
        pass

    # Fallback to np.load
    try:
        bio = io.BytesIO(data)
        arr = np.load(bio, allow_pickle=True)
        if isinstance(arr, np.ndarray):
            if arr.ndim == 2:
                return [row.astype(np.float32) for row in arr]
            elif arr.ndim == 1:
                return [arr.astype(np.float32)]
            elif arr.dtype == object:
                return [np.asarray(x, dtype=np.float32) for x in arr.tolist()]
    except Exception:
        pass

    return []


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


def save_voice_embedding(
    user_id: int,
    embedding_bytes: bytes | None = None,
    embeddings_list_bytes: bytes | None = None,
) -> bool:
    """
    Save the serialized voice embedding(s) and update enrollment status.
    Supports storing up to 3 voice embeddings in voice_embeddings_list while
    maintaining backward compatibility with the single voice_embedding column.
    """
    db = get_db()
    cursor = db.cursor()
    if embeddings_list_bytes is not None and embedding_bytes is not None:
        cursor.execute(
            """
            UPDATE users
            SET voice_embedding = ?, voice_embeddings_list = ?, is_enrolled = 1, enrolled_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (embedding_bytes, embeddings_list_bytes, user_id),
        )
    elif embeddings_list_bytes is not None:
        cursor.execute(
            """
            UPDATE users
            SET voice_embeddings_list = ?, is_enrolled = 1, enrolled_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (embeddings_list_bytes, user_id),
        )
    else:
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


def save_voice_embeddings_list(
    user_id: int,
    embeddings_list_bytes: bytes,
    latest_embedding_bytes: bytes | None = None,
) -> bool:
    """Save serialized list of embeddings and optional single latest embedding."""
    return save_voice_embedding(
        user_id=user_id,
        embedding_bytes=latest_embedding_bytes,
        embeddings_list_bytes=embeddings_list_bytes,
    )


def delete_voice_embedding(user_id: int) -> bool:
    """
    Remove voice enrollment data for a user.
    Clears voice_embedding, voice_embeddings_list, and failure/lockout counters.
    """
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """
        UPDATE users
        SET voice_embedding = NULL, voice_embeddings_list = NULL,
            is_enrolled = 0, enrolled_at = NULL,
            voice_fail_count = 0, voice_locked_until = NULL
        WHERE id = ?
        """,
        (user_id,),
    )
    db.commit()
    return cursor.rowcount > 0


class VoiceLockStatus:
    """
    Represents voice lockout state.
    Supports boolean check `if is_voice_locked(user_id):`,
    unpacking `locked, mins = is_voice_locked(user_id)`,
    and direct attribute access `status.remaining_minutes`.
    """

    def __init__(self, locked: bool, remaining_minutes: int = 0):
        self.locked = bool(locked)
        self.remaining_minutes = int(remaining_minutes)

    def __bool__(self):
        return self.locked

    def __iter__(self):
        return iter((self.locked, self.remaining_minutes))

    def __getitem__(self, index):
        return (self.locked, self.remaining_minutes)[index]

    def __repr__(self):
        return f"VoiceLockStatus(locked={self.locked}, remaining_minutes={self.remaining_minutes})"


class VoiceFailResult:
    """
    Represents result of incrementing voice failure count.
    Supports unpacking, boolean check, integer conversion, and attributes.
    """

    def __init__(self, count: int, locked: bool, remaining_minutes: int = 0):
        self.count = int(count)
        self.locked = bool(locked)
        self.remaining_minutes = int(remaining_minutes)

    def __bool__(self):
        return self.locked

    def __int__(self):
        return self.count

    def __iter__(self):
        return iter((self.count, self.locked, self.remaining_minutes))

    def __getitem__(self, index):
        return (self.count, self.locked, self.remaining_minutes)[index]

    def __repr__(self):
        return f"VoiceFailResult(count={self.count}, locked={self.locked}, remaining_minutes={self.remaining_minutes})"


def _parse_timestamp(val) -> datetime | None:
    """Safely parse a timestamp value from SQLite into a UTC datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    if isinstance(val, str):
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                dt = datetime.strptime(val, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                pass
        try:
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    return None


def is_voice_locked(user_id: int) -> VoiceLockStatus:
    """
    Check if a user is currently locked out from voice authentication.
    Returns VoiceLockStatus object indicating lockout and remaining minutes.
    If the lockout period has expired, automatically resets the failure count.
    """
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT voice_locked_until FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    if not row or row["voice_locked_until"] is None:
        return VoiceLockStatus(locked=False, remaining_minutes=0)

    locked_until_dt = _parse_timestamp(row["voice_locked_until"])
    if not locked_until_dt:
        return VoiceLockStatus(locked=False, remaining_minutes=0)

    now = datetime.now(timezone.utc)
    if now < locked_until_dt:
        rem_sec = (locked_until_dt - now).total_seconds()
        rem_min = max(1, math.ceil(rem_sec / 60))
        return VoiceLockStatus(locked=True, remaining_minutes=rem_min)

    # Lock period has expired -> auto-reset
    reset_voice_fail(user_id)
    return VoiceLockStatus(locked=False, remaining_minutes=0)


def increment_voice_fail(user_id: int) -> VoiceFailResult:
    """
    Increment the voice failure counter for a user.
    After 5 consecutive failures, lock voice authentication for 15 minutes.
    Returns VoiceFailResult with (count, locked, remaining_minutes).
    """
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "SELECT voice_fail_count, voice_locked_until FROM users WHERE id = ?",
        (user_id,),
    )
    row = cursor.fetchone()
    if not row:
        return VoiceFailResult(count=0, locked=False, remaining_minutes=0)

    # Check if already locked
    current_lock = is_voice_locked(user_id)
    if current_lock:
        return VoiceFailResult(
            count=row["voice_fail_count"] or 5,
            locked=True,
            remaining_minutes=current_lock.remaining_minutes,
        )

    current_count = row["voice_fail_count"] or 0
    new_count = current_count + 1

    if new_count >= 5:
        # Lock for 15 minutes
        lock_until = datetime.now(timezone.utc) + timedelta(minutes=15)
        lock_str = lock_until.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            UPDATE users
            SET voice_fail_count = ?, voice_locked_until = ?
            WHERE id = ?
            """,
            (new_count, lock_str, user_id),
        )
        db.commit()
        return VoiceFailResult(count=new_count, locked=True, remaining_minutes=15)

    cursor.execute(
        "UPDATE users SET voice_fail_count = ? WHERE id = ?",
        (new_count, user_id),
    )
    db.commit()
    return VoiceFailResult(count=new_count, locked=False, remaining_minutes=0)


def reset_voice_fail(user_id: int) -> bool:
    """
    Reset the failed voice attempts count and clear voice lockout for a user.
    """
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """
        UPDATE users
        SET voice_fail_count = 0, voice_locked_until = NULL
        WHERE id = ?
        """,
        (user_id,),
    )
    db.commit()
    return cursor.rowcount > 0

