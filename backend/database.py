import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime


def _utc_now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@contextmanager
def get_connection(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path):
    with get_connection(db_path) as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                student_id TEXT DEFAULT '',
                face_encoding TEXT NOT NULL,
                user_type TEXT DEFAULT 'permanent',
                expiry_datetime TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                student_id TEXT DEFAULT '',
                user_type TEXT DEFAULT 'unknown',
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                denied_reason TEXT DEFAULT ''
            )
        """)

        # --- Safe column migrations (won't fail if column already exists) ---
        user_cols = [row[1] for row in cur.execute("PRAGMA table_info(users)").fetchall()]
        if "student_id" not in user_cols:
            cur.execute("ALTER TABLE users ADD COLUMN student_id TEXT DEFAULT ''")
        if "user_type" not in user_cols:
            cur.execute("ALTER TABLE users ADD COLUMN user_type TEXT DEFAULT 'permanent'")
        if "expiry_datetime" not in user_cols:
            cur.execute("ALTER TABLE users ADD COLUMN expiry_datetime TEXT")

        log_cols = [row[1] for row in cur.execute("PRAGMA table_info(logs)").fetchall()]
        if "user_type" not in log_cols:
            cur.execute("ALTER TABLE logs ADD COLUMN user_type TEXT DEFAULT 'unknown'")
        if "denied_reason" not in log_cols:
            cur.execute("ALTER TABLE logs ADD COLUMN denied_reason TEXT DEFAULT ''")


def add_user(db_path, name, face_encoding, user_type, expiry_datetime, student_id=""):
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO users (name, student_id, face_encoding, user_type, expiry_datetime)
            VALUES (?, ?, ?, ?, ?)
        """, (name, student_id, json.dumps(face_encoding), user_type, expiry_datetime))
        return cur.lastrowid


def update_user_encoding(db_path, user_id, encoding):
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE users SET face_encoding = ? WHERE id = ?
        """, (json.dumps(encoding), user_id))


def find_user_id_by_name(db_path, name):
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT id FROM users WHERE name = ? ORDER BY id DESC LIMIT 1",
            (name,),
        ).fetchone()
        return int(row["id"]) if row else None


def _alphanumeric_key(name):
    return "".join(c for c in name if c.isalnum()).lower()


def find_user_id_by_alphanumeric_name(db_path, key):
    want = _alphanumeric_key(key)
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute("SELECT id, name FROM users").fetchall()
    for r in rows:
        if _alphanumeric_key(r["name"]) == want:
            return int(r["id"])
    return None


def list_users_with_encodings(db_path):
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute("""
            SELECT id, name, student_id, face_encoding, user_type, expiry_datetime
            FROM users
        """).fetchall()
        users = []
        for r in rows:
            d = dict(r)
            d["face_encoding"] = json.loads(d["face_encoding"])
            users.append(d)
        return users


def add_log(db_path, name, status, student_id="", user_type="unknown", denied_reason=""):
    """
    Persist one access event to SQLite.
    This is the ONLY source of truth — the frontend must fetch /logs on load
    to restore history across restarts.
    """
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO logs (name, student_id, user_type, timestamp, status, denied_reason)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, student_id, user_type, _utc_now_iso(), status, denied_reason or ""))


def list_logs(db_path, limit=500):
    """
    Return up to `limit` most-recent log entries, newest first.
    Frontend calls GET /logs on page load to rehydrate history.
    """
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute("""
            SELECT id, name, student_id, user_type, timestamp, status, denied_reason
            FROM logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def update_user_face_encoding(db_path, user_id, encoding):
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET face_encoding = ? WHERE id = ?",
            (json.dumps(encoding), user_id),
        )


def delete_user(db_path, user_id):
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return cur.rowcount > 0


def list_users(db_path):
    with get_connection(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute("""
            SELECT id, name, student_id, user_type, expiry_datetime
            FROM users
            ORDER BY id DESC
        """).fetchall()
        return [dict(r) for r in rows]