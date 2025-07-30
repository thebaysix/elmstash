import sqlite3
from datetime import datetime
import json
import uuid
import os

DB_PATH = os.getenv("DB_PATH", "data/sessions.sqlite")

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            session_id TEXT,
            step INTEGER,
            timestamp TEXT,
            input TEXT,
            action TEXT,
            output TEXT,
            metadata TEXT
        )
    ''')
    conn.commit()
    return conn

def log_interaction(conn, session_id, step, input_str, action, output_str, metadata=None):
    cursor = conn.cursor()
    timestamp = datetime.utcnow().isoformat()
    meta_json = json.dumps(metadata or {})
    cursor.execute('''
        INSERT INTO interactions (session_id, step, timestamp, input, action, output, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, step, timestamp, input_str, action, output_str, meta_json))
    conn.commit()
