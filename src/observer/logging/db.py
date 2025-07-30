import sqlite3

def init_db(path="data/sessions.sqlite"):
    conn = sqlite3.connect(path)
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
