"""Simple SQLite storage for uploaded meetings and results."""
import sqlite3
from typing import Optional, Dict, Any
from pathlib import Path

DB_PATH = Path(__file__).parent / 'meetings.db'


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS meetings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        transcript TEXT,
        summary TEXT,
        actions TEXT
    )
    ''')
    conn.commit()
    conn.close()


def create_meeting(filename: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('INSERT INTO meetings (filename) VALUES (?)', (filename,))
    mid = cur.lastrowid
    conn.commit()
    conn.close()
    return mid


def update_transcript(mid: int, transcript: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('UPDATE meetings SET transcript = ? WHERE id = ?', (transcript, mid))
    conn.commit()
    conn.close()


def update_summary_actions(mid: int, summary: str, actions: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('UPDATE meetings SET summary = ?, actions = ? WHERE id = ?', (summary, actions, mid))
    conn.commit()
    conn.close()


def get_meeting(mid: int) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, filename, uploaded_at, transcript, summary, actions FROM meetings WHERE id = ?', (mid,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return dict(id=row[0], filename=row[1], uploaded_at=row[2], transcript=row[3], summary=row[4], actions=row[5])


def list_meetings() -> list:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, filename, uploaded_at, summary FROM meetings ORDER BY uploaded_at DESC')
    rows = cur.fetchall()
    conn.close()
    return [{'id': r[0], 'filename': r[1], 'uploaded_at': r[2], 'summary': r[3]} for r in rows]
