"""
memory.py — persistent conversation memory for LINGO.

Production-beta default: SQLite, safe for concurrent calls on a single worker.
Legacy JSON mode is still available with MEMORY_BACKEND=json.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

MAX_TURNS: int = int(os.getenv("MAX_TURNS", "10"))
MEMORY_BACKEND: str = os.getenv("MEMORY_BACKEND", "sqlite").lower()
MEMORY_FILE: str = os.getenv("MEMORY_FILE", "conversation_memory.json")
MEMORY_DB: str = os.getenv("MEMORY_DB", "/data/lingo_memory.sqlite3")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    path = Path(MEMORY_DB)
    _ensure_parent(path)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS caller_memory (
            phone TEXT PRIMARY KEY,
            turns_json TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS call_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    return conn


def _load_json_store() -> Dict[str, List[Dict[str, str]]]:
    path = Path(MEMORY_FILE)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read memory file %s: %s — starting fresh.", MEMORY_FILE, exc)
        return {}


def _save_json_store(store: Dict[str, List[Dict[str, str]]]) -> None:
    path = Path(MEMORY_FILE)
    tmp_path = path.with_suffix(".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2, ensure_ascii=False)
        tmp_path.replace(path)
    except OSError as exc:
        logger.error("Failed to write memory file %s: %s", MEMORY_FILE, exc)


def load(phone: str) -> List[Dict[str, str]]:
    """Return stored conversation messages for a caller."""
    if MEMORY_BACKEND == "json":
        return _load_json_store().get(phone, [])

    with _connect() as conn:
        row = conn.execute(
            "SELECT turns_json FROM caller_memory WHERE phone = ?",
            (phone,),
        ).fetchone()
    if row is None:
        return []
    try:
        data = json.loads(row[0])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        logger.warning("Corrupt memory payload for %s — returning empty history.", phone)
        return []


def save(phone: str, turns: List[Dict[str, str]]) -> None:
    """Persist caller messages, trimmed to MAX_TURNS conversation turns."""
    trimmed = turns[-(MAX_TURNS * 2):]

    if MEMORY_BACKEND == "json":
        store = _load_json_store()
        store[phone] = trimmed
        _save_json_store(store)
        return

    payload = json.dumps(trimmed, ensure_ascii=False)
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO caller_memory(phone, turns_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(phone) DO UPDATE SET
                turns_json = excluded.turns_json,
                updated_at = excluded.updated_at
            """,
            (phone, payload, now),
        )


def clear(phone: str) -> None:
    """Delete all stored history for a caller."""
    if MEMORY_BACKEND == "json":
        store = _load_json_store()
        if phone in store:
            del store[phone]
            _save_json_store(store)
        return

    with _connect() as conn:
        conn.execute("DELETE FROM caller_memory WHERE phone = ?", (phone,))


def all_callers() -> List[str]:
    """Return all callers with stored history."""
    if MEMORY_BACKEND == "json":
        return list(_load_json_store().keys())

    with _connect() as conn:
        rows = conn.execute(
            "SELECT phone FROM caller_memory ORDER BY updated_at DESC"
        ).fetchall()
    return [row[0] for row in rows]


def log_event(phone: str, event_type: str, payload: Dict[str, Any] | None = None) -> None:
    """Append an operational event for dashboards/debugging."""
    if MEMORY_BACKEND == "json":
        return

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO call_events(phone, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (phone, event_type, json.dumps(payload or {}, ensure_ascii=False), time.time()),
        )


def recent_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent call events for the health/status API."""
    if MEMORY_BACKEND == "json":
        return []

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT phone, event_type, payload_json, created_at
            FROM call_events
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    events: List[Dict[str, Any]] = []
    for phone, event_type, payload_json, created_at in rows:
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            payload = {}
        events.append(
            {
                "phone": phone,
                "event_type": event_type,
                "payload": payload,
                "created_at": created_at,
            }
        )
    return events
