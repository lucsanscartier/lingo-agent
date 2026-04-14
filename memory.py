"""
memory.py — Persistent conversation memory for LINGO.

Stores per-caller conversation history in a local JSON file keyed by
phone number.  Keeps the last MAX_TURNS turns so the LLM context stays
manageable.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

# Maximum number of conversation turns (one turn = one user message + one
# assistant reply) to keep per caller.
MAX_TURNS: int = 10

# Path to the JSON file used as the backing store.
MEMORY_FILE: str = os.getenv("MEMORY_FILE", "conversation_memory.json")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_store() -> Dict[str, List[Dict[str, str]]]:
    """Read the entire memory store from disk.  Returns an empty dict on
    first run or if the file is corrupted."""
    path = Path(MEMORY_FILE)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read memory file %s: %s — starting fresh.", MEMORY_FILE, exc)
        return {}


def _save_store(store: Dict[str, List[Dict[str, str]]]) -> None:
    """Atomically write the full memory store back to disk."""
    path = Path(MEMORY_FILE)
    tmp_path = path.with_suffix(".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2, ensure_ascii=False)
        tmp_path.replace(path)
    except OSError as exc:
        logger.error("Failed to write memory file %s: %s", MEMORY_FILE, exc)


# ── Public API ────────────────────────────────────────────────────────────────

def load(phone: str) -> List[Dict[str, str]]:
    """Return the stored conversation turns for *phone*.

    Each turn is a dict with keys ``"role"`` (``"user"`` or ``"assistant"``)
    and ``"content"`` (the message text).  Returns an empty list if there
    is no history for this caller.

    Args:
        phone: The caller's E.164 phone number, e.g. ``"+14155551234"``.

    Returns:
        A list of message dicts compatible with the OpenAI-style chat format.
    """
    store = _load_store()
    history = store.get(phone, [])
    logger.debug("Loaded %d turns for %s", len(history), phone)
    return history


def save(phone: str, turns: List[Dict[str, str]]) -> None:
    """Persist *turns* for *phone*, trimming to the last MAX_TURNS entries.

    Args:
        phone: The caller's E.164 phone number.
        turns: The full (or updated) list of message dicts to store.
    """
    # Keep only the most recent MAX_TURNS turns (each turn is 2 messages)
    trimmed = turns[-(MAX_TURNS * 2):]
    store = _load_store()
    store[phone] = trimmed
    _save_store(store)
    logger.debug("Saved %d messages for %s", len(trimmed), phone)


def clear(phone: str) -> None:
    """Delete all stored history for *phone*.  No-op if not found.

    Args:
        phone: The caller's E.164 phone number.
    """
    store = _load_store()
    if phone in store:
        del store[phone]
        _save_store(store)
        logger.info("Cleared history for %s", phone)


def all_callers() -> List[str]:
    """Return a list of all phone numbers that have stored history."""
    store = _load_store()
    return list(store.keys())
