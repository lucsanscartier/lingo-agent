"""
health_server.py — tiny FastAPI status surface for Hugging Face Spaces.

This lets the Space expose /health while the LiveKit worker runs in the same
container. It intentionally avoids exposing secrets or caller transcripts.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI

import memory

logger = logging.getLogger(__name__)

STARTED_AT = time.time()
STATE: Dict[str, Any] = {
    "worker_started": False,
    "last_error": None,
    "active_calls": 0,
    "calls_started": 0,
    "calls_ended": 0,
    "escalations": 0,
}

app = FastAPI(title="LINGO Runtime", version="0.2.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "lingo-agent",
        "version": "0.2.0",
        "uptime_seconds": round(time.time() - STARTED_AT, 2),
        "worker_started": STATE["worker_started"],
    }


@app.get("/status")
def status() -> Dict[str, Any]:
    return {
        "ok": True,
        "state": STATE,
        "callers_known": len(memory.all_callers()),
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    return {
        "ok": True,
        "calls_started": STATE["calls_started"],
        "calls_ended": STATE["calls_ended"],
        "active_calls": STATE["active_calls"],
        "escalations": STATE["escalations"],
        "recent_events": memory.recent_events(limit=20),
    }


def mark_worker_started() -> None:
    STATE["worker_started"] = True


def set_last_error(error: str | None) -> None:
    STATE["last_error"] = error


def record_call_start() -> None:
    STATE["calls_started"] += 1
    STATE["active_calls"] += 1


def record_call_end() -> None:
    STATE["calls_ended"] += 1
    STATE["active_calls"] = max(0, STATE["active_calls"] - 1)


def record_escalation() -> None:
    STATE["escalations"] += 1


def start_health_server(host: str = "0.0.0.0", port: int = 7860) -> threading.Thread:
    """Start the HTTP health server in a daemon thread."""
    def _run() -> None:
        logger.info("Starting health server on %s:%s", host, port)
        uvicorn.run(app, host=host, port=port, log_level="info")

    thread = threading.Thread(target=_run, name="lingo-health-server", daemon=True)
    thread.start()
    return thread
