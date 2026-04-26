"""
config.py — environment-driven runtime settings for LINGO.

Keep secrets in Hugging Face Space secrets, LiveKit secrets, or a local .env
file during development. Never commit real tokens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # LiveKit / SIP
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    # Speech-to-text
    deepgram_api_key: str

    # Hugging Face
    hf_token: str
    hf_chat_model: str = "Qwen/Qwen2.5-7B-Instruct:fastest"
    hf_chat_url: str = "https://router.huggingface.co/v1/chat/completions"
    hf_tts_model: str = "hexgrad/Kokoro-82M"
    hf_tts_url: str = "https://router.huggingface.co/hf-inference/models/hexgrad/Kokoro-82M"

    # Runtime
    log_level: str = "INFO"
    sample_rate: int = 24_000
    channels: int = 1

    # Memory
    memory_backend: str = "sqlite"
    memory_db: str = "/data/lingo_memory.sqlite3"
    memory_file: str = "conversation_memory.json"
    max_turns: int = 10

    # Health/API
    health_enabled: bool = True
    health_host: str = "0.0.0.0"
    health_port: int = 7860

    # Optional hooks for production behavior
    escalation_webhook_url: str = ""
    owner_alert_email: str = ""
    business_name: str = "the business"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            livekit_url=os.getenv("LIVEKIT_URL", ""),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", ""),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY", ""),
            hf_token=os.getenv("HF_TOKEN", ""),
            hf_chat_model=os.getenv("HF_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct:fastest"),
            hf_chat_url=os.getenv("HF_CHAT_URL", "https://router.huggingface.co/v1/chat/completions"),
            hf_tts_model=os.getenv("HF_TTS_MODEL", "hexgrad/Kokoro-82M"),
            hf_tts_url=os.getenv(
                "HF_TTS_URL",
                "https://router.huggingface.co/hf-inference/models/hexgrad/Kokoro-82M",
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            sample_rate=_int("SAMPLE_RATE", 24_000),
            channels=_int("CHANNELS", 1),
            memory_backend=os.getenv("MEMORY_BACKEND", "sqlite").lower(),
            memory_db=os.getenv("MEMORY_DB", "/data/lingo_memory.sqlite3"),
            memory_file=os.getenv("MEMORY_FILE", "conversation_memory.json"),
            max_turns=_int("MAX_TURNS", 10),
            health_enabled=_bool("HEALTH_ENABLED", True),
            health_host=os.getenv("HEALTH_HOST", "0.0.0.0"),
            health_port=_int("PORT", _int("HEALTH_PORT", 7860)),
            escalation_webhook_url=os.getenv("ESCALATION_WEBHOOK_URL", ""),
            owner_alert_email=os.getenv("OWNER_ALERT_EMAIL", ""),
            business_name=os.getenv("BUSINESS_NAME", "the business"),
        )

    def missing_required(self) -> List[str]:
        required = {
            "LIVEKIT_URL": self.livekit_url,
            "LIVEKIT_API_KEY": self.livekit_api_key,
            "LIVEKIT_API_SECRET": self.livekit_api_secret,
            "DEEPGRAM_API_KEY": self.deepgram_api_key,
            "HF_TOKEN": self.hf_token,
        }
        return [name for name, value in required.items() if not value]
