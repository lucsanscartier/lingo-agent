#!/usr/bin/env python3
"""
hf_space_diagnostic.py — Hugging Face Space deployment probe for LINGO.

Run this from a local terminal, Codespace, or Hugging Face runtime that has
network access. It prints a sanitized JSON report and never prints token values.

Example:
  HF_OWNER=lucsanscartier HF_SPACE=lingo-agent python tools/hf_space_diagnostic.py

Optional:
  HF_TOKEN=... python tools/hf_space_diagnostic.py --test-inference
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests


DEFAULT_OWNER = "lucsanscartier"
DEFAULT_SPACE = "lingo-agent"
DEFAULT_TIMEOUT = 20


REQUIRED_SECRET_NAMES = [
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "DEEPGRAM_API_KEY",
    "HF_TOKEN",
]


def mask_status(ok: bool, detail: str = "") -> Dict[str, Any]:
    return {"ok": ok, "detail": detail}


def auth_headers(token: str | None) -> Dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def get_json(url: str, token: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    started = time.time()
    try:
        response = requests.get(url, headers=auth_headers(token), timeout=timeout)
        elapsed = round(time.time() - started, 3)
        content_type = response.headers.get("content-type", "")
        body: Any
        if "application/json" in content_type:
            body = response.json()
        else:
            body = response.text[:800]
        return {
            "ok": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "elapsed_seconds": elapsed,
            "content_type": content_type,
            "body": body,
        }
    except requests.RequestException as exc:
        return {
            "ok": False,
            "status_code": None,
            "elapsed_seconds": round(time.time() - started, 3),
            "content_type": None,
            "body": f"request_error: {type(exc).__name__}: {exc}",
        }


def post_json(url: str, payload: Dict[str, Any], token: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    started = time.time()
    headers = {"Content-Type": "application/json", **auth_headers(token)}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed = round(time.time() - started, 3)
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            body: Any = response.json()
        elif "audio" in content_type or "octet-stream" in content_type:
            body = {"audio_bytes": len(response.content)}
        else:
            body = response.text[:800]
        return {
            "ok": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "elapsed_seconds": elapsed,
            "content_type": content_type,
            "body": body,
        }
    except requests.RequestException as exc:
        return {
            "ok": False,
            "status_code": None,
            "elapsed_seconds": round(time.time() - started, 3),
            "content_type": None,
            "body": f"request_error: {type(exc).__name__}: {exc}",
        }


def infer_404_cause(repo_result: Dict[str, Any], health_result: Dict[str, Any]) -> str:
    repo_code = repo_result.get("status_code")
    health_code = health_result.get("status_code")

    if repo_code == 404:
        return "Space repo is not publicly visible, private without token access, wrong owner/name, or does not exist."
    if repo_result.get("ok") and health_code == 404:
        return "Space exists, but live app URL is wrong, app is private/protected routing is not ready, build has not completed, or app_port/runtime is mismatched."
    if repo_result.get("ok") and health_code in {502, 503, 504}:
        return "Space exists but the app is sleeping, building, crashed, or not listening on the expected port."
    if repo_result.get("ok") and health_result.get("ok"):
        return "No 404. Space and health endpoint are reachable."
    return "Unknown from HTTP probes alone. Check Space build/runtime logs."


def test_chat(token: str, model: str, url: str) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say LINGO_OK in one word."}],
        "max_tokens": 16,
        "temperature": 0,
    }
    return post_json(url, payload, token=token, timeout=45)


def test_tts(token: str, url: str) -> Dict[str, Any]:
    modern = post_json(url, {"text_inputs": "LINGO test."}, token=token, timeout=75)
    if modern.get("ok"):
        return {"preferred_payload": "text_inputs", "result": modern}
    legacy = post_json(url, {"inputs": "LINGO test."}, token=token, timeout=75)
    return {"preferred_payload": "inputs" if legacy.get("ok") else None, "modern": modern, "legacy": legacy}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe LINGO Hugging Face Space deployment.")
    parser.add_argument("--owner", default=os.getenv("HF_OWNER", DEFAULT_OWNER))
    parser.add_argument("--space", default=os.getenv("HF_SPACE", DEFAULT_SPACE))
    parser.add_argument("--token", default=os.getenv("HF_TOKEN", ""))
    parser.add_argument("--test-inference", action="store_true")
    parser.add_argument("--chat-model", default=os.getenv("HF_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct:fastest"))
    parser.add_argument("--chat-url", default=os.getenv("HF_CHAT_URL", "https://router.huggingface.co/v1/chat/completions"))
    parser.add_argument("--tts-url", default=os.getenv("HF_TTS_URL", "https://router.huggingface.co/hf-inference/models/hexgrad/Kokoro-82M"))
    args = parser.parse_args()

    owner = args.owner.strip()
    space = args.space.strip()
    token: Optional[str] = args.token.strip() or None

    repo_url = f"https://huggingface.co/spaces/{owner}/{space}"
    api_url = f"https://huggingface.co/api/spaces/{owner}/{space}"
    app_url = f"https://{owner}-{space}.hf.space"

    api_probe = get_json(api_url, token=token)
    repo_probe = get_json(repo_url, token=token)
    health_probe = get_json(f"{app_url}/health", token=token)
    status_probe = get_json(f"{app_url}/status", token=token) if health_probe.get("ok") else mask_status(False, "skipped because /health failed")
    metrics_probe = get_json(f"{app_url}/metrics", token=token) if health_probe.get("ok") else mask_status(False, "skipped because /health failed")

    report: Dict[str, Any] = {
        "name": "LINGO-HF-DIAGNOSTIC REPORT",
        "space": {
            "owner": owner,
            "space": space,
            "repo_url": repo_url,
            "api_url": api_url,
            "app_url": app_url,
        },
        "access": {
            "hf_token_present": bool(token),
            "required_secret_names": REQUIRED_SECRET_NAMES,
            "note": "This script cannot list Space secrets. It only reports required names and runtime symptoms.",
        },
        "probes": {
            "space_api": api_probe,
            "space_repo": repo_probe,
            "health": health_probe,
            "status": status_probe,
            "metrics": metrics_probe,
        },
        "likely_404_cause": infer_404_cause(repo_probe, health_probe),
        "next_action": "Check Space visibility, exact slug, build logs, runtime logs, README sdk/app_port, and required secrets.",
    }

    if args.test_inference:
        if not token:
            report["inference"] = {"ok": False, "detail": "HF_TOKEN missing, inference tests skipped."}
        else:
            report["inference"] = {
                "chat": test_chat(token, args.chat_model, args.chat_url),
                "tts": test_tts(token, args.tts_url),
            }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    # Exit nonzero if health failed; useful in CI/diagnostic bots.
    return 0 if health_probe.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
