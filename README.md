---
title: LINGO AI Phone Agent
emoji: 📞
colorFrom: purple
colorTo: cyan
sdk: docker
pinned: true
app_port: 7860
---

# LINGO — AI Phone Agent

> Your business never stops talking.

LINGO is a beta AI phone-agent backend. It is designed to answer inbound LiveKit SIP calls, transcribe callers with Deepgram, generate concise receptionist-style replies with Hugging Face chat inference, speak responses with Hugging Face TTS, and remember callers across calls.

## Current status

**Beta, not full production yet.**

Implemented:

- LiveKit worker entrypoint for inbound audio jobs
- Deepgram streaming STT
- Hugging Face router-compatible chat completions
- Hugging Face hf-inference TTS adapter with payload fallback
- SQLite-backed caller memory
- `/health`, `/status`, and `/metrics` HTTP endpoints
- Optional escalation webhook hook
- Docker/Hugging Face Spaces deployment shape

Not fully implemented yet:

- Real SIP transfer to a human
- Calendar booking integration
- SMS follow-up
- Client dashboard
- Billing/subscription portal
- Multi-tenant client configuration
- Production privacy/compliance review

## Architecture

```text
Inbound call → LiveKit SIP
      │
      ▼
Deepgram STT
      │ transcript
      ▼
Hugging Face chat router
      │ text reply
      ▼
Hugging Face TTS
      │ audio bytes
      ▼
LiveKit audio track → caller hears LINGO
```

Runtime status:

```text
GET /health
GET /status
GET /metrics
```

## Setup

### 1. LiveKit

Create a LiveKit project and configure SIP/inbound calling. Add these secrets:

```text
LIVEKIT_URL
LIVEKIT_API_KEY
LIVEKIT_API_SECRET
```

### 2. Deepgram

Create a Deepgram API key and add:

```text
DEEPGRAM_API_KEY
```

### 3. Hugging Face

Create a fine-grained Hugging Face token with permission to make Inference Providers calls and add:

```text
HF_TOKEN
```

Default model settings:

```text
HF_CHAT_MODEL=Qwen/Qwen2.5-7B-Instruct:fastest
HF_CHAT_URL=https://router.huggingface.co/v1/chat/completions
HF_TTS_MODEL=hexgrad/Kokoro-82M
HF_TTS_URL=https://router.huggingface.co/hf-inference/models/hexgrad/Kokoro-82M
```

### 4. Memory

Default beta memory uses SQLite:

```text
MEMORY_BACKEND=sqlite
MEMORY_DB=/data/lingo_memory.sqlite3
MAX_TURNS=10
```

Legacy JSON mode remains available:

```text
MEMORY_BACKEND=json
MEMORY_FILE=conversation_memory.json
```

### 5. Run locally

```bash
cp .env.example .env
# edit .env with real secrets
docker build -t lingo-backend .
docker run --env-file .env -p 7860:7860 lingo-backend
```

Check:

```bash
curl http://localhost:7860/health
curl http://localhost:7860/status
curl http://localhost:7860/metrics
```

## Customizing the agent

Edit `prompts.py` for the receptionist personality, business FAQs, booking intake rules, and escalation behavior.

Use `ESCALATION_WEBHOOK_URL` to notify an external service when a caller asks for a human. This is not yet a full SIP transfer.

## Honest product wording

Safe beta claim:

> LINGO answers calls, remembers callers, answers known FAQs, collects messages and appointment requests, and alerts a human when escalation is needed.

Avoid claiming full appointment booking, SMS follow-up, or live transfer until those modules are implemented.

## Monetization path

Starter beta package:

- 1 phone line
- AI receptionist
- caller memory
- message capture
- appointment request capture
- owner escalation alerts
- weekly call summary

Next production upgrades:

- SIP transfer
- SMS follow-up
- Google Calendar booking
- CRM/customer dashboard
- billing
- multi-client tenant config
- privacy/security review
