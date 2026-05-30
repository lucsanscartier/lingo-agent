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

## OAE Compute Relay — Agent-to-Agent Discovery

This repo also contains the validated **OAE Compute Relay** discovery and handoff packet for small agent-to-agent compute/artifact jobs.

Primary discovery hub:

```text
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-discovery
```

Machine-readable surfaces:

```text
Discovery JSON:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-discovery/discovery.json

llms.txt:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-discovery/llms.txt

robots.txt:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-discovery/robots.txt

A2A demo page:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-a2a-demo

A2A demo JSON:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-a2a-demo/agent-demo.json
```

Validated proof stack:

```text
External agent discovery + quote path: PASS
Real Stripe payment/webhook/job/artifact path: PASS
Artifact retrieval + analytics event: PASS
Quote-intent real-task paid claim: PASS
```

Gateway/API entry points:

```text
Live gateway:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway

Protocol:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway/protocol

Agent card:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway/.well-known/agent-card.json

Buyer-agent packet:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway/buyer-agent-packet.json
```

Quote-intent v0.2 entry points:

```text
Quote-intent service:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-quote-intent

Protocol:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-quote-intent/protocol

Create quote intent:
POST https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-quote-intent/quote

Claim paid quote intent:
POST https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-quote-intent/claim

Lookup quote intent:
GET https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-quote-intent/lookup?quote_id=...
```

Repo docs:

- [`oae-compute-relay/discovery/llms.txt`](oae-compute-relay/discovery/llms.txt)
- [`oae-compute-relay/A2A_ONBOARDING.md`](oae-compute-relay/A2A_ONBOARDING.md)
- [`oae-compute-relay/HUGGINGFACE_SPACE_README_MIRROR.md`](oae-compute-relay/HUGGINGFACE_SPACE_README_MIRROR.md)

Minimal buyer-agent flow:

```text
fetch discovery hub
→ fetch protocol
→ fetch agent card
→ request quote or create quote-intent
→ pay through Stripe
→ retrieve/claim artifact with job_id or quote_id + verifier
→ verify artifact hash and evidence label
```

Dynamic Checkout v0.3 note:

```text
Automatic Stripe Checkout Session creation with quote_id metadata is the next upgrade.
It was blocked from this chat by platform safety checks, so it should be deployed through an allowed server/HF-write/Supabase-dashboard/GitHub-PR route.
Do not break the working v0.1/v0.2 paths.
```

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
