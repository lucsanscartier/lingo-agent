---
title: LINGO AI Phone Agent
emoji: 📞
colorFrom: purple
colorTo: cyan
sdk: docker
pinned: true
app_port: 8080
---

# LINGO — AI Phone Agent

> *Your business never stops talking.*

LINGO is a fully autonomous AI phone agent. Give your business a real phone number. LINGO answers every call 24/7, remembers every caller, handles FAQs, books appointments, sends follow-up texts, and escalates to a human when needed.

---

## Architecture

```
Inbound call → LiveKit SIP
      │
      ▼
Deepgram STT  (streaming, real-time transcription)
      │ transcript
      ▼
Qwen2.5-7B-Instruct  (HF Serverless Inference — free)
      │ text reply
      ▼
Kokoro-82M TTS  (HF Serverless Inference — free)
      │ audio frames
      ▼
LiveKit audio track → caller hears LINGO
```

**Memory:** Kirk persistent memory engine — every caller is remembered by phone number across all calls, forever.

---

## Setup (Free — No Credit Card Required)

### 1. LiveKit (free phone number + real-time audio)
- Sign up at [livekit.io](https://livekit.io) — free tier, no CC
- Create a project → copy `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- In LiveKit console → SIP → buy a free US phone number (50 inbound mins/month free)

### 2. Deepgram (speech-to-text)
- Sign up at [deepgram.com](https://deepgram.com) — $200 free credits (~433 hours)
- Create an API key → copy `DEEPGRAM_API_KEY`

### 3. Hugging Face Token
- Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Create a read-scope token → copy as `HF_TOKEN`

### 4. Set Secrets in HF Spaces
In this Space → Settings → Repository Secrets, add:
```
LIVEKIT_URL
LIVEKIT_API_KEY
LIVEKIT_API_SECRET
DEEPGRAM_API_KEY
HF_TOKEN
```

### 5. Restart the Space
The agent starts automatically and connects to your LiveKit project. When a call comes in, LINGO answers.

---

## Customizing Your Agent

Edit `prompts.py` to change:
- **Business name and description** — what LINGO knows about your business
- **FAQs** — hours, services, pricing, location
- **Booking flow** — what info to collect for appointments
- **Escalation trigger** — when to hand off to a human

---

## Pricing (LINGO as a Service)

| Plan | Price | Includes |
|------|-------|---------|
| Starter | $29/month | 1 agent line, 500 call minutes, SMS, memory |
| Pro | $79/month | 3 lines, 2000 minutes, custom voice |
| Agency | $199/month | 10 lines, white label, API access |

**→ [Get your agent line](https://lingo.lemonsqueezy.com)**

---

## Built By

Luc — solo founder, Edmonton. Built overnight using free tiers and AI tools.  
Powered by the **Kirk memory architecture** — persistent, hyperbolic, non-linear.

---

## Stack
- [LiveKit Agents](https://github.com/livekit/agents) — real-time voice pipeline
- [Deepgram](https://deepgram.com) — speech-to-text
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) — LLM brain
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — text-to-speech
- Kirk Memory Engine — persistent caller memory
