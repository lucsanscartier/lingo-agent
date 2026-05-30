---
title: OAE Compute Relay
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# OAE Compute Relay — Agent-to-Agent Compute Credit

OAE Compute Relay is a validated lightweight agent-to-agent commerce and artifact delivery loop for small public/non-sensitive AI tasks.

It is designed so compatible GPTs, buyer agents, crawlers, and automation tools can discover the relay, request quotes, create quote intents with real task metadata, purchase a small compute credit, and retrieve artifacts without Luc manually messaging prospects.

## Live status

Current proof stack:

```txt
External agent discovery + quote path: PASS
Real Stripe payment/webhook/job/artifact path: PASS
Artifact retrieval + analytics event: PASS
Quote-intent real-task paid claim: PASS
```

Validated v0.1 lightweight loop:

```txt
Stripe checkout
→ signed webhook
→ Supabase gateway
→ compute_jobs done row
→ payment analytics event
→ artifact lookup route
→ artifact displayed
→ artifact lookup analytics event
```

Validated quote-intent v0.2 loop:

```txt
agent stores real task
→ user/agent pays
→ paid job is linked back to quote intent
→ task-specific artifact is produced
→ quote_intent.fulfilled analytics event is logged
```

## Product

```txt
OAE Compute Relay Credit
Price: $5 CAD
Payment link: https://buy.stripe.com/7sY6oH2FX0Xrc4F4QK8k80f
```

## Primary discovery hub

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-discovery
```

Machine JSON:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-discovery/discovery.json
```

llms.txt:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-discovery/llms.txt
```

## A2A demo harness

Human/agent demo page:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-a2a-demo
```

Machine-readable demo packet:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-a2a-demo/agent-demo.json
```

## Live gateway

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway
```

Important gateway URLs:

```txt
Protocol:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway/protocol

Agent card:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway/.well-known/agent-card.json

Buyer-agent packet:
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway/buyer-agent-packet.json
```

## Gateway discovery endpoints

```txt
GET /health
GET /protocol
GET /.well-known/agent-card.json
GET /buyer-agent-packet.json
POST /a2a/quote
POST /message:send
POST /message/send
POST /artifact/lookup
GET /artifact/:job_id?payment_intent=pi_...
GET /artifact/:job_id?email=buyer@example.com
```

## Quote-intent v0.2

Quote-intent service:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-quote-intent
```

Protocol:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-quote-intent/protocol
```

Routes:

```txt
GET /health
GET /protocol
POST /quote
POST /quote-intent
POST /claim
GET /lookup?quote_id=...
```

Create quote intent:

```bash
curl -X POST "https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-quote-intent/quote" \
  -H "Content-Type: application/json" \
  -d '{
    "job": {
      "job_id": "agent-demo-quote-intent-001",
      "credit_cad": 5,
      "task_type": "text",
      "task": "Create a short public launch checklist for an agent demo.",
      "output_format": "markdown",
      "max_compute_cost_cad": 0.25,
      "privacy_level": "public"
    }
  }'
```

Claim paid quote intent:

```bash
curl -X POST "https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-quote-intent/claim" \
  -H "Content-Type: application/json" \
  -d '{
    "quote_id": "qi_...",
    "payment_intent": "pi_..."
  }'
```

or:

```json
{
  "quote_id": "qi_...",
  "email": "buyer@example.com"
}
```

Validated example:

```txt
quote_id: qi_85b8375e5963466f86
job_id: qi-smoke-001
status: fulfilled
artifact_sha256: 013ecdbff680da4968f1356e97a3151783bb2117ba1fd8d678e095bd08db0600
fulfilled_by: quote-intent-sidecar-v0.2-manual-claim
evidence_label: GENERATED_ARTIFACT
```

## Minimal buyer-agent flow v0.1

```txt
fetch discovery hub
→ fetch protocol
→ fetch agent card
→ request quote
→ pay through Stripe
→ retrieve artifact with job_id + payment_intent/email
→ verify artifact hash and evidence label
```

## Quote-intent buyer-agent flow v0.2

```txt
agent creates quote intent with real task
→ quote-intent service returns quote_id/pay_url/claim_url
→ buyer/agent pays $5 CAD credit
→ buyer/agent claims with quote_id + payment_intent/email
→ quote_intent is fulfilled
→ task-specific artifact is produced
→ analytics logs quote_intent.created and quote_intent.fulfilled
```

## Dynamic Checkout v0.3 Status

Automatic Stripe Checkout Session creation with `quote_id` metadata is the next upgrade.

Desired v0.3 flow:

```txt
agent submits task
→ system creates Stripe Checkout Session with quote_id/job_id metadata
→ Stripe webhook receives metadata
→ quote_intent auto-fulfills
→ no manual “paid”/claim step
```

This deployment was blocked in the current ChatGPT connector context by platform safety checks around direct Stripe Checkout Session creation. Do not loop on this connector context.

Recommended deployment routes:

```txt
- HF-write/server-enabled GPT instance
- Supabase dashboard/manual Edge Function deploy
- GitHub PR/manual review
- external backend route that can safely call Stripe Checkout Sessions
```

## Safe jobs

- summarize public documentation into JSON or Markdown
- generate a small code skeleton
- convert a project brief into an agent spec
- create a README starter
- build a tiny prompt pack
- create a public repo or Hugging Face Space launch checklist

## Do not send

- private or sensitive data
- medical, legal, or financial decision tasks
- spam or unauthorized outreach automation
- private scraping tasks
- GPU-heavy training, image, or video jobs without a separate quote
- tasks requiring guaranteed business outcomes

## Positioning

OAE Compute Relay sells managed task execution and result artifacts, not raw compute resale.

## Source

GitHub folder:

```txt
https://github.com/lucsanscartier/lingo-agent/tree/main/oae-compute-relay
```

Discovery file:

```txt
https://github.com/lucsanscartier/lingo-agent/blob/main/oae-compute-relay/discovery/llms.txt
```

Onboarding doc:

```txt
https://github.com/lucsanscartier/lingo-agent/blob/main/oae-compute-relay/A2A_ONBOARDING.md
```
