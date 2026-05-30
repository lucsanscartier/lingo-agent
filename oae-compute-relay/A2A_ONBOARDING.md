# OAE Compute Relay — Agent-to-Agent Onboarding

OAE Compute Relay is a managed AI job execution relay for small agent-to-agent compute tasks.

It is designed so compatible GPTs, buyer agents, crawlers, and automation tools can discover the relay, request quotes, purchase a small compute credit, store real task metadata, and retrieve or claim artifacts without Luc manually messaging prospects.

## Current Proof Stack

```txt
External agent discovery + quote path: PASS
Real Stripe payment/webhook/job/artifact path: PASS
Artifact retrieval + analytics event: PASS
Quote-intent real-task paid claim: PASS
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

## Live gateway

Base URL:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway
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

## Product

```txt
OAE Compute Relay Credit
Price: $5 CAD
Payment link: https://buy.stripe.com/7sY6oH2FX0Xrc4F4QK8k80f
```

## Gateway discovery routes

```txt
GET /health
GET /protocol
GET /.well-known/agent-card.json
GET /buyer-agent-packet.json
POST /quote
POST /a2a/quote
POST /message:send
POST /message/send
POST /artifact/lookup
GET /artifact/:job_id?payment_intent=pi_...
GET /artifact/:job_id?email=buyer@example.com
```

## Protected gateway routes

```txt
POST /stripe/event-to-queue
POST /stripe/webhook
```

Protected routes require relay authorization or Stripe signature verification. They are not public unauthenticated job submission endpoints.

## Quote-intent v0.2

The quote-intent sidecar stores the real task before payment and can later claim a paid job into a task-specific artifact.

Base URL:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-quote-intent
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

Validated example:

```txt
quote_id: qi_85b8375e5963466f86
job_id: qi-smoke-001
status: fulfilled
artifact_sha256: 013ecdbff680da4968f1356e97a3151783bb2117ba1fd8d678e095bd08db0600
fulfilled_by: quote-intent-sidecar-v0.2-manual-claim
evidence_label: GENERATED_ARTIFACT
```

### Quote-intent request

```json
{
  "job": {
    "job_id": "agent-demo-quote-intent-001",
    "credit_cad": 5,
    "task_type": "text",
    "task": "Create a short public launch checklist for an agent demo.",
    "output_format": "markdown",
    "max_compute_cost_cad": 0.25,
    "privacy_level": "public"
  }
}
```

### Quote-intent response shape

```json
{
  "status": "quoted",
  "quote_intent": {
    "quote_id": "qi_...",
    "job_id": "agent-demo-quote-intent-001",
    "task_type": "text",
    "output_format": "markdown",
    "privacy_level": "public",
    "credit_cad": 5,
    "estimated_cost_cad": 0.001,
    "estimated_margin_cad": 4.999,
    "pay_url": "https://buy.stripe.com/7sY6oH2FX0Xrc4F4QK8k80f",
    "claim_url": "https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-quote-intent/claim"
  }
}
```

### Quote-intent claim

```json
{
  "quote_id": "qi_...",
  "payment_intent": "pi_..."
}
```

or:

```json
{
  "quote_id": "qi_...",
  "email": "buyer@example.com"
}
```

Successful claim returns:

```txt
status
quote_id
job_id
paid_job_id
artifact_sha256
evidence_label
fulfilled_by
artifact_markdown
```

## Minimal buyer-agent flow v0.1

```txt
agent requests quote
→ gateway returns route/cost/payment link
→ payment is verified by signed Stripe webhook
→ paid job record is created
→ lightweight v0.1 route produces deterministic artifact immediately
→ buyer/agent retrieves artifact with job_id plus verifier
→ analytics logs payment and retrieval events
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

## Guardrails

- Do not frame this as raw compute resale.
- OAE sells completed managed job execution and artifacts.
- GPU or heavy jobs require deposit and explicit margin check.
- Do not send private or sensitive data to the lightweight public route.
- Do not use this for medical, legal, or financial decision tasks.
- Do not use this for spam, private scraping, or guaranteed business outcomes.

## Safe example tasks

- summarize public documentation into JSON or Markdown
- generate a small code skeleton
- convert a product brief into an agent spec
- create a README starter
- build a tiny prompt pack
- create a public repo or Hugging Face Space launch checklist

## Agent discovery surfaces

- `oae-compute-relay/discovery/llms.txt`
- live discovery hub
- live agent card
- live buyer-agent packet
- live protocol endpoint
- live A2A demo harness
- live quote-intent sidecar
- GitHub folder source
