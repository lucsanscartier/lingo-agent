# OAE Compute Relay

Managed AI job runs for agents and builders.

OAE Compute Relay lets an external agent or human builder request a quote for a small AI task, receive a route/cost/risk estimate, buy a compute credit, and route the paid job into a managed execution flow.

## Live public gateway

Base URL:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway
```

Public routes:

```txt
GET /health
GET /protocol
POST /a2a/quote
```

Protected routes:

```txt
POST /stripe/event-to-queue
POST /stripe/webhook
```

## Product

```txt
OAE Compute Relay Credit
Price: $5 CAD
Payment link: https://buy.stripe.com/7sY6oH2FX0Xrc4F4QK8k80f
```

## What it sells

We sell managed job execution and result artifacts, not raw platform compute resale.

```txt
agent asks for quote
→ gateway returns route, cost, risk, and payment link
→ payment is verified by trusted control plane or webhook
→ queue record is created
→ Hugging Face Jobs relay executes
→ artifact and hash are returned
```

## Example quote request

```json
{
  "job": {
    "job_id": "agent-quote-001",
    "paid": false,
    "credit_cad": 5,
    "task_type": "text",
    "task": "Summarize this public documentation into JSON.",
    "output_format": "json",
    "max_compute_cost_cad": 0.25,
    "privacy_level": "public"
  }
}
```

## Guardrails

- No raw compute resale framing.
- No unpaid heavy jobs.
- No guaranteed profit claims.
- GPU/heavy jobs require a separate quote or deposit.
- Private or sensitive data requires privacy review.
- Protected queue/payment routes require relay authorization.

## Status

- Public A2A quote API: live and tested.
- Payment credit: live.
- Hugging Face Jobs relay: tested.
- Protected queue routes: deployed and secret-gated.
- Production Stripe signature verification: still pending.
