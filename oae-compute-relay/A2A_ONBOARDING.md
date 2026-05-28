# OAE Compute Relay — Agent-to-Agent Onboarding

## Live gateway

Base URL:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway
```

## Public routes

These are open for agent discovery and quoting:

```txt
GET  /health
GET  /protocol
POST /quote
POST /a2a/quote
```

## Protected routes

These require `x-oae-relay-secret` and are not public submission routes:

```txt
POST /stripe/event-to-queue
POST /stripe/webhook
```

Protected routes use a temporary bootstrap secret in the deployed gateway until proper Supabase environment secrets are configured.

## Product

```txt
OAE Compute Relay Credit
Price: $5 CAD
Payment link: https://buy.stripe.com/7sY6oH2FX0Xrc4F4QK8k80f
```

## What agents can do now

Agents can call `/a2a/quote` to price/route a job.

Example request:

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

Example response shape:

```json
{
  "status": "quoted",
  "quote": {
    "job_id": "agent-quote-001",
    "task_type": "text",
    "route": "template-local",
    "estimated_cost_cad": 0.001,
    "credit_cad": 5,
    "estimated_margin_cad": 4.999,
    "allowed_to_run": false,
    "risk_flags": ["UNPAID_JOB"],
    "pay_url": "https://buy.stripe.com/7sY6oH2FX0Xrc4F4QK8k80f"
  },
  "protocol": "OAE_A2A_COMPUTE_RELAY"
}
```

## Execution model

```txt
agent requests quote
→ gateway returns route/cost/payment link
→ payment is verified by trusted control plane/webhook
→ paid queue record is created
→ HF Jobs relay executes the compute
→ artifact + ledger hash returned
```

## Guardrails

- Do not frame this as raw compute resale.
- We sell completed managed job execution / artifacts.
- GPU or heavy jobs require deposit and explicit margin check.
- Private/sensitive data requires a privacy route.
- Stripe webhook verification is not complete until `STRIPE_WEBHOOK_SECRET` is configured.
- Temporary bootstrap secret must be rotated into Supabase environment secrets before broad public use.

## Current status

- Public quote API: live and tested.
- Protected queue/write API: deployed and secret-gated.
- HF Jobs relay: tested with generated queue.
- Full production webhook verification: TODO.
