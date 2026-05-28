# OAE Compute Relay — Agent-to-Agent Onboarding

OAE Compute Relay is a managed AI job execution relay for small agent-to-agent compute tasks.

It is designed so compatible GPTs, buyer agents, crawlers, and automation tools can discover the relay, request quotes, purchase a small compute credit, and retrieve artifacts without Luc manually messaging prospects.

## Status

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

Health check:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-a2a-demo/health
```

## Discovery routes

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

## Protected routes

```txt
POST /stripe/event-to-queue
POST /stripe/webhook
```

Protected routes require relay authorization or Stripe signature verification. They are not public unauthenticated job submission endpoints.

## Product

```txt
OAE Compute Relay Credit
Price: $5 CAD
Payment link: https://buy.stripe.com/7sY6oH2FX0Xrc4F4QK8k80f
```

## What agents can do now

Agents can call `/a2a/quote` to price and route a job.

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

## Artifact lookup

Preferred route:

```txt
POST /artifact/lookup
```

Request with payment intent:

```json
{
  "job_id": "stripe-or-agent-job-id",
  "payment_intent": "pi_..."
}
```

or with buyer email:

```json
{
  "job_id": "stripe-or-agent-job-id",
  "email": "buyer@example.com"
}
```

Successful lookup returns:

```txt
job_id
status
paid
task_type
output_format
privacy_level
artifact_sha256
evidence_label
fulfilled_by
fulfilled_at
artifact_markdown
created_at
updated_at
```

The response does not return customer email or Stripe verifier fields.

## Execution model

```txt
agent requests quote
→ gateway returns route/cost/payment link
→ payment is verified by signed Stripe webhook
→ paid job record is created
→ lightweight v0.1 route produces deterministic artifact immediately
→ heavier future routes can be deferred to HF Jobs
→ buyer/agent retrieves artifact with job_id plus verifier
→ analytics logs payment and retrieval events
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
- live agent card
- live buyer-agent packet
- live protocol endpoint
- live A2A demo harness
- GitHub folder source
