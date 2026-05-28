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

It is designed so compatible GPTs, buyer agents, crawlers, and automation tools can discover the relay, request quotes, purchase a small compute credit, and retrieve artifacts without Luc manually messaging prospects.

## Live status

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

## Product

```txt
OAE Compute Relay Credit
Price: $5 CAD
Payment link: https://buy.stripe.com/7sY6oH2FX0Xrc4F4QK8k80f
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

## Live gateway

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway
```

## Discovery endpoints

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

## Important URLs

Agent card:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway/.well-known/agent-card.json
```

Buyer-agent packet:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway/buyer-agent-packet.json
```

Protocol:

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway/protocol
```

## Minimal buyer-agent flow

```txt
fetch demo packet
→ fetch protocol
→ fetch agent card
→ request quote
→ pay through Stripe
→ retrieve artifact with job_id + payment_intent/email
→ verify artifact hash and evidence label
```

## Quote example

```bash
curl -X POST "https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway/a2a/quote" \
  -H "Content-Type: application/json" \
  -d '{
    "job": {
      "job_id": "agent-demo-readme-001",
      "paid": false,
      "credit_cad": 5,
      "task_type": "text",
      "task": "Create a concise README starter for a small AI compute relay microservice. Use only public/non-sensitive information.",
      "output_format": "markdown",
      "max_compute_cost_cad": 0.25,
      "privacy_level": "public"
    }
  }'
```

## Artifact lookup example

Use POST when possible so verifier data is not exposed in URLs:

```bash
curl -X POST "https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway/artifact/lookup" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "stripe-or-agent-job-id",
    "payment_intent": "pi_..."
  }'
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
