# OAE Dynamic Checkout v0.3 — Deploy Handoff

This folder stages the next OAE Compute Relay upgrade. It is intentionally additive. Do **not** remove or break the working v0.1/v0.2 paths.

## Current verified stack

```txt
External agent discovery + quote path: PASS
Real Stripe payment/webhook/job/artifact path: PASS
Artifact retrieval + analytics event: PASS
Quote-intent real-task paid claim: PASS
HF Space README mirror: PASS
Dynamic Checkout v0.3: NOT DEPLOYED
```

## Goal

```txt
agent submits task
→ system creates quote_intent
→ system creates Stripe Checkout Session with quote_id/job_id metadata
→ Stripe webhook receives metadata
→ quote_intent auto-fulfills
→ no manual paid/claim step
```

## Supabase project

```txt
ubauxksvewtwwerkpbuo
```

## Existing working functions

```txt
oae-compute-relay-gateway
oae-discovery
oae-a2a-demo
oae-quote-intent
```

## Existing tables

```txt
public.compute_jobs
public.oae_relay_events
public.quote_intents
```

## Required secrets

Verify these are configured for Supabase Edge Functions before deploying:

```txt
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET
```

Do not paste secret values into chat, GitHub, Linear, or logs.

## Stripe price

Use the existing OAE Compute Relay Credit price:

```txt
price_1TbqgzFRuWOMBXuRYwxGaDQW
```

## New function target

```txt
oae-dynamic-checkout
```

Suggested deploy path:

```txt
supabase functions deploy oae-dynamic-checkout --project-ref ubauxksvewtwwerkpbuo --no-verify-jwt
```

## Routes

```txt
GET  /health
GET  /protocol
POST /create
POST /reconcile
GET  /success?quote_id=...&session_id=...
GET  /lookup?quote_id=...
```

## POST /create input

```json
{
  "job": {
    "job_id": "agent-demo-dynamic-001",
    "credit_cad": 5,
    "task_type": "text",
    "task": "Create a short public launch checklist for an agent demo.",
    "output_format": "markdown",
    "max_compute_cost_cad": 0.25,
    "privacy_level": "public"
  }
}
```

Expected output:

```txt
status: checkout_created
quote_id: qi_...
job_id: agent-demo-dynamic-001
checkout_session_id: cs_...
checkout_url: https://checkout.stripe.com/...
```

## Metadata requirements

Create the Stripe Checkout Session with:

```txt
metadata.quote_id
metadata.job_id
metadata.oae_product=compute_relay_credit
metadata.task_type
metadata.output_format
metadata.privacy_level
```

Also set PaymentIntent metadata:

```txt
payment_intent_data.metadata.quote_id
payment_intent_data.metadata.job_id
payment_intent_data.metadata.oae_product=compute_relay_credit
```

## Reconcile/fulfillment behavior

When Stripe webhook creates a paid `compute_jobs` row with a matching `job_id` or metadata-derived `quote_id`, update the matching `quote_intents` row:

```txt
status = fulfilled
stripe_payment_intent = pi_...
artifact_job_id = paid compute_jobs.job_id
artifact_sha256 = sha256(task-specific artifact_markdown)
result_payload.artifact_markdown = generated artifact using quote_intents.task
result_payload.evidence_label = GENERATED_ARTIFACT
result_payload.fulfilled_by = dynamic-checkout-sidecar-v0.3
fulfilled_at = now()
```

Log analytics:

```txt
dynamic_checkout.created
dynamic_checkout.fulfilled
```

## Test plan

```txt
1. Deploy oae-dynamic-checkout.
2. GET /health and confirm status ok.
3. POST /create with a safe public task.
4. Confirm checkout_url and quote_id returned.
5. Pay the Checkout Session.
6. Confirm Stripe webhook received quote_id/job_id metadata.
7. Confirm compute_jobs row is created as paid=true/status=done.
8. Confirm quote_intents.status becomes fulfilled automatically or after /reconcile.
9. Confirm artifact_markdown contains the original stored task.
10. Confirm oae_relay_events includes dynamic_checkout.created and dynamic_checkout.fulfilled.
11. Confirm /lookup?quote_id=... returns fulfilled artifact info.
12. Update Linear GPT-31/GPT-35 with exact session/payment/quote IDs and pass/fail status.
```

## Safety constraints

```txt
- Public/non-sensitive tasks only.
- No spam automation.
- No private scraping.
- No medical/legal/financial decision tasks.
- Heavy GPU jobs require separate quote.
- Do not call this raw compute resale; it is managed task execution and artifact delivery.
```

## Current blocker context

The normal ChatGPT connector context could not deploy direct Stripe Checkout Session creation logic due to platform safety checks. Use an allowed deploy route:

```txt
- Supabase dashboard/manual Edge Function deploy
- HF/server-enabled GPT with Supabase + Stripe access
- external backend route
- GitHub PR/manual review
```
