# OAE Compute Relay v0.1

This is the canonical compute-selling path. It replaces the earlier report-first framing.

## Product

**OAE Compute Relay Credit**

- Stripe product: `prod_Ub2m82KsbFXy3T`
- Stripe price: `price_1TbqgzFRuWOMBXuRYwxGaDQW`
- Stripe payment link: `https://buy.stripe.com/7sY6oH2FX0Xrc4F4QK8k80f`
- Price: `$5 CAD`

## What we sell

We sell **managed AI compute/job execution**, not raw free compute resale.

```txt
customer/agent pays
→ submits task
→ OAE quotes/routes job
→ cheapest allowed compute executes
→ result artifact + ledger hash returned
```

## Core rule

Do not sell platform compute directly. Sell completed work performed by the relay.

## Modes

- `quote` — estimate route, risk, and price class
- `run` — run a paid job from JSON input
- `queue` — process paid jobs from queue JSON
- `ledger` — print ledger summary

## Compute routes

Initial routes:

- `template-local` — deterministic low-cost logic
- `hf-jobs-cpu-basic` — Hugging Face Jobs CPU
- `hf-search` — Hugging Face model/dataset/Space discovery
- `future-hf-inference` — gated behind token/cost caps
- `future-gpu` — only after paid deposit and margin check

## Guardrails

- No unpaid heavy compute.
- No spam/outreach automation.
- No guaranteed profit claims.
- No raw compute resale framing.
- Evidence Firewall labels are required.
- Jobs that involve private/sensitive data must declare privacy route before execution.

## Example job

```json
{
  "job_id": "compute-demo-001",
  "paid": true,
  "credit_cad": 5,
  "task_type": "text",
  "task": "Create a 500-word product summary from this brief.",
  "output_format": "markdown",
  "max_compute_cost_cad": 0.25
}
```

## Next steps

1. Connect Stripe payment confirmation to queue creation.
2. Add an intake form or agent-facing JSON endpoint.
3. Resume HF scheduled worker only after budget guardrails are final.
4. Add delivery sink: Drive/Gmail/GitHub artifact page.
5. Add cost/margin dashboard.
