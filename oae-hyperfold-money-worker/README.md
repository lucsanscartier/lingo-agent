# OAE Hyperfold Money Worker v0.1

This folder is separate from LINGO. LINGO remains beta/parked.

The purpose of this module is to create an autonomous-ish artifact vending worker:

```txt
Stripe payment -> queued artifact job -> Hugging Face compute -> artifact output -> ledger/delivery
```

## Current live rail

- Stripe product: `ARCH Money Agent Starter Report`
- Stripe payment link: `https://buy.stripe.com/00wfZh3K1gWpgkVgzs8k80e`
- HF scheduled job scaffold: `6a1778d05c8d10ffa1104dd0`
- Status: scheduled but suspended

## Evidence Firewall

- VERIFIED: ChatGPT connector can create Stripe products/prices/payment links.
- VERIFIED: ChatGPT connector can create and run Hugging Face Jobs under `lucsanscartier`.
- VERIFIED: HF scheduled worker scaffold exists but is suspended.
- INFERRED: Full automation is possible once the worker has its own payment/order source.
- BLOCKER: HF Jobs do not automatically inherit ChatGPT's Stripe connector session.

## Modes

### Probe mode

No secrets required. It prints the route, validates config, and simulates one artifact job.

```bash
python worker.py --mode probe
```

### Queue mode

Reads jobs from a local JSON queue file and writes artifacts to `artifacts/`.

```bash
python worker.py --mode queue --queue queue.example.json
```

### Stripe mode

Requires a Stripe API key inside the HF Job environment:

```txt
STRIPE_SECRET_KEY
STRIPE_PAYMENT_LINK_ID=plink_1Tbq2nFRuWOMBXuRWGwCmhIg
```

Then:

```bash
python worker.py --mode stripe
```

Stripe mode is designed to poll paid checkout sessions, convert completed payments into artifact jobs, then generate outputs.

## Guardrails

- Do not run paid compute unless payment is confirmed.
- Do not send unsolicited outreach.
- Do not claim guaranteed revenue.
- Do not use GPU unless the job price/margin justifies it.
- Every artifact must include Evidence Firewall labels.

## First product

`HF Opportunity Report`

Input:

```json
{
  "idea": "I want to make money with a Hugging Face Space",
  "target_customer": "small businesses",
  "budget": "free/cheap"
}
```

Output:

- Model ideas
- Dataset ideas
- Space examples
- Build path
- Monetization path
- Risk notes
- Next upsell

## Next integration steps

1. Add Stripe API/webhook secret to the HF Job environment, or provide a webhook receiver.
2. Add an intake form or custom fields to checkout so each paid order includes job input.
3. Resume the suspended HF scheduled job after budget guardrails are confirmed.
4. Add delivery: Gmail, Drive link, or hosted artifact page.
