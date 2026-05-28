# OAE Compute Relay Gateway Security Notes

## Current gateway posture

The gateway source supports:

- public quote/protocol routes
- Stripe/payment-event to queue conversion routes
- optional shared-secret protection using `OAE_RELAY_SHARED_SECRET`

## Important production rule

Do **not** trust `/stripe/webhook` or `/stripe/event-to-queue` as authoritative payment verification unless one of these is configured:

1. `OAE_RELAY_SHARED_SECRET` is set on the Edge Function and callers provide `x-oae-relay-secret`, or
2. Stripe webhook signature verification is implemented using `STRIPE_WEBHOOK_SECRET`, or
3. event JSON is passed in by a trusted connector/control-plane that has already verified payment.

## Why this matters

A public endpoint that accepts arbitrary JSON cannot prove payment by itself. It can only transform a trusted payment event into a queue record.

## Next hardening patch

- Public: `/health`, `/protocol`, `/quote`, `/a2a/quote`
- Protected: `/stripe/event-to-queue`, `/stripe/webhook`
- Block protected routes if no server-side secret is configured.
- Add Stripe signature verification when `STRIPE_WEBHOOK_SECRET` is available.
