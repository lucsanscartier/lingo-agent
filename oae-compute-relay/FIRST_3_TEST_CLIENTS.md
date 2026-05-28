# OAE Compute Relay — First 3 Pilot Clients

## Goal

Get 3 tiny public/non-sensitive test jobs through the relay to prove that people understand the offer, will pay or seriously consider paying, and can receive a useful artifact.

## Current offer

```txt
$5 CAD managed compute test run
```

Buyer sends one small public/non-sensitive task. OAE Compute Relay quotes the route, runs the job through the control-plane/HF relay path, and returns an artifact plus hash or explains why the job is blocked.

## Target 1: Hugging Face / AI builder

Best ask:

```txt
Do you have a tiny Hugging Face Space, model, dataset, README, or prompt task you want turned into a short JSON/Markdown artifact?
```

Good job examples:

- summarize a Space README into JSON
- create a launch checklist for a Space
- generate a small model/dataset scouting brief
- convert a project idea into a build spec

## Target 2: indie hacker / automation builder

Best ask:

```txt
Do you have one tiny automation or AI-product task I can run through a managed compute relay as a $5 test?
```

Good job examples:

- turn product idea into a one-page API spec
- make a prompt pack for a tiny feature
- generate a README starter
- create a JSON task manifest

## Target 3: small business / creator

Best ask:

```txt
Do you have a small public/non-sensitive business task — a landing page blurb, offer summary, FAQ, or content outline — that could be generated as a test artifact?
```

Good job examples:

- landing page starter
- offer summary
- FAQ draft
- email/DM draft pack
- social post idea pack

## Accept only

- public or non-sensitive text/code/doc tasks
- tasks that can be delivered as markdown, JSON, code skeleton, or simple file
- jobs with low expected compute cost

## Reject or defer

- private personal data
- medical/legal/financial advice
- spam/outreach automation
- scraping private systems
- GPU-heavy image/video/training
- tasks requiring guaranteed business outcomes

## Pilot success criteria

For each pilot, record:

```txt
Name / handle:
Task:
Did they pay? yes/no
If not, why?
Artifact delivered:
Artifact hash:
Time to fulfill:
Reaction:
Upsell opportunity:
```

## First outreach line

```txt
I’m testing a small agent-to-agent compute relay. It quotes tiny AI jobs, routes them through a managed worker, and returns a result artifact with a ledger hash. First test credits are $5 CAD. Want to try one tiny public/non-sensitive job?
```

## Fulfillment path for now

```txt
lead says yes
→ get task details
→ quote via /a2a/quote
→ send $5 Stripe credit link
→ verify payment through Stripe connector
→ create queue job
→ run HF relay
→ return artifact + hash
```
