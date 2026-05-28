# OAE Compute Relay — Client Acquisition Playbook

## Positioning

OAE Compute Relay sells small managed AI job runs. The first offer is a low-friction $5 CAD compute credit for agents/builders who want a small task quoted and routed.

Do not pitch this as unlimited compute or raw compute resale. Pitch it as managed execution with a quote, route, risk check, and artifact hash.

## First customer segments

1. AI builders who need quick prototype tasks
2. Hugging Face Space builders who need small helper jobs
3. indie hackers experimenting with agents
4. automation consultants who need one-off artifact generation
5. people with broken scripts, docs, or small JSON/code tasks

## Best first offer

```txt
$5 CAD test run
Submit one small public/non-sensitive task.
We quote the route, run the relay, and return a result artifact or explain why it is blocked.
```

## Good first job types

- Summarize public docs into JSON
- Generate a small code skeleton
- Convert a task brief into an agent spec
- Make a prompt pack for a narrow use case
- Create a README or API usage example
- Review a simple public repo/file excerpt

## Jobs to reject or upsell

Reject or require custom quote:

- private/sensitive personal data
- medical/legal/financial decisions
- scraping private systems
- spam/outreach automation
- GPU-heavy image/video/model training
- anything promising guaranteed revenue

## Simple funnel

```txt
1. Show the live quote endpoint/docs.
2. Offer a $5 test run.
3. Buyer pays Stripe compute credit.
4. Buyer sends task details.
5. Relay processes the job.
6. Return artifact and hash.
7. Upsell larger managed job if useful.
```

## Discovery message

```txt
I’m testing a small agent-to-agent compute relay.

It quotes tiny AI jobs, routes them through a managed worker, and returns a result artifact with a ledger hash. First test credits are $5 CAD.

Best fit: public/non-sensitive tasks like summarizing docs, generating a small code skeleton, converting a brief into JSON, or creating an agent spec.

Want to try one small job?
```

## Short social post

```txt
Testing a tiny AI compute relay:

- agent asks for quote
- gateway returns route/cost/risk
- $5 CAD credit pays for a small managed job
- worker returns artifact + hash

Not raw compute resale. Managed task execution.

Looking for 3 small public/non-sensitive test jobs.
```

## DM follow-up

```txt
Nice — send me one small task with:

1. task type: text, code, hf-search, llm, or gpu
2. the task
3. desired output: markdown, JSON, code, or file
4. privacy level: public / non-sensitive / private / sensitive
5. max compute budget

For the first test, keep it public or non-sensitive.
```

## Success criteria for first 10 users

Track:

- Did they understand the offer?
- Did they pay or hesitate?
- What job type did they request?
- Did the relay return something useful?
- How long did fulfillment take?
- Was the $5 price too low, too high, or just right?
- What upsell naturally appeared?

## First upsells

- $19 standard managed job
- $49 batch of 5 small jobs
- $99 custom relay route / GPU deposit
- $149 agent spec + implementation scaffold
- $299+ done-for-you mini automation

## Proof assets to show

- Live quote endpoint
- FigJam architecture diagram
- GitHub source folder
- Stripe compute credit
- HF Jobs smoke test result
- Linear GPT Compute board

## Immediate action

Find 3 people/builders/agents with tiny public tasks and run them manually through the current control-plane path while the Stripe webhook is being hardened.
