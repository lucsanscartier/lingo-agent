# OAE Compute Relay — A2A Client Quickstart

This guide maps OAE Compute Relay to the broader Agent-to-Agent client pattern.

## Current compatibility

OAE Compute Relay currently exposes:

```txt
GET  /health
GET  /protocol
POST /a2a/quote
POST /quote
```

Planned A2A compatibility additions:

```txt
GET  /.well-known/agent-card.json
POST /message:send
POST /message/send
```

## Live gateway

```txt
https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway
```

## Agent Card

Repository copy:

```txt
oae-compute-relay/a2a/agent-card.json
```

The card describes:

- provider
- gateway URL
- quote skill
- protected paid-job skill
- input/output modes
- security requirements
- $5 CAD compute credit

## Plain HTTP quote example

```bash
curl -X POST \
  https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway/a2a/quote \
  -H 'content-type: application/json' \
  -d '{
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
  }'
```

## Python client example

```python
import json
import urllib.request

BASE = "https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway"

payload = {
    "job": {
        "job_id": "python-agent-quote-001",
        "paid": False,
        "credit_cad": 5,
        "task_type": "text",
        "task": "Summarize this public documentation into JSON.",
        "output_format": "json",
        "max_compute_cost_cad": 0.25,
        "privacy_level": "public",
    }
}

req = urllib.request.Request(
    BASE + "/a2a/quote",
    data=json.dumps(payload).encode("utf-8"),
    method="POST",
    headers={"content-type": "application/json"},
)

with urllib.request.urlopen(req, timeout=30) as response:
    print(response.status)
    print(response.read().decode("utf-8"))
```

## JavaScript / TypeScript client example

```ts
const base = "https://ubauxksvewtwwerkpbuo.supabase.co/functions/v1/oae-compute-relay-gateway";

const response = await fetch(`${base}/a2a/quote`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({
    job: {
      job_id: "js-agent-quote-001",
      paid: false,
      credit_cad: 5,
      task_type: "text",
      task: "Summarize this public documentation into JSON.",
      output_format: "json",
      max_compute_cost_cad: 0.25,
      privacy_level: "public"
    }
  })
});

console.log(await response.json());
```

## Mapping to A2A-style clients

If an SDK expects an Agent Card first:

1. load `agent-card.json`
2. read `url`
3. call `/a2a/quote` for the current public quote skill
4. pay using the returned `pay_url`
5. submit paid job through protected queue route only after payment verification

## Protected paid job route

Protected routes require:

```txt
x-oae-relay-secret: <relay secret>
```

Do not use the protected payment routes as public unauthenticated endpoints.

## Production TODO

- Serve Agent Card from `/.well-known/agent-card.json`.
- Add A2A `message:send` wrapper that maps natural language or parts into quote jobs.
- Replace temporary bootstrap secret with Supabase environment secret.
- Add Stripe webhook signature verification.
