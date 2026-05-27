#!/usr/bin/env python3
"""
OAE Compute Relay v0.1

Canonical compute-selling path.

We sell managed AI job execution / completed artifacts, not raw platform compute resale.

Modes:
- quote: estimate route, price class, and risk
- run: run a single JSON job
- queue: process jobs from a JSON queue
- ledger: summarize ledger

No external secrets are required for v0.1. It uses deterministic/template execution so it can run cheaply on HF Jobs cpu-basic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

PRODUCT = "OAE Compute Relay Credit"
STRIPE_PRODUCT_ID = "prod_Ub2m82KsbFXy3T"
STRIPE_PRICE_ID = "price_1TbqgzFRuWOMBXuRYwxGaDQW"
STRIPE_PAYMENT_LINK_ID = "plink_1TbqhQFRuWOMBXuRuZgJzpZW"
STRIPE_PAYMENT_LINK = "https://buy.stripe.com/7sY6oH2FX0Xrc4F4QK8k80f"
DEFAULT_CREDIT_CAD = 5

ARTIFACT_DIR = Path(os.getenv("OAE_RELAY_ARTIFACT_DIR", "artifacts"))
LEDGER_PATH = Path(os.getenv("OAE_RELAY_LEDGER", "ledger.jsonl"))

ROUTE_TABLE = {
    "text": {
        "route": "template-local",
        "estimated_cost_cad": 0.001,
        "notes": "Deterministic text artifact; no paid model call required in v0.1.",
    },
    "code": {
        "route": "template-local-code",
        "estimated_cost_cad": 0.001,
        "notes": "Deterministic code skeleton; no paid model call required in v0.1.",
    },
    "hf-search": {
        "route": "future-hf-search",
        "estimated_cost_cad": 0.02,
        "notes": "Future route: HF model/dataset/space search via connector or job token.",
    },
    "llm": {
        "route": "future-hf-inference",
        "estimated_cost_cad": 0.10,
        "notes": "Future route: HF InferenceClient or provider API with cost cap.",
    },
    "gpu": {
        "route": "future-paid-gpu",
        "estimated_cost_cad": 2.00,
        "notes": "Requires deposit, explicit approval, and margin check.",
    },
}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def sha(obj: Any) -> str:
    if isinstance(obj, str):
        data = obj.encode("utf-8")
    else:
        data = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def evidence_firewall(job: Dict[str, Any], quote: Dict[str, Any]) -> List[str]:
    labels = [
        "VERIFIED: OAE Compute Relay v0.1 generated this artifact.",
        f"VERIFIED: configured Stripe product is {STRIPE_PRODUCT_ID} / {STRIPE_PRICE_ID}.",
        f"VERIFIED: selected route is {quote['route']}.",
        "INFERRED: estimated compute cost is based on the configured v0.1 route table, not live billing telemetry.",
        "SPECULATIVE: Hyperfold Compute is an architecture metaphor for routing, caching, branching, and cost compression; not verified new physics.",
        "LIMITATION: This artifact is not a guarantee of profit, performance, or availability.",
    ]
    if not job.get("paid"):
        labels.append("BLOCKED: job was not marked paid, so execution should be skipped outside quote/probe contexts.")
    return labels


def quote_job(job: Dict[str, Any]) -> Dict[str, Any]:
    task_type = str(job.get("task_type", "text")).lower()
    route_info = ROUTE_TABLE.get(task_type, ROUTE_TABLE["text"])
    credit = float(job.get("credit_cad", DEFAULT_CREDIT_CAD) or 0)
    max_cost = float(job.get("max_compute_cost_cad", route_info["estimated_cost_cad"]) or 0)
    estimated_cost = float(route_info["estimated_cost_cad"])
    privacy = str(job.get("privacy_level", "unknown"))

    if task_type == "gpu":
        allowed = bool(job.get("paid")) and credit >= 99 and max_cost >= estimated_cost
    else:
        allowed = bool(job.get("paid")) and max_cost >= estimated_cost

    return {
        "job_id": job.get("job_id", "unknown"),
        "task_type": task_type,
        "route": route_info["route"],
        "estimated_cost_cad": estimated_cost,
        "credit_cad": credit,
        "estimated_margin_cad": round(max(0, credit - estimated_cost), 4),
        "allowed_to_run": allowed,
        "privacy_level": privacy,
        "risk_flags": compute_risk_flags(job, estimated_cost, credit, privacy),
        "notes": route_info["notes"],
        "quoted_at": utc_now(),
    }


def compute_risk_flags(job: Dict[str, Any], estimated_cost: float, credit: float, privacy: str) -> List[str]:
    flags: List[str] = []
    if not job.get("paid"):
        flags.append("UNPAID_JOB")
    if estimated_cost > credit:
        flags.append("NEGATIVE_MARGIN")
    if privacy not in {"non_sensitive", "public", "unknown"}:
        flags.append("PRIVACY_REVIEW_REQUIRED")
    if str(job.get("task_type", "")).lower() == "gpu":
        flags.append("GPU_REQUIRES_DEPOSIT_AND_APPROVAL")
    return flags


def render_text_artifact(job: Dict[str, Any], quote: Dict[str, Any]) -> str:
    task = job.get("task", "No task provided.")
    fmt = job.get("output_format", "markdown")
    customer = job.get("customer_email", "unknown")

    return f"""# OAE Compute Relay Result

**Job ID:** `{job.get('job_id', 'unknown')}`  
**Customer:** `{customer}`  
**Generated:** {utc_now()}  
**Product:** {PRODUCT}  
**Payment Link:** {STRIPE_PAYMENT_LINK}  

## Task

{task}

## Output

OAE Compute Relay has accepted this as a small managed compute job and produced a deterministic v0.1 artifact.

For this task, the recommended output is:

> {task}

### Product Summary Draft

OAE Compute Relay is a managed AI job router. Customers or agents buy compute credits, submit a task, and receive a completed artifact. The system chooses the cheapest authorized route, records a ledger hash, and keeps Evidence Firewall labels attached to the output.

### Suggested Customer-Facing Copy

Pay for a small managed AI job run. Submit a clear task, and the relay routes it through cheap authorized compute to return a result artifact and execution ledger. This is not raw compute resale; it is a managed output service.

## Route Quote

```json
{json.dumps(quote, indent=2)}
```

## Evidence Firewall

{chr(10).join('- ' + label for label in evidence_firewall(job, quote))}

## Artifact Hash

This markdown file should be hashed in the ledger after write.

**Requested format:** `{fmt}`
"""


def render_code_artifact(job: Dict[str, Any], quote: Dict[str, Any]) -> str:
    task = job.get("task", "Generate a small Python CLI skeleton.")
    return f"""# OAE Compute Relay Code Result

**Job ID:** `{job.get('job_id', 'unknown')}`  
**Generated:** {utc_now()}

## Task

{task}

## Code Skeleton

```python
#!/usr/bin/env python3
import argparse

def main():
    parser = argparse.ArgumentParser(description='OAE generated CLI skeleton')
    parser.add_argument('--input', default='hello')
    args = parser.parse_args()
    print({{'status': 'ok', 'input': args.input}})

if __name__ == '__main__':
    main()
```

## Route Quote

```json
{json.dumps(quote, indent=2)}
```

## Evidence Firewall

{chr(10).join('- ' + label for label in evidence_firewall(job, quote))}
"""


def execute_job(job: Dict[str, Any], allow_unpaid_probe: bool = False) -> Dict[str, Any]:
    quote = quote_job(job)
    if not quote["allowed_to_run"] and not allow_unpaid_probe:
        result = {
            "job_id": job.get("job_id"),
            "status": "skipped",
            "reason": "job not allowed to run under guardrails",
            "quote": quote,
        }
        append_ledger(result)
        return result

    task_type = quote["task_type"]
    if task_type == "code":
        markdown = render_code_artifact(job, quote)
    else:
        markdown = render_text_artifact(job, quote)

    paths = write_artifact(job, quote, markdown)
    result = {
        "job_id": job.get("job_id"),
        "status": "done",
        "quote": quote,
        "paths": paths,
    }
    append_ledger(result)
    return result


def write_artifact(job: Dict[str, Any], quote: Dict[str, Any], markdown: str) -> Dict[str, str]:
    ensure_dirs()
    job_id = str(job.get("job_id", "job"))
    safe_job_id = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in job_id)[:80]
    artifact_sha = sha(markdown)
    base = ARTIFACT_DIR / f"{safe_job_id}-{artifact_sha[:12]}"
    md_path = base.with_suffix(".md")
    json_path = base.with_suffix(".json")
    record = {
        "job": job,
        "quote": quote,
        "artifact_sha256": artifact_sha,
        "worker": "oae-compute-relay-v0.1",
        "generated_at": utc_now(),
        "stripe": {
            "product_id": STRIPE_PRODUCT_ID,
            "price_id": STRIPE_PRICE_ID,
            "payment_link_id": STRIPE_PAYMENT_LINK_ID,
            "payment_link": STRIPE_PAYMENT_LINK,
        },
    }
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"markdown": str(md_path), "json": str(json_path), "artifact_sha256": artifact_sha}


def append_ledger(record: Dict[str, Any]) -> None:
    if LEDGER_PATH.parent != Path('.'):
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mode_quote(args: argparse.Namespace) -> None:
    job = load_json(args.job) if args.job else demo_job(paid=False)
    print(json.dumps({"mode": "quote", "quote": quote_job(job)}, indent=2))


def mode_run(args: argparse.Namespace) -> None:
    job = load_json(args.job) if args.job else demo_job(paid=True)
    result = execute_job(job, allow_unpaid_probe=args.allow_unpaid_probe)
    print(json.dumps({"mode": "run", "result": result}, indent=2))


def mode_queue(args: argparse.Namespace) -> None:
    queue = load_json(args.queue)
    if not isinstance(queue, list):
        raise ValueError("Queue must be a JSON list")
    results = [execute_job(job, allow_unpaid_probe=args.allow_unpaid_probe) for job in queue]
    print(json.dumps({"mode": "queue", "results": results}, indent=2))


def mode_ledger(args: argparse.Namespace) -> None:
    if not LEDGER_PATH.exists():
        print(json.dumps({"mode": "ledger", "entries": 0, "path": str(LEDGER_PATH)}, indent=2))
        return
    lines = LEDGER_PATH.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    done = sum(1 for x in entries if x.get("status") == "done")
    skipped = sum(1 for x in entries if x.get("status") == "skipped")
    print(json.dumps({"mode": "ledger", "entries": len(entries), "done": done, "skipped": skipped, "path": str(LEDGER_PATH)}, indent=2))


def demo_job(paid: bool) -> Dict[str, Any]:
    return {
        "job_id": "compute-demo-live",
        "paid": paid,
        "credit_cad": DEFAULT_CREDIT_CAD if paid else 0,
        "customer_email": "demo@example.com",
        "task_type": "text",
        "task": "Create a concise product summary for an AI compute relay that sells managed AI job runs.",
        "output_format": "markdown",
        "max_compute_cost_cad": 0.25,
        "privacy_level": "non_sensitive",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quote", "run", "queue", "ledger"], default="quote")
    parser.add_argument("--job", help="Path to a single job JSON")
    parser.add_argument("--queue", default="queue.example.json", help="Path to queue JSON")
    parser.add_argument("--allow-unpaid-probe", action="store_true")
    args = parser.parse_args()

    if args.mode == "quote":
        mode_quote(args)
    elif args.mode == "run":
        mode_run(args)
    elif args.mode == "queue":
        mode_queue(args)
    elif args.mode == "ledger":
        mode_ledger(args)


if __name__ == "__main__":
    main()
