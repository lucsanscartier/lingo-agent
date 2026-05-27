#!/usr/bin/env python3
"""
Stripe Event Bridge for OAE Compute Relay v0.1

Purpose:
- Convert Stripe Checkout/payment-like JSON into OAE Compute Relay queue jobs.
- Portable: can run in HF Jobs, GitHub Actions, local scripts, or a webhook service.
- Safe: does not execute compute. It only writes queue entries.

Input options:
1. A saved Stripe event JSON file.
2. A simplified paid order JSON.

This does not require Stripe SDK for v0.1 because it consumes event payloads already provided
by a webhook, connector, or export. Later versions can poll Stripe API with STRIPE_SECRET_KEY.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_QUEUE = Path("queue.generated.json")
DEFAULT_CREDIT_CAD = 5
EXPECTED_PRICE_ID = "price_1TbqgzFRuWOMBXuRYwxGaDQW"
EXPECTED_PRODUCT_ID = "prod_Ub2m82KsbFXy3T"
EXPECTED_PAYMENT_LINK_ID = "plink_1TbqhQFRuWOMBXuRuZgJzpZW"


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_queue(queue_path: Path, jobs: List[Dict[str, Any]]) -> None:
    queue_path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")


def append_queue(queue_path: Path, new_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing: List[Dict[str, Any]] = []
    if queue_path.exists():
        data = json.loads(queue_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            existing = data
    seen = {job.get("job_id") for job in existing}
    for job in new_jobs:
        if job.get("job_id") not in seen:
            existing.append(job)
    save_queue(queue_path, existing)
    return existing


def extract_checkout_object(event: Dict[str, Any]) -> Dict[str, Any]:
    # Stripe webhook shape: {type, data: {object: {...}}}
    if "data" in event and isinstance(event.get("data"), dict):
        obj = event["data"].get("object")
        if isinstance(obj, dict):
            return obj
    # Simplified/direct shape.
    return event


def is_paid_checkout(obj: Dict[str, Any]) -> bool:
    payment_status = str(obj.get("payment_status", "")).lower()
    status = str(obj.get("status", "")).lower()
    paid_flag = obj.get("paid")
    amount_total = obj.get("amount_total") or obj.get("amount")
    if paid_flag is True:
        return True
    if payment_status == "paid":
        return True
    if status in {"complete", "succeeded"} and amount_total:
        return True
    return False


def product_matches(obj: Dict[str, Any]) -> bool:
    # Payment links and metadata are the simplest reliable v0.1 markers.
    payment_link = str(obj.get("payment_link") or obj.get("payment_link_id") or "")
    if payment_link and EXPECTED_PAYMENT_LINK_ID in payment_link:
        return True

    metadata = obj.get("metadata") or {}
    if isinstance(metadata, dict):
        if metadata.get("product_id") == EXPECTED_PRODUCT_ID:
            return True
        if metadata.get("price_id") == EXPECTED_PRICE_ID:
            return True
        if metadata.get("oae_product") == "compute_relay_credit":
            return True

    # Some simplified order records can pass price/product directly.
    if obj.get("price_id") == EXPECTED_PRICE_ID or obj.get("product_id") == EXPECTED_PRODUCT_ID:
        return True

    # Fail open only for explicit simplified demo orders that say they are compute credits.
    if obj.get("product") == "OAE Compute Relay Credit":
        return True

    return False


def checkout_to_job(event_or_order: Dict[str, Any]) -> Dict[str, Any]:
    obj = extract_checkout_object(event_or_order)
    if not is_paid_checkout(obj):
        raise ValueError("Stripe/order object is not paid/complete")
    if not product_matches(obj):
        raise ValueError("Stripe/order object does not match OAE Compute Relay Credit")

    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    customer_email = (
        obj.get("customer_email")
        or (obj.get("customer_details") or {}).get("email")
        or metadata.get("customer_email")
        or "unknown"
    )

    task = (
        metadata.get("task")
        or obj.get("task")
        or "Run a small OAE Compute Relay text job. Customer did not provide task metadata yet."
    )

    task_type = metadata.get("task_type") or obj.get("task_type") or "text"
    output_format = metadata.get("output_format") or obj.get("output_format") or "markdown"
    privacy_level = metadata.get("privacy_level") or obj.get("privacy_level") or "unknown"

    amount_total = obj.get("amount_total") or obj.get("amount") or DEFAULT_CREDIT_CAD * 100
    credit_cad = round(float(amount_total) / 100.0, 2)

    source_id = obj.get("id") or obj.get("payment_intent") or sha(obj)[:16]
    job_id = metadata.get("job_id") or f"stripe-{source_id}"

    job = {
        "job_id": job_id,
        "paid": True,
        "credit_cad": credit_cad,
        "customer_email": customer_email,
        "task_type": task_type,
        "task": task,
        "output_format": output_format,
        "max_compute_cost_cad": float(metadata.get("max_compute_cost_cad") or obj.get("max_compute_cost_cad") or 0.25),
        "privacy_level": privacy_level,
        "stripe_checkout_session": obj.get("id") if str(obj.get("object", "")).lower() == "checkout.session" else obj.get("stripe_checkout_session"),
        "stripe_payment_intent": obj.get("payment_intent") or obj.get("stripe_payment_intent"),
        "stripe_payment_link_id": obj.get("payment_link") or obj.get("payment_link_id") or EXPECTED_PAYMENT_LINK_ID,
        "source_event_sha256": sha(event_or_order),
        "created_at": utc_now(),
        "delivery": {
            "method": metadata.get("delivery_method") or obj.get("delivery_method") or "manual",
            "target": metadata.get("delivery_target") or customer_email,
        },
    }
    job["queue_record_sha256"] = sha(job)
    return job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Stripe event/order JSON path")
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE), help="Output queue JSON path")
    parser.add_argument("--append", action="store_true", help="Append to existing queue instead of replacing")
    args = parser.parse_args()

    payload = load_json(args.input)
    payloads = payload if isinstance(payload, list) else [payload]
    jobs = []
    errors = []
    for item in payloads:
        try:
            jobs.append(checkout_to_job(item))
        except Exception as exc:
            errors.append({"error": str(exc), "source_sha256": sha(item)})

    queue_path = Path(args.queue)
    final_queue = append_queue(queue_path, jobs) if args.append else jobs
    if not args.append:
        save_queue(queue_path, final_queue)

    print(json.dumps({
        "status": "ok" if jobs else "no_jobs_created",
        "jobs_created": len(jobs),
        "errors": errors,
        "queue_path": str(queue_path),
        "queue_count": len(final_queue),
        "job_ids": [job.get("job_id") for job in jobs],
    }, indent=2))


if __name__ == "__main__":
    main()
