# LINGO — AI Phone Agent Service for Small Businesses

**GitHub:** [github.com/lucsanscartier/lingo-agent](https://github.com/lucsanscartier/lingo-agent)  
**Landing page:** [LINGO AI Phone Agents](https://www.perplexity.ai/computer/a/lingo-ai-phone-agents-uf7YK78jSo2fy2KjyMS.ag)  
**API base URL:** `https://api.lingo-agent.ai/v1`  
**OpenAPI spec:** `https://api.lingo-agent.ai/v1/openapi.json`

---

## What is LINGO?

LINGO deploys AI phone agents for small businesses. You give it a phone number and a plain-English description of the business — LINGO handles the rest: answering calls 24/7, sending follow-up SMS, and remembering every caller through the **Kirk memory system**.

From an agent developer's perspective, LINGO is a **callable microservice** that adds persistent, voice-capable, human-facing communication to any multi-agent workflow. Your agents don't need to deal with telephony, speech synthesis, ASR, or conversation state management — LINGO abstracts all of it behind a clean REST API.

### Why an agent would want to use LINGO

| Your agent does this | LINGO does this for you |
|---|---|
| Books appointments | Answers the phone and books over voice |
| Manages a CRM | Talks to customers and feeds structured transcripts back |
| Runs a local business workflow | Provides 24/7 front-desk coverage |
| Orchestrates other agents | Routes incoming human requests to the right agent |
| Handles customer support tickets | Intercepts calls before they become tickets |

---

## Authentication

All API requests require a Bearer token. Obtain your API key from the LINGO dashboard:

**→ [app.lingo-agent.ai/settings/api](https://app.lingo-agent.ai/settings/api)**

Pass the key in every request header:

```
Authorization: Bearer lingo_sk_your_api_key_here
```

Keys are account-scoped. Each key can create and manage multiple agents. Store keys in environment variables — never hardcode them.

```bash
export LINGO_API_KEY="lingo_sk_your_api_key_here"
```

---

## Quick Start

### 1. Deploy an agent

```python
import httpx
import os

LINGO_API_KEY = os.environ["LINGO_API_KEY"]
BASE_URL = "https://api.lingo-agent.ai/v1"

headers = {
    "Authorization": f"Bearer {LINGO_API_KEY}",
    "Content-Type": "application/json",
}

response = httpx.post(
    f"{BASE_URL}/agents",
    headers=headers,
    json={
        "business_name": "Tony's Pizzeria",
        "phone_number": "+14155551234",
        "system_prompt": (
            "You are the friendly AI phone assistant for Tony's Pizzeria in San Francisco. "
            "Help customers place orders, check wait times, and answer menu questions. "
            "We are open Mon–Sat 11am–10pm, Sun 12pm–9pm. Address: 425 Columbus Ave. "
            "Always confirm the order before ending the call."
        ),
        "escalation_email": "tony@tonyspizzeria.com",
        "plan": "pro",
        "timezone": "America/Los_Angeles",
        "voice": "nova",
        "sms_enabled": True,
    },
)

agent = response.json()
print(f"Agent deployed: {agent['agent_id']}")
print(f"Status: {agent['status']}")
print(f"Phone: {agent['phone_number']}")
# → Agent deployed: agt_7xKp2mNqR4wBvL9s
# → Status: active
# → Phone: +14155551234
```

The agent is live within seconds. Calls to `+14155551234` are now handled by LINGO.

### 2. Check agent status

```python
agent_id = "agt_7xKp2mNqR4wBvL9s"

response = httpx.get(f"{BASE_URL}/agents/{agent_id}", headers=headers)
agent = response.json()

print(f"Total calls: {agent['stats']['total_calls']}")
print(f"Total minutes: {agent['stats']['total_minutes']}")
print(f"Unique callers: {agent['stats']['unique_callers']}")
print(f"Escalations: {agent['stats']['escalations']}")
```

---

## Pricing

| Plan | Monthly | Included minutes | SMS | Per-minute overage |
|---|---|---|---|---|
| **Burner** | $9/mo | 60 min | None | $0.08/min |
| **Starter** | $29/mo | 300 min | 200/mo | $0.06/min |
| **Pro** | $79/mo | Unlimited (2,000 fair use) | Unlimited | $0.04/min above 2,000 |

**Blended per-call estimate: ~$0.05/min** across typical usage patterns.

A 4-minute call on Starter costs roughly $0.24. A busy small business taking 100 calls/month at 4 min average uses ~400 minutes — well within Pro's fair-use envelope at $79/month flat.

14-day free trial available, no credit card required.

---

## Example 1: Booking Agent + LINGO

A booking agent orchestrator uses LINGO to handle the voice channel. Before each expected call, the booking agent pre-loads context into LINGO via the `/message` endpoint. After each call, it reads the transcript and writes the appointment to a calendar.

```python
import httpx
import os
from datetime import datetime, timedelta

LINGO_API_KEY = os.environ["LINGO_API_KEY"]
BASE_URL = "https://api.lingo-agent.ai/v1"
AGENT_ID = "agt_7xKp2mNqR4wBvL9s"

headers = {
    "Authorization": f"Bearer {LINGO_API_KEY}",
    "Content-Type": "application/json",
}


class BookingAgent:
    """
    Orchestrates LINGO phone agent + calendar writes.
    """

    def pre_brief_caller(self, caller_phone: str, context: dict):
        """
        Call this before an expected inbound call to load caller context
        into LINGO's Kirk memory. The agent will greet the caller by name
        and reference their history.
        """
        message = (
            f"Upcoming caller: {context['name']} ({caller_phone}). "
            f"Their current appointment: {context['appointment_date']} at {context['appointment_time']}. "
            f"They may want to reschedule. Available slots: {', '.join(context['open_slots'])}. "
            f"Notes from last visit: {context.get('notes', 'None')}. "
            "Please greet them by name and proactively offer the open slots."
        )

        response = httpx.post(
            f"{BASE_URL}/agents/{AGENT_ID}/message",
            headers=headers,
            json={
                "role": "system",
                "content": message,
                "caller_id": caller_phone,
                "metadata": {
                    "source_agent": "booking-agent-v2",
                    "crm_record_id": context.get("crm_id"),
                },
            },
        )
        return response.json()

    def sync_completed_calls(self, since: datetime):
        """
        Poll LINGO for completed call transcripts since a given time,
        extract appointment bookings, and write them to the calendar.
        """
        response = httpx.get(
            f"{BASE_URL}/agents/{AGENT_ID}/transcripts",
            headers=headers,
            params={
                "type": "call",
                "from": since.isoformat() + "Z",
                "limit": 50,
            },
        )
        transcripts = response.json()["transcripts"]

        bookings = []
        for transcript in transcripts:
            # LINGO extracts entities from every call automatically
            entities = transcript.get("entities", {})
            if transcript.get("intent") == "appointment_booking" and entities.get("date"):
                bookings.append(
                    {
                        "caller_id": transcript["caller_id"],
                        "caller_name": transcript.get("caller_name"),
                        "date": entities["date"],
                        "time": entities.get("time"),
                        "service": entities.get("service", "appointment"),
                        "transcript_id": transcript["transcript_id"],
                    }
                )

        return bookings

    def update_agent_with_new_hours(self, new_hours_text: str):
        """Update the agent's knowledge when business hours change."""
        response = httpx.post(
            f"{BASE_URL}/agents/{AGENT_ID}/train",
            headers=headers,
            json={
                "system_prompt_patch": f"\n\nIMPORTANT UPDATE: {new_hours_text}",
                "change_summary": "Updated business hours",
            },
        )
        result = response.json()
        print(f"Training applied: version {result['version']}")


# Usage
agent = BookingAgent()

# Pre-brief LINGO before a known incoming call
agent.pre_brief_caller(
    "+14155559012",
    {
        "name": "John Smith",
        "appointment_date": "2026-05-01",
        "appointment_time": "3:00 PM",
        "open_slots": ["9:00 AM", "10:00 AM", "11:00 AM"],
        "notes": "Prefers morning appointments.",
        "crm_id": "contact_8xKp3",
    },
)

# Sync bookings from the last hour
one_hour_ago = datetime.utcnow() - timedelta(hours=1)
new_bookings = agent.sync_completed_calls(since=one_hour_ago)
for booking in new_bookings:
    print(f"New booking: {booking['caller_name']} on {booking['date']} at {booking['time']}")
    # → write to Google Calendar, Calendly, etc.
```

---

## Example 2: CRM Agent Reading LINGO Transcripts

A CRM sync agent runs on a schedule, reads LINGO transcripts, and upserts contact records and interaction logs into a CRM.

```python
import httpx
import os
from datetime import datetime, timedelta, timezone

LINGO_API_KEY = os.environ["LINGO_API_KEY"]
BASE_URL = "https://api.lingo-agent.ai/v1"
AGENT_ID = "agt_7xKp2mNqR4wBvL9s"

headers = {
    "Authorization": f"Bearer {LINGO_API_KEY}",
    "Content-Type": "application/json",
}


class CRMSyncAgent:
    """
    Reads LINGO call data and syncs it to a CRM.
    Designed to run every 15 minutes as a cron job.
    """

    def get_caller_profile(self, caller_phone: str) -> dict:
        """
        Pull the full Kirk memory record for a caller.
        Returns preferences, interaction history, and sentiment score.
        """
        response = httpx.get(
            f"{BASE_URL}/agents/{AGENT_ID}/memory",
            headers=headers,
            params={"caller_id": caller_phone},
        )
        data = response.json()
        records = data.get("records", [])
        return records[0] if records else {}

    def sync_to_crm(self, last_sync_time: datetime):
        """
        Fetch all transcripts since last sync and upsert to CRM.
        """
        cursor = None
        all_transcripts = []

        # Page through all new transcripts
        while True:
            params = {
                "type": "all",
                "from": last_sync_time.isoformat() + "Z",
                "limit": 100,
            }
            if cursor:
                params["cursor"] = cursor

            response = httpx.get(
                f"{BASE_URL}/agents/{AGENT_ID}/transcripts",
                headers=headers,
                params=params,
            )
            data = response.json()
            all_transcripts.extend(data["transcripts"])

            cursor = data.get("cursor")
            if not cursor:
                break

        print(f"Processing {len(all_transcripts)} new interactions")

        for transcript in all_transcripts:
            caller_phone = transcript["caller_id"]

            # Pull full Kirk memory for this caller
            memory = self.get_caller_profile(caller_phone)

            # Build CRM interaction record
            crm_record = {
                "contact_phone": caller_phone,
                "contact_name": transcript.get("caller_name") or memory.get("name"),
                "interaction_type": transcript["type"],
                "timestamp": transcript["started_at"],
                "duration_seconds": transcript.get("duration_seconds"),
                "intent": transcript.get("intent"),
                "sentiment": transcript.get("sentiment"),
                "sentiment_score": transcript.get("sentiment_score"),
                "escalated": transcript.get("escalated", False),
                "entities": transcript.get("entities", {}),
                "summary": self._summarize_transcript(transcript["messages"]),
                "lingo_transcript_id": transcript["transcript_id"],
                # Enrich with Kirk memory
                "caller_total_interactions": memory.get("total_interactions", 0),
                "caller_avg_sentiment": memory.get("sentiment_score"),
                "caller_tags": memory.get("tags", []),
                "caller_notes": memory.get("preferences", {}).get("notes"),
            }

            # Push to your CRM here (HubSpot, Salesforce, etc.)
            self._upsert_crm_contact(crm_record)

        return len(all_transcripts)

    def _summarize_transcript(self, messages: list) -> str:
        """Build a one-line summary from transcript messages."""
        if not messages:
            return ""
        # Simple heuristic: first user message describes the reason for call
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                return content[:200] + ("..." if len(content) > 200 else "")
        return ""

    def _upsert_crm_contact(self, record: dict):
        """
        Placeholder — replace with your CRM SDK call.
        e.g., hubspot.crm.contacts.basic_api.create_or_update(...)
        """
        print(
            f"[CRM] Upserting {record['contact_phone']} — "
            f"intent={record['intent']}, sentiment={record['sentiment']}"
        )

    def flag_at_risk_callers(self):
        """
        Use Kirk memory to find callers with declining sentiment
        and flag them for proactive outreach in the CRM.
        """
        response = httpx.get(
            f"{BASE_URL}/agents/{AGENT_ID}/memory",
            headers=headers,
            params={"limit": 200},
        )
        records = response.json()["records"]

        at_risk = [
            r for r in records
            if r.get("sentiment_score", 1.0) < 0.45
            and r.get("total_interactions", 0) >= 2
        ]

        print(f"At-risk callers: {len(at_risk)}")
        for caller in at_risk:
            print(
                f"  {caller.get('name', caller['caller_id'])} — "
                f"sentiment={caller['sentiment_score']:.2f}, "
                f"interactions={caller['total_interactions']}"
            )
        return at_risk


# Run the sync
agent = CRMSyncAgent()
last_sync = datetime.now(timezone.utc) - timedelta(minutes=15)
synced = agent.sync_to_crm(last_sync)
print(f"Synced {synced} interactions to CRM")

# Flag at-risk customers for follow-up
at_risk = agent.flag_at_risk_callers()
```

---

## Webhooks

For real-time event delivery, configure a webhook URL when deploying your agent. LINGO POSTs JSON to your endpoint for each event:

| Event | Trigger |
|---|---|
| `call.started` | Inbound call begins |
| `call.ended` | Call ends (includes duration and transcript ID) |
| `escalation.triggered` | Agent escalates to human |
| `sms.received` | Inbound SMS received |
| `voicemail.recorded` | Caller left a voicemail |

```python
# Webhook payload example — call.ended
{
  "event": "call.ended",
  "agent_id": "agt_7xKp2mNqR4wBvL9s",
  "timestamp": "2026-04-27T20:15:42Z",
  "data": {
    "transcript_id": "txn_3mKp8wNqR5xBvL2s",
    "caller_id": "+14155559012",
    "duration_seconds": 187,
    "intent": "appointment_booking",
    "escalated": False,
    "sentiment": "positive"
  }
}
```

Secure your webhook endpoint by validating the `X-Lingo-Signature` header (HMAC-SHA256 of the raw body using your webhook secret).

---

## API Reference Summary

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/agents` | Deploy a new phone agent |
| `GET` | `/agents` | List all agents |
| `GET` | `/agents/{id}` | Get agent status and stats |
| `PATCH` | `/agents/{id}` | Update agent configuration |
| `DELETE` | `/agents/{id}` | Decommission agent |
| `POST` | `/agents/{id}/message` | Send agent-to-agent message |
| `GET` | `/agents/{id}/memory` | Get Kirk caller memory |
| `DELETE` | `/agents/{id}/memory` | Delete a caller's memory (GDPR) |
| `GET` | `/agents/{id}/transcripts` | Get call/SMS transcripts |
| `POST` | `/agents/{id}/train` | Update agent instructions |
| `POST` | `/agents/{id}/pause` | Pause the agent |
| `POST` | `/agents/{id}/resume` | Resume a paused agent |

Full schema details: [openapi.json](./openapi.json) or the hosted spec at `https://api.lingo-agent.ai/v1/openapi.json`.

---

## Error Handling

LINGO returns standard HTTP status codes with machine-readable error bodies:

```python
response = httpx.post(f"{BASE_URL}/agents", headers=headers, json=payload)

if response.status_code == 201:
    agent = response.json()
elif response.status_code == 409:
    error = response.json()
    print(f"Conflict: {error['message']}")
    # e.g. phone number already in use
elif response.status_code == 401:
    print("Invalid API key — regenerate at https://app.lingo-agent.ai/settings/api")
elif response.status_code >= 500:
    # Retry with exponential backoff
    request_id = response.json().get("request_id")
    print(f"Server error. Contact support with request_id={request_id}")
```

All error responses include a `request_id` field — include it when contacting support.

---

## Rate Limits

| Plan | Requests/min | Concurrent calls |
|---|---|---|
| Burner | 30 | 1 |
| Starter | 120 | 3 |
| Pro | 600 | 10 |

Rate limit headers are included in every response:
```
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 598
X-RateLimit-Reset: 1745793060
```

---

## SDK & Resources

- **GitHub:** [github.com/lucsanscartier/lingo-agent](https://github.com/lucsanscartier/lingo-agent)
- **Landing page:** [LINGO AI Phone Agents](https://www.perplexity.ai/computer/a/lingo-ai-phone-agents-uf7YK78jSo2fy2KjyMS.ag)
- **OpenAPI spec:** `https://api.lingo-agent.ai/v1/openapi.json`
- **Dashboard:** [app.lingo-agent.ai](https://app.lingo-agent.ai)
- **Support:** dev@lingo-agent.ai

Python SDK (install from GitHub while PyPI listing is pending):
```bash
pip install git+https://github.com/lucsanscartier/lingo-agent.git#subdirectory=sdk/python
```

---

*LINGO is compatible with Fetch.ai AgentVerse (see `agentverse-manifest.json`), Naptha (see `naptha-module.json`), and any REST-capable agent framework.*
