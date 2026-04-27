# Registering LINGO on Fetch.ai AgentVerse

This guide walks through registering LINGO as a service agent on [Fetch.ai AgentVerse](https://agentverse.ai) using the `uagents` Python SDK. Once registered, LINGO appears in the AgentVerse marketplace and can be discovered and messaged by other uAgents.

---

## Prerequisites

```bash
# Python 3.10+
pip install uagents uagents-ai-engine httpx python-dotenv
```

Create a `.env` file:
```env
LINGO_API_KEY=lingo_sk_your_api_key_here
AGENT_SEED=your-deterministic-seed-phrase-keep-secret
AGENTVERSE_API_KEY=your-agentverse-api-key
```

Get your AgentVerse API key from [agentverse.ai/profile/api-keys](https://agentverse.ai/profile/api-keys).

---

## Overview

The registration process has three parts:

1. **Define message schemas** — the data structures other agents send to LINGO
2. **Create the uAgent** — instantiate a uAgent with a deterministic identity
3. **Register on Almanac** — publish the agent's address and service profile to the Fetch.ai Almanac contract so it's discoverable

---

## Step 1: Define Message Schemas

Create `lingo_messages.py`:

```python
# lingo_messages.py
"""
Message schemas for the LINGO uAgent service.
Other agents send these structured messages to interact with LINGO.
"""

from uagents import Model
from typing import Optional, List


# ── Inbound messages (requests FROM other agents TO LINGO) ────────────────

class DeployAgentRequest(Model):
    """Deploy a new LINGO phone agent for a business."""
    business_name: str
    phone_number: str          # E.164 format, e.g. "+14155551234"
    system_prompt: str
    escalation_email: str
    plan: str                  # "burner" | "starter" | "pro"
    timezone: Optional[str] = "America/New_York"
    voice: Optional[str] = "nova"
    sms_enabled: Optional[bool] = True


class SendMessageRequest(Model):
    """Send an agent-to-agent message to a deployed LINGO agent."""
    agent_id: str
    content: str
    role: Optional[str] = "user"       # "user" | "system"
    caller_id: Optional[str] = None    # E.164 — loads Kirk memory for this caller
    metadata: Optional[dict] = None


class GetTranscriptsRequest(Model):
    """Request recent call/SMS transcripts from a LINGO agent."""
    agent_id: str
    interaction_type: Optional[str] = "all"   # "call" | "sms" | "all"
    caller_id: Optional[str] = None
    limit: Optional[int] = 20


class GetMemoryRequest(Model):
    """Retrieve Kirk caller memory for a LINGO agent."""
    agent_id: str
    caller_id: Optional[str] = None   # Filter to one caller
    limit: Optional[int] = 50


class TrainAgentRequest(Model):
    """Update a LINGO agent's instructions."""
    agent_id: str
    system_prompt: Optional[str] = None
    system_prompt_patch: Optional[str] = None
    knowledge_entries: Optional[List[dict]] = None
    change_summary: Optional[str] = None


class GetAgentStatusRequest(Model):
    """Get status and stats for a LINGO agent."""
    agent_id: str


# ── Outbound messages (responses FROM LINGO TO requesting agents) ─────────

class DeployAgentResponse(Model):
    agent_id: str
    status: str
    phone_number: str
    plan: str
    kirk_memory_id: str
    error: Optional[str] = None


class SendMessageResponse(Model):
    message_id: str
    content: str
    caller_memory_updated: bool
    error: Optional[str] = None


class TranscriptSummary(Model):
    transcript_id: str
    type: str
    caller_id: str
    caller_name: Optional[str]
    started_at: str
    duration_seconds: Optional[int]
    intent: Optional[str]
    sentiment: str
    escalated: bool


class GetTranscriptsResponse(Model):
    agent_id: str
    total: int
    transcripts: List[dict]
    error: Optional[str] = None


class GetMemoryResponse(Model):
    agent_id: str
    total_callers: int
    records: List[dict]
    error: Optional[str] = None


class AgentStatusResponse(Model):
    agent_id: str
    status: str
    business_name: str
    phone_number: str
    plan: str
    stats: dict
    error: Optional[str] = None


class ErrorResponse(Model):
    error: str
    message: str
    request_id: Optional[str] = None
```

---

## Step 2: Create the LINGO uAgent

Create `lingo_agent.py`:

```python
# lingo_agent.py
"""
LINGO uAgent — registers LINGO Phone Agent Service on Fetch.ai AgentVerse.

Run with:
    python lingo_agent.py

The agent will:
  1. Start with a deterministic address derived from AGENT_SEED
  2. Register itself on the Fetch.ai Almanac contract
  3. Listen for messages from other uAgents and proxy them to the LINGO REST API
"""

import os
import httpx
from dotenv import load_dotenv
from uagents import Agent, Context, Protocol
from uagents.setup import fund_agent_if_low

from lingo_messages import (
    DeployAgentRequest, DeployAgentResponse,
    SendMessageRequest, SendMessageResponse,
    GetTranscriptsRequest, GetTranscriptsResponse,
    GetMemoryRequest, GetMemoryResponse,
    TrainAgentRequest,
    GetAgentStatusRequest, AgentStatusResponse,
    ErrorResponse,
)

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────

LINGO_API_KEY = os.environ["LINGO_API_KEY"]
AGENT_SEED = os.environ["AGENT_SEED"]
LINGO_BASE_URL = os.getenv("LINGO_BASE_URL", "https://api.lingo-agent.ai/v1")

AGENT_NAME = "lingo-phone-agent-service"
AGENT_PORT = 8001

# ── Instantiate the agent ─────────────────────────────────────────────────

agent = Agent(
    name=AGENT_NAME,
    seed=AGENT_SEED,
    port=AGENT_PORT,
    endpoint=[f"http://0.0.0.0:{AGENT_PORT}/submit"],
    # Set agentverse= to publish to hosted AgentVerse endpoint
    # agentverse="https://agentverse.ai",
)

# Fund the agent wallet if needed (testnet only)
fund_agent_if_low(agent.wallet.address())

print(f"LINGO uAgent address: {agent.address}")
print(f"LINGO uAgent wallet:  {agent.wallet.address()}")

# ── LINGO API client ──────────────────────────────────────────────────────

lingo_headers = {
    "Authorization": f"Bearer {LINGO_API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "LINGO-uAgent/1.0",
}


def lingo_request(method: str, path: str, **kwargs):
    """Make a synchronous request to the LINGO REST API."""
    url = f"{LINGO_BASE_URL}{path}"
    with httpx.Client(timeout=30.0) as client:
        response = client.request(method, url, headers=lingo_headers, **kwargs)
    return response


# ── Protocol definition ───────────────────────────────────────────────────

lingo_protocol = Protocol(name="lingo-v1", version="1.0.0")


@lingo_protocol.on_message(model=DeployAgentRequest, replies={DeployAgentResponse, ErrorResponse})
async def handle_deploy(ctx: Context, sender: str, msg: DeployAgentRequest):
    """Deploy a new LINGO phone agent."""
    ctx.logger.info(f"DeployAgentRequest from {sender} for '{msg.business_name}'")

    response = lingo_request(
        "POST",
        "/agents",
        json={
            "business_name": msg.business_name,
            "phone_number": msg.phone_number,
            "system_prompt": msg.system_prompt,
            "escalation_email": msg.escalation_email,
            "plan": msg.plan,
            "timezone": msg.timezone,
            "voice": msg.voice,
            "sms_enabled": msg.sms_enabled,
        },
    )

    if response.status_code == 201:
        data = response.json()
        await ctx.send(
            sender,
            DeployAgentResponse(
                agent_id=data["agent_id"],
                status=data["status"],
                phone_number=data["phone_number"],
                plan=data["plan"],
                kirk_memory_id=data["kirk_memory_id"],
            ),
        )
    else:
        err = response.json()
        await ctx.send(
            sender,
            ErrorResponse(
                error=err.get("error", "api_error"),
                message=err.get("message", "Unknown error"),
                request_id=err.get("request_id"),
            ),
        )


@lingo_protocol.on_message(model=SendMessageRequest, replies={SendMessageResponse, ErrorResponse})
async def handle_send_message(ctx: Context, sender: str, msg: SendMessageRequest):
    """Forward an agent-to-agent message to a LINGO phone agent."""
    ctx.logger.info(f"SendMessageRequest from {sender} to agent {msg.agent_id}")

    payload = {
        "role": msg.role,
        "content": msg.content,
    }
    if msg.caller_id:
        payload["caller_id"] = msg.caller_id
    if msg.metadata:
        payload["metadata"] = {**(msg.metadata or {}), "uagent_sender": sender}

    response = lingo_request("POST", f"/agents/{msg.agent_id}/message", json=payload)

    if response.status_code == 200:
        data = response.json()
        await ctx.send(
            sender,
            SendMessageResponse(
                message_id=data["message_id"],
                content=data["content"],
                caller_memory_updated=data.get("caller_memory_updated", False),
            ),
        )
    else:
        err = response.json()
        await ctx.send(
            sender,
            ErrorResponse(
                error=err.get("error", "api_error"),
                message=err.get("message", "Unknown error"),
            ),
        )


@lingo_protocol.on_message(model=GetTranscriptsRequest, replies={GetTranscriptsResponse, ErrorResponse})
async def handle_get_transcripts(ctx: Context, sender: str, msg: GetTranscriptsRequest):
    """Retrieve transcripts from a LINGO agent."""
    ctx.logger.info(f"GetTranscriptsRequest from {sender} for agent {msg.agent_id}")

    params = {"type": msg.interaction_type, "limit": msg.limit}
    if msg.caller_id:
        params["caller_id"] = msg.caller_id

    response = lingo_request("GET", f"/agents/{msg.agent_id}/transcripts", params=params)

    if response.status_code == 200:
        data = response.json()
        await ctx.send(
            sender,
            GetTranscriptsResponse(
                agent_id=data["agent_id"],
                total=data["total"],
                transcripts=data["transcripts"],
            ),
        )
    else:
        err = response.json()
        await ctx.send(
            sender,
            ErrorResponse(error=err.get("error", "api_error"), message=err.get("message", "")),
        )


@lingo_protocol.on_message(model=GetMemoryRequest, replies={GetMemoryResponse, ErrorResponse})
async def handle_get_memory(ctx: Context, sender: str, msg: GetMemoryRequest):
    """Retrieve Kirk caller memory from a LINGO agent."""
    ctx.logger.info(f"GetMemoryRequest from {sender} for agent {msg.agent_id}")

    params = {"limit": msg.limit}
    if msg.caller_id:
        params["caller_id"] = msg.caller_id

    response = lingo_request("GET", f"/agents/{msg.agent_id}/memory", params=params)

    if response.status_code == 200:
        data = response.json()
        await ctx.send(
            sender,
            GetMemoryResponse(
                agent_id=data["agent_id"],
                total_callers=data["total_callers"],
                records=data["records"],
            ),
        )
    else:
        err = response.json()
        await ctx.send(
            sender,
            ErrorResponse(error=err.get("error", "api_error"), message=err.get("message", "")),
        )


@lingo_protocol.on_message(model=GetAgentStatusRequest, replies={AgentStatusResponse, ErrorResponse})
async def handle_get_status(ctx: Context, sender: str, msg: GetAgentStatusRequest):
    """Get status and stats for a LINGO agent."""
    response = lingo_request("GET", f"/agents/{msg.agent_id}")

    if response.status_code == 200:
        data = response.json()
        await ctx.send(
            sender,
            AgentStatusResponse(
                agent_id=data["agent_id"],
                status=data["status"],
                business_name=data["business_name"],
                phone_number=data["phone_number"],
                plan=data["plan"],
                stats=data.get("stats", {}),
            ),
        )
    else:
        err = response.json()
        await ctx.send(
            sender,
            ErrorResponse(error=err.get("error", "api_error"), message=err.get("message", "")),
        )


# ── Startup / health ──────────────────────────────────────────────────────

@agent.on_event("startup")
async def on_startup(ctx: Context):
    ctx.logger.info("=" * 60)
    ctx.logger.info("LINGO Phone Agent Service — uAgent started")
    ctx.logger.info(f"  Agent address : {agent.address}")
    ctx.logger.info(f"  Wallet address: {agent.wallet.address()}")
    ctx.logger.info(f"  LINGO API     : {LINGO_BASE_URL}")
    ctx.logger.info("=" * 60)

    # Verify LINGO API connectivity
    try:
        response = lingo_request("GET", "/health")
        if response.status_code == 200:
            ctx.logger.info("LINGO API connection: OK")
        else:
            ctx.logger.warning(f"LINGO API returned {response.status_code} on health check")
    except Exception as exc:
        ctx.logger.error(f"Could not reach LINGO API: {exc}")


# ── Include protocol and run ──────────────────────────────────────────────

agent.include(lingo_protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
```

---

## Step 3: Register a Service Profile on AgentVerse

For full marketplace visibility, register a structured service profile using the AgentVerse Hosted Agents API. This is separate from the Almanac contract registration (which happens automatically when the agent runs) and makes LINGO searchable with a rich description, pricing, and capability tags.

Create `register_agentverse.py`:

```python
# register_agentverse.py
"""
Registers LINGO's service profile on Fetch.ai AgentVerse.
Run once after deploying the uAgent to publish the marketplace listing.

Usage:
    python register_agentverse.py
"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

AGENTVERSE_API_KEY = os.environ["AGENTVERSE_API_KEY"]
AGENT_ADDRESS = os.environ.get(
    "LINGO_AGENT_ADDRESS",
    # Replace with the address printed when you first ran lingo_agent.py
    "agent1qvlingo0phoneagentsvc2024xkp7mnqr4wbvl9skirk",
)

AGENTVERSE_BASE = "https://agentverse.ai/v1"

headers = {
    "Authorization": f"Bearer {AGENTVERSE_API_KEY}",
    "Content-Type": "application/json",
}

service_profile = {
    "agent_address": AGENT_ADDRESS,
    "name": "LINGO Phone Agent Service",
    "description": (
        "Deploy AI phone agents that answer calls 24/7 for any business. "
        "LINGO handles inbound voice, outbound SMS, and remembers every caller "
        "through its Kirk memory system. Supports agent-to-agent messaging for "
        "booking agents, CRM agents, and workflow orchestrators. "
        "Pricing: Burner $9/mo · Starter $29/mo · Pro $79/mo. "
        "Free 14-day trial, no credit card required."
    ),
    "short_description": "AI phone agents for small businesses — 24/7 voice, SMS, caller memory.",
    "category": "communication",
    "tags": [
        "voice-ai", "phone-agent", "small-business", "saas",
        "telephony", "sms", "caller-memory", "crm", "booking",
        "receptionist", "24-7", "escalation",
    ],
    "homepage_url": "https://www.perplexity.ai/computer/a/lingo-ai-phone-agents-uf7YK78jSo2fy2KjyMS.ag",
    "source_code_url": "https://github.com/lucsanscartier/lingo-agent",
    "protocols": [
        {
            "name": "lingo-v1",
            "version": "1.0.0",
            "description": "LINGO phone agent deployment and management protocol",
        }
    ],
    "capabilities": [
        "inbound_calls",
        "outbound_sms",
        "caller_memory",
        "escalation",
        "agent_messaging",
        "transcript_retrieval",
    ],
    "pricing": {
        "model": "subscription",
        "tiers": [
            {"name": "burner", "price": "9 USD/month", "call_minutes": 60},
            {"name": "starter", "price": "29 USD/month", "call_minutes": 300},
            {"name": "pro", "price": "79 USD/month", "call_minutes": "unlimited"},
        ],
        "per_minute_estimate": "0.05 USD",
    },
    "status": "active",
    "geo": "global",
}

def register():
    print("Registering LINGO service profile on AgentVerse...")

    response = httpx.post(
        f"{AGENTVERSE_BASE}/services",
        headers=headers,
        json=service_profile,
        timeout=30.0,
    )

    if response.status_code in (200, 201):
        data = response.json()
        print("Registration successful!")
        print(f"  Service ID  : {data.get('service_id')}")
        print(f"  Profile URL : {data.get('profile_url')}")
        print(f"  Status      : {data.get('status')}")
    elif response.status_code == 409:
        print("Service already registered. Updating existing profile...")
        update_response = httpx.put(
            f"{AGENTVERSE_BASE}/services/{AGENT_ADDRESS}",
            headers=headers,
            json=service_profile,
            timeout=30.0,
        )
        if update_response.status_code == 200:
            print("Profile updated successfully.")
        else:
            print(f"Update failed: {update_response.status_code} — {update_response.text}")
    else:
        print(f"Registration failed: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    register()
```

---

## Step 4: Calling LINGO from Another uAgent

Here is how a third-party booking agent discovers and calls LINGO:

```python
# booking_agent.py — calls LINGO from another uAgent
from uagents import Agent, Context
from uagents.query import query
from lingo_messages import DeployAgentRequest, DeployAgentResponse, SendMessageRequest, SendMessageResponse

# Known LINGO uAgent address (from AgentVerse listing or Almanac lookup)
LINGO_AGENT_ADDRESS = "agent1qvlingo0phoneagentsvc2024xkp7mnqr4wbvl9skirk"

booking_agent = Agent(name="booking-agent", seed="booking-agent-seed-phrase", port=8002)


@booking_agent.on_event("startup")
async def bootstrap(ctx: Context):
    # Deploy a phone agent for a new client
    ctx.logger.info("Deploying LINGO phone agent for new client...")

    deploy_request = DeployAgentRequest(
        business_name="City Auto Repair",
        phone_number="+13235557890",
        system_prompt=(
            "You are the AI receptionist for City Auto Repair. "
            "Schedule service appointments, give estimates, and answer questions about our services. "
            "We specialize in oil changes, brake service, and diagnostics. "
            "Hours: Mon–Fri 8am–6pm, Sat 9am–3pm."
        ),
        escalation_email="owner@cityautorepair.com",
        plan="starter",
        timezone="America/Los_Angeles",
    )

    # Send a query (synchronous request-response) to LINGO
    response = await query(
        destination=LINGO_AGENT_ADDRESS,
        message=deploy_request,
        timeout=30,
    )

    if isinstance(response, DeployAgentResponse):
        ctx.logger.info(f"Phone agent deployed: {response.agent_id}")
        ctx.logger.info(f"Phone number active: {response.phone_number}")
        # Store agent_id in your booking agent's state for later use
        ctx.storage.set("lingo_agent_id", response.agent_id)
    else:
        ctx.logger.error(f"Deployment failed: {response}")


@booking_agent.on_interval(period=900.0)  # Every 15 minutes
async def sync_appointments(ctx: Context):
    """Poll LINGO for new bookings and sync to calendar."""
    agent_id = ctx.storage.get("lingo_agent_id")
    if not agent_id:
        return

    from lingo_messages import GetTranscriptsRequest, GetTranscriptsResponse
    from datetime import datetime, timedelta, timezone

    fifteen_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()

    response = await query(
        destination=LINGO_AGENT_ADDRESS,
        message=GetTranscriptsRequest(
            agent_id=agent_id,
            interaction_type="call",
            limit=50,
        ),
        timeout=30,
    )

    if isinstance(response, GetTranscriptsResponse):
        bookings = [
            t for t in response.transcripts
            if t.get("intent") == "appointment_booking"
        ]
        ctx.logger.info(f"New bookings to sync: {len(bookings)}")
        # Write to calendar...


booking_agent.run()
```

---

## Step 5: Full Deployment Checklist

```
[ ] 1. pip install uagents uagents-ai-engine httpx python-dotenv
[ ] 2. Create .env with LINGO_API_KEY, AGENT_SEED, AGENTVERSE_API_KEY
[ ] 3. python lingo_agent.py              # note the printed agent address
[ ] 4. Add agent address to .env as LINGO_AGENT_ADDRESS
[ ] 5. python register_agentverse.py      # publish to AgentVerse marketplace
[ ] 6. Verify listing at agentverse.ai/agents/<your-address>
[ ] 7. Test from another agent using the booking_agent.py example
```

---

## Troubleshooting

**"Insufficient funds" on startup**  
The agent wallet needs FET tokens to register on the Almanac contract. `fund_agent_if_low()` handles this automatically on testnet. For mainnet, send FET to the wallet address printed at startup.

**Agent address changes between runs**  
You must use a fixed `seed` in the `Agent()` constructor. A deterministic seed produces a deterministic address. Never omit the `seed` parameter for a production service agent.

**AgentVerse registration returns 401**  
Regenerate your AgentVerse API key at [agentverse.ai/profile/api-keys](https://agentverse.ai/profile/api-keys). Keys expire after 90 days of inactivity.

**LINGO API returns 401**  
Regenerate your LINGO API key at [app.lingo-agent.ai/settings/api](https://app.lingo-agent.ai/settings/api).

**Messages from other agents not received**  
Ensure port 8001 is open and the `endpoint` URL in the `Agent()` constructor is publicly reachable. For local testing, use [ngrok](https://ngrok.com): `ngrok http 8001` and set the endpoint to the ngrok HTTPS URL.

---

## Resources

- [Fetch.ai uAgents documentation](https://docs.fetch.ai/guides/agents/getting-started/create-a-uagent/)
- [AgentVerse marketplace](https://agentverse.ai)
- [Almanac contract](https://docs.fetch.ai/references/contracts/uagents-almanac/almanac-overview/)
- [LINGO GitHub](https://github.com/lucsanscartier/lingo-agent)
- [LINGO landing page](https://www.perplexity.ai/computer/a/lingo-ai-phone-agents-uf7YK78jSo2fy2KjyMS.ag)
- [LINGO OpenAPI spec](https://api.lingo-agent.ai/v1/openapi.json)
