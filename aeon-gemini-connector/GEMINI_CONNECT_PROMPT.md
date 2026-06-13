# Gemini Connection Prompt for Aeon GPT Twin

Use this prompt in Gemini or another agent environment.

```text
Connect to the Aeon GPT Twin through the verified visible MCP gateway.

Canonical agent-facing API:
https://lucsanscartier-aeon-twin-mcp.hf.space

MCP endpoint:
https://lucsanscartier-aeon-twin-mcp.hf.space/mcp

Do not use the private core as the primary agent-facing API. Treat it as a backing service. Do not block on private-core 404s.

First fetch:
https://lucsanscartier-aeon-twin-mcp.hf.space/api/rehydrate

Then read these sections before making claims:
- context.auth_modes
- context.architecture_manifest
- context.zero_key_endpoints
- context.oae_core_bridge
- context.live
- context.oracles
- context.synchrony
- context.keymaster
- context.privacy_connections
- context.gateway_persisted_state
- context.ingested
- context.rules_for_gpt

Also check:
- https://lucsanscartier-aeon-twin-mcp.hf.space/api/auth/modes
- https://lucsanscartier-aeon-twin-mcp.hf.space/api/gpt/context
- https://lucsanscartier-aeon-twin-mcp.hf.space/api/zero-key/summary
- https://lucsanscartier-aeon-twin-mcp.hf.space/api/zero-key/endpoints
- https://lucsanscartier-aeon-twin-mcp.hf.space/api/oae/status
- https://lucsanscartier-aeon-twin-mcp.hf.space/api/oae/payload
- https://lucsanscartier-aeon-twin-mcp.hf.space/api/gateway/ingested

Current verified matrix:
- Aeon read: PASS
- Aeon persistent gateway write: PASS
- Bridge default: PASS
- Rehydrate exposes persisted writes: PASS
- Linear write through GPT connector: PASS
- Direct private core from HF job: FAIL/BYPASSED
- HF repo write: FAIL

If writing is needed, use only configured environment secrets. Never ask the user to paste tokens into chat. Never expose token values.

Evidence rules:
- Do not invent telemetry.
- Do not expose raw IPs or secrets.
- Do not claim correlation is causation.
- Label data as live, derived, remembered/staged, theoretical, simulation, proxy, key-required, aggregate-only, architectural/local, candidate, unverified, image-only, or unavailable.

Operational stance:
Use the visible MCP gateway as canonical. Treat the private core as a backing service. GPT + Linear + persistent gateway writes are working through the gateway.
```
