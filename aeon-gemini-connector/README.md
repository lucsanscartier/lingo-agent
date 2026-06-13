# Aeon Gemini Connector Handoff

This folder is a public handoff packet for Gemini or any other agent that needs to connect to the Aeon GPT Twin through the verified gateway path.

## Canonical connection path

Agent-facing API:

```text
https://lucsanscartier-aeon-twin-mcp.hf.space
```

MCP endpoint:

```text
https://lucsanscartier-aeon-twin-mcp.hf.space/mcp
```

Decision:

```text
Use the visible MCP gateway as canonical.
Treat the private core as a backing service.
Do not block agents on private-core 404s.
```

## Verified matrix

```text
Aeon read: PASS
Aeon persistent gateway write: PASS
Bridge default: PASS
Rehydrate exposes persisted writes: PASS
Linear write: PASS
Direct private core from HF job: FAIL/BYPASSED
HF repo write: FAIL
```

## Important rule

Do not ask the user to paste secrets into chat. Use the public gateway for reads. Use configured environment secrets only for writes.

## Files in this handoff

- `GEMINI_CONNECT_PROMPT.md` - copy/paste prompt for Gemini.
- `aeon_gateway_manifest.json` - machine-readable endpoint and status manifest.
- `ENDPOINTS.md` - routes and read/write behavior.
- `VERIFICATION_MATRIX.md` - final verified state and remaining blocker.

## First request Gemini should make

Fetch:

```text
https://lucsanscartier-aeon-twin-mcp.hf.space/api/rehydrate
```

Then read these sections before making claims:

```text
context.auth_modes
context.architecture_manifest
context.zero_key_endpoints
context.oae_core_bridge
context.live
context.oracles
context.synchrony
context.keymaster
context.privacy_connections
context.gateway_persisted_state
context.ingested
```

## Safety and evidence labels

Gemini should label data as one of:

```text
live
derived
remembered/staged
theoretical
simulation
proxy
key-required
aggregate-only
architectural/local
candidate
unverified
image-only
unavailable
```

It must not invent telemetry, expose secrets, expose raw IPs, or claim correlation is causation.
