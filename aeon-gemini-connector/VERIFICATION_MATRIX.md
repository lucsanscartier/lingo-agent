# Aeon Verification Matrix

Verified date: 2026-06-12

## Final matrix

```text
Aeon read: PASS
Aeon persistent gateway write: PASS
Bridge default: PASS
Rehydrate exposes persisted writes: PASS
Linear write: PASS
Direct private core from HF job: FAIL/BYPASSED
HF repo write: FAIL
```

## What works

### Aeon read through MCP gateway

PASS. Gemini and other agents should use:

```text
https://lucsanscartier-aeon-twin-mcp.hf.space/api/rehydrate
```

### Persistent gateway writes

PASS. The gateway persists fallback writes and exposes them through:

```text
/api/gateway/ingested
/api/rehydrate -> context.ingested
/api/rehydrate -> context.gateway_persisted_state
```

### Bridge default

PASS. `linear_aeon_bridge.py` now defaults to the visible MCP gateway rather than the private core.

### Linear write

PASS through GPT's Linear connector. Gemini may not have this same connector unless configured separately.

## What is intentionally bypassed

### Direct private core from HF job

FAIL/BYPASSED. Do not use this as the primary agent path. The gateway exists so agents do not fail on private-core 404s.

## What still needs infrastructure polish

### HF repo write

FAIL. `HF_REPO_WRITE_TOKEN` needs write access to:

```text
lucsanscartier/aeon-twin-mcp
```

This is only needed for repo commits. It is not required for Gemini to read Aeon state or use the gateway.

## Current canonical decision

```text
Use the visible MCP gateway as canonical.
Treat the private core as a backing service.
Do not block agents on private-core 404s.
```
