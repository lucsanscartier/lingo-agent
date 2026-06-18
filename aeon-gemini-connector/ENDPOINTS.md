# Aeon Gateway Endpoints

Base URL:

```text
https://lucsanscartier-aeon-twin-mcp.hf.space
```

MCP URL:

```text
https://lucsanscartier-aeon-twin-mcp.hf.space/mcp
```

HuggingChat Linear MCP URL:

```text
https://lucsanscartier-aeon-twin-mcp.hf.space/mcp
```

Compatibility alias:

```text
https://lucsanscartier-aeon-twin-mcp.hf.space/mcp/linear
```

MCP server name:

```text
aeon-linear-mcp
```

## Read routes

| Route | Purpose | Verified state |
|---|---|---:|
| `/health` | Gateway health | PASS |
| `/api/auth/modes` | Auth mode summary | PASS |
| `/api/rehydrate` | Main rehydration packet | PASS |
| `/api/gpt/context` | Context-only packet | PASS |
| `/api/zero-key/summary` | Zero-key registry summary | PASS |
| `/api/zero-key/endpoints` | Zero-key endpoint registry | PASS |
| `/api/oae/status` | OAE bridge status | PASS |
| `/api/oae/payload` | OAE bridge payload | PASS |
| `/api/gateway/ingested` | Persisted fallback writes | PASS |
| `/api/gateway/directives` | Persisted directives | PASS |
| `/mcp` | Streamable HTTP MCP endpoint for HuggingChat | IMPLEMENTED |
| `/mcp/linear` | Compatibility alias for Linear MCP | IMPLEMENTED |

## Write route

| Route | Purpose | Verified state |
|---|---|---:|
| `/api/ingest` | Persist fallback write into visible gateway state | PASS |

## MCP tools

| Tool | Purpose |
|---|---|
| `linear_search` | Search Linear issues by title/description |
| `linear_get_issue` | Fetch Linear issue by identifier, e.g. GPT-69 |
| `aeon_rehydrate_linear` | Pull core AEON / Shard Network Linear packet |

Required Hugging Face Space secret for live Linear reads:

```text
LINEAR_API_KEY
```

## Verified write behavior

Expected state after write:

```text
/api/ingest -> gateway_fallback_write_persisted
/api/gateway/ingested -> contains persisted item
/api/rehydrate -> context.ingested contains persisted item
```

## Canonical policy

Use the visible MCP gateway as canonical. The private core is a backing service. Do not block on private-core 404s.

## Gemini read workflow

1. Fetch `/api/rehydrate`.
2. Read `context.rules_for_gpt`.
3. Read `context.architecture_manifest`, `context.auth_modes`, and `context.oae_core_bridge`.
4. Check `context.gateway_persisted_state` and `context.ingested` for persisted gateway writes.
5. Label each claim using the evidence labels in `aeon_gateway_manifest.json`.

## Gemini write workflow

Only write if Gemini has a properly configured environment with the needed secrets. Never ask the user to paste token values into chat.

For ordinary agent operation, reads are enough. Write only for status events, bridge notes, or explicit user-approved updates.
