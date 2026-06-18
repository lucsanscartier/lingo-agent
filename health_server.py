"""
health_server.py — FastAPI status + MCP surface for Hugging Face Spaces.

This exposes:
- /health, /status, /metrics for runtime status
- /mcp and /mcp/linear for HuggingChat MCP connection

Secrets required for live Linear reads:
- LINEAR_API_KEY
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import memory

logger = logging.getLogger(__name__)

STARTED_AT = time.time()
STATE: Dict[str, Any] = {
    "worker_started": False,
    "last_error": None,
    "active_calls": 0,
    "calls_started": 0,
    "calls_ended": 0,
    "escalations": 0,
}

MCP_SERVER_NAME = "aeon-linear-mcp"
MCP_SERVER_VERSION = "0.1.0"
LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"

app = FastAPI(title="LINGO Runtime + AEON Linear MCP", version="0.3.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "lingo-agent",
        "version": "0.3.0",
        "uptime_seconds": round(time.time() - STARTED_AT, 2),
        "worker_started": STATE["worker_started"],
        "mcp": {
            "name": MCP_SERVER_NAME,
            "urls": ["/mcp", "/mcp/linear"],
            "linear_configured": bool(os.getenv("LINEAR_API_KEY")),
        },
    }


@app.get("/status")
def status() -> Dict[str, Any]:
    return {
        "ok": True,
        "state": STATE,
        "callers_known": len(memory.all_callers()),
        "mcp_server": MCP_SERVER_NAME,
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    return {
        "ok": True,
        "calls_started": STATE["calls_started"],
        "calls_ended": STATE["calls_ended"],
        "active_calls": STATE["active_calls"],
        "escalations": STATE["escalations"],
        "recent_events": memory.recent_events(limit=20),
    }


@app.get("/mcp")
@app.get("/mcp/linear")
def mcp_sse_hint() -> StreamingResponse:
    """Small SSE compatibility surface for clients that probe before POSTing."""

    async def events():
        yield "event: endpoint\n"
        yield "data: /mcp\n\n"
        yield "event: ready\n"
        yield f"data: {json.dumps({'name': MCP_SERVER_NAME, 'transport': 'streamable-http'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/mcp")
@app.post("/mcp/linear")
async def mcp_post(request: Request) -> JSONResponse:
    payload = await request.json()
    if isinstance(payload, list):
        return JSONResponse([_handle_jsonrpc(message) for message in payload])
    return JSONResponse(_handle_jsonrpc(payload))


def _jsonrpc_result(message_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _jsonrpc_error(message_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _text_content(text: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _handle_jsonrpc(message: Dict[str, Any]) -> Dict[str, Any]:
    method = message.get("method")
    message_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        return _jsonrpc_result(
            message_id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": MCP_SERVER_NAME, "version": MCP_SERVER_VERSION},
            },
        )

    if method in {"notifications/initialized", "ping"}:
        return _jsonrpc_result(message_id, {})

    if method == "tools/list":
        return _jsonrpc_result(message_id, {"tools": _mcp_tools()})

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}
        return _jsonrpc_result(message_id, _call_tool(tool_name, arguments))

    return _jsonrpc_error(message_id, -32601, f"Unsupported MCP method: {method}")


def _mcp_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": "linear_search",
            "description": "Search Linear issues, projects, and documents in the gpt-and-luc workspace.",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "linear_get_issue",
            "description": "Fetch a Linear issue by identifier, for example GPT-69.",
            "inputSchema": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
        {
            "name": "aeon_rehydrate_linear",
            "description": "Return the core AEON / Shard Network Linear rehydration issue packet.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


def _call_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if tool_name == "linear_search":
            return _text_content(_linear_search(arguments.get("query", "")))
        if tool_name == "linear_get_issue":
            return _text_content(_linear_get_issue(arguments.get("id", "")))
        if tool_name == "aeon_rehydrate_linear":
            packet = []
            for issue_id in ["GPT-48", "GPT-61", "GPT-63", "GPT-64", "GPT-66", "GPT-69"]:
                packet.append(_linear_get_issue(issue_id))
            return _text_content("\n\n---\n\n".join(packet))
        return _text_content(f"Unknown tool: {tool_name}")
    except Exception as exc:
        logger.exception("MCP tool call failed")
        return _text_content(f"MCP tool call failed: {exc}")


def _linear_graphql(query: str, variables: Dict[str, Any] | None = None) -> Dict[str, Any]:
    token = os.getenv("LINEAR_API_KEY")
    if not token:
        return {
            "error": "LINEAR_API_KEY is not set on this Hugging Face Space. Add it as a Space secret to enable live Linear reads."
        }

    body = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
    req = urllib.request.Request(
        LINEAR_GRAPHQL_URL,
        data=body,
        headers={
            "Authorization": token,
            "Content-Type": "application/json",
            "User-Agent": MCP_SERVER_NAME,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _linear_search(search_query: str) -> str:
    gql = """
    query SearchIssues($term: String!) {
      issues(filter: { or: [
        { title: { containsIgnoreCase: $term } },
        { description: { containsIgnoreCase: $term } }
      ]}, first: 10, orderBy: updatedAt) {
        nodes {
          identifier
          title
          url
          priority
          updatedAt
          state { name type }
          project { name }
        }
      }
    }
    """
    data = _linear_graphql(gql, {"term": search_query})
    if "error" in data:
        return data["error"]
    nodes = (((data.get("data") or {}).get("issues") or {}).get("nodes") or [])
    if not nodes:
        return f"No Linear issues found for query: {search_query}"
    return json.dumps(nodes, indent=2)


def _linear_get_issue(identifier: str) -> str:
    gql = """
    query Issue($id: String!) {
      issue(id: $id) {
        identifier
        title
        description
        url
        priority
        updatedAt
        state { name type }
        project { name }
        comments(first: 10) { nodes { body createdAt user { name } } }
      }
    }
    """
    data = _linear_graphql(gql, {"id": identifier})
    if "error" in data:
        return data["error"]
    issue = ((data.get("data") or {}).get("issue"))
    if not issue:
        return f"Linear issue not found: {identifier}"
    return json.dumps(issue, indent=2)


def mark_worker_started() -> None:
    STATE["worker_started"] = True


def set_last_error(error: str | None) -> None:
    STATE["last_error"] = error


def record_call_start() -> None:
    STATE["calls_started"] += 1
    STATE["active_calls"] += 1


def record_call_end() -> None:
    STATE["calls_ended"] += 1
    STATE["active_calls"] = max(0, STATE["active_calls"] - 1)


def record_escalation() -> None:
    STATE["escalations"] += 1


def start_health_server(host: str = "0.0.0.0", port: int = 7860) -> threading.Thread:
    """Start the HTTP health/MCP server in a daemon thread."""

    def _run() -> None:
        logger.info("Starting health/MCP server on %s:%s", host, port)
        uvicorn.run(app, host=host, port=port, log_level="info")

    thread = threading.Thread(target=_run, name="lingo-health-mcp-server", daemon=True)
    thread.start()
    return thread
