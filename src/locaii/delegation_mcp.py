"""Newline JSON-RPC MCP adapter for a running loCAII delegation connection.

Run as ``python -m locaii.delegation_mcp --connection /private/connection.json``.
Only protocol JSON goes to stdout. This adapter never launches its own model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

from .delegation import INSTRUCTIONS, MAX_BODY


def _schema(properties, required):
    """Disallow implicit scope or permission overrides in all public tools."""
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TEXT = {"type": "string", "minLength": 1, "maxLength": 32000}
IDENTIFIER = {"type": "string", "minLength": 1, "maxLength": 160}
TOOLS = [
    {
        "name": "local_model_info",
        "description": "Inspect the connected subordinate model, workspace and frozen permission preferences.",
        "inputSchema": _schema({}, []),
    },
    {
        "name": "delegate_task",
        "description": "Delegate a bounded simple or moderate task. Returns an immediate job receipt. Use a unique stable request_id for safe retries, then task_status(wait_seconds=20).",
        "inputSchema": _schema(
            {"task": TEXT, "title": IDENTIFIER, "request_id": IDENTIFIER},
            ["task", "request_id"],
        ),
    },
    {
        "name": "task_status",
        "description": "Inspect actual output, tool receipts, progress, local token usage and status. Use wait_seconds=20 to wait efficiently for completion; do not poll rapidly.",
        "inputSchema": _schema(
            {
                "job_id": IDENTIFIER,
                "wait_seconds": {"type": "number", "minimum": 0, "maximum": 25},
            },
            ["job_id"],
        ),
    },
    {
        "name": "continue_task",
        "description": "Request a bounded correction or verification in a terminal task's saved session. Returns a new job receipt; original evidence is retained.",
        "inputSchema": _schema(
            {"job_id": IDENTIFIER, "task": TEXT, "request_id": IDENTIFIER},
            ["job_id", "task", "request_id"],
        ),
    },
    {
        "name": "cancel_task",
        "description": "Cancel queued work or request cooperative interruption of a running subordinate task. Check task_status for confirmed completion of cancellation.",
        "inputSchema": _schema({"job_id": IDENTIFIER}, ["job_id"]),
    },
]


class _NoRedirect(HTTPRedirectHandler):
    """Never forward a bridge credential to a redirect destination."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class BridgeClient:
    """Load one frozen local connection; revocation requires a new MCP process."""

    def __init__(self, path: Path):
        raw = Path(path).read_bytes()
        if len(raw) > 16384:
            raise ValueError("Invalid connection file.")
        connection = json.loads(raw)
        endpoint = urlparse(connection.get("endpoint", ""))
        if (
            endpoint.scheme != "http"
            or endpoint.hostname != "127.0.0.1"
            or endpoint.path != "/rpc"
            or not endpoint.port
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("Connection must use the local loCAII loopback endpoint.")
        token = connection.get("token")
        if (
            not isinstance(token, str)
            or not 32 <= len(token) <= 128
            or not token.isascii()
            or any(char.isspace() for char in token)
        ):
            raise ValueError("Invalid local connection credential.")
        self.endpoint = connection["endpoint"]
        self._token = token
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def call(self, method, arguments=None):
        """Send bounded authenticated requests without environment proxy routing."""
        data = json.dumps(
            {"method": method, "arguments": {} if arguments is None else arguments}
        ).encode()
        if len(data) > MAX_BODY:
            raise ValueError("Delegation request is too large.")
        request = Request(
            self.endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self._token,
            },
        )
        try:
            with self._opener.open(request, timeout=35) as response:
                raw = response.read(1048577)
                if len(raw) > 1048576:
                    raise ValueError("Local response exceeds its size limit.")
                value = json.loads(raw)
        except HTTPError as error:
            try:
                message = json.loads(error.read(4096)).get(
                    "error", "Local request rejected."
                )
            except (ValueError, TypeError):
                message = "Local request rejected."
            raise ValueError(message) from None
        except (URLError, OSError):
            raise ValueError(
                "loCAII is disconnected or unavailable. Reconnect from the application."
            ) from None
        if "error" in value:
            raise ValueError(str(value["error"]))
        return value["result"]


def handle(message, client):
    """Implement the MCP lifecycle and tool calls without executing arbitrary RPC."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "Invalid JSON-RPC request."},
        }
    if "id" not in message:
        return None  # Notifications, including initialized/cancelled, need no reply.
    response = {"jsonrpc": "2.0", "id": message["id"]}
    method, params = message.get("method"), message.get("params", {})
    if not isinstance(params, dict):
        response["error"] = {"code": -32602, "message": "params must be an object."}
        return response
    if method == "initialize":
        requested = params.get("protocolVersion")
        protocol = (
            requested
            if requested in {"2024-11-05", "2025-03-26", "2025-06-18"}
            else "2024-11-05"
        )
        response["result"] = {
            "protocolVersion": protocol,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "locaii-subordinate", "version": "1.0.0"},
            "instructions": INSTRUCTIONS,
        }
    elif method == "ping":
        response["result"] = {}
    elif method == "tools/list":
        response["result"] = {"tools": TOOLS}
    elif method == "tools/call":
        try:
            name = params.get("name")
            if name not in {tool["name"] for tool in TOOLS}:
                raise ValueError("Unknown local model tool.")
            result = client.call(name, params.get("arguments", {}))
            response["result"] = {
                "content": [
                    {"type": "text", "text": json.dumps(result, ensure_ascii=False)}
                ],
                "isError": False,
            }
        except (ValueError, TypeError, KeyError) as error:
            response["result"] = {
                "content": [{"type": "text", "text": str(error)[:600]}],
                "isError": True,
            }
    else:
        response["error"] = {"code": -32601, "message": "Method not found."}
    return response


def main(argv=None):
    """Serve one JSON object per line until the supervisor closes stdin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        client = BridgeClient(args.connection)
    except (OSError, ValueError, TypeError):
        print(
            "loCAII connection is unavailable. Connect a supervisor in the application first.",
            file=sys.stderr,
        )
        return 1
    stream = sys.stdin.buffer
    while True:
        line = stream.readline(MAX_BODY + 2)
        if not line:
            break
        if len(line) > MAX_BODY:
            # Drain the oversized line to preserve subsequent message framing.
            while line and not line.endswith(b"\n"):
                line = stream.readline(MAX_BODY + 2)
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Request exceeds maximum size."},
            }
        else:
            try:
                response = handle(json.loads(line), client)
            except (ValueError, UnicodeDecodeError):
                response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error."},
                }
        if response is not None:
            print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
