"""Persistent HTTP surface for the BrowseComp-Plus Standard search tool."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from rag_system.retrieval.standard import StandardSearchTool


def trace_payload(trace) -> dict[str, Any]:
    return {
        "top_k": trace.top_k,
        "snippet_max_tokens": trace.snippet_max_tokens,
        "hits": [
            {
                "docid": hit.document_id,
                "score": hit.score,
                "snippet": hit.snippet,
                "snippet_token_count": hit.snippet_token_count,
            }
            for hit in trace.hits
        ],
    }


def make_standard_search_handler(search_tool: StandardSearchTool):
    class StandardSearchHandler(BaseHTTPRequestHandler):
        server_version = "rag-system-standard-search/0.1"

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json(404, {"error": "not found"})
                return
            self._send_json(
                200,
                {
                    "status": "ok",
                    "top_k": search_tool.top_k,
                    "snippet_max_tokens": search_tool.snippet_max_tokens,
                },
            )

        def do_POST(self) -> None:
            if self.path != "/search":
                self._send_json(404, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 1_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                query = payload.get("query") if isinstance(payload, dict) else None
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("query must be a non-empty string")
                result = trace_payload(search_tool.search(query))
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(502, {"error": type(exc).__name__})
                return
            self._send_json(200, result)

        def log_message(self, format: str, *args: object) -> None:
            # The path and status are safe; query bodies are never logged.
            print(f"{self.address_string()} {format % args}", flush=True)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return StandardSearchHandler


def serve_standard_search(
    search_tool: StandardSearchTool,
    host: str,
    port: int,
) -> None:
    server = HTTPServer((host, port), make_standard_search_handler(search_tool))
    print(
        f"standard search listening on {host}:{port} "
        f"(top_k={search_tool.top_k}, snippet_max_tokens={search_tool.snippet_max_tokens})",
        flush=True,
    )
    server.serve_forever()
