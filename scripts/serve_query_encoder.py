#!/usr/bin/env python3
"""Serve exact Tevatron Qwen3 query embeddings from the PCIe A100 on g3."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer "
    "the query\nQuery:"
)


class Qwen3TevatronQueryEncoder:
    def __init__(
        self,
        model_path: Path,
        *,
        max_length: int,
        attention_backend: str | None,
    ) -> None:
        try:
            import torch
            from tevatron.retriever.driver.encode import DenseModel
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "run this script in the BrowseComp-Plus environment with "
                "Tevatron, Transformers, and PyTorch installed"
            ) from exc

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self._torch = torch
        self._device = torch.device("cuda")
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            padding_side="left",
        )
        load_options = {
            "pooling": "eos",
            "normalize": True,
            "torch_dtype": torch.float16,
        }
        if attention_backend:
            load_options["attn_implementation"] = attention_backend
        self._model = DenseModel.load(str(model_path), **load_options)
        self._model = self._model.to(self._device)
        self._model.eval()

    def encode(self, query: str) -> list[float]:
        batch = self._tokenizer(
            QUERY_PREFIX + query,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        batch = {name: value.to(self._device) for name, value in batch.items()}
        with self._torch.inference_mode():
            representation = self._model.encode_query(batch)
        return representation[0].float().cpu().tolist()


def make_handler(encoder: Qwen3TevatronQueryEncoder):
    class QueryEncoderHandler(BaseHTTPRequestHandler):
        server_version = "rag-system-query-encoder/0.1"

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json(404, {"error": "not found"})
                return
            self._send_json(200, {"status": "ok"})

        def do_POST(self) -> None:
            if self.path != "/encode":
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
                embedding = encoder.encode(query.strip())
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(
                200,
                {"dimension": len(embedding), "embedding": embedding},
            )

        def log_message(self, format: str, *args: object) -> None:
            # Do not place benchmark query text in access logs.
            print(f"{self.address_string()} {format % args}", flush=True)

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return QueryEncoderHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--attention-backend",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    args = parser.parse_args()
    encoder = Qwen3TevatronQueryEncoder(
        args.model_path.expanduser().resolve(),
        max_length=args.max_length,
        attention_backend=args.attention_backend,
    )
    server = HTTPServer((args.host, args.port), make_handler(encoder))
    print(f"query encoder listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
