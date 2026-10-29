#!/usr/bin/env python3
"""Minimal OpenAI-compatible stub for testing the eval path.

Serves /v1/models, /v1/chat/completions and /v1/completions (with echo +
logprobs, which lm-eval needs for multiple-choice tasks), so the whole
lm-eval -> HTTP -> results pipeline can be exercised without a GPU.

Usage: python stub_openai_server.py <port>
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
ANSWER = "The answer is 42."


def _logprobs_for(text):
    """Fake per-token logprobs for an echoed prompt."""
    text = str(text)
    tokens = text.split(" ") or [""]
    return {
        "tokens": tokens,
        "token_logprobs": [-0.1] * len(tokens),
        "top_logprobs": [{tokens[0]: -0.1}] * len(tokens),
        "text_offset": [0] * len(tokens),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send({"object": "list", "data": [{"id": "stub", "object": "model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        try:
            self._handle_post()
        except Exception as e:  # never drop the connection: lm-eval reports it as a network error
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)

    def _handle_post(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            req = {}

        if self.path.rstrip("/").endswith("/chat/completions"):
            self._send({
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "model": "stub",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ANSWER},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })
            return

        # /v1/completions — echo the prompt so loglikelihood tasks work
        prompts = req.get("prompt", "")
        if isinstance(prompts, str):
            prompts = [prompts]
        choices = []
        for i, p in enumerate(prompts):
            if isinstance(p, list):  # token ids
                p = " ".join(str(t) for t in p)
            text = str(p) + ANSWER
            choices.append({
                "index": i,
                "text": text,
                "logprobs": _logprobs_for(text),
                "finish_reason": "stop",
            })
        self._send({
            "id": "cmpl-stub",
            "object": "text_completion",
            "model": "stub",
            "choices": choices,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })


if __name__ == "__main__":
    from http.server import ThreadingHTTPServer
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
