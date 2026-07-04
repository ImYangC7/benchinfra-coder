#!/usr/bin/env python3
"""Trivial round-robin proxy forwarding /v1/* POSTs across N backend HF servers.
Lets verilog-eval's `make -jN` fan out across 8 single-GPU backends."""
import itertools
import json
import os
import re
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKENDS = sys.argv[2:] if len(sys.argv) > 2 else []
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
_rr = itertools.cycle(BACKENDS)

# THINK=1 (default): let the model reason. We DON'T force enable_thinking=false,
# and we strip <think>...</think> from the response content so the downstream
# bench code-extractors see only the post-reasoning answer.
# THINK=0: force enable_thinking=false (old non-thinking behavior).
THINK = os.environ.get("THINK", "1") not in ("0", "false", "False", "")


def strip_think(text):
    if not text:
        return text
    # remove well-formed <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Qwen chat template injects the opening <think> itself, so the returned
    # content usually starts mid-reasoning and only emits a closing </think>.
    # Strip everything up to and including the first orphan </think>.
    text = re.sub(r"\A.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.lstrip("\n")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, json.dumps({"status": "ok", "backends": BACKENDS}).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        is_chat = self.path.endswith("/chat/completions")
        # THINK=0: force enable_thinking=false so <think> never appears (old
        # non-thinking behavior). THINK=1: leave thinking on (chat template
        # default) and strip <think> from the response instead.
        if is_chat and not THINK:
            try:
                body = json.loads(data)
                kw = body.setdefault("chat_template_kwargs", {})
                kw.setdefault("enable_thinking", False)
                data = json.dumps(body).encode()
            except Exception:
                pass
        backend = next(_rr)
        url = backend.rstrip("/") + self.path
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                resp = r.read()
            if is_chat and THINK:
                # strip <think> from each choice so bench extractors see clean code
                try:
                    obj = json.loads(resp)
                    for ch in obj.get("choices", []):
                        msg = ch.get("message")
                        if msg and isinstance(msg.get("content"), str):
                            msg["content"] = strip_think(msg["content"])
                        # /completions-style choices carry .text
                        if isinstance(ch.get("text"), str):
                            ch["text"] = strip_think(ch["text"])
                    resp = json.dumps(obj).encode()
                except Exception:
                    pass
            elif self.path.endswith("/completions") and THINK:
                try:
                    obj = json.loads(resp)
                    for ch in obj.get("choices", []):
                        if isinstance(ch.get("text"), str):
                            ch["text"] = strip_think(ch["text"])
                    resp = json.dumps(obj).encode()
                except Exception:
                    pass
            self._send(200, resp)
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}).encode())


if __name__ == "__main__":
    print(f"[proxy] :{PORT} think={THINK} -> {BACKENDS}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
