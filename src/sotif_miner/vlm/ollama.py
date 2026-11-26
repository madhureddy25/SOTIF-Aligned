"""Ollama backend (local GPU). Native /api/chat with `format=<JSON schema>` (constrained
decoding), `think=false`, temperature 0. Use an *-instruct tag: untagged Qwen3-VL builds may load
a thinking variant that can exhaust the token budget."""

from __future__ import annotations

import warnings

import httpx

from .base import VLMError, _RetryingBackend


class OllamaBackend(_RetryingBackend):
    def __init__(self, model: str = "qwen3-vl:4b-instruct", base_url: str = "http://localhost:11434",
                 num_ctx: int = 16384, timeout_s: float = 600.0, max_side: int = 768, keep_alive: str = "30m",
                 client: httpx.Client | None = None):
        super().__init__(max_side)
        if "instruct" not in model:
            warnings.warn(f"Ollama model '{model}' is not an -instruct tag; it may run in thinking mode",
                          stacklevel=2)
        self.model, self.base_url, self.num_ctx, self.keep_alive = model, base_url.rstrip("/"), num_ctx, keep_alive
        self.client = client or httpx.Client(timeout=timeout_s)
        self.name = f"ollama:{model}"

    def _call(self, images_b64, prompt, schema, system):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": images_b64})
        body = {"model": self.model, "messages": messages, "stream": False, "format": schema, "think": False,
                "keep_alive": self.keep_alive, "options": {"temperature": 0, "num_ctx": self.num_ctx}}
        try:
            r = self.client.post(f"{self.base_url}/api/chat", json=body)
        except httpx.ConnectError as e:
            raise VLMError(f"cannot reach Ollama at {self.base_url}; is `ollama serve` running?") from e
        if r.status_code == 404:
            raise VLMError(f"Ollama model '{self.model}' not found; run: ollama pull {self.model}")
        if r.status_code == 400:
            detail = r.text[:300]
            # usually "context length exceeded": retry once with a bigger window
            if self.num_ctx < 16384 and "context" in detail.lower() or "exceed" in detail.lower():
                body["options"]["num_ctx"] = 16384
                r = self.client.post(f"{self.base_url}/api/chat", json=body)
                if r.status_code == 200:
                    return r.json().get("message", {}).get("content", "")
            raise VLMError(f"Ollama rejected the request (400): {detail}")
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")
