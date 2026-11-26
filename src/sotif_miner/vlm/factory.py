from __future__ import annotations

from .base import VLMBackend


def build_backend(cfg: dict) -> VLMBackend:
    v = cfg["vlm"]
    kind = v["backend"]
    if kind == "ollama":
        from .ollama import OllamaBackend

        o = v["ollama"]
        return OllamaBackend(o["model"], o["base_url"], o.get("num_ctx", 16384), o.get("timeout_s", 600.0),
                             v.get("image_max_side", 768))
    if kind == "vllm":
        from .openai_compat import OpenAICompatBackend

        o = v["vllm"]
        return OpenAICompatBackend(o["model"], o["base_url"], disable_thinking=o.get("disable_thinking", True),
                                   max_tokens=v.get("max_tokens", 2048), max_side=v.get("image_max_side", 768))
    raise ValueError(f"unknown vlm backend '{kind}' (use ollama or vllm; tests use ScriptedBackend)")
