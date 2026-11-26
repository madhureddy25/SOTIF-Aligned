"""OpenAI-compatible backend for a vLLM / SGLang server (e.g. Qwen3-VL-8B-Instruct on a larger GPU).
Structured output via `response_format={"type": "json_schema", ...}`; thinking disabled through
the chat template."""

from __future__ import annotations

from .base import _RetryingBackend


class OpenAICompatBackend(_RetryingBackend):
    def __init__(self, model: str = "Qwen/Qwen3-VL-8B-Instruct", base_url: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY", disable_thinking: bool = True, max_tokens: int = 2048,
                 max_side: int = 768, client=None):
        super().__init__(max_side)
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client, self.model = client, model
        self.disable_thinking, self.max_tokens = disable_thinking, max_tokens
        self.name = f"openai-compat:{model}"

    def _call(self, images_b64, prompt, schema, system):
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in images_b64]
        content.append({"type": "text", "text": prompt})
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        extra = {"chat_template_kwargs": {"enable_thinking": False}} if self.disable_thinking else {}
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=self.max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": True}},
            extra_body=extra)
        return r.choices[0].message.content or ""
