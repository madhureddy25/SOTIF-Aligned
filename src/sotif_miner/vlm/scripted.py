"""Deterministic backend for tests and the CPU demo: a function decides the reply."""

from __future__ import annotations

from collections.abc import Callable

from .base import _RetryingBackend


class ScriptedBackend(_RetryingBackend):
    name = "scripted"

    def __init__(self, respond: Callable[[str, dict, int], str]):
        """respond(prompt, json_schema, n_images) -> raw text (normally JSON)."""
        super().__init__()
        self.respond = respond
        self.prompts: list[str] = []

    def generate(self, images, prompt, schema, system=""):
        self._n_images = len(images)
        self.prompts.append(prompt)
        return super().generate([], prompt, schema, system)

    def _call(self, images_b64, prompt, schema, system):
        return self.respond(prompt, schema, self._n_images)
