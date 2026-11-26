"""Vision-language backend interface.

Every backend takes images + a prompt + a pydantic schema and returns a validated instance of
that schema (constrained JSON decoding where the server supports it, then strict validation).
Invalid JSON is retried once with an explicit correction message, then raised as VLMError, and
counted, so JSON-validity rates can be reported per backend (one of the ablation metrics).
"""

from __future__ import annotations

import base64
import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

import cv2
import numpy as np
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class VLMError(RuntimeError):
    pass


@dataclass
class VLMStats:
    calls: int = 0
    invalid_json: int = 0
    failures: int = 0
    seconds: float = 0.0
    per_call: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"calls": self.calls, "invalid_json_first_try": self.invalid_json, "failures": self.failures,
                "json_validity": 1 - self.invalid_json / self.calls if self.calls else None,
                "mean_s": self.seconds / self.calls if self.calls else None}


class VLMBackend(Protocol):
    name: str
    stats: VLMStats

    def generate(self, images: list[str | np.ndarray], prompt: str, schema: type[T], system: str = "") -> T: ...


def encode_image(img: str | Path | np.ndarray, max_side: int = 768, quality: int = 88) -> str:
    """Base64 JPEG, downscaled so the long side is <= max_side (keeps VLM token cost bounded)."""
    if isinstance(img, (str, Path)):
        arr = cv2.imread(str(img))
        if arr is None:
            raise FileNotFoundError(img)
    else:
        arr = img
    h, w = arr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VLMError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def inline_schema(schema: type[BaseModel]) -> dict:
    """JSON schema with every $ref inlined and $defs removed (some local servers' grammar
    converters do not resolve references)."""
    raw = schema.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                merged = copy.deepcopy(defs[name])
                merged.update({k: v for k, v in node.items() if k != "$ref"})
                return walk(merged)
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(raw)


def parse_or_none(schema: type[T], text: str) -> T | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    if "{" in text and not text.startswith("{"):
        text = text[text.find("{"):]
    if text.rfind("}") >= 0:
        text = text[: text.rfind("}") + 1]
    try:
        return schema.model_validate_json(text)
    except (ValidationError, ValueError):
        return None


class _RetryingBackend:
    """Shared retry/validation loop; subclasses implement `_call(images_b64, prompt, schema, system) -> str`."""

    name = "base"

    def __init__(self, max_side: int = 768):
        self.stats = VLMStats()
        self.max_side = max_side

    def _call(self, images_b64: list[str], prompt: str, schema: dict, system: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def generate(self, images, prompt, schema, system=""):
        imgs = [encode_image(i, self.max_side) for i in images]
        js = inline_schema(schema)
        t0 = time.time()
        self.stats.calls += 1
        try:
            text = self._call(imgs, prompt, js, system)
            out = parse_or_none(schema, text)
            if out is None:
                self.stats.invalid_json += 1
                fix = (prompt + "\n\nYour previous answer was not valid JSON for the required schema. "
                       "Return ONLY one JSON object that matches the schema exactly, with no other text.")
                out = parse_or_none(schema, self._call(imgs, fix, js, system))
            if out is None:
                self.stats.failures += 1
                raise VLMError(f"{self.name}: invalid JSON after retry for {schema.__name__}")
            return out
        finally:
            dt = time.time() - t0
            self.stats.seconds += dt
            self.stats.per_call.append(dt)
