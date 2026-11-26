"""Image / text embedders. `TinyEmbedder` is CPU-only and dependency-free (colour
thumbnail + gradient-orientation histogram) and is good enough for scene-level
rarity on small corpora; `HFImageEmbedder` (SigLIP / DINOv2) is the GPU default."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import cv2
import numpy as np


def l2n(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


class ImageEmbedder(Protocol):
    name: str
    dim: int

    def embed(self, images: list[np.ndarray]) -> np.ndarray: ...


class TextEmbedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class TinyEmbedder:
    name, dim = "tiny", 8 * 8 * 3 + 4 * 4 * 8 + 3

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        out = []
        for img in images:
            thumb = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
            gray = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)
            hog = []
            for by in range(4):
                for bx in range(4):
                    m = mag[by * 32:(by + 1) * 32, bx * 32:(bx + 1) * 32].ravel()
                    a = ang[by * 32:(by + 1) * 32, bx * 32:(bx + 1) * 32].ravel()
                    h, _ = np.histogram(a, bins=8, range=(0, 2 * np.pi), weights=m)
                    hog.append(h / (h.sum() + 1e-6))
            stats = np.array([gray.mean() / 255, gray.std() / 128, (gray > 250).mean()])
            out.append(np.r_[thumb - thumb.mean(), np.concatenate(hog), stats])
        return l2n(np.array(out, dtype=np.float32))


class HashTextEmbedder:
    """Feature-hashed bag of words/bigrams; deterministic CPU fallback for text."""

    name, dim = "hash-text", 256

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            toks = re.findall(r"[a-z0-9_]+", t.lower())
            for g in toks + [a + "_" + b for a, b in zip(toks, toks[1:], strict=False)]:
                h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return l2n(out)


class HFImageEmbedder:  # pragma: no cover - needs torch + weights
    def __init__(self, model_id: str = "google/siglip2-base-patch16-224", device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.name = model_id
        self.dim = 0

    def embed(self, images):
        rgb = [img[:, :, ::-1] for img in images]
        with self.torch.no_grad():
            inp = self.proc(images=rgb, return_tensors="pt").to(self.device)
            if hasattr(self.model, "get_image_features"):
                f = self.model.get_image_features(**inp)
            else:  # DINOv2
                f = self.model(**inp).pooler_output
        v = l2n(f.float().cpu().numpy())
        self.dim = v.shape[1]
        return v


class STTextEmbedder:  # pragma: no cover
    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer(model_id)
        self.name, self.dim = model_id, self.m.get_sentence_embedding_dimension()

    def embed(self, texts):
        return l2n(np.asarray(self.m.encode(texts), dtype=np.float32))


def build_image_embedder(kind: str = "tiny") -> ImageEmbedder:
    return TinyEmbedder() if kind == "tiny" else HFImageEmbedder(kind)


def build_text_embedder(kind: str = "hash") -> TextEmbedder:
    return HashTextEmbedder() if kind == "hash" else STTextEmbedder(kind)
