"""Live check against a local Ollama server. Run with:  pytest -m gpu tests/test_ollama_live.py"""

import httpx
import numpy as np
import pytest

from sotif_miner.config import load_config
from sotif_miner.schemas import RoiAudit, SceneAudit
from sotif_miner.vlm.factory import build_backend

pytestmark = pytest.mark.gpu


def _server_up(url):
    try:
        return httpx.get(f"{url}/api/tags", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def test_ollama_structured_output():
    cfg = load_config(overrides={"vlm": {"backend": "ollama"}})
    url = cfg["vlm"]["ollama"]["base_url"]
    if not _server_up(url):
        pytest.skip(f"Ollama not reachable at {url}")
    import cv2

    img = np.full((360, 640, 3), 90, np.uint8)
    cv2.rectangle(img, (280, 150), (360, 330), (30, 200, 230), -1)  # a figure-like block
    be = build_backend(cfg)
    r = be.generate([img], "Describe the object in the centre.", RoiAudit)
    assert 0 <= r.visibility <= 3
    s = be.generate([img, img], "These are 2 frames from a road video. Describe the scene.", SceneAudit)
    assert s.description
    print(be.stats.as_dict())
