"""Config loading: packaged defaults deep-merged with an optional user YAML."""

from __future__ import annotations

import copy
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def _load_packaged(name: str) -> dict[str, Any]:
    text = resources.files("sotif_miner.configs").joinpath(name).read_text()
    return yaml.safe_load(text) or {}


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _load_packaged("default.yaml")
    if path:
        cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


@lru_cache(maxsize=1)
def ontology() -> dict[str, Any]:
    return _load_packaged("ontology.yaml")


@lru_cache(maxsize=1)
def catalog() -> dict[str, Any]:
    return _load_packaged("sotif_catalog.yaml")


def map_class(dataset: str, name: str) -> str:
    """Map a dataset-native class name onto the unified ontology ("other" if unknown)."""
    onto = ontology()
    if name in onto["classes"]:
        return name
    return onto["maps"].get(dataset, {}).get(name.lower(), "other")


def is_vru(cls: str) -> bool:
    return cls in ontology()["vru"]
