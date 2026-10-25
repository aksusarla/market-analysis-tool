"""Config loading with dotted-key overrides (e.g. backtest.top_k=20)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_config(path: str | Path = "config/default.yaml", overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def set_dotted(cfg: dict, key: str, value) -> None:
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def get_dotted(cfg: dict, key: str):
    node = cfg
    for p in key.split("."):
        node = node[p]
    return node


def with_overrides(cfg: dict, changes: dict) -> dict:
    new = copy.deepcopy(cfg)
    for k, v in changes.items():
        set_dotted(new, k, v)
    return new
