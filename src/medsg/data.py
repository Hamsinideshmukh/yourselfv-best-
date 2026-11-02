"""Pair discovery (by ID, never by sort order), seeding and subject-level splits."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config


@dataclass(frozen=True)
class Pair:
    pid: str
    pre: Path
    post: Path


def seed_everything(seed: int) -> None:
    """Fix python, numpy and (if installed) torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def list_pairs(cfg: Config) -> list[Pair]:
    """Return all pre/post pairs matched by numeric ID. Raises on orphans or unknown files."""
    pat = re.compile(rf"^(\d+)({re.escape(cfg.pre_suffix)}|{re.escape(cfg.post_suffix)}){re.escape(cfg.img_ext)}$")
    found: dict[str, dict[str, Path]] = {}
    unknown = []
    for p in cfg.data_dir.iterdir():
        m = pat.match(p.name)
        if not m:
            unknown.append(p.name)
            continue
        found.setdefault(m[1], {})[m[2]] = p
    if unknown:
        raise ValueError(f"Unrecognised files in data dir: {unknown[:10]}")
    orphans = [k for k, v in found.items() if len(v) != 2]
    if orphans:
        raise ValueError(f"IDs missing pre or post image: {orphans}")
    pairs = [Pair(k, v[cfg.pre_suffix], v[cfg.post_suffix]) for k, v in found.items()]
    return sorted(pairs, key=lambda p: int(p.pid))


def make_split(pairs: list[Pair], cfg: Config, path: Path) -> dict[str, list[str]]:
    """Seeded dev/test split by pair ID, saved to `path`. Reuses an existing file if present.

    Note: HDA pair IDs are treated as subjects. The paper reports 638 subjects for
    662 pairs, so a few subjects may appear twice (to be checked with face embeddings).
    """
    if path.exists():
        split = json.loads(path.read_text())
        assert set(split["dev"]) | set(split["test"]) == {p.pid for p in pairs}, "stale split file"
        return split
    ids = [p.pid for p in pairs]
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(ids)
    n_dev = round(cfg.dev_frac * len(ids))
    split = {"dev": sorted(ids[:n_dev], key=int), "test": sorted(ids[n_dev:], key=int),
             "seed": cfg.seed, "dev_frac": cfg.dev_frac}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1))
    return split
