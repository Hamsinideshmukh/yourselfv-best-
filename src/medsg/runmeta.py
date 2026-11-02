"""Run metadata: versions, seed, config and weight hashes saved next to every output."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def write_run_meta(out_dir: Path, cfg_dict: dict, weights: dict[str, Path], extra: dict | None = None) -> Path:
    """Write run_meta.json with package versions, platform, config and sha256 of each weight file."""
    pkgs = {}
    for name in ("numpy", "scipy", "pandas", "opencv-python", "opencv-python-headless",
                 "mediapipe", "torch", "torchvision"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    meta = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
        "config": cfg_dict,
        "weights_sha256": {k: sha256(Path(v)) for k, v in weights.items()},
        **(extra or {}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=1))
    return path
