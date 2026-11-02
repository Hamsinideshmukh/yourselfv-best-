"""Single configuration source for the v2 pipeline.

Every path, filename convention and pre-declared threshold lives here.
No other module may hardcode "_a.jpg"/"_b.jpg" or /kaggle/... paths.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# HDA flattened copy: NN_b.jpg = pre-op, NN_a.jpg = post-op.
# Confirmed 2026-10-29 by visual inspection of pairs 01-12
# (b shows untreated ageing signs, a shows the tightened result).
PRE_SUFFIX = "_b"
POST_SUFFIX = "_a"
IMG_EXT = ".jpg"

MEDIAPIPE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
TDDFA_REPO_URL = "https://github.com/cleardusk/3DDFA_V2.git"


@dataclass(frozen=True)
class Config:
    data_dir: Path
    out_dir: Path
    models_dir: Path
    repos_dir: Path
    pre_suffix: str = PRE_SUFFIX
    post_suffix: str = POST_SUFFIX
    img_ext: str = IMG_EXT
    seed: int = 1234
    dev_frac: float = 0.30          # subject-level dev split; rest is held-out test
    # Pre-declared pair inclusion thresholds
    max_abs_dyaw_deg: float = 10.0
    max_abs_dpitch_deg: float = 10.0
    # Noise-floor protocol
    noise_n_images: int = 60
    n_boot: int = 10_000
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}


def _find_kaggle_data_dir() -> Path:
    """Locate the flattened HDA folder under /kaggle/input (raises if absent/ambiguous)."""
    hits = sorted({p.parent for p in Path("/kaggle/input").rglob(f"*{PRE_SUFFIX}{IMG_EXT}")})
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one HDA image folder under /kaggle/input, found {hits}")
    return hits[0]


def get_config(**overrides) -> Config:
    """Build the config for the current environment (Kaggle or local).

    Local runs read MEDSG_DATA_DIR (folder with NN_a.jpg / NN_b.jpg) and
    MEDSG_ROOT (project root; default: two levels above this file's package).
    """
    if Path("/kaggle/input").exists():
        root = Path("/kaggle/working")
        data_dir = _find_kaggle_data_dir()
    else:
        root = Path(os.environ.get("MEDSG_ROOT", Path(__file__).resolve().parents[2]))
        if "MEDSG_DATA_DIR" not in os.environ:
            raise EnvironmentError("Set MEDSG_DATA_DIR to the extracted flattened_images folder")
        data_dir = Path(os.environ["MEDSG_DATA_DIR"])
    base = dict(
        data_dir=data_dir,
        out_dir=root / "outputs",
        models_dir=root / "models",
        repos_dir=root / "repos",
    )
    base.update(overrides)
    return Config(**base)
