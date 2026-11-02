"""Photo check for the app: do two independent landmark detectors agree on this photo?

Every measurement and every warp edit is placed on MediaPipe's landmarks. When a photo is blurred, small, turned,
occluded or oddly lit, those landmarks drift, and nothing in a single detector tells you. FAN (a separately trained
network) locates the same anatomical anchors; if the two disagree much more than they do on ordinary clinical photos,
the app warns (it never blocks). Calibrated on HDA dev photos (tools/make_quality_asset.py -> assets/quality_v1.json:
only summary numbers). FAN plays no part in the edit itself and stays an independent judge in the evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ASSET = Path(__file__).resolve().parent / "assets" / "quality_v1.json"
# (MediaPipe index, FAN/iBUG-68 index, region): eye corners, nose tip, subnasale, mouth corners, chin, brow ends
ANCHORS = [(33, 36, "eyes"), (133, 39, "eyes"), (362, 42, "eyes"), (263, 45, "eyes"), (4, 30, "nose"),
           (2, 33, "nose"), (61, 48, "mouth"), (291, 54, "mouth"), (152, 8, "chin"), (70, 17, "brows"),
           (107, 21, "brows"), (336, 22, "brows"), (300, 26, "brows")]


def offsets(L_mp: np.ndarray, P_fan: np.ndarray) -> np.ndarray:
    """FAN minus MediaPipe position of each anchor, in a face frame (iris axis, units of inter-pupil distance).
    Scale- and rotation-free, so photos of any size compare."""
    a, b = L_mp[468, :2], L_mp[473, :2]
    ipd = float(np.linalg.norm(b - a))
    if ipd < 10:
        raise ValueError("face too small to check")
    u = (b - a) / ipd
    R = np.stack([u, [-u[1], u[0]]])                       # rows: across the face, down the face
    return np.array([R @ (P_fan[j, :2] - L_mp[i, :2]) for i, j, _ in ANCHORS]) / ipd


def disagreement(L_mp: np.ndarray, P_fan: np.ndarray, bias: np.ndarray) -> tuple[float, np.ndarray]:
    """(mean anchor disagreement, per-anchor disagreement), after removing each anchor's usual offset (the two
    detectors define some points slightly differently, e.g. brow upper edge vs brow centre line)."""
    per = np.linalg.norm(offsets(L_mp, P_fan) - bias, axis=1)
    return float(per.mean()), per


class PhotoCheck:
    def __init__(self, fan, asset: Path = ASSET):
        q = json.loads(Path(asset).read_text())
        assert [tuple(x) for x in q["anchors"]] == [(i, j) for i, j, _ in ANCHORS], "quality asset / anchors mismatch"
        self.fan, self.q = fan, q
        self.bias, self.anchor_p99 = np.array(q["bias"]), np.array(q["anchor_p99"])

    def __call__(self, rgb: np.ndarray, L_mp: np.ndarray, box: np.ndarray) -> dict:
        """-> {"level": "ok" | "caution" | "warn", "score", "regions": [...], "text": markdown}."""
        P = self.fan.landmarks(rgb, box)
        off = offsets(L_mp, P)
        per = np.linalg.norm(off - self.bias, axis=1)
        score = float(per.mean())
        regions = sorted({r for (_, _, r), x, p in zip(ANCHORS, per, self.anchor_p99) if x > p})
        level = "warn" if score > self.q["warn"] else "caution" if score > self.q["caution"] else "ok"
        pct = 100 * score
        if level == "ok":
            text = (f"**Photo check: OK.** Two independent landmark detectors agree (mean difference {pct:.1f} % of the "
                    f"inter-pupil distance; typical clinical photos: up to {100 * self.q['caution']:.1f} %).")
        else:
            where = f" Largest disagreement: {', '.join(regions)}." if regions else ""
            text = (f"**Photo check: {'⚠ landmarks unreliable' if level == 'warn' else 'caution'}.** Two independent "
                    f"landmark detectors disagree by {pct:.1f} % of the inter-pupil distance (typical clinical photos: up "
                    f"to {100 * self.q['caution']:.1f} %).{where} Measurements and edit placement may be off; a sharper, "
                    "front-facing, evenly lit photo with nothing covering the face is recommended.")
        return {"level": level, "score": score, "regions": regions, "text": text, "offsets": off}
