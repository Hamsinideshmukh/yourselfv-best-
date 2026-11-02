"""Photo check (medsg.quality): scale/rotation invariance and that displaced landmarks are flagged."""
import numpy as np

from medsg.quality import ANCHORS, PhotoCheck, disagreement, offsets


def _face(seed=0):
    rng = np.random.default_rng(seed)
    L = rng.uniform(100, 400, (478, 2))
    L[468], L[473] = [200, 250], [300, 250]                 # IPD 100 px
    P = np.zeros((68, 2))
    for i, j, _ in ANCHORS:
        P[j] = L[i] + rng.normal(0, 1, 2)                   # FAN ~ MediaPipe + 1 % IPD noise
    return L, P


def _rot(X, a, s, t):
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return s * X @ R.T + t


def test_offsets_invariant_to_scale_rotation_shift():
    L, P = _face()
    o1 = offsets(L, P)
    o2 = offsets(_rot(L, 0.4, 2.5, [30, -7]), _rot(P, 0.4, 2.5, [30, -7]))
    assert np.allclose(o1, o2, atol=1e-9)


def test_photo_check_levels(tmp_path):
    import json
    L, P = _face()
    q = {"anchors": [[i, j] for i, j, _ in ANCHORS], "bias": np.zeros((len(ANCHORS), 2)).tolist(),
         "anchor_p99": [0.04] * len(ANCHORS), "caution": 0.03, "warn": 0.045}
    f = tmp_path / "q.json"
    f.write_text(json.dumps(q))

    class Fan:
        def __init__(self, P):
            self.P = P

        def landmarks(self, rgb, box):
            return self.P
    ok = PhotoCheck(Fan(P), f)(None, L, None)
    assert ok["level"] == "ok", ok
    Pbad = P.copy()
    Pbad[8] += [0, 40]                                          # chin 40 % IPD off
    Pbad[48] += [15, 0]
    bad = PhotoCheck(Fan(Pbad), f)(None, L, None)
    assert bad["level"] in ("caution", "warn") and "chin" in bad["regions"], bad
    assert disagreement(L, Pbad, np.zeros((len(ANCHORS), 2)))[0] > ok["score"]
