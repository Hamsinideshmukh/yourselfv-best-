"""Deformation transfer: exact request, protected metrics unchanged (first order), iris pinned, zero = identity."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")


def _real_landmarks():
    """MediaPipe landmarks of a synthetic example face (public fixture; no dataset image needed)."""
    from pathlib import Path
    p = Path(__file__).resolve().parent / "fixtures" / "synthetic_adult_1_landmarks.csv"
    return np.loadtxt(p, delimiter=",").astype(np.float64)


@pytest.mark.parametrize("target,req", [("nose_width", -0.055), ("brow_height", 0.047), ("mrd1", 0.0126),
                                        ("jaw_width", -0.126)])
def test_exact_request_and_protection(target, req):
    from medsg.transfer import COUPLED, NAMES, Transfer
    L = _real_landmarks()
    e = Transfer().edit(L, target, req)
    assert e.after[target] - e.before[target] == pytest.approx(req, abs=1e-6)
    prot = [m for m in NAMES if m != target and m not in COUPLED[target]]
    worst = max(abs(e.after[m] - e.before[m]) for m in prot)
    assert worst < 0.1 * abs(req), (target, worst)          # first-order zero; second-order residual small
    assert np.abs(e.field_px[468:478]).max() < 1e-6       # iris pinned


def test_zero_request_is_identity():
    from medsg.transfer import warp_photo
    rgb = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    L = np.random.default_rng(1).uniform(10, 50, (478, 2))
    assert np.array_equal(warp_photo(rgb, L, np.zeros((478, 2))), rgb)


def test_multi_target_exact_and_independent():
    from medsg.transfer import COUPLED, NAMES, Transfer
    L = _real_landmarks()
    req = {"nose_width": -0.05, "jaw_width": -0.10, "mrd1": 0.012, "mouth_width": 0.02}
    e = Transfer().edit_multi(L, req)
    for t, r in req.items():
        assert e.after[t] - e.before[t] == pytest.approx(r, abs=1e-6), t
    coupled = set().union(*[COUPLED[t] for t in req])
    prot = [m for m in NAMES if m not in req and m not in coupled]
    assert max(abs(e.after[m] - e.before[m]) for m in prot) < 0.1 * 0.012
    assert np.abs(e.field_px[468:478]).max() < 1e-6


def test_multi_zero_is_identity():
    from medsg.transfer import Transfer
    e = Transfer().edit_multi(_real_landmarks(), {"nose_width": 0.0})
    assert not np.any(e.field_px)
