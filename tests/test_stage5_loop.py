"""The closed loop converges on a monotone but under-responsive target (gain 0.3, measurement noise)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")


def _run(monkeypatch, gain, noise, r, seed=0):
    from types import SimpleNamespace
    from medsg.stages import stage5_overnight as S5
    rng = np.random.default_rng(seed)
    D = np.zeros((18, 512), np.float32); D[4:8] = 1.0
    monkeypatch.setattr(S5, "render", lambda gan, codes: codes)          # "image" = the code itself
    monkeypatch.setattr(S5, "measure_avg", lambda img, lmk: (np.r_[gain * float(img[4, 0]) + rng.normal(0, noise), np.zeros(14)],
                                       {"lm": None, "pose": np.zeros(3), "bs": np.zeros(10)}))
    loop = S5._closed_loop(SimpleNamespace(gan=None, lmk=None), np.zeros((18, 512), np.float32), D, a_cap=10.0, r=r, tol=0.004,
                           m0=np.zeros(15), ti=0)
    recs = []
    while True:
        try:
            recs.append(next(loop))
        except StopIteration as s:
            return s.value, recs


def test_converges_with_low_gain_and_noise(monkeypatch):
    status, recs = _run(monkeypatch, gain=0.3, noise=0.001, r=0.04)
    assert status == "converged", status
    assert abs(recs[-1][2][0] - 0.04) <= 0.004 and len(recs) <= 4


def test_negative_request(monkeypatch):
    status, recs = _run(monkeypatch, gain=0.5, noise=0.0005, r=-0.03)
    assert status == "converged" and recs[-1][1] < 0


def test_reports_non_monotone_instead_of_faking(monkeypatch):
    status, _ = _run(monkeypatch, gain=-0.5, noise=0.0, r=0.03)
    assert status == "non_monotone"


@pytest.mark.parametrize("gain,noise,r", [(0.3, 0.001, 0.04), (0.5, 0.0005, -0.03), (-0.5, 0.0, 0.03), (1.0, 0.002, 0.02)])
def test_app_editor_matches_stage5_loop(monkeypatch, gain, noise, r):
    """The app's secant_edit must reproduce the evaluated Stage-5 loop exactly (same amplitudes, same status)."""
    from medsg.editor import secant_edit
    status5, recs5 = _run(monkeypatch, gain, noise, r, seed=3)
    rng = np.random.default_rng(3)
    recs, status = secant_edit(lambda a: gain * a + rng.normal(0, noise), r, 0.004, 10.0)
    assert status == status5
    assert np.allclose([x[0] for x in recs], [x[1] for x in recs5])
