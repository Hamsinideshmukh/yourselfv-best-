"""Second candidate direction for eye opening (fine layers 8-17), rebuilt from the stored Stage 4b Jacobians.

Stage 4b kept, per target, only the direction of its best layer group (eye opening: middle 4-7). For Stage 8 arm C
the fine-group direction is rebuilt with exactly Stage 4b's rule (same protected set, same scales, same projector).
Check first: rebuilding the stored middle-group direction must reproduce assets/editor_v1.npz (cosine ~ 1).
Writes src/medsg/assets/candidates_v1.npz.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medsg import editspec as E  # noqa: E402
from medsg.directions import constrained_direction  # noqa: E402

M2 = list(E.EDIT_METRICS)


def direction(z, t: str, g: int) -> tuple[np.ndarray, float]:
    names = [str(n) for n in z["names"]]
    qi = {n: i for i, n in enumerate(names)}
    p = [m for m in M2 if m != t and m not in E.COUPLED.get(t, [])]
    p += [n for n in E.NUISANCE if n not in E.COUPLED_NUISANCE.get(t, [])]
    P = [qi[x] for x in p] + [qi["ipd_px"]]
    J = z["J"][g]
    P = [i for i in P if np.isfinite(J[i]).all()]
    u, kappa = constrained_direction(J[qi[t]], J[P], z["scale"][P])
    return u * z["w_std"], kappa


def main() -> None:
    z = np.load(ROOT / "outputs/stage4b/directions.npz")
    a = np.load(ROOT / "src/medsg/assets/editor_v1.npz")
    tg = [str(t) for t in a["targets"]]
    for t in ("palpebral_height", "nose_width", "brow_height"):
        k = tg.index(t)
        wd, _ = direction(z, t, int(a["groups"][k]))
        cos = float(wd @ a["w_delta"][k] / np.linalg.norm(wd) / np.linalg.norm(a["w_delta"][k]))
        rel = float(np.linalg.norm(wd - a["w_delta"][k]) / np.linalg.norm(a["w_delta"][k]))
        print(f"rebuild check {t} group {int(a['groups'][k])}: cosine {cos:.6f}, relative difference {rel:.2e}")
        assert cos > 0.9999, "rebuild does not reproduce the stored direction"
    wd, kappa = direction(z, "palpebral_height", 2)
    k = tg.index("palpebral_height")
    print(f"fine-group eye direction: kappa {kappa:.3f}, cosine to middle-group direction (different layers) n/a")
    np.savez(ROOT / "src/medsg/assets/candidates_v1.npz", targets=np.array(["palpebral_height"]),
             groups=np.array([2]), w_delta=wd[None].astype(np.float32), kappa=np.array([kappa]),
             source=np.array("Stage 4b Jacobians (outputs/stage4b/directions.npz, J[group 2]); tools/make_candidate_asset.py"))
    print("wrote candidates_v1.npz")


if __name__ == "__main__":
    main()
