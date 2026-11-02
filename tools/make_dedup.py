from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DUP_THR, LINK_THR = 0.80, 0.70


def main() -> None:
    E = np.load(ROOT / "outputs/stage4a/private/facenet_embeddings.npz")
    split = json.load(open(ROOT / "outputs/splits.json"))
    sp = {p: s for s in ("dev", "test") for p in split[s]}
    keys = sorted(k for k in E.files if k.endswith("_real"))
    X = np.stack([E[k] for k in keys]).astype(np.float64)
    assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-3)
    pid = np.array([k.split("_")[0] for k in keys])
    S = X @ X.T
    ids = sorted(set(pid), key=int)
    best = {}                                   # (a, b) -> max cosine over their images
    iu = np.triu_indices(len(keys), 1)
    for i, j in zip(*iu):
        if pid[i] != pid[j] and S[i, j] >= LINK_THR:
            k = tuple(sorted((pid[i], pid[j]), key=int))
            best[k] = max(best.get(k, -1.0), float(S[i, j]))
    # duplicate groups (connected components over >= DUP_THR links)
    parent = {p: p for p in ids}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    for (a, b), c in best.items():
        if c >= DUP_THR:
            parent[find(a)] = find(b)
    groups = {}
    for p in ids:
        groups.setdefault(find(p), []).append(p)
    dup_groups = [sorted(g, key=int) for g in groups.values() if len(g) > 1]
    exclude = {}
    for g in dup_groups:
        devs = [p for p in g if sp[p] == "dev"]
        keep = devs[0] if devs else g[0]
        for p in g:
            if p != keep:
                exclude[p] = f"duplicate of {keep} (cos >= {DUP_THR})"
    for (a, b), c in best.items():
        if sp[a] != sp[b]:
            t = a if sp[a] == "test" else b
            d = b if t == a else a
            exclude.setdefault(t, f"test pair linked to dev pair {d} (cos {c:.3f} >= {LINK_THR})")
    out = {"rule": __doc__.split("Run:")[0].strip(), "dup_thr": DUP_THR, "link_thr": LINK_THR,
           "duplicate_groups": dup_groups,
           "links": [{"a": a, "b": b, "max_cos": round(c, 4), "split_a": sp[a], "split_b": sp[b]}
                     for (a, b), c in sorted(best.items(), key=lambda kv: -kv[1])],
           "exclude": dict(sorted(exclude.items(), key=lambda kv: int(kv[0]))),
           "n_exclude_dev": sum(sp[p] == "dev" for p in exclude), "n_exclude_test": sum(sp[p] == "test" for p in exclude)}
    json.dump(out, open(ROOT / "outputs/dedup.json", "w"), indent=1)
    print(f"{len(dup_groups)} duplicate groups; excluded dev {out['n_exclude_dev']}, test {out['n_exclude_test']}")
    print("EXCLUDE = {")
    for p, why in out["exclude"].items():
        print(f'    "{p}": "{why}",')
    print("}")


if __name__ == "__main__":
    main()
