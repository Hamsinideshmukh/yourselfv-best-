"""Stage 4b: controllability atlas + edit directions (CPU, minutes, tables only).

Modes
- "full": Stage 3 + Stage 4a outputs present. Uses the synthetic noise floor, linearity and real-code data.
  Only this mode exports directions that the Stage-5 editor accepts.
- "preliminary": Stage 3 only. The noise ceiling uses the Stage-1 real-photo SEM as a proxy, and there is
  no transfer/linearity check. Every output file carries the mode; nothing preliminary feeds an evaluation.

Steps
 4b local Jacobians per layer group: d_quantity ~ J . u, u = whitened latent step (see directions.py);
    held-out R² vs the noise ceiling R²_max = 1 - 2 SEM² / Var(d).
 4c constrained (null-space) direction per 2D metric in its best group; kappa (share of effect left after
    protection) and *cross-fitted* predictions: direction built on half the faces, effect predicted with the
    Jacobian of the other half (honest guess of open-loop gain and leakage before any GPU validation).
 4d transfer: does a synthetic-face Jacobian predict real-code (HDA e4e) changes?
 4e linearity: does the change grow in proportion to the step (sigma -0.5 / 0.25 / 0.5 / 1.0)?
 4f HDA latent deltas: predicted change J . (E(post) - E(pre)) vs measured change, dev pairs.
 D10 HDA envelope + E3 requests (dev pairs only).
 D8 identity yardstick: tau = 5th percentile of FaceNet cos(real pre, real post) on dev pairs.

Writes outputs/stage4b/*.csv, directions.npz, envelope.json.  Run: python -m medsg.stages.stage4b_analysis
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import editspec as E
from ..config import get_config
from ..directions import constrained_direction, fit_jacobian, fit_ridge_cv, global_direction
from .stage3b_analysis import GROUP_NAMES, load_chunks

POSE_MP = ["yaw", "pitch", "roll"]           # MediaPipe pose = first three POSE columns


def quantities(res: dict, meta: dict) -> tuple[list[str], np.ndarray]:
    """Stack the measured quantities of a sample set: 22 metrics, MediaPipe pose, derived blendshapes, IPD px."""
    names = list(meta["METRICS"]) + POSE_MP + list(E.BS_DERIVED)
    bs_idx = {f"bs_{b}": i for i, b in enumerate(meta["BLENDSHAPES"])}
    cols = [res["metrics"], res["pose"][:, :3]]
    cols += [res["bs"][:, [bs_idx[c] for c in src]].mean(1, keepdims=True) for src in E.BS_DERIVED.values()]
    if "lm" in res:
        lm = res["lm"].astype(np.float64) * meta["MEAS_RES"]
        cols.append(np.linalg.norm(lm[:, 468] - lm[:, 473], axis=1, keepdims=True))
    else:
        cols.append(np.full((len(res["metrics"]), 1), np.nan))
    names.append("ipd_px")
    Q = np.concatenate(cols, 1).astype(np.float64)
    Q[~res["ok"]] = np.nan
    return names, Q


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(pd.Series(a[ok]).rank(), pd.Series(b[ok]).rank())[0, 1])


def load_sets(cfg, meta3: dict, w_std: np.ndarray) -> tuple[list[str], dict]:
    """Return quantity names and per-source arrays {src: dict(U, dQ, group, base)} of whitened steps/changes."""
    d3, d4 = cfg.out_dir / "stage3", cfg.out_dir / "stage4a"
    g, loc = load_chunks(d3, "global"), load_chunks(d3, "local")
    names, Qg = quantities(g, meta3)
    _, Ql = quantities(loc, meta3)
    sets = {"s3": {"U": loc["delta"] / w_std, "dQ": Ql - Qg[loc["base_idx"]], "group": loc["group"],
                   "base": loc["base_idx"]}}
    if (d4 / "linearity.npz").exists():
        b, lin = dict(np.load(d4 / "bases.npz")), dict(np.load(d4 / "linearity.npz"))
        _, Qb = quantities(b, meta3)
        _, Qlin = quantities(lin, meta3)
        sets["lin"] = {"U": lin["eps"] * lin["sigma"][:, None] / w_std, "dQ": Qlin - Qb[lin["base"]],
                       "group": lin["group"], "base": lin["base"], "sigma": lin["sigma"], "eps": lin["eps"]}
        rl = dict(np.load(d4 / "real_local.npz"))
        _, Qr = quantities(rl, meta3)
        base_row = {k: i for i, (k, gr) in enumerate(zip(rl["base"], rl["group"])) if gr == -1}
        pert = rl["group"] >= 0
        ref = np.array([base_row[k] for k in rl["base"][pert]])
        sets["real"] = {"U": rl["delta"][pert] / w_std, "dQ": Qr[pert] - Qr[ref], "group": rl["group"][pert],
                        "base": rl["base"][pert]}
    return names, sets


def _sub(s: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in s.items()}


def _concat(parts: list[dict]) -> dict:
    keys = set.intersection(*(set(p) for p in parts)) & {"U", "dQ", "group", "base"}
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def _fit_rows(U, dQ, rows, seed):
    """Fit J for the given quantity columns; columns that are all-NaN (e.g. IPD on real codes) get NaN rows."""
    J = np.full((dQ.shape[1], U.shape[1]), np.nan)
    r2 = np.full(dQ.shape[1], np.nan)
    have = [j for j in rows if np.isfinite(dQ[:, j]).mean() > 0.9]
    ok = np.isfinite(dQ[:, have]).all(1)
    f = fit_jacobian(U[ok], dQ[ok][:, have], seed)
    J[have], r2[have] = f["J"], f["r2"]
    return J, r2


def main() -> None:
    cfg = get_config()
    out = cfg.out_dir / "stage4b"
    out.mkdir(parents=True, exist_ok=True)
    meta3 = json.load(open(cfg.out_dir / "stage3" / "meta.json"))
    w_std = np.load(cfg.out_dir / "stage3" / "w_std.npy").astype(np.float64)
    assert w_std.shape == (512,), w_std.shape
    names, sets = load_sets(cfg, meta3, w_std)
    full = "real" in sets
    mode = "full" if full else "preliminary"
    print(f"stage4b mode: {mode} (sources: {', '.join(sets)})", flush=True)
    qi = {n: i for i, n in enumerate(names)}
    M2 = E.EDIT_METRICS

    # noise floor used for ceilings / MDC units
    if full:
        nf = pd.read_csv(cfg.out_dir / "stage4a" / "noise_synth.csv").set_index("metric")
        sem_src = "synthetic (Stage 4a)"
    else:
        nf = pd.read_csv(cfg.out_dir / "stage1" / "noise_floor.csv").set_index("metric")
        sem_src = "real-photo proxy (Stage 1)"
    sem = nf["sem"].reindex(names).to_numpy()
    mdc = nf["mdc95"].reindex(names).to_numpy()
    scale = mdc.copy()
    for n, tol in E.NUISANCE_TOL.items():
        scale[qi[n]] = tol
    scale[qi["ipd_px"]] = E.IPD_TOL_PX

    # ---- 4b/4d: Jacobians per group and source -------------------------------------------------
    syn = _concat([sets[k] for k in ("s3", "lin") if k in sets])
    rows_all = list(range(len(names)))
    jac, atlas = {}, []
    for gi, gname in GROUP_NAMES.items():
        S = _sub(syn, syn["group"] == gi)
        J_syn, r2_syn = _fit_rows(S["U"], S["dQ"], rows_all, cfg.seed)
        use = "synthetic"
        J, r2, data = J_syn, r2_syn, S
        transfer = {}
        if full:
            R = _sub(sets["real"], sets["real"]["group"] == gi)
            J_real, r2_real = _fit_rows(R["U"], R["dQ"], rows_all, cfg.seed)
            # synthetic J applied to real-code steps: R² on real data
            pred = R["U"] @ J_syn.T
            ss = np.nansum((R["dQ"] - np.nanmean(R["dQ"], 0)) ** 2, 0)
            r2_transfer = 1 - np.nansum((R["dQ"] - pred) ** 2, 0) / ss
            cos = np.array([J_syn[j] @ J_real[j] / (np.linalg.norm(J_syn[j]) * np.linalg.norm(J_real[j]) + 1e-12)
                            for j in range(len(names))])
            transfer = {"r2_real_cv": r2_real, "r2_syn_on_real": r2_transfer, "cos_syn_real": cos}
            ratio = np.nanmedian(r2_transfer[[qi[m] for m in M2]] / r2_real[[qi[m] for m in M2]])
            # D4d rule: if synthetic J explains < half of what a real-code J explains, use real-code data
            use = "pooled" if ratio >= 0.5 else "real"
            data = _concat([S, R]) if use == "pooled" else R
            J, r2 = _fit_rows(data["U"], data["dQ"], rows_all, cfg.seed)
            J[qi["ipd_px"]], r2[qi["ipd_px"]] = J_syn[qi["ipd_px"]], r2_syn[qi["ipd_px"]]   # IPD: synthetic only
        var = np.nanvar(data["dQ"], 0)
        r2max = 1 - 2 * sem ** 2 / var
        u_norm95 = float(np.quantile(np.linalg.norm(data["U"], axis=1), 0.95))
        jac[gi] = {"J": J, "r2": r2, "r2max": r2max, "data": data, "use": use, "u_norm95": u_norm95}
        for j, n in enumerate(names):
            atlas.append({"group": gname, "quantity": n, "jacobian_data": use, "n": len(data["U"]),
                          "r2_heldout": r2[j], "r2_max": r2max[j], "sd_change": np.sqrt(var[j]),
                          **{k: v[j] for k, v in transfer.items()}})
    atlas = pd.DataFrame(atlas)
    atlas.to_csv(out / "atlas_jacobian.csv", index=False)

    # ---- 4c: constrained directions + cross-fitted predictions -------------------------------------
    def protected(t: str) -> list[int]:
        p = [m for m in M2 if m != t and m not in E.COUPLED.get(t, [])]
        p += [n for n in E.NUISANCE if n not in E.COUPLED_NUISANCE.get(t, [])]
        if t in E.NUISANCE:
            p = [m for m in M2] + [n for n in E.NUISANCE if n != t]
        return [qi[x] for x in p] + [qi["ipd_px"]]

    targets = M2 + E.CORRECTION_NUISANCE
    dirs, rows = {}, []
    for t in targets:
        best = None
        for gi in GROUP_NAMES:
            jg = jac[gi]
            P_idx = [i for i in protected(t) if np.isfinite(jg["J"][i]).all()]
            try:
                u, kappa = constrained_direction(jg["J"][qi[t]], jg["J"][P_idx], scale[P_idx])
            except ValueError:
                continue
            # cross-fit: split faces into halves A/B; direction from A, effect predicted by J of B (and back)
            data = jg["data"]
            half = (pd.Series(data["base"]).rank(method="dense").to_numpy().astype(int) % 2) == 0
            cf = []
            for a_mask in (half, ~half):
                Ja, _ = _fit_rows(data["U"][a_mask], data["dQ"][a_mask], [qi[t]] + P_idx, cfg.seed)
                Jb, _ = _fit_rows(data["U"][~a_mask], data["dQ"][~a_mask], [qi[t]] + P_idx, cfg.seed)
                Pa = [i for i in P_idx if np.isfinite(Ja[i]).all() and np.isfinite(Jb[i]).all()]
                ua, _ = constrained_direction(Ja[qi[t]], Ja[Pa], scale[Pa])
                leak = np.abs(Jb[Pa] @ ua) * scale[qi[t]] / scale[Pa]      # per 1-unit-of-scale request
                cf.append((float(Jb[qi[t]] @ ua), leak, Pa))
            r_unit = scale[qi[t]]                                           # 1 MDC (or 1 tolerance) request
            rec = {"target": t, "group": GROUP_NAMES[gi], "r2_heldout": jg["r2"][qi[t]], "r2_max": jg["r2max"][qi[t]],
                   "kappa": kappa, "u_norm_per_2mdc": float(np.linalg.norm(u) * 2 * r_unit),
                   "u_norm95_sampled": jg["u_norm95"],
                   "crossfit_gain": np.mean([c[0] for c in cf]),
                   "crossfit_median_leak_mdc": np.mean([np.median(c[1]) for c in cf]),
                   "crossfit_max_leak_mdc": np.mean([np.max(c[1]) for c in cf]),
                   "crossfit_worst_protected": names[cf[0][2][int(np.argmax(cf[0][1]))]],
                   "n_protected": len(P_idx)}
            rows.append(rec)
            if best is None or rec["r2_heldout"] > best[0]["r2_heldout"]:
                best = (rec, gi, u)
        if best is None:
            raise RuntimeError(f"no usable direction for {t}")
        rec, gi, u = best
        dirs[t] = {"group": gi, "u": u, "w_delta": u * w_std, "kappa": rec["kappa"]}
    cand = pd.DataFrame(rows)
    cand.to_csv(out / "atlas_directions_all_groups.csv", index=False)
    chosen = pd.DataFrame([{**cand[(cand.target == t) & (cand.group == GROUP_NAMES[d["group"]])].iloc[0].to_dict()}
                           for t, d in dirs.items()])
    chosen["entangled"] = chosen.kappa < 0.1
    chosen["c1_r2"] = chosen.r2_heldout >= E.CTRL_R2_FRAC * chosen.r2_max
    chosen["c2_pred_gain"] = chosen.crossfit_gain.between(*E.CTRL_SLOPE)
    chosen["c3_pred_leak"] = chosen.crossfit_median_leak_mdc < E.CTRL_LEAK_MDC
    chosen["c4_in_range"] = chosen.u_norm_per_2mdc <= chosen.u_norm95_sampled
    chosen["mode"] = mode
    chosen["sem_source"] = sem_src
    chosen.to_csv(out / "atlas_directions.csv", index=False)

    # ---- B4 baseline: global ridge W -> metric on the Stage-3 global set (all 18 layers) -------------
    g = load_chunks(cfg.out_dir / "stage3", "global")
    _, Qg = quantities(g, meta3)
    fg = fit_ridge_cv(g["w"].astype(np.float64), Qg[:, [qi[m] for m in M2]], cfg.seed)
    b4 = {m: global_direction(fg["coef"][:, k]) for k, m in enumerate(M2)}

    # ---- 4e linearity --------------------------------------------------------------------------
    if full:
        lin = sets["lin"]
        rows = []
        for gi, gname in GROUP_NAMES.items():
            inst = {}
            for i in np.where(lin["group"] == gi)[0]:
                inst.setdefault(int(lin["base"][i]), []).append(i)      # one direction per (base, group)
            for m in M2:
                j = qi[m]
                sse_l = sse_q = sst = 0.0
                sym = []
                for idx in inst.values():
                    yi, si = lin["dQ"][idx, j], lin["sigma"][idx].astype(np.float64)
                    if not np.isfinite(yi).all():
                        continue
                    b = (si @ yi) / (si @ si)
                    A = np.c_[si, si ** 2]
                    c = np.linalg.lstsq(A, yi, rcond=None)[0]
                    sse_l += ((yi - b * si) ** 2).sum()
                    sse_q += ((yi - A @ c) ** 2).sum()
                    sst += (yi ** 2).sum()
                    d = dict(zip(np.round(si, 2), yi))
                    sym.append((d.get(-0.5, np.nan), d.get(0.5, np.nan)))
                sym = np.array(sym)
                ok = np.isfinite(sym).all(1)
                rows.append({"group": gname, "metric": m, "n_directions": len(inst),
                             "r2_linear_through_0": 1 - sse_l / sst, "r2_quadratic": 1 - sse_q / sst,
                             "symmetry_corr": float(np.corrcoef(sym[ok, 0], -sym[ok, 1])[0, 1])})
        pd.DataFrame(rows).to_csv(out / "linearity.csv", index=False)

    # ---- 4f HDA latent deltas (dev, pose-included) ---------------------------------------------
    s1 = cfg.out_dir / "stage1"
    pairs = pd.read_csv(s1 / "pairs.csv", dtype={"pid": str}).set_index("pid")
    split = json.load(open(cfg.out_dir / "splits.json"))
    dev_inc = [p for p in split["dev"] if p in pairs.index and pairs.loc[p, "include_pose"]]
    lat_path = cfg.out_dir / "stage2" / "private" / "latents.npz"
    if lat_path.exists():
        lat = np.load(lat_path)
        s2 = pd.read_csv(cfg.out_dir / "stage2" / "images.csv", dtype={"pid": str}).set_index(["pid", "role"])
        groups = {int(k): v for k, v in meta3["GROUPS"].items()}
        rows, Uq = [], []
        for p in dev_inc:
            dc = lat[f"{p}_post"].astype(np.float64) - lat[f"{p}_pre"].astype(np.float64)
            assert dc.shape == (18, 512), dc.shape
            # J_g was fitted for one vector added to every layer of g; for a per-layer change we use the
            # group-mean change (first-order approximation, stated in the report)
            Uq.append({gi: dc[a:b].mean(0) / w_std for gi, (a, b) in groups.items()})
        for m in M2:
            j = qi[m]
            pred = np.array([sum(jac[gi]["J"][j] @ u[gi] for gi in GROUP_NAMES) for u in Uq])
            d_real = pairs.loc[dev_inc, f"d_{m}"].to_numpy()
            d_e4e = np.array([s2.loc[(p, "post"), f"e4e_{m}"] - s2.loc[(p, "pre"), f"e4e_{m}"] for p in dev_inc])
            rows.append({"metric": m, "n": len(dev_inc), "spearman_pred_vs_real": _spearman(pred, d_real),
                         "spearman_pred_vs_e4e": _spearman(pred, d_e4e),
                         "spearman_e4e_vs_real": _spearman(d_e4e, d_real)})
        pd.DataFrame(rows).to_csv(out / "hda_latent_deltas.csv", index=False)
    else:
        print("note: stage2 private latents not found; 4f skipped", flush=True)

    # ---- D10 envelope + E3 requests (dev, pose-included) -----------------------------------------
    s1img = pd.read_csv(s1 / "images.csv", dtype={"pid": str})
    pre = s1img[s1img.role == "pre"].set_index("pid").loc[dev_inc]
    post = s1img[s1img.role == "post"].set_index("pid").loc[dev_inc]
    D = pairs.loc[dev_inc, [f"d_{m}" for m in M2]].to_numpy()
    smile = ["bs_mouthSmileLeft", "bs_mouthSmileRight"]
    X = np.c_[np.ones(len(dev_inc)), post[smile].mean(1) - pre[smile].mean(1), post.bs_jawOpen - pre.bs_jawOpen]
    req = {}
    for k, m in enumerate(M2):
        req[m] = float(np.linalg.lstsq(X, D[:, k], rcond=None)[0][0]) if m == "mouth_width" else float(D[:, k].mean())
    env = {"n_pairs": len(dev_inc), "metrics": M2,
           "p025": np.quantile(D, 0.025, 0).tolist(), "p975": np.quantile(D, 0.975, 0).tolist(),
           "mean": D.mean(0).tolist(), "cov": np.cov(D.T).tolist(),
           "chi2_95_df": len(M2), "e3_requests": {t: req[t] for t in E.E3_TARGETS},
           "e3_request_note": "dev-pair mean real change; mouth_width expression-adjusted (smile, jawOpen)"}
    json.dump(env, open(out / "envelope.json", "w"), indent=1)

    # ---- D8 identity yardstick ------------------------------------------------------------------
    id_path = cfg.out_dir / "stage4a" / "identity_pairs.csv"
    tau = np.nan
    if id_path.exists():
        idp = pd.read_csv(id_path, dtype={"pid": str})
        dev_id = idp[idp.split == "dev"]
        tau = float(dev_id.cos_real_pre_post.quantile(0.05))
        summ = pd.read_csv(cfg.out_dir / "stage4a" / "identity_summary.csv")
        e4e = pd.read_csv(cfg.out_dir / "stage2" / "images.csv").id_cos
        pti = pd.read_csv(cfg.out_dir / "stage2" / "pti.csv").id_cos
        imp_p99 = float(summ.set_index("quantity").loc["cos_different_person_pre_pre", "p99"])
        top = pd.read_csv(cfg.out_dir / "stage4a" / "identity_top_impostors.csv")
        pd.DataFrame([{
            "tau_dev_p05_real_pre_post": tau, "n_dev": len(dev_id),
            "median_real_pre_post_all": idp.cos_real_pre_post.median(),
            "median_rec_pre_post_all": idp.cos_rec_pre_post.median(),
            "different_person_p99": imp_p99,
            "e4e_median_cos_x_rec": e4e.median(), "pti_median_cos_x_rec": pti.median(),
            "e4e_frac_above_tau": float((e4e >= tau).mean()), "pti_frac_above_tau": float((pti >= tau).mean()),
            "n_different_id_pairs_above_tau": int((top.max_cos >= tau).sum()),
            "note": "FaceNet stands in for buffalo_l ArcFace (D8, planned); top list capped at 40 pairs"}]
        ).to_csv(out / "identity_yardstick.csv", index=False)

    # ---- export ---------------------------------------------------------------------------------
    np.savez_compressed(
        out / "directions.npz", mode=mode, sem_source=sem_src, names=np.array(names), targets=np.array(targets),
        groups=np.array([dirs[t]["group"] for t in targets]),
        u=np.stack([dirs[t]["u"] for t in targets]), w_delta=np.stack([dirs[t]["w_delta"] for t in targets]),
        kappa=np.array([dirs[t]["kappa"] for t in targets]),
        J=np.stack([jac[gi]["J"] for gi in GROUP_NAMES]), w_std=w_std, mdc=mdc, scale=scale,
        b4_metrics=np.array(M2), b4_w_delta=np.stack([b4[m] for m in M2]), tau=tau)
    pd.set_option("display.width", 250)
    show = ["target", "group", "r2_heldout", "r2_max", "kappa", "crossfit_gain", "crossfit_median_leak_mdc",
            "crossfit_max_leak_mdc", "crossfit_worst_protected", "u_norm_per_2mdc", "u_norm95_sampled",
            "c1_r2", "c2_pred_gain", "c3_pred_leak", "c4_in_range"]
    print(chosen[show].round(3).to_string(index=False))
    for f in ["linearity.csv", "hda_latent_deltas.csv", "identity_yardstick.csv"]:
        if (out / f).exists():
            print(pd.read_csv(out / f).round(3).to_string(index=False))
    print(json.dumps({k: env[k] for k in ["n_pairs", "e3_requests"]}, indent=1))


if __name__ == "__main__":
    main()
