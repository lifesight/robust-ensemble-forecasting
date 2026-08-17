"""Leaderboard, significance testing, tail bootstraps and unit-of-aggregation checks.

Produces Tables 1, 2, 4 and 6, and writes the per-cell metrics, leaderboard, tail
bootstrap and unit-of-aggregation files that the figure scripts read.
"""

import os, glob, importlib.util, collections
import numpy as np, pandas as pd
from scipy import stats
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE


HERE = str(CODE)
spec = importlib.util.spec_from_file_location("a", str(CODE / "core.py"))
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)

BASE = a.BASE
REF = "ens_trimmed"
PRETTY = dict(a.PRETTY)
PRETTY.update({"sel_global": "Global validation-selected", "theta": "Theta"})
ORDER = BASE + ["theta", "sel_val", "sel_global"] + a.ENS
# Studentised range / sqrt(2) constants for the Nemenyi test at alpha = 0.05
Q05 = {10: 3.164, 11: 3.219, 12: 3.268}


def rank_biserial(d):
    d = np.asarray(d, float); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d < 0].sum() - r[d > 0].sum()) / r.sum())


def holm(pvals):
    p = np.asarray(pvals, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (n - i) * p[k])
        adj[k] = min(run, 1.0)
    return adj


def boot_stat_diff(x, y, series, fn, B=4000, seed=0):
    """Series-clustered bootstrap CI for fn(x) - fn(y) on paired per-cell vectors."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(series, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(u))]
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(u), len(u))
        idx = np.concatenate([members[i] for i in pick])
        out[b] = fn(x[idx]) - fn(y[idx])
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5), float((out < 0).mean())


def main():
    cells = a.load_cells()
    theta = pd.read_csv(RESULTS / "theta_percell.csv").set_index(["series", "origin"])
    man = pd.read_csv(MANIFEST)
    man["sid10"] = man.series_id.str[:10]
    omap = dict(zip(man.sid10, man.org)); vmap = dict(zip(man.sid10, man.vertical))

    # ---- global validation winner (what a portfolio-level selector sees) -----------------
    vrows = [dict(series=c["series"], origin=c["origin"], **c["val"]) for c in cells]
    V = pd.DataFrame(vrows).set_index(["series", "origin"])[BASE]
    print("=== AGGREGATE VALIDATION MAPE BY MODEL (the global selector's information) ===")
    agg = pd.DataFrame({"mean": V.mean(), "median": V.median()}).sort_values("median")
    print(agg.round(3).to_string())
    per_origin = {o: V.xs(o, level="origin").median().idxmin() for o in sorted({c["origin"] for c in cells})}
    print("global validation winner by origin:", per_origin)
    GLOBAL = V.median().idxmin()
    print("GLOBAL PICK =", GLOBAL)

    # ---- per-cell table over all methods -------------------------------------------------
    rows, sel_hit, glob_hit, hz = [], 0, 0, []
    matched_rho, mismatched_rho, matched_hit = [], [], 0
    for c in cells:
        F = a.forecasts(c)
        F["sel_global"] = c["base"][GLOBAL]
        key = (c["series"], c["origin"])
        F["theta"] = None  # scored from the cached file, not a forecast path
        oracle = min(a.mae(c["y"], c["base"][m]) / c["nmae"] for m in BASE)
        truewin = min(BASE, key=lambda m: a.mae(c["y"], c["base"][m]))
        sel_hit += (F["_pick"] == truewin); glob_hit += (GLOBAL == truewin)
        hz.append(dict(series=c["series"], origin=c["origin"], n_test=len(c["y"])))

        # matched-loss diagnostic: validation MAPE vs test MAPE, same loss on both sides
        vv = np.array([c["val"][m] for m in BASE], float)
        tm = np.array([a.mape(c["y"], c["base"][m]) for m in BASE], float)
        ts = np.array([a.mae(c["y"], c["base"][m]) / c["nmae"] for m in BASE], float)
        if np.isfinite(vv).all():
            matched_rho.append(stats.spearmanr(vv, tm).statistic)
            mismatched_rho.append(stats.spearmanr(vv, ts).statistic)
            matched_hit += (BASE[int(np.argmin(vv))] == BASE[int(np.argmin(tm))])

        for k in ORDER:
            if k == "theta":
                if key not in theta.index:
                    continue
                sk = float(theta.loc[key, "skill"]); mp = float(theta.loc[key, "mape"])
            else:
                sk = a.mae(c["y"], F[k]) / c["nmae"]; mp = a.mape(c["y"], F[k])
            rows.append(dict(series=c["series"], origin=c["origin"], method=k, skill=sk,
                             mape=mp, regret=sk - oracle, oracle=oracle))

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["series", "origin"], columns="method", values="skill")[ORDER].dropna()
    df = df[df.set_index(["series", "origin"]).index.isin(piv.index)]
    df.to_csv(RESULTS / "percell_test_metrics.csv", index=False)
    N = len(piv)
    ser = np.array([s for s, _ in piv.index])
    print(f"\ncells scored on all {len(ORDER)} methods: {N}")
    print(f"validation-to-test hit rate, per-cell selection : {sel_hit/len(cells):.3f}")
    print(f"validation-to-test hit rate, constant global rule: {glob_hit/len(cells):.3f}")
    print(f"matched-loss (val MAPE -> test MAPE) hit rate    : {matched_hit/len(matched_rho):.3f}")
    print(f"within-cell Spearman  val MAPE -> test MAPE  (matched)   : "
          f"mean {np.mean(matched_rho):.3f} median {np.median(matched_rho):.3f} frac>0 {np.mean(np.array(matched_rho)>0):.2f}")
    print(f"within-cell Spearman  val MAPE -> test RelMAE (mismatched): "
          f"mean {np.mean(mismatched_rho):.3f} median {np.median(mismatched_rho):.3f} frac>0 {np.mean(np.array(mismatched_rho)>0):.2f}")

    # ---- leaderboard ---------------------------------------------------------------------
    mp = df.pivot_table(index=["series", "origin"], columns="method", values="mape")[ORDER]
    rg = df.pivot_table(index=["series", "origin"], columns="method", values="regret")[ORDER]
    rank = piv.rank(axis=1)
    g = pd.DataFrame({
        "mean": piv.mean(), "geomean": np.exp(np.log(piv.clip(lower=1e-9)).mean()),
        "median": piv.median(), "p90": piv.quantile(.90), "p95": piv.quantile(.95),
        "p99": piv.quantile(.99), "max": piv.max(), "beats_naive": (piv < 1).mean(),
        "mape": mp.mean(), "regret_mean": rg.mean(), "regret_max": rg.max(),
        "avg_rank": rank.mean(),
    }).sort_values("mean")

    sb = piv.groupby(level=0).mean()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(sb), size=(10000, len(sb)))
    boot = sb.values[idx].mean(1)
    ci = pd.DataFrame(np.percentile(boot, [2.5, 97.5], axis=0).T, index=sb.columns, columns=["lo", "hi"])
    g = g.join(ci)
    g.to_csv(RESULTS / "leaderboard.csv")
    print("\n=== LEADERBOARD v4 (RelMAE = MAE / seasonal-naive MAE; lower is better) ===")
    print(g.round(3).to_string())

    # ---- Friedman / Nemenyi --------------------------------------------------------------
    # The global validation-selected strategy resolves to a single base model on this corpus,
    # so it is an ex-ante RELABELLING of that model's forecasts, not a distinct column. Including
    # both would enter one forecast path twice, inflate k, and distort every average rank.
    # It is therefore excluded from the rank analysis and reported in the leaderboard with a note.
    RANKSET = [m for m in ORDER if m != "sel_global"]
    identical = bool(np.allclose(piv["sel_global"], piv[GLOBAL]))
    print(f"\nsel_global is numerically identical to '{GLOBAL}': {identical} "
          f"-> excluded from the rank analysis to avoid double-counting")
    rank = piv[RANKSET].rank(axis=1)
    k = len(RANKSET)
    chi2, pf = stats.friedmanchisquare(*[piv[c] for c in RANKSET])
    CD = Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
    print(f"\n=== FRIEDMAN / NEMENYI (k={k}, N={N}) ===")
    print(f"chi2 = {chi2:.1f}  p = {pf:.3g}   CD = {CD:.3f}")
    ar = rank.mean().sort_values()
    print(ar.round(3).to_string())
    print(f"\nrank gap {REF} vs global selection (= {GLOBAL}) : {ar[GLOBAL]-ar[REF]:.3f}  "
          f"({'SEPARABLE' if ar[GLOBAL]-ar[REF] > CD else 'not separable'})")
    print(f"rank gap {REF} vs sel_val    : {ar['sel_val']-ar[REF]:.3f}  "
          f"({'SEPARABLE' if ar['sel_val']-ar[REF] > CD else 'not separable'})")
    print(f"rank gap {REF} vs catboost   : {ar['catboost']-ar[REF]:.3f}  "
          f"({'SEPARABLE' if ar['catboost']-ar[REF] > CD else 'not separable'})")

    # ---- pairwise Wilcoxon with Holm and effect sizes -------------------------------------
    comps = [m for m in RANKSET if m != REF]
    raw, eff, wins = [], [], []
    for m in comps:
        d = (piv[REF] - piv[m]).values
        raw.append(stats.wilcoxon(d).pvalue); eff.append(rank_biserial(d))
        wins.append(int((d < 0).sum()))
    adj = holm(raw)
    W = pd.DataFrame({"wins": wins, "n": N, "mean_diff": [(piv[REF]-piv[m]).mean() for m in comps],
                      "rank_biserial": eff, "p_raw": raw, "p_holm": adj}, index=comps).sort_values("p_holm")
    print(f"\n=== PAIRWISE vs {REF}: Wilcoxon, Holm over {len(comps)} comparisons, with effect sizes ===")
    print(W.round(4).to_string())

    # ---- tail metrics with bootstrap CIs on the difference --------------------------------
    print(f"\n=== TAIL METRICS: series-bootstrap CI on the DIFFERENCE ({REF} - competitor) ===")
    fns = {"P90": lambda v: np.percentile(v, 90), "P95": lambda v: np.percentile(v, 95),
           "P99": lambda v: np.percentile(v, 99), "Max": np.max}
    trows = []
    for m in ["catboost", "sel_val"]:
        for nm, fn in fns.items():
            mu, lo, hi, pb = boot_stat_diff(piv[REF].values, piv[m].values, ser, fn)
            trows.append(dict(competitor=m, stat=nm, ref_level=fn(piv[REF].values),
                              comp_level=fn(piv[m].values), diff=mu, lo=lo, hi=hi,
                              p_ref_better=pb, significant=(hi < 0)))
    T = pd.DataFrame(trows)
    T.to_csv(RESULTS / "tail_bootstrap.csv", index=False)
    print(T.round(3).to_string(index=False))

    # ---- three units of aggregation --------------------------------------------------------
    print("\n=== UNIT OF AGGREGATION ===")
    org = np.array([omap.get(s, s) for s in ser])
    urows = []
    for m in ["sel_val", "catboost"]:
        d = (piv[REF] - piv[m])
        r = {"comparison": f"{REF} vs {m}"}
        for unit, lab in [(ser, "cells"), (ser, "series"), (org, "orgs")]:
            v = d.values if lab == "cells" else d.groupby(unit if lab == "series" else pd.Index(org, name="u")).mean().values \
                if lab != "cells" else d.values
            if lab == "series":
                v = d.groupby(level=0).mean().values
            if lab == "orgs":
                v = pd.Series(d.values, index=org).groupby(level=0).mean().values
            r[f"{lab}_wins"] = f"{int((v<0).sum())}/{len(v)}"
            r[f"{lab}_p"] = stats.wilcoxon(v).pvalue
        urows.append(r)
    U = pd.DataFrame(urows)
    U.to_csv(RESULTS / "unit_of_aggregation.csv", index=False)
    print(U.to_string(index=False))

    # ---- horizon vs training length --------------------------------------------------------
    H = pd.DataFrame(hz).set_index(["series", "origin"]).reindex(piv.index)
    s_raw = pd.read_csv(SUMMARY_CLEAN)
    ntr = s_raw.dropna(subset=["n_train"]).groupby(["series", "origin"]).n_train.first().reindex(piv.index)
    y = (piv[REF] - piv["catboost"]).values
    X = np.column_stack([np.ones(N), np.log(H.n_test.values), np.log(ntr.values)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X) * (resid @ resid) / (N - 3)))
    print("\n=== HORIZON vs TRAINING LENGTH (dep. var: trimmed - catboost RelMAE) ===")
    for nm, coef, s_ in zip(["intercept", "log n_test", "log n_train"], b, se):
        print(f"  {nm:14s} {coef:+.4f}  (se {s_:.4f}, t {coef/s_:+.2f})")
    print(f"  corr(log n_test, log n_train) = {np.corrcoef(np.log(H.n_test), np.log(ntr))[0,1]:.3f}")

    ter = pd.qcut(H.n_test, 3, labels=["short", "mid", "long"])
    cols = [REF, "catboost", "sel_global", "sel_val", "theta"]
    tt = piv[cols].groupby(ter, observed=True).mean()
    tt["mean_len"] = H.n_test.groupby(ter, observed=True).mean()
    tt["mean_train"] = ntr.groupby(ter, observed=True).mean()
    tt["oracle"] = piv[BASE].min(axis=1).groupby(ter, observed=True).mean()
    print("\n=== HORIZON TERCILES ===")
    print(tt.round(3).to_string())

    # ---- drift-benchmark robustness ---------------------------------------------------------
    print("\n=== SECOND REFERENCE DENOMINATOR: seasonal-naive-with-drift ===")
    drows = []
    for c in cells:
        nt_nv = int(ntr.get((c["series"], c["origin"]), np.nan) or 0)
        drows.append((c["series"], c["origin"]))
    dref = {}
    ci_raw = pd.read_csv(CORPUS)
    ci_raw["sid10"] = ci_raw.SERIES_ID.str[:10]
    ci_raw = ci_raw.sort_values(["sid10", "DATE_DAY"])
    hist = {s: gg.ALL_PURCHASES.to_numpy(float) for s, gg in ci_raw.groupby("sid10")}
    splits = (s_raw.dropna(subset=["n_train"]).groupby(["series", "origin"])[["n_train", "n_val", "n_test"]]
              .first().astype(int))
    for c in cells:
        key = (c["series"], c["origin"])
        if key not in splits.index or c["series"] not in hist:
            continue
        nt, nv, ns = splits.loc[key]
        yfull = hist[c["series"]]; o = nt + nv
        wk = yfull[o - 7:o]
        base_nv = np.array([wk[i % 7] for i in range(ns)])
        drift = (yfull[o - 1] - yfull[0]) / max(o - 1, 1)
        dnaive = np.clip(base_nv + drift * (np.arange(ns) + 1), 0.0, None)
        dref[key] = a.mae(c["y"], dnaive)
    D = pd.Series(dref).reindex(piv.index)
    nm_ = pd.Series({(c["series"], c["origin"]): c["nmae"] for c in cells}).reindex(piv.index)
    scale = (nm_ / D)
    piv_d = piv.mul(scale, axis=0)
    comp = pd.DataFrame({"RelMAE vs naive": piv.mean(), "RelMAE vs naive+drift": piv_d.mean(),
                         "rank vs naive": piv.rank(axis=1).mean(),
                         "rank vs naive+drift": piv_d.rank(axis=1).mean()}).sort_values("RelMAE vs naive")
    print(f"drift benchmark is harder than plain naive in {(D < nm_).mean():.0%} of cells")
    print(comp.round(3).to_string())
    print("(within-cell ranks are invariant to the denominator by construction; only the "
          "cross-cell aggregation weights change)")

    # ---- per-vertical with bootstrap intervals ------------------------------------------------
    print("\n=== PER-VERTICAL (bootstrap CI on trimmed - catboost, resampling series) ===")
    vert = pd.Series([vmap.get(s, None) for s in ser], index=piv.index)
    vrows2 = []
    for v, grp in piv.groupby(vert.values):
        sids = np.array([s for s, _ in grp.index])
        if len(set(sids)) < 3:
            continue
        mu, lo, hi, _ = boot_stat_diff(grp[REF].values, grp["catboost"].values, sids, np.mean, B=2000)
        vrows2.append(dict(vertical=v, n_series=len(set(sids)), n_cells=len(grp),
                           trimmed=grp[REF].mean(), catboost=grp["catboost"].mean(),
                           sel_val=grp["sel_val"].mean(), diff=mu, lo=lo, hi=hi))
    print(pd.DataFrame(vrows2).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
