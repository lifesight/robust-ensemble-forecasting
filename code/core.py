"""Core evaluation layer: loads every out-of-fold cell, applies the non-negativity
constraint and the family de-duplication, and builds the four combination rules and
the per-series selection baseline.

The target (daily purchases) is a strictly positive count, so a negative point
forecast is not admissible: all base-model forecasts are clipped at zero before any
evaluation or combination, uniformly across members. Error diversity is measured on
signed per-observation errors, the quantity the Bates-Granger diversification
argument concerns, with the per-cell error magnitude correlation as a secondary view.

Imported by analysis.py, figures.py, figure01.py and figure02.py. Running it directly prints
the core diagnostics and writes the error-correlation matrices of Section 5.7.
"""

import glob, os, re, json, collections
import numpy as np, pandas as pd
from scipy import stats

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE

PPDIR = str(FORECASTS)

BASE   = ["sarimax", "bayesianpymc", "prophet", "neuralprophet", "catboost"]
FAMILY = ["prophet", "neuralprophet"]
TAU    = 0.40
CLIP   = True                      # non-negativity constraint on all point forecasts

PRETTY = {"sarimax":"SARIMAX","bayesianpymc":"BayesianPyMC","prophet":"Prophet",
          "neuralprophet":"NeuralProphet","catboost":"CatBoost",
          "sel_val":"Validation-selected single",
          "ens_mean":"Ensemble (Mean)","ens_wtd":"Ensemble (Weighted)",
          "ens_trimmed":"Ensemble (Trimmed)","ens_screened":"Ensemble (Screened ≥2)"}
ENS   = ["ens_mean","ens_wtd","ens_trimmed","ens_screened"]
ORDER = BASE + ["sel_val"] + ENS


def mae(a, f):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(f, float))))


def mape(a, f):
    a = np.asarray(a, float); f = np.asarray(f, float); m = a != 0
    return float(np.mean(np.abs((a[m] - f[m]) / a[m]))) if m.any() else np.nan


def load_valmap():
    s = pd.concat([pd.read_csv(f) for f in [VALIDATION_SUMMARY]])
    s = s[s.model.notna() & (s.model != "model")]
    s["val_mape"] = pd.to_numeric(s["val_mape"], errors="coerce")
    return s.dropna(subset=["val_mape"]).set_index(["series", "origin", "model"])["val_mape"].to_dict()


def load_cells():
    val = load_valmap()
    cells = []
    for f in sorted(glob.glob(os.path.join(PPDIR, "*_origin*.csv"))):
        m = re.search(r"([^/\\]+)_origin(\d+)\.csv$", f)
        sid, o = m.group(1), int(m.group(2))
        d = pd.read_csv(f)
        if any(f"{b}__yhat" not in d.columns for b in BASE):
            continue
        y = d["actual"].to_numpy(float)
        yh = {b: d[f"{b}__yhat"].to_numpy(float) for b in BASE}
        n_neg = int(sum((v < 0).sum() for v in yh.values()))
        if CLIP:
            yh = {b: np.clip(v, 0.0, None) for b, v in yh.items()}
        cells.append(dict(series=sid, origin=o, ds=d["ds"].tolist(), y=y,
                          nmae=mae(y, d["naive"].to_numpy(float)), base=yh, n_neg=n_neg,
                          val={b: float(val.get((sid, o, b), np.nan)) for b in BASE}))
    return cells


def members_of(c):
    """Family de-duplication: keep the validation-better of the additive-decomposition
    family, plus every non-family model."""
    vp, vn = c["val"]["prophet"], c["val"]["neuralprophet"]
    pf = "prophet" if (not np.isfinite(vn)) or (np.isfinite(vp) and vp <= vn) else "neuralprophet"
    return [pf, "catboost", "sarimax", "bayesianpymc"]


def forecasts(c):
    """All nine evaluated methods for one cell."""
    out = dict(c["base"])
    mem = members_of(c)
    M = np.vstack([c["base"][m] for m in mem])
    out["ens_mean"] = M.mean(0)
    w = np.array([1.0 / max(c["val"][m], 1e-9) if np.isfinite(c["val"][m]) else np.nan for m in mem])
    if not np.isfinite(w).any():
        w = np.ones(len(mem))
    w = np.where(np.isfinite(w), w, np.nanmin(w[np.isfinite(w)])); w = w / w.sum()
    out["ens_wtd"] = np.average(M, 0, weights=w)
    out["ens_trimmed"] = np.sort(M, 0)[1:-1].mean(0)
    keep = [m for m in mem if np.isfinite(c["val"][m]) and c["val"][m] < TAU]
    out["ens_screened"] = np.vstack([c["base"][m] for m in keep]).mean(0) if len(keep) >= 2 else M.mean(0)
    # ex-ante selection: lowest validation MAPE among ALL five base models
    pick = min(BASE, key=lambda m: c["val"][m] if np.isfinite(c["val"][m]) else np.inf)
    out["sel_val"] = c["base"][pick]
    out["_pick"] = pick
    out["_mem"] = mem
    return out


def main():
    cells = load_cells()
    print(f"cells with all five base models: {len(cells)}   series: {len({c['series'] for c in cells})}")
    print(f"negative point forecasts clipped: {sum(c['n_neg'] for c in cells)} "
          f"of {sum(len(c['y']) for c in cells) * len(BASE)} member-observations")
    H = np.array([len(c["y"]) for c in cells])
    print(f"test horizon (days): min {H.min()} median {int(np.median(H))} max {H.max()}")

    rows, picks, fampicks, fallback, sel_hit = [], collections.Counter(), collections.Counter(), 0, 0
    for c in cells:
        F = forecasts(c)
        oracle = min(mae(c["y"], c["base"][m]) / c["nmae"] for m in BASE)
        truewin = min(BASE, key=lambda m: mae(c["y"], c["base"][m]))
        picks[truewin] += 1
        fampicks[F["_mem"][0]] += 1
        sel_hit += (F["_pick"] == truewin)
        mem = F["_mem"]
        if sum(1 for m in mem if np.isfinite(c["val"][m]) and c["val"][m] < TAU) < 2:
            fallback += 1
        for k in ORDER:
            sk = mae(c["y"], F[k]) / c["nmae"]
            rows.append(dict(series=c["series"], origin=c["origin"], method=k, skill=sk,
                             mape=mape(c["y"], F[k]), regret=sk - oracle, oracle=oracle))
    df = pd.DataFrame(rows)

    piv = df.pivot_table(index=["series", "origin"], columns="method", values="skill")[ORDER]
    rank = piv.rank(axis=1)
    g = pd.DataFrame({
        "mean": piv.mean(), "median": piv.median(), "p90": piv.quantile(.9), "max": piv.max(),
        "beats_naive": (piv < 1).mean(),
        "mape": df.pivot_table(index=["series", "origin"], columns="method", values="mape")[ORDER].mean(),
        "regret_mean": df.pivot_table(index=["series", "origin"], columns="method", values="regret")[ORDER].mean(),
        "regret_max": df.pivot_table(index=["series", "origin"], columns="method", values="regret")[ORDER].max(),
        "avg_rank": rank.mean(),
    }).sort_values("mean")
    print("\n=== LEADERBOARD (skill = MAE / seasonal-naive MAE; lower better) ===")
    print(g.round(3).to_string())

    # bootstrap CI on mean skill, resampling SERIES (respects within-series dependence)
    ser = piv.groupby(level=0).mean()
    rng = np.random.default_rng(0); B = 10000
    idx = rng.integers(0, len(ser), size=(B, len(ser)))
    boot = ser.values[idx].mean(1)
    ci = pd.DataFrame(np.percentile(boot, [2.5, 97.5], axis=0).T, index=ser.columns, columns=["lo", "hi"])
    print("\n=== 95% series-bootstrap CI on mean skill ===")
    print(ci.reindex(g.index).round(3).to_string())

    # Friedman + Nemenyi
    k, N = piv.shape[1], piv.shape[0]
    chi2, p = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
    CD = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    print(f"\nFriedman: chi2 = {chi2:.1f}, p = {p:.3g}  (k = {k}, N = {N})")
    print(f"Nemenyi CD (alpha = 0.05) = {CD:.3f}")
    av = g["avg_rank"].sort_values()
    print("\nrank gaps vs CatBoost (%.3f):" % av["catboost"])
    for m in av.index:
        if m != "catboost":
            d = av["catboost"] - av[m]
            print(f"  {PRETTY[m]:<28s} {d:+.3f}  {'SIGNIFICANT' if abs(d) > CD else 'n.s.'}")
    print("\nrank gaps vs Validation-selected single (%.3f):" % av["sel_val"])
    for m in ENS:
        d = av["sel_val"] - av[m]
        print(f"  {PRETTY[m]:<28s} {d:+.3f}  {'SIGNIFICANT' if abs(d) > CD else 'n.s.'}")

    # pairwise Wilcoxon
    print("\n=== Wilcoxon signed-rank, reference = Ensemble (Trimmed) ===")
    for opp in ["catboost", "sel_val"]:
        a, b = piv["ens_trimmed"], piv[opp]
        w = stats.wilcoxon(a, b)
        sa = piv.groupby(level=0).mean()
        ws = stats.wilcoxon(sa["ens_trimmed"], sa[opp])
        print(f"  vs {PRETTY[opp]}")
        print(f"    cell level   (N=293): better in {int((a < b).sum())}/{len(a)}, "
              f"mean diff {float((a - b).mean()):+.4f}, p = {w.pvalue:.4g}")
        print(f"    series level (n= 99): better in {int((sa['ens_trimmed'] < sa[opp]).sum())}/{len(sa)}, "
              f"mean diff {float((sa['ens_trimmed'] - sa[opp]).mean()):+.4f}, p = {ws.pvalue:.4g}")

    # winner rotation / selection quality / family split
    tot = sum(picks.values())
    print("\nbest single model per cell:", {PRETTY[m]: f"{n} ({n/tot:.0%})" for m, n in picks.most_common()})
    print(f"validation-selected model is the ex-post winner in {sel_hit}/{tot} ({sel_hit/tot:.0%}) of cells")
    print("family representative:", {PRETTY[m]: f"{n/tot:.0%}" for m, n in fampicks.most_common()})
    print(f"screened fallback rate: {fallback/tot:.0%}")
    print(f"oracle mean skill: {df[df.method=='ens_trimmed'].oracle.mean():.3f}")

    # ---- error diversity -------------------------------------------------
    E = pd.concat([pd.DataFrame({m: (c["y"] - c["base"][m]) / max(c["nmae"], 1e-9) for m in BASE})
                   for c in cells])
    Csigned = E.corr().loc[BASE, BASE]
    Cmag = pd.DataFrame({m: [mae(c["y"], c["base"][m]) / c["nmae"] for c in cells]
                         for m in BASE}).corr().loc[BASE, BASE]
    iu = np.triu_indices(len(BASE), 1)
    print(f"\n=== SIGNED per-observation error correlation (scaled by cell naive MAE) ===")
    print(Csigned.round(2).to_string())
    print(f"mean pairwise = {Csigned.values[iu].mean():.2f}   "
          f"range {Csigned.values[iu].min():.2f} to {Csigned.values[iu].max():.2f}")
    print(f"\n=== per-cell error MAGNITUDE correlation (secondary) ===")
    print(Cmag.round(2).to_string())
    print(f"mean pairwise = {Cmag.values[iu].mean():.2f}")

    Csigned.to_csv(RESULTS / "error_correlation_signed.csv")
    Cmag.to_csv(RESULTS / "error_correlation_magnitude.csv")
    return g, piv


if __name__ == "__main__":
    main()
