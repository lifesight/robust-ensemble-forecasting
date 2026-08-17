"""Figure 5: tail differences with series-bootstrap intervals.
"""

import os
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE


HERE = str(CODE)
OUT = str(FIGURES)

mpl.rcParams.update({"figure.dpi":120,"savefig.dpi":300,"font.size":14,"font.family":"DejaVu Sans",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,
    "axes.axisbelow":True,"legend.frameon":False,"axes.titlesize":16,"axes.labelsize":14})

REF = "ens_trimmed"
pc = pd.read_csv(RESULTS / "percell_test_metrics.csv")
piv = pc.pivot_table(index=["series", "origin"], columns="method", values="skill")
ser = np.array([s for s, _ in piv.index])


def boot_diff(x, y, series, fn, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(series, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(u))]
    out = np.empty(B)
    for b in range(B):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(u), len(u))])
        out[b] = fn(x[idx]) - fn(y[idx])
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5)


FNS = {"P90": lambda v: np.percentile(v, 90), "P95": lambda v: np.percentile(v, 95),
       "P99": lambda v: np.percentile(v, 99), "Max": np.max}
PANELS = [("catboost", "vs CatBoost (= global val-selected)"),
          ("sel_val", "vs per-series val-selected")]

res = {}
for comp, _ in PANELS:
    res[comp] = [(nm,) + boot_diff(piv[REF].values, piv[comp].values, ser, fn)
                 for nm, fn in FNS.items()]
    for nm, mu, lo, hi in res[comp]:
        print(f"{comp:10s} {nm:4s} {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"{'  *excludes zero' if hi < 0 else ''}")

allv = [v for comp, _ in PANELS for _, _, lo, hi in res[comp] for v in (lo, hi)]
pad = (max(allv) - min(allv)) * 0.06
xlim = (min(allv) - pad, max(allv) + pad)

fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True, sharex=True, layout="constrained")
for axi, (comp, title) in zip(axs, PANELS):
    rows = res[comp]
    n_sig = sum(1 for _, _, _, hi in rows if hi < 0)
    yp = np.arange(len(rows))[::-1]
    for (nm, mu, lo, hi), y in zip(rows, yp):
        sig = hi < 0
        axi.errorbar(mu, y, xerr=[[mu - lo], [hi - mu]], fmt="o", ms=9, lw=1.9, capsize=5,
                     color="#1a7a4f" if sig else "#9a9a9a", zorder=4)
    axi.axvline(0, ls="--", lw=1.3, color="#333")
    axi.set_yticks(yp); axi.set_yticklabels(list(FNS))
    axi.set_xlim(*xlim)
    axi.set_xlabel("Difference in tail RelMAE  (negative = ensemble better)")
    sub = "no interval excludes zero" if n_sig == 0 else \
          f"{n_sig} of {len(rows)} intervals exclude zero"
    axi.set_title(f"{title}\n{sub}", fontsize=12.5, pad=8)

fig.suptitle("Tail differences with series-bootstrap 95% intervals", fontsize=15)

for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure05.{e}"), bbox_inches="tight",
                facecolor="white")
plt.close(fig)
print("wrote figure05.png and .pdf")
