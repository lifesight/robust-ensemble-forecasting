"""Figure 2: why decision granularity matters.
"""

import os
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import spearmanr
import core as A
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE


HERE = str(CODE)
OUT = str(FIGURES)

mpl.rcParams.update({"figure.dpi":120,"savefig.dpi":300,"font.size":14,"font.family":"DejaVu Sans",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,
    "axes.axisbelow":True,"legend.frameon":False,"axes.titlesize":16,"axes.labelsize":14})

BASE = A.BASE
GLOBAL = "catboost"

cells = A.load_cells()
theta = pd.read_csv(RESULTS / "theta_percell.csv").set_index(["series", "origin"])

rho, hit_cell, hit_glob, n = [], 0, 0, 0
for c in cells:
    if (c["series"], c["origin"]) not in theta.index:
        continue
    n += 1
    truewin = min(BASE, key=lambda m: A.mae(c["y"], c["base"][m]))
    pick = min(BASE, key=lambda m: c["val"][m] if np.isfinite(c["val"][m]) else np.inf)
    hit_cell += (pick == truewin)
    hit_glob += (GLOBAL == truewin)
    vv = np.array([c["val"][m] for m in BASE], float)
    tm = np.array([A.mape(c["y"], c["base"][m]) for m in BASE], float)
    if np.isfinite(vv).all():
        rho.append(spearmanr(vv, tm).statistic)

rho = np.array(rho)
pos = float((rho > 0).mean())
print(f"n={n}  per-cell hits={hit_cell} ({hit_cell/n:.1%})  global hits={hit_glob} ({hit_glob/n:.1%})")
print(f"mean rho={rho.mean():.3f}  positive in {pos:.0%} of cells")

fig, ax = plt.subplots(1, 2, figsize=(14.4, 5.9), layout="constrained")

# ---------------- left: hit rates
a0 = ax[0]
labels = ["Blind draw\n(1 of 5)", "Per-series selection\non validation",
          "Constant rule:\nalways the portfolio leader"]
vals = [20.0, hit_cell / n * 100, hit_glob / n * 100]
subs = ["expected", f"{hit_cell} of {n} cells", f"{hit_glob} of {n} cells"]
cols = ["#bbbbbb", "#B07AA1", "#1a7a4f"]
bars = a0.bar(labels, vals, color=cols, width=0.6)
top = max(vals) * 1.30
a0.set_ylim(0, top)
for r, v, sub in zip(bars, vals, subs):
    x = r.get_x() + r.get_width() / 2
    a0.text(x, v + top * 0.055, f"{v:.1f}%", ha="center", va="bottom",
            fontsize=14, fontweight="bold")
    a0.text(x, v + top * 0.015, sub, ha="center", va="bottom", fontsize=10, color="#666")
a0.set_ylabel("Picks the eventual best model (% of cells)", fontsize=12.5)
a0.tick_params(axis="x", labelsize=11.5)
a0.set_title("A constant rule beats the adaptive rule\nat the adaptive rule's own job",
             fontsize=13, pad=10)

# ---------------- right: within-cell rank correlation
a1 = ax[1]
counts, bins, _ = a1.hist(rho, bins=25, color="#4C72B0", alpha=0.9)
hmax = counts.max()
a1.set_ylim(0, hmax * 1.42)                      # reserved headroom for the annotation
a1.axvspan(-1.02, 0, color="#C44E52", alpha=0.07, zorder=0)
a1.axvline(0, ls="--", lw=1.3, color="#666")

m = rho.mean()
a1.plot([m, m], [0, hmax * 1.06], lw=2.2, color="#D55E00", zorder=4)
a1.annotate(f"mean ρ = {m:.2f}", xy=(m, hmax * 1.06), xytext=(m + 0.30, hmax * 1.30),
            fontsize=13, color="#D55E00", va="center",
            arrowprops=dict(arrowstyle="-", lw=1.2, color="#D55E00"))
a1.text(-0.98, hmax * 1.30, f"negative in {1-pos:.0%}\nof cells", fontsize=11,
        color="#8a3a3a", ha="left", va="center")
a1.text(1.00, hmax * 1.06, f"positive in {pos:.0%} of cells", fontsize=11,
        color="#3a5a8a", ha="right", va="center")
a1.set_xlim(-1.02, 1.02)
a1.set_xlabel("Within-cell rank correlation:  validation MAPE vs test MAPE")
a1.set_ylabel("Cells")
a1.set_title("Validation ranks the members almost at random\n(the same loss on both sides)",
             fontsize=13, pad=10)

for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure02.{e}"),
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote figure02.png and .pdf")
