"""Figure 7: per-cell RelMAE distribution by method.
"""

import os, math
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

LABEL = {"ens_trimmed":"Ensemble (Trimmed)","ens_wtd":"Ensemble (Weighted)",
         "ens_mean":"Ensemble (Mean)","ens_screened":"Ensemble (Screened ≥2)",
         "catboost":"CatBoost (= global val-selected)","sel_val":"Per-series val-selected",
         "bayesianpymc":"BayesianPyMC","neuralprophet":"NeuralProphet","prophet":"Prophet",
         "sarimax":"SARIMAX","theta":"Theta"}
PALETTE = {"sarimax":"#4C72B0","bayesianpymc":"#DD8452","prophet":"#8C8C8C",
    "neuralprophet":"#C44E52","catboost":"#8172B3","sel_val":"#B07AA1","theta":"#A0522D",
    "ens_mean":"#4FA37A","ens_wtd":"#2E8B57","ens_trimmed":"#1a7a4f","ens_screened":"#7CC6A0"}

pc = pd.read_csv(RESULTS / "percell_test_metrics.csv")
piv = pc.pivot_table(index=["series", "origin"], columns="method", values="skill")
methods = [m for m in piv.columns if m != "sel_global"]
geo = {m: float(np.exp(np.log(piv[m].clip(lower=1e-9)).mean())) for m in methods}
ORDER = sorted(methods, key=lambda m: geo[m])            # matches Table 1a

data = [piv[m].values for m in ORDER]
allv = np.concatenate(data)
cap = float(np.percentile(allv, 97)) * 1.10
top = cap * 1.26

fig, ax = plt.subplots(figsize=(14.6, 6.6))
ax.set_ylim(0, top)
ax.axhline(cap, ls=(0, (4, 3)), lw=1, color="#B0B0B0")

# whiskers clipped at the cap so nothing is drawn past it
clipped = [np.minimum(v, cap) for v in data]
bp = ax.boxplot(clipped, patch_artist=True, widths=0.6, showfliers=False,
                medianprops=dict(color="black", lw=1.6), whis=(5, 95))
for patch, m in zip(bp["boxes"], ORDER):
    patch.set_facecolor(PALETTE[m]); patch.set_alpha(0.85)

marker_y = cap + (top - cap) * 0.34
label_y = cap + (top - cap) * 0.74
for i, (m, v) in enumerate(zip(ORDER, data), 1):
    over = v[v > cap]
    p95 = float(np.percentile(v, 95))
    if p95 > cap:                                        # note the truncated whisker
        ax.plot([i, i], [cap, cap + (top - cap) * 0.14], lw=1.4, color=PALETTE[m], zorder=6)
        ax.text(i, cap + (top - cap) * 0.16, f"p95 {p95:.1f}", ha="center", va="bottom",
                fontsize=7.5, color=PALETTE[m])
    if len(over) == 0:
        continue
    n_draw = min(len(over), 6)
    xs = np.full(n_draw, i, float) if n_draw == 1 else i + np.linspace(-0.19, 0.19, n_draw)
    ax.scatter(xs, np.full(n_draw, marker_y), marker="^", s=26, color=PALETTE[m], zorder=8)
    ax.text(i, label_y, f"{len(over)}  (max {over.max():.2f})", ha="center", va="center",
            fontsize=8, color=PALETTE[m])

# no inline legend for the markers: the title already defines them, and any text placed on the
# marker row collides with the denser cases at one end or the other.
ax.axhline(1.0, ls="--", lw=1, color="#D55E00")
ax.text(0.55, 1.03, "beats naive", fontsize=10, color="#D55E00")
ax.set_ylabel("RelMAE  (error / naive;  lower = better)")
ax.set_xticklabels([LABEL[m] for m in ORDER], rotation=22, ha="right", fontsize=10)
ax.set_title("Ensembles are tight; single models, Theta and per-series selection have long tails\n"
             f"(box = IQR, whiskers 5–95th pct clipped at the cap of {cap:.2f}; "
             "▲ = cases above the cap, with the count and true maximum)", pad=14)

for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure07.{e}"), bbox_inches="tight",
                facecolor="white")
plt.close(fig)
print(f"cap = {cap:.3f}")
for m in ORDER:
    v = piv[m].values
    print(f"  {m:14s} geo {geo[m]:.3f}  n>cap {int((v>cap).sum()):3d}  max {v.max():6.2f}  "
          f"p95 {np.percentile(v,95):5.2f}")
print("wrote figure07.png and .pdf")
