"""Figure 8: mean RelMAE against 90th-percentile RelMAE.
"""

import os
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE


HERE = str(CODE)
OUT = str(FIGURES)

mpl.rcParams.update({"figure.dpi":120,"savefig.dpi":300,"font.size":14,"font.family":"DejaVu Sans",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,
    "axes.axisbelow":True,"legend.frameon":False,"axes.titlesize":16,"axes.labelsize":13})

LABEL = {"ens_trimmed":"Ensemble (Trimmed)","ens_wtd":"Ensemble (Weighted)",
         "ens_mean":"Ensemble (Mean)","ens_screened":"Ensemble (Screened ≥2)",
         "catboost":"CatBoost = Global val-selected","sel_val":"Per-series val-selected",
         "bayesianpymc":"BayesianPyMC","neuralprophet":"NeuralProphet","prophet":"Prophet",
         "sarimax":"SARIMAX","theta":"Theta"}
PALETTE = {"sarimax":"#4C72B0","bayesianpymc":"#DD8452","prophet":"#8C8C8C",
    "neuralprophet":"#C44E52","catboost":"#8172B3","sel_val":"#B07AA1","theta":"#A0522D",
    "ens_mean":"#4FA37A","ens_wtd":"#2E8B57","ens_trimmed":"#1a7a4f","ens_screened":"#7CC6A0"}
ENS = {"ens_trimmed","ens_wtd","ens_mean","ens_screened"}
REC = "ens_trimmed"

pc = pd.read_csv(RESULTS / "percell_test_metrics.csv")
piv = pc.pivot_table(index=["series", "origin"], columns="method", values="skill")
M = [m for m in piv.columns if m != "sel_global"]
stat = pd.DataFrame({"mean": [piv[m].mean() for m in M],
                     "p90": [piv[m].quantile(.90) for m in M]}, index=M)

ZOOM = [m for m in M if stat.loc[m, "mean"] <= 0.82]      # leading group + per-series selection
pad_x = (stat.loc[ZOOM, "mean"].max() - stat.loc[ZOOM, "mean"].min()) * 0.28
pad_y = (stat.loc[ZOOM, "p90"].max() - stat.loc[ZOOM, "p90"].min()) * 0.30
zx = (stat.loc[ZOOM, "mean"].min() - pad_x, stat.loc[ZOOM, "mean"].max() + pad_x * 2.6)
zy = (stat.loc[ZOOM, "p90"].min() - pad_y, stat.loc[ZOOM, "p90"].max() + pad_y)


def marker(m):
    return "*" if m in ENS else ("D" if m == "sel_val" else "o")


def size(m):
    return 420 if m == REC else (260 if m in ENS else 110)


fig, axs = plt.subplots(1, 2, figsize=(15.0, 6.4), layout="constrained",
                        gridspec_kw={"width_ratios": [1, 1.12]})

# ---------------- left: full range
a0 = axs[0]
for m in M:
    a0.scatter(stat.loc[m, "mean"], stat.loc[m, "p90"], s=size(m) * 0.7, color=PALETTE[m],
               marker=marker(m), zorder=3, edgecolor="black", linewidth=0.6)
for m in ["theta", "prophet", "neuralprophet", "sarimax", "bayesianpymc"]:
    a0.annotate(LABEL[m], (stat.loc[m, "mean"], stat.loc[m, "p90"]), textcoords="offset points",
                xytext=(10, -4), fontsize=10.5)
a0.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False,
                       ec="#333", lw=1.2, ls="--", zorder=5))
a0.annotate("leading group\n(right panel)", (zx[1], zy[1]), textcoords="offset points",
            xytext=(14, 16), fontsize=10, color="#333",
            arrowprops=dict(arrowstyle="->", lw=1.1, color="#333"))
a0.set_xlabel("Mean RelMAE  (typical accuracy → lower better)")
a0.set_ylabel("90th-pct RelMAE  (tail risk → lower better)")
a0.set_title("All eleven methods", fontsize=13, pad=8)

# ---------------- right: zoom on the leading group
a1 = axs[1]
for m in ZOOM:
    a1.scatter(stat.loc[m, "mean"], stat.loc[m, "p90"], s=size(m), color=PALETTE[m],
               marker=marker(m), zorder=3, edgecolor="black", linewidth=0.7)
    a1.annotate(LABEL[m], (stat.loc[m, "mean"], stat.loc[m, "p90"]), textcoords="offset points",
                xytext=(13, -4), fontsize=11,
                fontweight="bold" if m == REC else "normal")
a1.set_xlim(*zx); a1.set_ylim(*zy)
a1.set_xlabel("Mean RelMAE  (typical accuracy → lower better)")
a1.set_ylabel("90th-pct RelMAE  (tail risk → lower better)")
a1.set_title("Leading group, enlarged", fontsize=13, pad=8)

fig.suptitle("Accuracy versus tail risk — bottom-left is best: accurate and dependable", fontsize=15)

for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure08.{e}"), bbox_inches="tight",
                facecolor="white")
plt.close(fig)
print("zoom window:", tuple(round(v, 3) for v in zx), tuple(round(v, 3) for v in zy))
print(stat.loc[ZOOM].round(3).to_string())
print("wrote figure08.png and .pdf")
