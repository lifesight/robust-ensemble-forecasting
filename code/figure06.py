"""Figure 6: per-method rank distribution.
"""

import os
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import rankdata
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
         "catboost":"CatBoost = Global val-selected","sel_val":"Per-series val-selected",
         "bayesianpymc":"BayesianPyMC","neuralprophet":"NeuralProphet","prophet":"Prophet",
         "sarimax":"SARIMAX","theta":"Theta"}
REC = "ens_trimmed"

pc = pd.read_csv(RESULTS / "percell_test_metrics.csv")
piv = pc.pivot_table(index=["series", "origin"], columns="method", values="skill")
ORDER = [m for m in piv.columns if m != "sel_global"]
M = piv[ORDER].values
N, k = M.shape

R = np.array([rankdata(r) for r in M])          # average ranks, half-integers where tied
meanr = pd.Series(R.mean(0), index=ORDER).sort_values()
frac = np.zeros((k, k))                          # method x rank, tie mass split across neighbours
for j, m in enumerate(ORDER):
    for v in R[:, j]:
        lo, hi = int(np.floor(v)), int(np.ceil(v))
        if lo == hi:
            frac[j, lo - 1] += 1.0
        else:
            frac[j, lo - 1] += 0.5
            frac[j, hi - 1] += 0.5
frac = frac / N

# mean rank is preserved exactly by the split
check = (frac * np.arange(1, k + 1)).sum(1)
assert np.allclose(check, R.mean(0)), "tie split changed the mean rank"
print("tie split preserves mean rank: OK")
print("half-integer share of all ranks:", round(float((R % 1 != 0).mean()), 3))

meth = list(meanr.index)
idx = [ORDER.index(m) for m in meth]
F = frac[idx]

fig, ax = plt.subplots(figsize=(13.2, 7.0))
cmap = plt.cm.RdYlGn_r
left = np.zeros(len(meth))
for r in range(k):
    ax.barh(range(len(meth)), F[:, r] * 100, left=left * 100, height=0.68,
            color=cmap(r / (k - 1)), edgecolor="white", linewidth=0.6)
    for a_ in range(len(meth)):
        if F[a_, r] > 0.055:
            ax.text((left[a_] + F[a_, r] / 2) * 100, a_, str(r + 1), ha="center", va="center",
                    fontsize=9, color="white" if r > k * 0.55 or r < k * 0.2 else "black")
    left = left + F[:, r]

ax.set_yticks(range(len(meth)))
ax.set_yticklabels([f"{LABEL[m]}  ({meanr[m]:.2f})" for m in meth], fontsize=12)
for t, m in zip(ax.get_yticklabels(), meth):
    if m == REC:
        t.set_fontweight("bold")
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Share of cells (%)")
ax.grid(False)
ax.set_title("Where each method lands, cell by cell\n"
             f"Segments are ranks 1 (best, green) to {k} (worst, red); label shows mean rank",
             pad=12)

for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure06.{e}"), bbox_inches="tight",
                facecolor="white")
plt.close(fig)
print("wrote figure06.png and .pdf")
print("\nrank-1 share, top methods:")
for m in meth[:6]:
    print(f"  {LABEL[m]:32s} {frac[ORDER.index(m),0]*100:5.1f}%   mean rank {meanr[m]:.2f}")
