"""Figure 4: Nemenyi critical-difference diagram.
"""

import os
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata, studentized_range
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE


HERE = str(CODE)
OUT = str(FIGURES)

mpl.rcParams.update({"figure.dpi":120,"savefig.dpi":300,"font.size":14,"font.family":"DejaVu Sans",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,
    "axes.axisbelow":True,"legend.frameon":False,"axes.titlesize":16,"axes.labelsize":14})

LABEL = {
    "ens_trimmed": "Ensemble (Trimmed)",
    "ens_wtd": "Ensemble (Weighted)",
    "ens_mean": "Ensemble (Mean)",
    "ens_screened": "Ensemble (Screened ≥2)",
    "catboost": "CatBoost = Global val-selected",
    "sel_val": "Per-series val-selected",
    "bayesianpymc": "BayesianPyMC",
    "neuralprophet": "NeuralProphet",
    "prophet": "Prophet",
    "sarimax": "SARIMAX",
    "theta": "Theta",
}
REC = "ens_trimmed"

pc = pd.read_csv(RESULTS / "percell_test_metrics.csv")
piv = pc.pivot_table(index=["series", "origin"], columns="method", values="skill")
ORDER = [m for m in piv.columns if m != "sel_global"]
# Cells within a series are not independent, so per-cell ranks are averaged within
# series first and the test is run on the 99 series. This is the unit the paper's
# headline comparisons use (Section 5.9) and it is the conservative choice: it
# widens the critical difference from 0.88 to 1.52.
Rcell = np.array([rankdata(r) for r in piv[ORDER].values])
Rser = pd.DataFrame(Rcell, index=piv.index, columns=ORDER).groupby(level=0).mean()
k, N = len(ORDER), len(Rser)

_, p = friedmanchisquare(*[Rser[m].values for m in ORDER])
CDv = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
avg = Rser[ORDER].values.mean(0)
order = np.argsort(avg)
names = [LABEL[ORDER[i]] for i in order]
keys = [ORDER[i] for i in order]
av = avg[order]
print(f"k={k} N={N} CD={CDv:.3f} p={p:.3g}")
for nm, a_ in zip(names, av):
    print(f"  {a_:.3f}  {nm}")

fig, ax = plt.subplots(figsize=(13.4, 5.6))
ax.set_axis_off()
lo, hi, ay = 1, k, 0
ax.plot([lo, hi], [ay, ay], color="black", lw=1.2)
for x in range(lo, hi + 1):
    ax.plot([x, x], [ay, ay + 0.05], color="black", lw=1)
    ax.text(x, ay + 0.11, str(x), ha="center", fontsize=11)
ax.text((lo + hi) / 2, ay + 0.54, f"Average rank   (Friedman p = {p:.0e})", ha="center", fontsize=13)

cdy = ay + 0.34
ax.plot([lo, lo + CDv], [cdy, cdy], color="#D55E00", lw=3, solid_capstyle="round")
for xx in (lo, lo + CDv):
    ax.plot([xx, xx], [cdy - 0.025, cdy + 0.025], color="#D55E00", lw=1.4)
ax.text(lo + CDv / 2, cdy + 0.05, f"CD = {CDv:.2f}", ha="center", va="bottom",
        fontsize=11, color="#D55E00")

half = (k + 1) // 2
y0, step = -0.26, -0.16


def draw(i, y, side):
    a_ = av[i]
    ax.plot([a_, a_], [ay, y], color="black", lw=0.9)
    edge = lo - 0.35 if side == "l" else hi + 0.35
    ax.plot([a_, edge], [y, y], color="black", lw=0.9)
    bold = keys[i] == REC
    ax.text(lo - 0.42 if side == "l" else hi + 0.42, y, f"{names[i]} ({a_:.2f})",
            ha="right" if side == "l" else "left", va="center", fontsize=11,
            fontweight="bold" if bold else "normal")


for i in range(half):
    draw(i, y0 + step * i, "l")
for idx, i in enumerate(range(k - 1, half - 1, -1)):
    draw(i, y0 + step * idx, "r")

grp, i = [], 0
while i < k:
    j = i
    while j + 1 < k and (av[j + 1] - av[i]) <= CDv:
        j += 1
    if j > i:
        grp.append((i, j))
    i += 1
grp = [g for g in grp if not any(g != h and h[0] <= g[0] and g[1] <= h[1] for h in grp)]
for gi, (i0, i1) in enumerate(grp):
    yb = ay - 0.07 - gi * 0.05
    ax.plot([av[i0] - 0.05, av[i1] + 0.05], [yb, yb], color="#0072B2", lw=4, solid_capstyle="round")

ax.set_xlim(lo - 4.9, hi + 4.9)
ax.set_ylim(y0 + step * half - 0.12, ay + 0.76)
ax.set_title("Critical-difference diagram (lower rank = better; blue bars join methods with no significant difference)",
             fontsize=13, pad=2)

for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure04.{e}"), bbox_inches="tight",
                facecolor="white")
plt.close(fig)
print("wrote figure04.png and .pdf")
