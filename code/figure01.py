"""Figure 1: share of cells on which each single model is best.
"""

import os, collections
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
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
PRETTY = A.PRETTY

cells = A.load_cells()
theta = pd.read_csv(RESULTS / "theta_percell.csv").set_index(["series", "origin"])
picks = collections.Counter()
N = 0
for c in cells:
    if (c["series"], c["origin"]) not in theta.index:
        continue
    picks[min(BASE, key=lambda m: A.mae(c["y"], c["base"][m]))] += 1
    N += 1

win = pd.Series(picks).reindex(BASE).fillna(0).astype(int).sort_values(ascending=False)
pct = win.values / N * 100
lead_pct = pct.max()

fig, ax = plt.subplots(figsize=(9.4, 5.6))
bars = ax.bar([PRETTY[m] for m in win.index], pct, color="#4C72B0", width=0.62)

ax.set_ylim(0, lead_pct * 1.34)

# equal-share reference
ax.axhline(20, ls="--", lw=1, color="#999", zorder=1)
ax.text(len(win) - 0.45, 20 + lead_pct * 0.015, "equal share (20%)",
        ha="right", va="bottom", fontsize=11, color="#777")

# leader reference -- annotation right-aligned and below the line, clear of bars and title
ax.axhline(lead_pct, ls=":", lw=1.5, color="#D55E00", zorder=1)
ax.text(len(win) - 0.45, lead_pct - lead_pct * 0.02,
        f"always picking the portfolio leader: right {lead_pct:.0f}% of the time",
        ha="right", va="top", fontsize=10.5, color="#D55E00")

for r, cnt, p in zip(bars, win.values, pct):
    ax.text(r.get_x() + r.get_width() / 2, p + lead_pct * 0.018,
            f"{cnt}  ({p:.0f}%)", ha="center", va="bottom", fontsize=11.5, fontweight="bold")

ax.set_ylabel("Cells where model is best (%)")
ax.set_title("No single model dominates\nBest-model identity rotates across the 293 cells", pad=14)

for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure01.{e}"), bbox_inches="tight",
                facecolor="white")
plt.close(fig)
print("wrote figure01.png and .pdf")
print(f"N={N}  leader={lead_pct:.1f}%  counts={dict(win)}")
