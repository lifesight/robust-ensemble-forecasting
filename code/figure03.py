"""Figure 3: the trimmed ensemble against per-series selection.
"""

import os
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE


HERE = str(CODE)
OUT = str(FIGURES)

mpl.rcParams.update({"figure.dpi":120,"savefig.dpi":300,"font.size":14,"font.family":"DejaVu Sans",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,
    "axes.axisbelow":True,"legend.frameon":False,"axes.titlesize":16,"axes.labelsize":14})

pc = pd.read_csv(RESULTS / "percell_test_metrics.csv")
piv = pc.pivot_table(index=["series", "origin"], columns="method", values="skill")
ens, sel = piv["ens_trimmed"].values, piv["sel_val"].values
d = sel - ens
N = len(d)
wins = int((d > 0).sum())
LIM = 0.5
hi_out = int((d > LIM).sum()); lo_out = int((d < -LIM).sum())
inside = d[(d >= -LIM) & (d <= LIM)]
print(f"N={N} ensemble wins {wins}; beyond +{LIM}: {hi_out}; beyond -{LIM}: {lo_out}; mean {d.mean():+.4f}")

fig, ax = plt.subplots(1, 2, figsize=(14.6, 5.9), gridspec_kw={"width_ratios": [1.05, 1]},
                       layout="constrained")

# ---------------- left: per-cell scatter
a0 = ax[0]
a0.scatter(sel, ens, s=26, alpha=0.6, color="#1a7a4f", edgecolor="none")
lim = [0.15, 2.4]
a0.plot(lim, lim, ls="--", lw=1.2, color="#666")
a0.set_xlim(lim); a0.set_ylim(lim)
a0.set_xscale("log"); a0.set_yscale("log")
ticks = [0.25, 0.5, 1.0, 2.0]
for axis in (a0.xaxis, a0.yaxis):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(FixedLocator([]))
    axis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
a0.set_xlabel("Per-series validation-selected model  (RelMAE)")
a0.set_ylabel("Ensemble (Trimmed)  (RelMAE)")
a0.text(0.24, 1.95, "selection better", fontsize=11.5, color="#888")
a0.text(1.02, 0.215, "ensemble better", fontsize=11.5, color="#1a7a4f", fontweight="bold")
a0.set_title(f"Combining beats choosing per series:\nensemble wins {wins} of {N} cells",
             fontsize=13, pad=10)

# ---------------- right: distribution of the per-cell advantage
a1 = ax[1]
counts, _, _ = a1.hist(inside, bins=40, range=(-LIM, LIM), color="#1a7a4f", alpha=0.9)
hmax = counts.max()
a1.set_ylim(0, hmax * 1.30)
a1.set_xlim(-LIM * 1.12, LIM * 1.12)
a1.axvline(0, ls="--", lw=1.2, color="#666")
a1.plot([d.mean(), d.mean()], [0, hmax * 1.05], lw=2.2, color="#D55E00", zorder=4)
a1.annotate(f"mean {d.mean():+.3f}", xy=(d.mean(), hmax * 1.05),
            xytext=(d.mean() + 0.13, hmax * 1.21), fontsize=12.5, color="#D55E00", va="center",
            arrowprops=dict(arrowstyle="-", lw=1.2, color="#D55E00"))
# out-of-range cells reported rather than piled into the end bins
for x, n_, ha, col in ((LIM * 1.10, hi_out, "right", "#1a7a4f"),
                       (-LIM * 1.10, lo_out, "left", "#8a3a3a")):
    if n_:
        a1.annotate("", xy=(x, hmax * 0.44), xytext=(x - np.sign(x) * 0.055, hmax * 0.44),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color=col))
        a1.text(x, hmax * 0.52, f"{n_} cells\nbeyond {'+' if x>0 else '−'}{LIM}",
                ha=ha, va="bottom", fontsize=10.5, color=col)
a1.set_xlabel("Ensemble advantage  (selection − ensemble)")
a1.set_ylabel("Cells")
a1.set_title("Per-cell advantage\n(bars show the ±0.5 range; wider cases counted on the margins)",
             fontsize=13, pad=10)

for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure03.{e}"), bbox_inches="tight",
                facecolor="white")
plt.close(fig)
print("wrote figure03.png and .pdf")
