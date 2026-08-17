"""Figure 10: within-vertical mean RelMAE differences.
"""

import os
from math import comb
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE


HERE = str(CODE)
OUT = str(FIGURES)
MIN_RELIABLE = 6            # below this the percentile bootstrap is not interpretable

mpl.rcParams.update({"figure.dpi":120,"savefig.dpi":300,"font.size":14,"font.family":"DejaVu Sans",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,
    "axes.axisbelow":True,"legend.frameon":False,"axes.titlesize":15,"axes.labelsize":13})

pc = pd.read_csv(RESULTS / "percell_test_metrics.csv")
man = pd.read_csv(MANIFEST)
man["s"] = man.series_id.str[:10]
vmap = dict(zip(man.s, man.vertical))
piv = pc.pivot_table(index=["series", "origin"], columns="method", values="skill")
d = (piv["ens_trimmed"] - piv["catboost"]).rename("diff").reset_index()
d["v"] = d.series.map(vmap)
d = d[d.v.notna() & (d.v != "Unclassified")]

rng = np.random.default_rng(0)
rows = []
for v, g in d.groupby("v"):
    per_series = g.groupby("series")["diff"].mean()
    n = len(per_series)
    if n < 3:
        continue
    arr = per_series.values
    boots = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(4000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append(dict(vertical=v, n=n, diff=arr.mean(), lo=lo, hi=hi,
                     resamples=comb(2 * n - 1, n), reliable=n >= MIN_RELIABLE))
V = pd.DataFrame(rows).sort_values("diff", ascending=False).reset_index(drop=True)
print(V.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(12.2, 6.4))
yp = np.arange(len(V))[::-1]
for (_, r), y in zip(V.iterrows(), yp):
    sig = r.hi < 0
    col = "#1a7a4f" if sig else "#9a9a9a"
    ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt="none",
                ecolor=col, lw=1.6, capsize=4, alpha=1.0 if r.reliable else 0.45)
    if r.reliable:
        ax.scatter(r["diff"], y, s=120, color=col, zorder=5, edgecolor="black", linewidth=0.7)
    else:
        ax.scatter(r["diff"], y, s=120, facecolor="white", zorder=5, edgecolor=col,
                   linewidth=1.6, hatch="////")
        ax.text(r.hi + 0.012, y, f"only {r.resamples} distinct resamples",
                va="center", ha="left", fontsize=9, color="#8a3a3a")

ax.axvline(0, ls="--", lw=1.3, color="#333")
ax.set_yticks(yp)
ax.set_yticklabels([f"{r.vertical}  ({r.n} series)" for _, r in V.iterrows()])
for t, (_, r) in zip(ax.get_yticklabels(), V.iterrows()):
    if not r.reliable:
        t.set_color("#8a3a3a")
ax.set_xlabel("Mean RelMAE difference:  ensemble − CatBoost (= global val-selected)"
              "   (negative = ensemble better)")
ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 1.05 + 0.16)

handles = [
    Line2D([], [], marker="o", ls="none", color="#1a7a4f", ms=10, mec="black",
           label="interval excludes zero"),
    Line2D([], [], marker="o", ls="none", color="#9a9a9a", ms=10, mec="black",
           label="interval includes zero"),
    Line2D([], [], marker="o", ls="none", mfc="white", mec="#8a3a3a", mew=1.6, ms=10,
           label=f"fewer than {MIN_RELIABLE} series — bootstrap not interpretable"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
          fontsize=10.5, handletextpad=0.4, columnspacing=1.6)
ax.set_title("Within-vertical difference with series-bootstrap 95% intervals\n"
             "Most intervals cross zero, and the smallest verticals cannot support an interval at all",
             pad=12)

for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure10.{e}"), bbox_inches="tight",
                facecolor="white")
plt.close(fig)
rel = V[V.reliable]
print(f"\nreliable verticals: {len(rel)} of {len(V)}; of those, "
      f"{int((rel.hi < 0).sum())} exclude zero, {int((rel.lo > 0).sum())} exclude zero the other way")
print("wrote figure10.png and .pdf")
