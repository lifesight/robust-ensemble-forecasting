"""Figure generation. Writes every figure to figures/ as PNG and PDF.

Figure 9 is final as produced here; the remaining figures are refined by the
figureNN.py scripts, which run afterwards and overwrite their targets.
"""

import os, collections
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata, studentized_range, spearmanr
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

PRETTY = dict(A.PRETTY)
PRETTY.update({"sel_val":"Per-series val-selected","sel_global":"Global val-selected","theta":"Theta"})
PALETTE = {"sarimax":"#4C72B0","bayesianpymc":"#DD8452","prophet":"#8C8C8C",
    "neuralprophet":"#C44E52","catboost":"#8172B3","sel_val":"#B07AA1","theta":"#A0522D",
    "ens_mean":"#4FA37A","ens_wtd":"#2E8B57","ens_trimmed":"#1a7a4f","ens_screened":"#7CC6A0"}
BASE, ENS = A.BASE, A.ENS
ORDER = BASE + ["theta", "sel_val"] + ENS          # rank/comparison set (11 methods)
REC = "ens_trimmed"


def save(fig, n):
    for e in ("pdf", "png"):
        fig.savefig(f"{OUT}/{n}.{e}", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", n)


def cap_and_mark(ax, data, pct=97, pad=1.10, band=0.22):
    allv = np.concatenate([np.asarray(x, float) for x in data])
    cap = float(np.nanpercentile(allv, pct)) * pad; top = cap * (1 + band)
    ax.set_ylim(0, top); ax.axhline(cap, ls=(0,(4,3)), lw=1, color="#B0B0B0")
    ax.text(0.995, cap+(top-cap)*0.5, "cases above cap ▲", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=8.5, color="#8a8a8a")
    ax._top = top; return cap


def mark_top(ax, x, vals, cap, color):
    over = [v for v in vals if v > cap]
    if not over: return
    top = ax._top; y = cap + (top-cap)*0.42
    xs = np.full(len(over), x, float)
    if len(over) > 1: xs = x + np.linspace(-0.17, 0.17, len(over))
    ax.scatter(xs, np.full(len(over), y), marker="^", s=30, color=color, zorder=8)
    ax.text(x, cap+(top-cap)*0.80, f"{len(over)} (≤{max(over):.0f})", ha="center", va="center",
            fontsize=8, color=color)


def boot_diff(x, y, series, fn, B=3000, seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(series, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(u))]
    out = np.empty(B)
    for b in range(B):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(u), len(u))])
        out[b] = fn(x[idx]) - fn(y[idx])
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5)


# ------------------------------------------------------------------ data
cells = A.load_cells()
theta = pd.read_csv(RESULTS / "theta_percell.csv").set_index(["series", "origin"])
rows, picks = [], collections.Counter()
val_rows, rho_matched = [], []
for c in cells:
    F = A.forecasts(c)
    key = (c["series"], c["origin"])
    if key not in theta.index:
        continue
    picks[min(BASE, key=lambda m: A.mae(c["y"], c["base"][m]))] += 1
    r = dict(series=c["series"], origin=c["origin"],
             **{k: A.mae(c["y"], F[k])/c["nmae"] for k in BASE + ["sel_val"] + ENS})
    r["theta"] = float(theta.loc[key, "skill"])
    rows.append(r)
    vv = np.array([c["val"][m] for m in BASE], float)
    tm = np.array([A.mape(c["y"], c["base"][m]) for m in BASE], float)
    if np.isfinite(vv).all():
        rho_matched.append(spearmanr(vv, tm).statistic)
sk = pd.DataFrame(rows).set_index(["series", "origin"])[ORDER]
N = len(sk); k = len(ORDER)
ser = np.array([s for s, _ in sk.index])
print("cells", N, "methods", k)
GLOBAL = "catboost"

# ---------------------------------------------------------- FIG 1 rotation
win = pd.Series(picks).reindex(BASE).fillna(0).astype(int).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5.2))
b = ax.bar([PRETTY[m] for m in win.index], win.values/N*100, color="#4C72B0", width=0.62)
ax.axhline(20, ls="--", lw=1, color="#999")
ax.text(len(win)-0.5, 21, "equal share (20%)", ha="right", fontsize=11, color="#777")
ax.axhline(win.max()/N*100, ls=":", lw=1.4, color="#D55E00")
ax.text(0.02, win.max()/N*100+0.9, "a constant rule that always picks the leader\nis right this often",
        fontsize=10, color="#D55E00")
for r, v in zip(b, win.values):
    ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.6, f"{v}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Cells where model is best (%)")
ax.set_title("No single model dominates\nBest-model identity rotates across the 293 cells", pad=12)
save(fig, "figure01")

# --------------------------------------------- FIG 3 critical difference
M = sk[ORDER].values
statf, p = friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
avg = np.array([rankdata(r) for r in M]).mean(0)
CDv = studentized_range.ppf(0.95, k, np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*N))
order = np.argsort(avg); names = [PRETTY[ORDER[i]] for i in order]; av = avg[order]
fig, ax = plt.subplots(figsize=(12.5, 5.4)); ax.set_axis_off(); lo, hi = 1, k; ay = 0
ax.plot([lo, hi], [ay, ay], color="black", lw=1.2)
for x in range(lo, hi+1):
    ax.plot([x, x], [ay, ay+0.05], color="black", lw=1); ax.text(x, ay+0.11, str(x), ha="center", fontsize=11)
ax.text((lo+hi)/2, ay+0.54, f"Average rank   (Friedman p = {p:.0e})", ha="center", fontsize=13)
cdy = ay+0.34
ax.plot([lo, lo+CDv], [cdy, cdy], color="#D55E00", lw=3, solid_capstyle="round")
for xx in (lo, lo+CDv): ax.plot([xx, xx], [cdy-0.025, cdy+0.025], color="#D55E00", lw=1.4)
ax.text(lo+CDv/2, cdy+0.05, f"CD = {CDv:.2f}", ha="center", va="bottom", fontsize=11, color="#D55E00")
half = (k+1)//2; y0 = -0.26; step = -0.16
for i in range(half):
    aa = av[i]; y = y0+step*i
    ax.plot([aa, aa], [ay, y], color="black", lw=0.9); ax.plot([aa, lo-0.35], [y, y], color="black", lw=0.9)
    ax.text(lo-0.42, y, f"{names[i]} ({aa:.2f})", ha="right", va="center", fontsize=11,
            fontweight="bold" if "Trimmed" in names[i] else "normal")
for idx, i in enumerate(range(k-1, half-1, -1)):
    aa = av[i]; y = y0+step*idx
    ax.plot([aa, aa], [ay, y], color="black", lw=0.9); ax.plot([aa, hi+0.35], [y, y], color="black", lw=0.9)
    ax.text(hi+0.42, y, f"{names[i]} ({aa:.2f})", ha="left", va="center", fontsize=11,
            fontweight="bold" if "Trimmed" in names[i] else "normal")
grp = []; i = 0
while i < k:
    j = i
    while j+1 < k and (av[j+1]-av[i]) <= CDv: j += 1
    if j > i: grp.append((i, j))
    i += 1
grp = [g for g in grp if not any(g != h and h[0] <= g[0] and g[1] <= h[1] for h in grp)]
for gi, (i0, i1) in enumerate(grp):
    yb = ay-0.07-gi*0.05
    ax.plot([av[i0]-0.05, av[i1]+0.05], [yb, yb], color="#0072B2", lw=4, solid_capstyle="round")
ax.set_xlim(lo-3.9, hi+3.9); ax.set_ylim(y0+step*half-0.12, ay+0.76)
ax.set_title("Critical-difference diagram (lower rank = better; blue bars join methods with no significant difference)",
             fontsize=13, pad=2)
save(fig, "figure04")
print("Friedman p", p, "CD", round(CDv, 3))

# ------------------------------------------------- FIG 4 rank distribution
R = pd.DataFrame([rankdata(r) for r in sk[ORDER].values], columns=ORDER)
meanr = R.mean().sort_values(); meth = list(meanr.index)
frac = np.zeros((len(meth), k))
for a_, m in enumerate(meth):
    v = R[m].round().astype(int).value_counts(normalize=True)
    for r in range(1, k+1): frac[a_, r-1] = v.get(r, 0)
fig, ax = plt.subplots(figsize=(13, 7.0))
cmap = plt.cm.RdYlGn_r; left = np.zeros(len(meth))
for r in range(k):
    ax.barh(range(len(meth)), frac[:, r]*100, left=left*100, height=0.68,
            color=cmap(r/(k-1)), edgecolor="white", linewidth=0.6)
    for a_ in range(len(meth)):
        if frac[a_, r] > 0.055:
            ax.text((left[a_]+frac[a_, r]/2)*100, a_, str(r+1), ha="center", va="center", fontsize=9,
                    color="white" if r > k*0.55 or r < k*0.2 else "black")
    left = left+frac[:, r]
ax.set_yticks(range(len(meth)))
ax.set_yticklabels([f"{PRETTY[m]}  ({meanr[m]:.2f})" for m in meth], fontsize=12)
for t, m in zip(ax.get_yticklabels(), meth):
    if m == REC: t.set_fontweight("bold")
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Share of cells (%)"); ax.grid(False)
ax.set_title(f"Where each method lands, cell by cell\nSegments are ranks 1 (best, green) to {k} (worst, red); label shows mean rank", pad=12)
save(fig, "figure06")

# -------------------------------------------- FIG 5 RelMAE distribution
data = [sk[m].dropna().values for m in ORDER]
o = np.argsort([x.mean() for x in data]); meth_o = [ORDER[i] for i in o]; data_o = [data[i] for i in o]
fig, ax = plt.subplots(figsize=(13.5, 6.4))
cap = cap_and_mark(ax, data_o)
bp = ax.boxplot(data_o, patch_artist=True, widths=0.6, showfliers=False,
                medianprops=dict(color="black", lw=1.6), whis=(5, 95))
for patch, m in zip(bp["boxes"], meth_o): patch.set_facecolor(PALETTE[m]); patch.set_alpha(0.85)
for i, (m, vals) in enumerate(zip(meth_o, data_o), 1): mark_top(ax, i, vals, cap, PALETTE[m])
ax.axhline(1.0, ls="--", lw=1, color="#D55E00"); ax.text(0.6, 1.02, "beats naive", fontsize=10, color="#D55E00")
ax.set_ylabel("RelMAE  (error / naive;  lower = better)")
ax.set_xticklabels([PRETTY[m] for m in meth_o], rotation=20, ha="right")
ax.set_title("Ensembles are tight; single models, Theta and per-series selection have long tails\n(box = IQR, whiskers 5–95th pct; ▲ = cases above the axis cap)", pad=14)
save(fig, "figure07")

# ------------------------------------------------- FIG 6 mean vs tail
stat = pd.DataFrame({"mean": [sk[m].mean() for m in ORDER], "p90": [sk[m].quantile(.9) for m in ORDER]}, index=ORDER)
fig, ax = plt.subplots(figsize=(9.2, 6.4))
for m in ORDER:
    ens = m in ENS
    ax.scatter(stat.loc[m, "mean"], stat.loc[m, "p90"], s=240 if m == REC else (150 if ens else 95),
               color=PALETTE[m], marker="*" if ens else ("D" if m == "sel_val" else "o"),
               zorder=3, edgecolor="black", linewidth=0.6)
    ax.annotate(PRETTY[m], (stat.loc[m, "mean"], stat.loc[m, "p90"]), textcoords="offset points",
                xytext=(9, 4), fontsize=10.5, fontweight="bold" if m == REC else "normal")
ax.set_xlabel("Mean RelMAE  (typical accuracy → lower better)")
ax.set_ylabel("90th-pct RelMAE  (tail risk → lower better)")
ax.set_title("Accuracy vs tail risk\nBottom-left is best: accurate AND dependable")
save(fig, "figure08")

# --------------------------------- FIG 2 ensemble vs per-series selection
d = (sk["sel_val"]-sk["ens_trimmed"]).values
fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.8), gridspec_kw={"width_ratios": [1.1, 1]}, layout="constrained")
a0 = ax[0]
a0.scatter(sk["sel_val"], sk["ens_trimmed"], s=26, alpha=0.6, color="#1a7a4f", edgecolor="none")
lim = [0.15, 2.2]; a0.plot(lim, lim, ls="--", lw=1.2, color="#666")
a0.set_xlim(lim); a0.set_ylim(lim); a0.set_xscale("log"); a0.set_yscale("log")
a0.set_xlabel("Per-series validation-selected model  (RelMAE)")
a0.set_ylabel("Ensemble (Trimmed)  (RelMAE)")
a0.text(0.30, 1.75, "selection better", fontsize=11, color="#888")
a0.text(1.05, 0.24, "ensemble better", fontsize=11, color="#1a7a4f", fontweight="bold")
a0.set_title(f"Combining beats choosing per series:\nensemble wins {int((d>0).sum())} of {len(d)} cells", fontsize=13, pad=10)
a1 = ax[1]
a1.hist(np.clip(d, -0.5, 0.5), bins=45, color="#1a7a4f", alpha=0.85)
a1.axvline(0, ls="--", lw=1.2, color="#666"); a1.axvline(d.mean(), lw=2, color="#D55E00")
a1.text(d.mean(), a1.get_ylim()[1]*0.92, f"  mean {d.mean():+.3f}", color="#D55E00", fontsize=11)
a1.set_xlabel("Ensemble advantage  (selection − ensemble)")
a1.set_ylabel("Cells"); a1.set_title("Per-cell advantage\n(axis clipped at ±0.5)", fontsize=13, pad=10)
save(fig, "figure03")

# --------------------------------------------- FIG 7 error correlation
E = pd.concat([pd.DataFrame({m: (c["y"]-c["base"][m])/max(c["nmae"], 1e-9) for m in BASE}) for c in cells])
C = E.corr().loc[BASE, BASE]
fig, ax = plt.subplots(figsize=(8.2, 7.0))
im = ax.imshow(C.values, cmap="RdYlBu_r", vmin=0, vmax=1)
ax.set_xticks(range(len(BASE))); ax.set_yticks(range(len(BASE)))
ax.set_xticklabels([PRETTY[m] for m in BASE], rotation=28, ha="right")
ax.set_yticklabels([PRETTY[m] for m in BASE])
for i in range(len(BASE)):
    for j in range(len(BASE)):
        v = C.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=13,
                color="white" if (v > 0.72 or i == j) else "black",
                fontweight="bold" if i != j and v < 0.55 else "normal")
ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.82, label="Pearson correlation of signed error")
iu = np.triu_indices(len(BASE), 1)
ax.set_title("Members make correlated but far from redundant errors\n"
             f"Signed per-period forecast error; mean off-diagonal = {C.values[iu].mean():.2f} "
             f"(range {C.values[iu].min():.2f}–{C.values[iu].max():.2f})", pad=14, fontsize=14)
save(fig, "figure09")

# --------------------------------------------------- FIG 8 by vertical + CIs
man = pd.read_csv(MANIFEST); man["prefix"] = man.series_id.str[:10]
vmap = dict(zip(man.prefix, man.vertical))
sv = sk.reset_index(); sv["vertical"] = sv.series.map(vmap)
sv = sv[sv.vertical.notna() & (sv.vertical != "Unclassified")]
nser = sv.groupby("vertical").series.nunique(); keep = nser[nser >= 3].index
sv = sv[sv.vertical.isin(keep)]
vres = []
for v, g in sv.groupby("vertical"):
    mu, lo_, hi_ = boot_diff(g[REC].values, g["catboost"].values, g.series.values, np.mean, B=2000)
    vres.append(dict(vertical=v, ens=g[REC].mean(), cb=g["catboost"].mean(), sel=g["sel_val"].mean(),
                     diff=mu, lo=lo_, hi=hi_, n=nser[v]))
V = pd.DataFrame(vres).sort_values("diff")
fig, ax = plt.subplots(figsize=(11.0, 6.2))
yp = np.arange(len(V))
cols = ["#1a7a4f" if h < 0 else ("#C44E52" if l > 0 else "#9a9a9a") for l, h in zip(V.lo, V.hi)]
ax.errorbar(V["diff"], yp, xerr=[V["diff"]-V.lo, V.hi-V["diff"]], fmt="none", ecolor="#666", lw=1.4, capsize=4)
ax.scatter(V["diff"], yp, s=110, color=cols, zorder=5, edgecolor="black", linewidth=0.6)
ax.axvline(0, ls="--", lw=1.3, color="#333")
ax.set_yticks(yp); ax.set_yticklabels([f"{r.vertical}  ({r.n} series)" for r in V.itertuples()])
ax.set_xlabel("Mean RelMAE difference:  ensemble − CatBoost   (negative = ensemble better)")
ax.set_title("Within-vertical difference with series-bootstrap 95% intervals\n"
             "Most intervals cross zero: per-vertical differences are not resolvable at this sample size", pad=12)
save(fig, "figure10")
print(V.round(3).to_string(index=False))

# ------------------------------------------ FIG 9 tail metrics with CIs
fns = {"P90": lambda v: np.percentile(v, 90), "P95": lambda v: np.percentile(v, 95),
       "P99": lambda v: np.percentile(v, 99), "Max": np.max}
fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True, layout="constrained")
for axi, comp, title in zip(axs, ["catboost", "sel_val"],
                            ["vs CatBoost (= global val-selected)", "vs per-series val-selected"]):
    res = [(nm,)+boot_diff(sk[REC].values, sk[comp].values, ser, fn) for nm, fn in fns.items()]
    yp = np.arange(len(res))[::-1]
    for (nm, mu, lo_, hi_), y in zip(res, yp):
        sig = hi_ < 0
        axi.errorbar(mu, y, xerr=[[mu-lo_], [hi_-mu]], fmt="o", ms=9, lw=1.8, capsize=5,
                     color="#1a7a4f" if sig else "#9a9a9a", zorder=4)
    axi.axvline(0, ls="--", lw=1.3, color="#333")
    axi.set_yticks(yp); axi.set_yticklabels(list(fns))
    axi.set_xlabel("Difference in tail RelMAE  (negative = ensemble better)")
    axi.set_title(title, fontsize=13)
axs[0].text(0.02, 0.20, "all intervals cross zero:\nno tail advantage established",
            transform=axs[0].transAxes, fontsize=11, color="#8a3a3a")
fig.suptitle("Tail claims held to the same standard as mean claims\nSeries-bootstrap 95% intervals on the difference", fontsize=15)
save(fig, "figure05")

# ------------------------------- FIG 10 selection granularity (v4 headline)
val = pd.DataFrame([{**{"series": c["series"], "origin": c["origin"]}, **c["val"]} for c in cells]).set_index(["series", "origin"])[BASE]
truewin = pd.Series({(c["series"], c["origin"]): min(BASE, key=lambda m: A.mae(c["y"], c["base"][m])) for c in cells})
val = val.reindex(sk.index); truewin = truewin.reindex(sk.index)
hit_cell = float((val.idxmin(axis=1) == truewin).mean())
hit_glob = float((truewin == GLOBAL).mean())
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), layout="constrained")
a0 = ax[0]
bars = {"Blind draw\n(1 of 5)": 0.20, "Per-series selection\non validation": hit_cell,
        "Constant rule:\nalways the portfolio leader": hit_glob}
cols = ["#bbbbbb", "#B07AA1", "#1a7a4f"]
bb = a0.bar(list(bars), [v*100 for v in bars.values()], color=cols, width=0.6)
for r, v in zip(bb, bars.values()):
    a0.text(r.get_x()+r.get_width()/2, v*100+0.8, f"{v:.1%}", ha="center", fontsize=13, fontweight="bold")
a0.set_ylabel("Picks the eventual best model (% of cells)", fontsize=12.5)
a0.set_ylim(0, max(bars.values())*128)
a0.tick_params(axis="x", labelsize=11.5)
a0.set_title("A constant rule beats the adaptive rule\nat the adaptive rule's own job", fontsize=13, pad=10)
a1 = ax[1]
a1.hist(rho_matched, bins=25, color="#4C72B0", alpha=0.85)
a1.axvline(0, ls="--", lw=1.3, color="#666")
a1.axvline(np.mean(rho_matched), lw=2.2, color="#D55E00")
a1.text(np.mean(rho_matched), a1.get_ylim()[1]*0.93, f"  mean ρ = {np.mean(rho_matched):.2f}",
        color="#D55E00", fontsize=12)
a1.set_xlabel("Within-cell Spearman ρ:  validation MAPE vs test MAPE")
a1.set_ylabel("Cells")
a1.set_title("Validation ranks members almost at random\n(matched loss on both sides — not a metric artefact)", fontsize=13, pad=10)
save(fig, "figure02")
print(f"hit rates -- per-cell {hit_cell:.3f}  global {hit_glob:.3f}  matched-loss rho mean {np.mean(rho_matched):.3f}")
