"""Reliability of the test ranking, and membership vs operator in the de-duplication ablation.

Two supporting analyses:

  (1) How well does the test ranking of the five base models reproduce from one
      rolling origin to the next within the same series? This bounds how large the
      validation-to-test rank correlation of Section 5.3 could be even under a
      perfect estimator: a target that does not reproduce cannot be predicted.

  (2) The ablation of Section 5.8 varies membership (4 de-duplicated members vs 5)
      and operator (median of 4 vs mean of the middle 3) at the same time. This
      separates them by holding each fixed in turn.

Prints to stdout; writes no files.
"""

import sys, os
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from paths import VALIDATION_SUMMARY

BASE = core.BASE


def rank_stability():
    v = pd.read_csv(VALIDATION_SUMMARY)
    v = v[v.is_ensemble == 0]
    test = v.pivot_table(index=["series", "origin"], columns="model", values="mape")
    val = v.pivot_table(index=["series", "origin"], columns="model", values="val_mape")

    def adjacent(frame):
        out = []
        for _, g in frame.groupby(level=0):
            g = g.sort_index()
            for a, b in zip(g.index[:-1], g.index[1:]):
                if g.loc[a, BASE].notna().all() and g.loc[b, BASE].notna().all():
                    out.append(stats.spearmanr(g.loc[a, BASE], g.loc[b, BASE]).statistic)
        return np.array(out)

    tt = adjacent(test)
    vv = adjacent(val)
    vt = np.array([
        stats.spearmanr(val.loc[i, BASE], test.loc[i, BASE]).statistic
        for i in test.index
        if i in val.index and test.loc[i, BASE].notna().all() and val.loc[i, BASE].notna().all()
    ])

    print("=== RELIABILITY OF THE TEST RANKING (Section 5.3) ===")
    print(f"  validation -> test, same cell         : mean rho {vt.mean():+.3f}  (n={len(vt)})")
    print(f"  test -> test, adjacent origins        : mean rho {tt.mean():+.3f}  (n={len(tt)}, frac>0 {(tt > 0).mean():.2f})")
    print(f"  validation -> validation, adj origins : mean rho {vv.mean():+.3f}  (n={len(vv)})")
    ceiling = np.sqrt(max(tt.mean(), 0.0))
    print(f"  attenuation ceiling given that reliability: {ceiling:.2f}")
    print(f"  observed correlation as a share of ceiling: {vt.mean() / ceiling:.0%}")


def membership_vs_operator():
    cells = core.load_cells()
    rows = {k: [] for k in ("median4_dedup", "mean4_dedup", "median5_all", "trim5_all", "mean5_all")}
    for c in cells:
        mem = core.members_of(c)
        M4 = np.vstack([c["base"][m] for m in mem])
        M5 = np.vstack([c["base"][m] for m in BASE])
        rel = lambda f: core.mae(c["y"], f) / c["nmae"]
        rows["median4_dedup"].append(rel(np.sort(M4, 0)[1:-1].mean(0)))
        rows["mean4_dedup"].append(rel(M4.mean(0)))
        rows["median5_all"].append(rel(np.median(M5, 0)))
        rows["trim5_all"].append(rel(np.sort(M5, 0)[1:-1].mean(0)))
        rows["mean5_all"].append(rel(M5.mean(0)))

    print("\n=== MEMBERSHIP VS OPERATOR (Section 5.8) ===")
    print(f"  {'rule':16s} {'mean':>7s} {'median':>8s} {'P90':>6s} {'max':>6s}")
    for k, x in rows.items():
        x = np.array(x)
        print(f"  {k:16s} {x.mean():7.3f} {np.median(x):8.3f} {np.percentile(x, 90):6.2f} {x.max():6.2f}")
    m = {k: np.mean(v) for k, v in rows.items()}
    print(f"\n  operator held fixed (median): 4 de-duplicated {m['median4_dedup']:.3f} vs 5 members {m['median5_all']:.3f}"
          f"  -> membership worth {m['median5_all'] - m['median4_dedup']:+.3f}")
    print(f"  membership held fixed (5)   : median {m['median5_all']:.3f} vs equal mean {m['mean5_all']:.3f}"
          f"  -> operator worth {m['mean5_all'] - m['median5_all']:+.3f}")
    print(f"  estimation-free variant     : trimmed mean over all five members {m['trim5_all']:.3f}"
          f" (uses no validation input at any point)")


if __name__ == "__main__":
    rank_stability()
    membership_vs_operator()
