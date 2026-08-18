"""Sensitivity of the screened combination to the competence floor tau.

Reproduces the tau-sweep table in the Caveats section: mean and geometric-mean
RelMAE, P90, maximum, and the fallback rate at each threshold, over the same 293
cells as the main leaderboard.

The screened rule averages the de-duplicated members whose validation MAPE falls
below tau, requiring at least two qualifiers and otherwise falling back to the
equal mean of all members. Only tau varies here; membership, de-duplication, the
non-negativity constraint and the evaluation are unchanged.
"""

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, RESULTS  # noqa: F401

import importlib.util

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("a", str(CODE / "core.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

TAUS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.80]

cells = a.load_cells()


def screened(c, tau):
    """Equal-weighted mean over qualifying members; equal mean of all if < 2 qualify."""
    mem = a.members_of(c)
    keep = [m for m in mem if np.isfinite(c["val"][m]) and c["val"][m] < tau]
    fell_back = len(keep) < 2
    use = mem if fell_back else keep
    return np.mean([c["base"][m] for m in use], axis=0), fell_back


rows = []
for tau in TAUS:
    rel, fb = [], 0
    for c in cells:
        yh, f = screened(c, tau)
        rel.append(a.mae(c["y"], yh) / c["nmae"])
        fb += f
    rel = np.asarray(rel)
    rows.append({
        "tau": tau,
        "mean": rel.mean(),
        "geomean": np.exp(np.log(np.clip(rel, 1e-9, None)).mean()),
        "p90": np.quantile(rel, 0.90),
        "max": rel.max(),
        "fallback": fb / len(rel),
    })

t = pd.DataFrame(rows).set_index("tau")
print(f"=== SCREENED COMBINATION vs COMPETENCE FLOOR tau ({len(cells)} cells) ===")
print(t.round(3).to_string())

print("\nas reported in the paper:")
print(f"  mean RelMAE spans {t['mean'].min():.3f} to {t['mean'].max():.3f} "
      f"(total movement {t['mean'].max() - t['mean'].min():.3f})")
mono = bool(np.all(np.diff(t["mean"]) > 0) or np.all(np.diff(t["mean"]) < 0))
print(f"  monotone in tau: {mono}")
print(f"  fallback rate falls from {t['fallback'].iloc[0]:.0%} to {t['fallback'].iloc[-1]:.0%}")

# The recommended rule has no threshold; quoted for comparison on the same cells.
trim = np.asarray([a.mae(c["y"], np.mean(np.sort(
    np.array([c["base"][m] for m in a.members_of(c)]), axis=0)[1:-1], axis=0)) / c["nmae"]
    for c in cells])
print(f"\n  trimmed mean on the same cells: mean {trim.mean():.3f} "
      f"geo {np.exp(np.log(trim).mean()):.3f} P90 {np.quantile(trim,.9):.2f} max {trim.max():.2f}")
print(f"  screened trails the trimmed mean at every tau: "
      f"{bool((t['mean'] > trim.mean()).all())}")

t.to_csv(RESULTS / "tau_sensitivity.csv")
print(f"\nwrote {(RESULTS / 'tau_sensitivity.csv').relative_to(RESULTS.parent)}")
