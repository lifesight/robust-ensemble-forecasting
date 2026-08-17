"""Theta benchmark, evaluated per cell.

Theta is a competitor rather than an ensemble member, and uses none of the media-spend
covariates. Configured with weekly seasonality only.
"""

import os, glob, re, warnings
import numpy as np, pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel

warnings.filterwarnings("ignore")
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE

HERE = str(CODE)
PPDIR = str(FORECASTS)
CLEAN = str(CORPUS)
SUMM = str(SUMMARY_CLEAN)
PERIOD = 7


def mae(a, f):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(f, float))))


def main():
    ci = pd.read_csv(CLEAN)
    ci["sid10"] = ci.SERIES_ID.str[:10]
    ci = ci.sort_values(["sid10", "DATE_DAY"])
    hist = {s: g.ALL_PURCHASES.to_numpy(float) for s, g in ci.groupby("sid10")}

    s = pd.read_csv(SUMM)
    splits = (s.dropna(subset=["n_train"])
                .groupby(["series", "origin"])[["n_train", "n_val", "n_test"]]
                .first().astype(int))

    rows, bad, checked = [], 0, 0
    for f in sorted(glob.glob(os.path.join(PPDIR, "*_origin*.csv"))):
        m = re.search(r"([^/\\]+)_origin(\d+)\.csv$", f)
        sid, o = m.group(1), int(m.group(2))
        if (sid, o) not in splits.index or sid not in hist:
            continue
        d = pd.read_csv(f)
        nt, nv, ns = splits.loc[(sid, o)]
        y = hist[sid]
        if len(y) < nt + nv + ns:
            continue

        # verify the reconstructed split against the deposited per-period actuals
        truth = d["actual"].to_numpy(float)
        window = y[nt + nv: nt + nv + ns]
        if len(window) != len(truth) or not np.allclose(window, truth):
            bad += 1
            continue
        checked += 1

        train = y[:nt]
        try:
            fc = ThetaModel(train, period=PERIOD, deseasonalize=True).fit().forecast(nv + ns)
            yhat = np.clip(np.asarray(fc, float)[-ns:], 0.0, None)
            if not np.isfinite(yhat).all():
                raise ValueError("non-finite")
        except Exception:
            continue

        nmae = mae(truth, d["naive"].to_numpy(float))
        rows.append(dict(series=sid, origin=o, skill=mae(truth, yhat) / nmae,
                         mape=float(np.mean(np.abs((truth - yhat) / truth)[truth != 0]))))

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "theta_percell.csv", index=False)
    print(f"split verified against deposited actuals: {checked} cells, {bad} mismatches")
    print(f"theta fitted: {len(out)} cells")
    print(out.skill.describe().round(3).to_string())


if __name__ == "__main__":
    main()
