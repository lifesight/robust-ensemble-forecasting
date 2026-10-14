"""Qualification screen: from the public source dataset to the deposited corpus.

Reproduces data/candidate_manifest_142_series.csv and data/corpus_99_series.csv
from the source dataset described in Section 3.1 of the paper:

    Anderson, Anthony (2024). Multi-Region Marketing Mix Modelling (MMM) Dataset
    for Several eCommerce Brands. figshare. Dataset, version 3.
    https://doi.org/10.6084/m9.figshare.25314841.v3

The source file (conjura_mmm_data.csv, about 37 MB) is not redistributed here.
Download it from the DOI above and pass its path with --raw.

Steps, in the order Section 3.2 describes them:

  1. Key each series on MMM_TIMESERIES_ID (organisation x territory). This
     dissolves the apparent (organisation, date) duplicates, which are multiple
     territories of one organisation.
  2. Remove data-quality failures before qualification: any series whose largest
     daily purchase count exceeds 1,000 times its median. On this source that is
     exactly one series, belonging to an organisation with no other series, and
     it is therefore absent from the candidate manifest.
  3. Sum the per-channel spend columns into three covariates
     (GOOGLE_SPEND_TOTAL, META_SPEND_TOTAL, TIKTOK_SPEND), treating a missing
     channel entry as zero spend.
  4. Carry the industry labels through, normalising missing, "Unknown" and
     "Other" to "Unclassified".
  5. Qualify each candidate: at least 600 days of history, no calendar gaps, no
     zero-target days, and paid-media spend present on at least 90% of days.
  6. Write the candidate manifest and the qualified corpus, then compare them
     against the deposited copies.

Usage:
    python code/qualify.py --raw path/to/conjura_mmm_data.csv

Outputs go to code/.cache/qualify/ by default, so a run never overwrites the
deposit. The comparison treats the manifest as exact, and the corpus as exact on
identifiers, dates, labels and the target.

The spend totals are compared to a relative tolerance of 1e-8 rather than
exactly. The deposited corpus was built from a copy of version 3 of the source
in which the currency columns had been re-exported at slightly reduced
precision: rows, columns, keys, missing values and the target are identical to
the published file, but spend values differ from it by at most about 5e-10 in
relative terms, i.e. they agree to nine significant figures. Against the
published file the spend totals therefore match to that precision rather than
digit for digit. Floating-point summation of the channel columns contributes
differences of the same order.
"""

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CACHE, CORPUS, MANIFEST

import argparse
import hashlib

import numpy as np
import pandas as pd

KEY = "MMM_TIMESERIES_ID"
DATE = "DATE_DAY"
TARGET = "ALL_PURCHASES"
VERT = "ORGANISATION_VERTICAL"
SUBVERT = "ORGANISATION_SUBVERTICAL"
COVS = ["GOOGLE_SPEND_TOTAL", "META_SPEND_TOTAL", "TIKTOK_SPEND"]
UNCLASSIFIED = {"unknown", "other", "", "nan", "none"}

# Published checksum of conjura_mmm_data.csv on figshare (file id 46779652).
SOURCE_MD5 = "84fc8f323644b2a5fc21d0cd1a4af985"

SPIKE_RATIO = 1000       # step 2: largest day vs median
MIN_DAYS = 600           # step 5
MAX_GAPS = 0
MAX_ZERO_FRAC = 0.0
MIN_COV_COVER = 0.90

REL_TOL = 1e-8          # spend totals, relative to max(|value|, 1)


def _label(series):
    """First non-null label for a series, normalised; missing/Unknown/Other -> Unclassified."""
    vals = series.dropna()
    if len(vals):
        v = str(vals.iloc[0]).strip()
        if v.lower() not in UNCLASSIFIED:
            return v
    return "Unclassified"


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def build(raw_path):
    df = pd.read_csv(raw_path, low_memory=False)
    df[DATE] = pd.to_datetime(df[DATE])

    # 2. data-quality removal, before any candidate is recorded
    ratio = df.groupby(KEY)[TARGET].agg(lambda y: y.max() / max(y.median(), 1e-9))
    removed = sorted(ratio[ratio > SPIKE_RATIO].index)
    for sid in removed:
        print(f"removed before qualification: {sid}  (max / median = {ratio[sid]:,.0f})")
    df = df[~df[KEY].isin(removed)]

    # 3. spend covariates
    g = [c for c in df.columns if c.startswith("GOOGLE_") and c.endswith("_SPEND")]
    m = [c for c in df.columns if c.startswith("META_") and c.endswith("_SPEND")]
    df["GOOGLE_SPEND_TOTAL"] = df[g].fillna(0).sum(axis=1)
    df["META_SPEND_TOTAL"] = df[m].fillna(0).sum(axis=1)
    df["TIKTOK_SPEND"] = df["TIKTOK_SPEND"].fillna(0)

    keep_rows, manifest = [], []
    for sid, d in df.groupby(KEY):
        d = d.sort_values(DATE)
        if d.duplicated([DATE]).any():
            d = d.groupby(DATE, as_index=False).first()
        dates = d[DATE]
        ndays = dates.nunique()
        gaps = len(pd.date_range(dates.min(), dates.max(), freq="D")) - ndays
        zero_frac = float((d[TARGET] == 0).mean())
        cov_cover = float((d[COVS].sum(axis=1) > 0).mean())
        vertical = _label(d[VERT]) if VERT in d else "Unclassified"
        subvertical = _label(d[SUBVERT]) if SUBVERT in d else "Unclassified"

        # 5. qualification
        fails = []
        if ndays < MIN_DAYS: fails.append(f"short({ndays}d)")
        if gaps > MAX_GAPS: fails.append(f"gaps({gaps})")
        if zero_frac > MAX_ZERO_FRAC: fails.append(f"zeros({zero_frac:.2f})")
        if cov_cover < MIN_COV_COVER: fails.append(f"low_cov({cov_cover:.2f})")
        qualifies = not fails

        manifest.append(dict(series_id=sid, org=d["ORGANISATION_ID"].iloc[0],
                             vertical=vertical, subvertical=subvertical,
                             ndays=ndays, gaps=gaps, zero_frac=round(zero_frac, 3),
                             cov_cover=round(cov_cover, 3),
                             qualifies=qualifies, excluded_reason=",".join(fails)))
        if qualifies:
            out = d[[DATE, TARGET] + COVS].copy()
            out.insert(0, SUBVERT, subvertical)
            out.insert(0, VERT, vertical)
            out.insert(0, "ORGANISATION_ID", d["ORGANISATION_ID"].iloc[0])
            out.insert(0, "SERIES_ID", sid)
            keep_rows.append(out)

    corpus = pd.concat(keep_rows, ignore_index=True)
    corpus[DATE] = corpus[DATE].dt.strftime("%Y-%m-%d")
    man = pd.DataFrame(manifest).sort_values(["qualifies", "ndays"], ascending=[False, False])
    return man, corpus


def compare(man, corpus):
    """Compare against the deposit. Returns True if it is reproduced."""
    ok = True
    with open(MANIFEST, newline="") as f:
        m_same = man.to_csv(index=False) == f.read()
    print(f"\nmanifest  {len(man)} candidates, {int(man.qualifies.sum())} qualify   "
          f"byte-identical to deposit: {m_same}")
    ok &= m_same

    dc = pd.read_csv(CORPUS)
    keys = ["SERIES_ID", DATE]
    a = corpus.sort_values(keys).reset_index(drop=True)
    b = dc.sort_values(keys).reset_index(drop=True)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        print(f"corpus    shape {a.shape} vs deposit {b.shape}: DIFFERS")
        return False
    exact = ["SERIES_ID", "ORGANISATION_ID", VERT, SUBVERT, DATE, TARGET]
    e_same = a[exact].astype(str).equals(b[exact].astype(str))
    x, y = a[COVS].to_numpy(float), b[COVS].to_numpy(float)
    dev = float(np.max(np.abs(x - y)))
    rel = float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1.0)))
    print(f"corpus    {a.SERIES_ID.nunique()} series, {len(a):,} rows   identifiers, dates, "
          f"labels and target identical: {e_same}\n"
          f"          spend totals: max |diff| {dev:.1e}, max relative {rel:.1e} "
          f"(tolerance {REL_TOL:.0e})")
    ok &= e_same and rel <= REL_TOL
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="path to conjura_mmm_data.csv")
    ap.add_argument("--out-dir", default=str(CACHE / "qualify"))
    a = ap.parse_args()

    digest = md5(a.raw)
    if digest != SOURCE_MD5:
        print(f"note: {a.raw} has md5 {digest}, not the published {SOURCE_MD5}.\n"
              f"      The comparison below still says whether it reproduces the deposit.")

    man, corpus = build(a.raw)
    out = _os.path.abspath(a.out_dir); _os.makedirs(out, exist_ok=True)
    man.to_csv(_os.path.join(out, "candidate_manifest.csv"), index=False)
    corpus.to_csv(_os.path.join(out, "corpus.csv"), index=False)
    print(f"wrote {out}/candidate_manifest.csv and corpus.csv")

    if compare(man, corpus):
        print("\nReproduced: the deposited manifest and corpus follow from the source.")
        return 0
    print("\nNOT reproduced. Check that --raw is version 3 of the source dataset.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
