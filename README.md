# Robust Ensemble Forecasting in Practice

Data, forecasts and analysis code for *Robust Ensemble Forecasting in Practice: Model Selection and Combination Across Heterogeneous Time Series* (Singh, Bhardwaj & Nair).

**Paper:** published as a white paper at [lifesight.io/resources/ensemble-forecasting](https://lifesight.io/resources/ensemble-forecasting/). The manuscript in this repository is [`paper/ensemble_forecasting.pdf`](paper/ensemble_forecasting.pdf), built from `paper/main.tex`.
**Forecasting library:** [Horizon](https://github.com/Lifesight-Software-Pvt-Ltd/horizon), whose pre-release version produced the base-model forecasts (see below).
**Source data:** Anderson (2024), [doi:10.6084/m9.figshare.25314841.v3](https://doi.org/10.6084/m9.figshare.25314841.v3), CC BY 4.0.

---

## What this repository contains

99 daily commercial time series from 65 organisations, the out-of-fold forecasts of five base models across 293 evaluation cells, and everything needed to regenerate every number and figure in the paper.

```
data/        corpus, candidate manifest, and the summary losses the analysis reads
forecasts/   per-cell, per-day forecasts for all 293 evaluated cells (297 files)
results/     derived per-cell metrics, leaderboard, bootstrap and unit-level outputs
code/        analysis and figure scripts
figures/     the ten figures as they appear in the paper (PNG + PDF)
paper/       LaTeX source and the PDF it builds
```

## What can and cannot be reproduced

**Can be reproduced.** Every table, figure and reported statistic in the paper, from the deposited forecasts, without refitting any model.

**Re-derivable from the public source.** The qualified corpus and the candidate manifest follow from the source dataset through `code/qualify.py`; see [Source data](#source-data).

**Not reproducible from released code: the forecasts themselves.** They were produced with a pre-release version of [Horizon](https://github.com/Lifesight-Software-Pvt-Ltd/horizon). Its public history at commit [`73127eb`](https://github.com/Lifesight-Software-Pvt-Ltd/horizon/tree/73127ebc46b87beaad64bcc9282727a7ee48cc41) (28 July 2026) records the model implementations, the hyperparameter grids in `horizon_forecaster/config/default_params.yaml`, the seeds, and the pinned dependency versions in `pyproject.toml`. The per-cell harness that split each series and issued forecasts from the end of its training window is not released. Horizon's first public release, 1.0.0, differs from that version in ways that change the forecasts: it refits on training and validation data before forecasting the test window, sizes the windows differently, and adds a trend term to SARIMAX on undifferenced series. The deposited forecasts are therefore the authoritative inputs; Section 7 of the paper discusses the SARIMAX point.

Validation-*window* forecasts are not retained by the pipeline, only validation-window summary losses (`data/validation_summary.csv`), which is what prevents the validation-MAE analysis listed as future work.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash run_all.sh
```

Or step by step:

```bash
python code/core.py        # error diversity, Section 5.7
python code/theta.py       # Theta benchmark
python code/analysis.py    # leaderboard, tests, tails, units
python code/figures.py     # all figures
for n in 01 02 03 04 05 06 07 08 10; do python "code/figure${n}.py"; done
python code/supp_load.py   # must precede the other supp_* scripts
for f in code/supp_*.py; do [ "$f" = "code/supp_load.py" ] || python "$f"; done
```

Random seeds are fixed: bootstrap intervals use 4,000 series-clustered resamples with `numpy.random.default_rng(0)`, except the arithmetic-mean intervals of Table 1, which use 10,000. The intervals are therefore reproducible rather than merely similar. `run_all.sh` also pins `SOURCE_DATE_EPOCH`, which stops matplotlib stamping a build date into each PDF — one avoidable source of spurious diffs, though not the only one.

### Verifying a run

```bash
bash run_all.sh          # regenerate first
python code/verify.py    # then compare
```

This compares every numeric column in `results/` against the committed copy (`git show HEAD:`) and reports the largest deviation per file. **Run the pipeline first.** On an untouched clone the working tree *is* the committed copy, so `verify.py` compares the deposit against itself and reports success without having checked anything.

**Do not expect byte-identical output, and do not use `git diff` as the test.** The analysis performs floating-point reductions whose final digits depend on the BLAS implementation and library versions, so a fresh run on a different machine rewrites the result files and every figure without changing anything that matters. Reproducing this deposit on macOS with Apple Accelerate, against the Linux/OpenBLAS reference the committed files came from, moves values by at most **1.1e-9** on quantities of order 1 — nine significant figures of agreement, against a paper that reports three. `verify.py` uses a 1e-6 tolerance, comfortably above that noise and comfortably below anything the paper claims.

Figures re-render differently across matplotlib versions for the same reason. Compare them by eye, not by checksum.

Two files in `results/` are deposited artefacts rather than pipeline outputs, because they are emitted during base-model fitting: `cell_split_boundaries.csv` and `percell_validation_and_test_metrics.csv`. `verify.py` skips them.

## Code

| Script | Produces |
|---|---|
| `paths.py` | Canonical locations of every input and output |
| `verify.py` | Checks a fresh run against the deposit within tolerance |
| `qualify.py` | Derives the corpus and candidate manifest from the source dataset (not run by `run_all.sh`; needs the source file) |
| `core.py` | Cell loading, non-negativity, family de-duplication, the combination rules and the selection baseline. Imported by the scripts below; run directly it writes the Section 5.7 error-correlation matrices |
| `theta.py` | Theta benchmark per cell |
| `analysis.py` | Tables 1, 2, 4 and 6; the leaderboard, tail bootstrap and unit-of-aggregation files |
| `figures.py` | All figures; Figure 9 is final as produced here |
| `figure01.py` … `figure10.py` | The final version of each remaining figure, overwriting the draft from `figures.py`. There is no `figure09.py` |
| `supp_load.py` | Shared data load; caches the frames the other `supp_*` scripts read. Run first |
| `supp_checks.py` | Cell counts and constant-rule setup (Section 5.3) |
| `supp_constant_rules.py` | Table 3 |
| `supp_ablation.py` | Table 5 and the tail differences of Section 5.5 |
| `supp_horizon.py` | Horizon against training length (Section 5.6) |
| `supp_pooling.py` | Table 7 |
| `supp_diversity.py` | Error diversity and trimmed-slot shares (Section 5.7) |
| `supp_terciles.py` | Tercile breakdown behind Table 4 |
| `supp_tau.py` | Sensitivity of the screened rule to the competence floor tau (Caveats) |
| `supp_stability.py` | Reproducibility of the test ranking across origins (Section 5.3); membership vs operator in the ablation (Section 5.8) |

The `supp_*` scripts print to stdout; only `supp_tau.py` also writes a file.

## Source data

The corpus is derived from version 3 of a public dataset, released under CC BY 4.0:

> Anderson, Anthony (2024). *Multi-Region Marketing Mix Modelling (MMM) Dataset for Several eCommerce Brands*. figshare. Dataset, version 3. https://doi.org/10.6084/m9.figshare.25314841.v3

The source file, `conjura_mmm_data.csv` (about 37 MB), is not redistributed here. To re-derive the corpus, download it from the DOI above and run:

```bash
python code/qualify.py --raw path/to/conjura_mmm_data.csv
```

The script removes the one series with a corrupt purchase spike, applies the screen of Section 3.2 of the paper, and compares its output with the deposit. Against the published file (md5 `84fc8f323644b2a5fc21d0cd1a4af985`) the candidate manifest reproduces byte for byte, and identifiers, dates, labels and the target reproduce exactly. The spend totals agree to about nine significant figures (relative difference at most 5e-10) rather than digit for digit, because the deposit was built from a copy of version 3 whose currency columns had been re-exported at slightly reduced precision. Outputs go to `code/.cache/` and never overwrite the deposit.

## Data dictionary

### `data/corpus_99_series.csv` — 106,924 rows

| Column | Description |
|---|---|
| `SERIES_ID` | Organisation–territory identifier, as published anonymised in the source (`MMM_TIMESERIES_ID`). 99 distinct values. |
| `ORGANISATION_ID` | Organisation identifier, as published anonymised in the source. 65 distinct values. |
| `ORGANISATION_VERTICAL` | Industry vertical. 13 named values plus `Unclassified` (8 series). |
| `ORGANISATION_SUBVERTICAL` | Subvertical. 30 named values plus `Unclassified`. |
| `DATE_DAY` | Calendar date, `YYYY-MM-DD`. Range 2019-07-21 to 2024-06-02. |
| `ALL_PURCHASES` | Forecast target: daily total purchases. Strictly positive by construction. |
| `GOOGLE_SPEND_TOTAL`, `META_SPEND_TOTAL`, `TIKTOK_SPEND` | Daily paid-media spend covariates, known in advance at forecast time. |

### `data/candidate_manifest_142_series.csv`

One row per candidate series. `qualifies` is the screening outcome; `excluded_reason` records `short(Nd)` and/or `low_cov(x)`. Of 142 candidates, 99 qualified: 33 excluded on history length alone, 6 on media-spend coverage alone, 4 on both. The corrupt-spike organisation described in Section 3.2 is removed *before* candidates are recorded, by the spike rule in `code/qualify.py`, and so does not appear here.

### `data/validation_summary.csv`

Per cell per model, the validation and test losses emitted during base-model fitting. `val_mape` is the criterion every selection, weighting, screening and de-duplication rule uses. `code/core.py` reads it, and it carries all five validation scores for every evaluated cell.

### `data/summary1-99_clean.csv`

A cleaned copy of the same run summary, read by `code/theta.py`, `code/analysis.py` (for window lengths) and `code/supp_load.py`. It lacks one row that `validation_summary.csv` has: Prophet for series `caf6f0773a` at origin 0. The main pipeline is unaffected and evaluates all 293 cells; the supplementary analyses built on `supp_load.py` use 292, which moves the matched-loss hit rate of Section 5.3 from 25.9% to 26.0%. `code/supp_checks.py` reports the count.

### `forecasts/{series}_origin{k}.csv`

One file per cell (series × rolling origin), 297 files, of which 293 have complete forecasts from all five base models and are the evaluated set. Columns: `ds`, `actual`, `naive`, and `{model}__yhat` for each of `prophet`, `catboost`, `sarimax`, `neuralprophet`, `bayesianpymc`, plus the precomputed `ens_mean__yhat`, `ens_wtd__yhat`, `trimmed_mean__yhat`.

Forecasts are stored **before** the non-negativity constraint of Section 4.2. Apply `max(yhat, 0)` before evaluation to match the paper; this affects 837 of 189,420 member-observations, all of them Prophet's.

### `results/`

| File | Contents |
|---|---|
| `percell_validation_and_test_metrics.csv` | Per cell per model: `validation_mape`, `test_mape`, `test_relmae`, and window lengths. |
| `cell_split_boundaries.csv` | Train / validation / test date boundaries for every cell. |
| `percell_test_metrics.csv` | Per cell per method: test RelMAE (`skill`), MAPE, regret, oracle. All twelve methods. |
| `leaderboard.csv` | Mean, geometric mean, median, P90/P95/P99, max, beats-naive rate, regret, average rank, bootstrap CI. |
| `tail_bootstrap.csv` | Series-clustered bootstrap on tail differences against the reference ensemble. |
| `unit_of_aggregation.csv` | Headline comparisons at cell, series and organisation level. |
| `error_correlation_signed.csv` | Pairwise Pearson correlation of signed per-period forecast error (Section 5.7). |
| `error_correlation_magnitude.csv` | The same for error magnitudes, reported for contrast only. |
| `theta_percell.csv` | Theta benchmark per cell. |
| `tau_sensitivity.csv` | Screened-rule accuracy and fallback rate across the competence floor tau. |
| `analysis_console_output.txt` | Full console output of the analysis run. |

## Table and figure map

| Paper item | Produced by | Reads |
|---|---|---|
| Table 1 (central tendency) | `code/analysis.py` | `results/leaderboard.csv` |
| Table 2 (tail risk and regret) | `code/analysis.py` | `results/leaderboard.csv` |
| Table 3 (constant rules) | `code/supp_constant_rules.py` | `results/percell_test_metrics.csv` |
| Table 4 (horizon terciles) | `code/analysis.py` | `results/percell_test_metrics.csv` |
| Table 5 (de-duplication ablation) | `code/supp_ablation.py` | `forecasts/` |
| Table 6 (unit of aggregation) | `code/analysis.py` | `results/unit_of_aggregation.csv` |
| Table 7 (pooling gradient) | `code/supp_pooling.py` | `results/percell_validation_and_test_metrics.csv`, `cell_split_boundaries.csv` |
| Figure 1 (winner rotation) | `code/figure01.py` | `results/percell_test_metrics.csv` |
| Figure 2 (selection granularity) | `code/figure02.py` | `results/percell_validation_and_test_metrics.csv` |
| Figure 3 (ensemble vs per-series) | `code/figure03.py` | `results/percell_test_metrics.csv` |
| Figure 4 (critical difference) | `code/figure04.py` | `results/percell_test_metrics.csv` |
| Figure 5 (tail bootstrap) | `code/figure05.py` | `results/tail_bootstrap.csv` |
| Figure 6 (rank distribution) | `code/figure06.py` | `results/percell_test_metrics.csv` |
| Figure 7 (RelMAE distribution) | `code/figure07.py` | `results/percell_test_metrics.csv` |
| Figure 8 (accuracy–tail trade-off) | `code/figure08.py` | `results/leaderboard.csv` |
| Figure 9 (error correlation) | `code/figures.py` | `results/error_correlation_signed.csv` |
| Figure 10 (by vertical) | `code/figure10.py` | `results/percell_test_metrics.csv` |

## Licence

The manuscript, figures and code are released under **CC BY 4.0**, with the code in `code/` additionally available under the **MIT License** at your option. See `LICENSE`.

The derived data in `data/`, `forecasts/` and `results/` is released under **CC BY 4.0**. See `LICENSE-DATA`. It derives from a CC BY 4.0 source, so reuse should credit both this repository and Anderson (2024).

GitHub's sidebar may not identify the licence, because `LICENSE` combines CC BY 4.0 with an MIT option; the file itself is authoritative.

## Privacy

No personal data is used. Organisation and series identifiers are the anonymised identifiers published with the source dataset, no organisation is named, and the source's territory names are not carried into the deposited data. The corpus is a screened subset of that publicly available dataset, which the authors neither collected nor own.

## Citation

See `CITATION.cff`. Please cite the paper and, if you use the data, the source dataset:

> Anderson, Anthony (2024). *Multi-Region Marketing Mix Modelling (MMM) Dataset for Several eCommerce Brands*. figshare. Dataset, version 3. https://doi.org/10.6084/m9.figshare.25314841.v3
