#!/usr/bin/env bash
# Regenerates every result and figure in the paper from the deposited forecasts.
# Base-model fitting is not included: forecasts/ holds inputs, not outputs.
set -euo pipefail
cd "$(dirname "$0")"

# Matplotlib stamps a creation date into every PDF it writes. Pinning it removes
# one avoidable source of differences between runs. To check a run, use
# code/verify.py rather than git diff: see "Verifying a run" in the README.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1735689600}"

echo "==> Error diversity (Section 5.7)"
python code/core.py

echo "==> Theta benchmark"
python code/theta.py

echo "==> Leaderboard, significance tests, tail bootstraps, unit-of-aggregation checks"
python code/analysis.py | tee results/analysis_console_output.txt

echo "==> Figures"
python code/figures.py
for n in 01 02 03 04 05 06 07 08 10; do python "code/figure${n}.py"; done

echo "==> Supplementary analyses"
python code/supp_load.py
for f in code/supp_*.py; do [ "$f" = "code/supp_load.py" ] || python "$f"; done

echo "==> Paper"
if command -v latexmk >/dev/null 2>&1; then
  (cd paper && latexmk -pdf -bibtex -interaction=nonstopmode main.tex)
  PAPER="paper/main.pdf"
else
  echo "latexmk not found: skipping the paper build. Everything above is complete;"
  echo "install a LaTeX distribution (e.g. TeX Live or MacTeX) to build the PDF too."
  PAPER="not built (no LaTeX); the deposited copy is paper/ensemble_forecasting.pdf"
fi

echo
echo "Done. Results in results/, figures in figures/, paper: ${PAPER}"
echo "Verify the run reproduced the published numbers:  python code/verify.py"
