#!/usr/bin/env bash
# Regenerates every result and figure in the paper from the deposited forecasts.
# Base-model fitting is not included: forecasts/ holds inputs, not outputs.
set -euo pipefail
cd "$(dirname "$0")"

# Matplotlib stamps a creation date into every PDF it writes, which would make
# otherwise identical figures differ between runs. Pinning this keeps figure
# output deterministic so `git diff figures/` is a meaningful check.
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
cd paper && latexmk -pdf -bibtex -interaction=nonstopmode main.tex && cd ..

echo
echo "Done. Results in results/, figures in figures/, paper in paper/main.pdf"
echo "Verify the run reproduced the published numbers:  git diff --stat results/ figures/"
