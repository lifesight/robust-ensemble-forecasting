"""Canonical locations of every input and output in this repository.

All scripts resolve paths through this module, so the layout is defined in one
place and nothing depends on the directory a script is launched from.
"""

from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent

DATA = ROOT / "data"
FORECASTS = ROOT / "forecasts"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CORPUS = DATA / "corpus_99_series.csv"
MANIFEST = DATA / "candidate_manifest_142_series.csv"
VALIDATION_SUMMARY = DATA / "validation_summary.csv"
SUMMARY_CLEAN = DATA / "summary1-99_clean.csv"

# Intermediates shared between the supp_*.py scripts. Not an artefact.
CACHE = CODE / ".cache"

for _d in (RESULTS, FIGURES, CACHE):
    _d.mkdir(exist_ok=True)
