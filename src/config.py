# -*- coding: utf-8 -*-
"""Shared paths for all scripts in this repo. See ../data/README.md for
what belongs in data/ (raw station data is not distributed with this repo)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
