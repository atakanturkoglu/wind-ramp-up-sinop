# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_ortak"))
from agac_onem_gorseli import render_tree_importance  # noqa: E402

HERE = Path(__file__).resolve().parent
render_tree_importance("lightgbm", "LightGBM", HERE)
