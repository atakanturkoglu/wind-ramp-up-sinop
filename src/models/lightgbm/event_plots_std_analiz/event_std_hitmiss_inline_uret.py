# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_ortak"))
from hitmiss_gorseli import render_hitmiss  # noqa: E402

HERE = Path(__file__).resolve().parent
render_hitmiss(
    pred_csv=HERE.parent / "predictions.csv",
    model_label="LightGBM",
    out_path=HERE / "test_events_hit_miss_inline.png",
)
