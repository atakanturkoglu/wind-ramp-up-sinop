# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_ortak"))
from olay_gorseli import render_event_pages  # noqa: E402

HERE = Path(__file__).resolve().parent
render_event_pages(
    predictions_csv=HERE / "predictions.csv",
    model_label="GRU",
    out_dir=HERE / "event_plots",
    note="GRU — tum 43 olay (34 train / 9 test, tek sabit bolme)",
)
