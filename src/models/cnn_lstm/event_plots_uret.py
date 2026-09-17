# -*- coding: utf-8 -*-
"""CNN-LSTM icin, tum 43 olayin OOF tahmin gorsellerini uretir."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_ortak"))
from olay_gorseli import render_event_pages  # noqa: E402

HERE = Path(__file__).resolve().parent
render_event_pages(
    predictions_csv=HERE / "predictions.csv",
    model_label="CNN-LSTM",
    out_dir=HERE / "event_plots",
    note="CNN-LSTM — out-of-fold predictions (5-fold x 5 repeats, block-aware CV)",
)
