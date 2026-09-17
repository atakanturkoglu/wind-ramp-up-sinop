# -*- coding: utf-8 -*-
"""
CNN-LSTM'in test olaylari (tek/sabit train-test bolmesinden gelen 9 olay)
icin iki ozet gorsel:
  - test_events_page.png : 9 test olayinin tek sayfada, olay-bazli paneli
  - test_tracking.png    : delta_ws / delta_t icin kronolojik gercek-vs-
                            tahmin cizgi grafigi

Onceki surumde bu iki gorsel sabit 6 olayluk bir test setine dayaniyordu;
simdi model/_ortak/degerlendirme.py:single_split'ten gelen 9 test olayi
kullanilir.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_ortak"))
from olay_gorseli import render_event_pages  # noqa: E402

HERE = Path(__file__).resolve().parent
pred = pd.read_csv(HERE / "predictions.csv", parse_dates=["t0", "t1"])
test = pred[pred["split"] == "test"].sort_values("t0").reset_index(drop=True)
print(f"Test olay sayisi: {len(test)}")

render_event_pages(
    pred_df=test,
    model_label="CNN-LSTM",
    out_dir=HERE,
    note=f"CNN-LSTM — Test Set Predictions ({len(test)} events, single train/test split, never seen during training)",
    single_file="test_events_page.png",
)

TARGETS = {"delta_ws": {"label": "Delta WS (ramp-up amplitude)", "unit": "m/s"},
           "delta_t": {"label": "Delta T (ramp-up duration)", "unit": "min"}}

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
for ax, tgt in zip(axes, TARGETS):
    y_true = test[f"{tgt}_gercek"].to_numpy(dtype=float)
    y_pred = test[f"{tgt}_tahmin_split"].to_numpy(dtype=float)
    x_axis = range(len(test))
    ax.plot(x_axis, y_true, marker="o", color="#2b7bba", label="Actual")
    ax.plot(x_axis, y_pred, marker="s", color="#d62728", label="Predicted")
    ax.set_ylabel(f"{TARGETS[tgt]['label']} ({TARGETS[tgt]['unit']})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
axes[0].set_title(f"CNN-LSTM Test Set — Actual vs Predicted (chronological order, n={len(test)})")
axes[1].set_xticks(list(range(len(test))))
axes[1].set_xticklabels(
    [f"#{int(eid)}\n{pd.Timestamp(t):%Y-%m-%d %H:%M}" for eid, t in zip(test["event_id"], test["t0"])],
    fontsize=8, rotation=30, ha="right")
fig.tight_layout()
fig.savefig(HERE / "test_tracking.png", dpi=140)
plt.close(fig)
print("Kaydedildi -> test_tracking.png")
