# -*- coding: utf-8 -*-
"""
oof_scatter.png'yi, train_cnn_lstm.py'yi tekrar calistirmadan (predictions.csv
ve metrics.csv zaten yeterli oldugu icin), baslikta MAE de gorunecek
sekilde yeniden cizer. Sayisal degerlere DOKUNULMAZ, sadece basliga
MAE eklenir (kullanici R^2, RMSE ve MAE'nin BIRLIKTE gorunmesini istedi).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
pred = pd.read_csv(HERE / "predictions.csv")
metrics = pd.read_csv(HERE / "metrics.csv")

TARGETS = {"delta_ws": {"label": "Delta WS (ramp-up amplitude)", "unit": "m/s"},
           "delta_t": {"label": "Delta T (ramp-up duration)", "unit": "min"}}

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ti, (tname, info) in enumerate(TARGETS.items()):
    ax = axes[ti]
    true = pred[f"{tname}_gercek"].to_numpy()
    p_mean = pred[f"{tname}_tahmin"].to_numpy()
    p_std = pred[f"{tname}_tahmin_std"].to_numpy()
    lo = min(true.min(), p_mean.min()) - 0.5
    hi = max(true.max(), p_mean.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.axhline(true.mean(), color="#d62728", ls=":", lw=1.2,
               label="Baseline (train mean)")
    ax.errorbar(true, p_mean, yerr=p_std, fmt="o", ms=6, color="#1f77b4",
                ecolor="#9ecae1", elinewidth=1, capsize=2, alpha=0.9)
    row = metrics[(metrics["target"] == tname) & (metrics["model"] == "CNN-LSTM")].iloc[0]
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Actual ({info['unit']})")
    ax.set_ylabel(f"Predicted ({info['unit']})")
    ax.set_title(f"{info['label']}\n"
                 f"R2 = {row['R2_ort']:+.3f} ± {row['R2_std']:.3f}   "
                 f"RMSE = {row['RMSE_ort']:.2f}   MAE = {row['MAE_ort']:.2f}",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("CNN-LSTM — Out-of-fold predictions (5-fold x 5 repeats, block-aware)",
             fontsize=12)
fig.tight_layout()
fig.savefig(HERE / "oof_scatter.png", dpi=140)
plt.close(fig)
print("Kaydedildi ->", HERE / "oof_scatter.png")
