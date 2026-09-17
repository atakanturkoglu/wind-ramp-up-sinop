# -*- coding: utf-8 -*-
"""
LSTM/GRU/XGBoost/CatBoost/LightGBM icin oof_scatter.png'yi, modeli tekrar
egitmeden (predictions.csv ve metrics.csv zaten yeterli oldugu icin)
degerlendirme.py:scatter_png ile BIREBIR ayni formatta yeniden cizer.
Tek amac: TARGETS etiketindeki "ramp amplitude/duration" -> "ramp-up
amplitude/duration" duzeltmesini, sayisal degerlere DOKUNMADAN, mevcut
PNG'lere yansitmak.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

TARGETS = {
    "delta_ws": {"label": "Delta WS (ramp-up amplitude)", "unit": "m/s"},
    "delta_t": {"label": "Delta T (ramp-up duration)", "unit": "min"},
}


def replot(model_dir, model_name):
    model_dir = Path(model_dir)
    pred = pd.read_csv(model_dir / "predictions.csv")
    metrics = pd.read_csv(model_dir / "metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ti, (tname, info) in enumerate(TARGETS.items()):
        ax = axes[ti]
        true = pred[f"{tname}_gercek"].to_numpy()
        pm = pred[f"{tname}_tahmin"].to_numpy()
        ps = pred[f"{tname}_tahmin_std"].to_numpy()
        lo = min(true.min(), pm.min()) - 0.5
        hi = max(true.max(), pm.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
        ax.axhline(true.mean(), color="#d62728", ls=":", lw=1.2, label="Baseline (mean)")
        ax.errorbar(true, pm, yerr=ps, fmt="o", ms=6, color="#1f77b4",
                    ecolor="#9ecae1", elinewidth=1, capsize=2, alpha=0.9)
        r = metrics[(metrics["target"] == tname) & (metrics["model"] == model_name)].iloc[0]
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Actual ({info['unit']})")
        ax.set_ylabel(f"Predicted ({info['unit']})")
        ax.set_title(f"{info['label']}\nR2 = {r['R2_ort']:+.3f} ± {r['R2_std']:.3f}   "
                     f"RMSE = {r['RMSE_ort']:.2f}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{model_name} — Out-of-fold predictions (5-fold x 5 repeats, block-aware)",
                 fontsize=12)
    fig.tight_layout()
    out_path = model_dir / "oof_scatter.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"{model_name}: {out_path}")


if __name__ == "__main__":
    MODEL_DIR = Path(PROJECT_ROOT, 'src', 'models')
    for key, label in [("lstm", "LSTM"), ("gru", "GRU"), ("xgboost", "XGBoost"),
                        ("catboost", "CatBoost"), ("lightgbm", "LightGBM")]:
        replot(MODEL_DIR / key, label)
