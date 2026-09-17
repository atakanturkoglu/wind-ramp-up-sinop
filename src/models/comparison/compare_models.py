# -*- coding: utf-8 -*-
"""
Alti modelin (CNN-LSTM, LSTM, GRU, XGBoost, CatBoost, LightGBM) ortak
cerceveden gelen sonuclarini tek tabloda ve tek gorselde toplar.

Tum modeller BIREBIR ayni bolmelerle (blok-farkinda 5 kat x 5 tekrar)
degerlendirildigi icin karsilastirma adildir.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

MODEL_DIR = Path(PROJECT_ROOT, 'src', 'models')
OUT_DIR = MODEL_DIR / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "CNN-LSTM": MODEL_DIR / "cnn_lstm" / "metrics.csv",
    "LSTM": MODEL_DIR / "lstm" / "metrics.csv",
    "GRU": MODEL_DIR / "gru" / "metrics.csv",
    "XGBoost": MODEL_DIR / "xgboost" / "metrics.csv",
    "CatBoost": MODEL_DIR / "catboost" / "metrics.csv",
    "LightGBM": MODEL_DIR / "lightgbm" / "metrics.csv",
}

frames = []
baseline = None
for name, path in SOURCES.items():
    if not path.exists():
        print(f"  atlandi (dosya yok): {path}")
        continue
    d = pd.read_csv(path)
    frames.append(d[d["model"] == name])
    if baseline is None:
        baseline = d[d["model"] == "Temel cizgi (ortalama)"].copy()

table = pd.concat(frames + [baseline], ignore_index=True)
table = table.sort_values(["target", "R2_ort"], ascending=[True, False])
table.to_csv(OUT_DIR / "comparison_table.csv", index=False)

print(table[["target", "model", "R2_ort", "R2_std", "RMSE_ort", "MAE_ort"]].to_string(index=False))

# --- gorseller ---
targets = ["delta_ws", "delta_t"]
labels = {"delta_ws": "Delta WS (ramp-up amplitude)", "delta_t": "Delta T (ramp-up duration)"}
models = [m for m in SOURCES if any(table["model"] == m)]
colors = {"CNN-LSTM": "#1f77b4", "LSTM": "#17becf", "GRU": "#8c564b",
          "XGBoost": "#2ca02c", "CatBoost": "#ff7f0e", "LightGBM": "#9467bd"}


def bar_metric(metric_col, err_col, ylabel, title_suffix, out_name, baseline_is_zero):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, tgt in zip(axes, targets):
        sub = table[table["target"] == tgt]
        xs = np.arange(len(models))
        vals = [float(sub[sub["model"] == m][metric_col].iloc[0]) for m in models]
        errs = [float(sub[sub["model"] == m][err_col].iloc[0]) for m in models]
        ax.bar(xs, vals, yerr=errs, capsize=4,
               color=[colors.get(m, "#777") for m in models], alpha=0.85,
               error_kw=dict(elinewidth=1.2, capthick=1.2))
        base_val = float(sub[sub["model"] == "Temel cizgi (ortalama)"][metric_col].iloc[0])
        ax.axhline(base_val, color="#d62728", ls="--", lw=1.4,
                   label="Baseline (predict mean)")
        ax.set_xticks(xs)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(labels[tgt], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Etiketler hata cubugunun UCUNUN disina, cubuktan acikca ayri
        # bir bosluk birakilarak yerlestirilir (once cubuk govdesinin
        # hemen ustunde/altinda idi, hata cubugunun dikey cizgisiyle
        # cakisip rakamlari okunmaz hale getiriyordu).
        span = (max(v + e for v, e in zip(vals, errs))
                - min(v - e for v, e in zip(vals, errs)))
        pad = 0.05 * span if span > 0 else 0.02
        for x, v, e in zip(xs, vals, errs):
            va = "top" if (baseline_is_zero and v < 0) else "bottom"
            y_text = (v - e - pad) if va == "top" else (v + e + pad)
            ax.text(x, y_text, f"{v:+.3f}" if baseline_is_zero else f"{v:.3f}",
                    ha="center", va=va, fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.8))
        y_lo, y_hi = ax.get_ylim()
        pad_frac = 0.10
        if baseline_is_zero:
            ax.set_ylim(y_lo - pad_frac * (y_hi - y_lo), y_hi)
        else:
            ax.set_ylim(y_lo, y_hi + pad_frac * (y_hi - y_lo))
    fig.suptitle(f"Model comparison — block-aware 5-fold x 5 repeats CV (43 events)\n{title_suffix}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=140)
    plt.close(fig)


bar_metric("R2_ort", "R2_std", "Out-of-fold R²",
           "higher is better — dashed line (R² = 0) is the mean-prediction baseline",
           "comparison_r2.png", baseline_is_zero=True)
bar_metric("RMSE_ort", "RMSE_std", "Out-of-fold RMSE",
           "lower is better — dashed line is the mean-prediction baseline",
           "comparison_rmse.png", baseline_is_zero=False)
bar_metric("MAE_ort", "MAE_std", "Out-of-fold MAE",
           "lower is better — dashed line is the mean-prediction baseline",
           "comparison_mae.png", baseline_is_zero=False)

print(f"\nKaydedildi -> {OUT_DIR}")
