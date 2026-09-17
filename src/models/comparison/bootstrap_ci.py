# -*- coding: utf-8 -*-
"""
Test seti R^2 degerlerinin ne kadar guvenilir oldugunu gostermek icin
non-parametrik (percentile) bootstrap guven araligi hesaplar.

Yontem: Efron & Tibshirani (1993), "An Introduction to the Bootstrap"
- klasik, standart percentile bootstrap: gozlemlenen (gercek, tahmin)
ciftleri yerine koyarak (with replacement) yeniden orneklenir, her
yeniden orneklemede R^2 hesaplanir, B=10.000 tekrarin %2,5 ve %97,5
persentilleri guven araligini verir.

Amac: test setinin yalnizca 9 olaydan olustugunu (comparison_table.csv'de
raporlanan tek nokta R^2 tahmininin) gizlememek, aksine bu tahminin ne
kadar genis bir belirsizlik tasidigini nicel olarak gostermek.

GUNCELLEME: Onceki surum 6 model (LSTM-32/LSTM-8 dahil, tek sabit 6
olayluk test seti) uzerine kuruluydu. Simdi 5 model (CNN-LSTM, LSTM,
XGBoost, CatBoost, LightGBM), 43 olayin tek/sabit train-test bolmesinden
gelen 9 test olayi kullaniliyor (model/_ortak/degerlendirme.py:
single_split). Yontem ve gorsel tasarim degismedi.

Girdi: model/<model>/predictions.csv ("split" == "test" satirlari)
Cikti: model/comparison/bootstrap_r2_test.csv, bootstrap_r2_test.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

MODEL_DIR = Path(PROJECT_ROOT, 'src', 'models')
OUT_DIR = MODEL_DIR / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("cnn_lstm", "CNN-LSTM"),
    ("lstm", "LSTM"),
    ("gru", "GRU"),
    ("xgboost", "XGBoost"),
    ("catboost", "CatBoost"),
    ("lightgbm", "LightGBM"),
]
TARGETS = ["delta_ws", "delta_t"]
TARGET_LABELS = {"delta_ws": "Delta WS (ramp-up amplitude)", "delta_t": "Delta T (ramp-up duration)"}

B = 10000
SEED = 42
rng = np.random.default_rng(SEED)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


rows = []
boot_store = {}  # (model_key, target) -> point_r2, ci_low, ci_high

for folder, label in MODELS:
    path = MODEL_DIR / folder / "predictions.csv"
    if not path.exists():
        print(f"  UYARI: {path} bulunamadi, atlaniyor.")
        continue
    df = pd.read_csv(path)
    test = df[df["split"] == "test"].reset_index(drop=True)
    n = len(test)

    for tgt in TARGETS:
        y_true = test[f"{tgt}_gercek"].to_numpy(dtype=float)
        y_pred = test[f"{tgt}_tahmin_split"].to_numpy(dtype=float)
        point_r2 = r2_score(y_true, y_pred)

        boots = []
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            val = r2_score(y_true[idx], y_pred[idx])
            if np.isfinite(val):
                boots.append(val)
        boots = np.array(boots)

        lo, med, hi = np.percentile(boots, [2.5, 50, 97.5])
        crosses_zero = (lo < 0.0) and (hi > 0.0)

        rows.append({
            "model": label,
            "model_key": folder,
            "target": tgt,
            "n_test": n,
            "point_R2": point_r2,
            "boot_median_R2": med,
            "ci_low_2.5pct": lo,
            "ci_high_97.5pct": hi,
            "ci_crosses_zero": crosses_zero,
            "n_boot_valid": len(boots),
            "n_boot_total": B,
        })
        boot_store[(folder, tgt)] = point_r2, lo, hi

result = pd.DataFrame(rows)
result.to_csv(OUT_DIR / "bootstrap_r2_test.csv", index=False)
print("bootstrap_r2_test.csv yazildi.")
print(result.to_string(index=False))

# ---------------------------------------------------------------------------
# Gorsel: model basina nokta tahmini + %95 bootstrap guven araligi
# ---------------------------------------------------------------------------
model_labels = [label for folder, label in MODELS if (MODEL_DIR / folder / "predictions.csv").exists()]
model_keys = [folder for folder, label in MODELS if (MODEL_DIR / folder / "predictions.csv").exists()]
x = np.arange(len(model_keys))
n_test_actual = rows[0]["n_test"] if rows else 0

# Model-kimligi renkleri, comparison_r2/mae/rmse.png ile BIREBIR ayni
# palet - okuyucu bir modeli tum gorseller boyunca ayni renkten takip
# edebilsin diye.
MODEL_COLORS = {"CNN-LSTM": "#1f77b4", "LSTM": "#17becf", "GRU": "#8c564b",
                 "XGBoost": "#2ca02c", "CatBoost": "#ff7f0e", "LightGBM": "#9467bd"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

Y_MIN = -3.2  # gorunur eksen alt siniri; bunun altina inen G.A.'lar kirpilip
              # gercek deger metin olarak yazilir (asil ilgi alani olan
              # sifir civarini sikistirmamak icin)

for ax, tgt in zip(axes, TARGETS):
    points = np.array([boot_store[(k, tgt)][0] for k in model_keys])
    los = np.array([boot_store[(k, tgt)][1] for k in model_keys])
    his = np.array([boot_store[(k, tgt)][2] for k in model_keys])

    los_clipped = np.maximum(los, Y_MIN)
    yerr_low = points - los_clipped
    yerr_high = his - points

    # Nokta rengi ISTATISTIKSEL anlami tasir (kirmizi = G.A. sifiri
    # iceriyor, lacivert = icermiyor); model kimligi ise x-ekseni
    # etiketlerinin rengiyle verilir (asagida), boylece iki ayri bilgi
    # ayni grafikte cakismadan bir arada durur.
    colors = ["#c00000" if (lo < 0 < hi) else "#1f4e79" for lo, hi in zip(los, his)]

    ax.axhline(0, color="black", lw=0.9, linestyle="-")
    ax.errorbar(x, points, yerr=[yerr_low, yerr_high], fmt="none",
                ecolor="#808080", elinewidth=1.4, capsize=5, zorder=2)
    ax.scatter(x, points, s=55, c=colors, zorder=3, edgecolors="black", linewidths=0.6)

    for xi, lo in zip(x, los):
        if lo < Y_MIN:
            ax.annotate(f"{lo:.1f}", xy=(xi, Y_MIN), xytext=(xi, Y_MIN + 0.16),
                        ha="center", va="bottom", fontsize=8.5, color="#222222",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.85))

    ax.set_ylim(Y_MIN, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9, rotation=15)
    for tick_label, m in zip(ax.get_xticklabels(), model_labels):
        tick_label.set_color(MODEL_COLORS.get(m, "#222222"))
        tick_label.set_fontweight("bold")
    ax.set_title(TARGET_LABELS[tgt], fontsize=11)
    ax.set_ylabel("Test R\u00b2 (point estimate \u00b1 95% bootstrap CI)" if tgt == TARGETS[0] else "")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(f"Bootstrap Confidence Interval of Test R\u00b2 (n={n_test_actual} events, B=10,000 resamples)\n"
             "Marker color: red = 95% CI includes zero, blue = 95% CI excludes zero",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT_DIR / "bootstrap_r2_test.png", dpi=150)
plt.close(fig)
print("bootstrap_r2_test.png kaydedildi ->", OUT_DIR / "bootstrap_r2_test.png")
