# -*- coding: utf-8 -*-
"""
CNN-LSTM icin OOF (out-of-fold, 5x5 tekrarli CV ortalamasi) artik
(residual = gercek - tahmin) dagilimi, her iki hedef icin. Amac:
oof_scatter.png'nin gosteremedigi bir seyi gostermek - hatalarin SIFIR
etrafinda simetrik mi dagildigi (yansiz), yoksa sistematik bir yone mi
kaydigi (onyargili, ör. hep dusuk tahmin). Sifir cizgisi ve ortalama
artik degeri isaretlenir. Veri dogrudan predictions.csv'den (delta_ws_
tahmin/delta_t_tahmin, 43 olayin tumu) okunur, yeniden hesap yapilmaz.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_NAME = "CNN-LSTM"
OUT_PATH = HERE / "residual_distribution.png"

pred = pd.read_csv(HERE / "predictions.csv")

TARGETS = {
    "delta_ws": {"label": "Delta WS (ramp-up amplitude)", "unit": "m/s", "color": "#1f77b4"},
    "delta_t": {"label": "Delta T (ramp-up duration)", "unit": "min", "color": "#d62728"},
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (tgt, info) in zip(axes, TARGETS.items()):
    resid = (pred[f"{tgt}_gercek"] - pred[f"{tgt}_tahmin"]).to_numpy()
    mean_resid = resid.mean()

    n_bins = 12
    ax.hist(resid, bins=n_bins, color=info["color"], edgecolor="white", alpha=0.85, zorder=2)
    ax.axvline(0, color="black", lw=1.4, linestyle="-", zorder=3, label="Zero (unbiased)")
    ax.axvline(mean_resid, color="#333333", lw=1.6, linestyle="--", zorder=3,
               label=f"Mean residual = {mean_resid:+.2f} {info['unit']}")

    ax.set_xlabel(f"Residual, Actual − Predicted ({info['unit']})", fontsize=10)
    ax.set_ylabel("Number of events", fontsize=10)
    ax.set_title(info["label"], fontsize=11.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(f"{MODEL_NAME} — Out-of-Fold Residual Distribution\n"
             "(block-aware 5-fold × 5 repeats, 43 events; positive = model underestimates)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Kaydedildi ->", OUT_PATH)
