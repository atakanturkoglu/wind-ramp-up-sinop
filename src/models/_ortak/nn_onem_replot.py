# -*- coding: utf-8 -*-
"""
CNN-LSTM/LSTM/GRU icin, egitim betiginin sonunda ONCEDEN hesaplanip
kaydedilmis olan feature_importance_{target}.csv'yi (4x6, istasyon x
degisken) yeniden CIZMEK icin kucuk bir yardimci - modeli tekrar
egitmeden. Tek amaci: "wl" istasyon etiketinin "WL" yerine "OB" olarak
gosterilmesini saglamak (agac_onem_gorseli.py'de yapilan ayni duzeltme,
NN modelleri icin). Sayisal degerlere DOKUNULMAZ, sadece etiket.

train_cnn_lstm.py/train_lstm.py/train_gru.py'nin kendi ciziminde
bulunan "(base RMSE=...)" alt basligi burada YOKTUR - bu deger CSV'de
saklanmadigi ve yeniden hesaplamak modelin tekrar egitilmesini
gerektirdigi icin (maliyetli), atlanmistir; bu sadece kozmetik bir
bilgi kaybidir, onem degerlerinin kendisini etkilemez.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

STATIONS = ["merkez", "inceburun", "airport", "wl"]
VARS = ["P", "T", "u", "v", "ws", "wd"]
DISPLAY_STATION_ORDER = ["inceburun", "airport", "merkez", "wl"]
DISPLAY_STATION_LABELS = {"inceburun": "Inceburun", "airport": "Airport",
                           "merkez": "Sinop", "wl": "OB"}
TARGET_LABELS = {"delta_ws": "Delta WS (ramp-up amplitude)", "delta_t": "Delta T (ramp-up duration)"}


def replot_nn_importance(model_label, out_dir):
    out_dir = Path(out_dir)
    for tname in ["delta_ws", "delta_t"]:
        csv_path = out_dir / f"feature_importance_{tname}.csv"
        grid = pd.read_csv(csv_path, index_col=0)
        grid = grid.reindex(index=STATIONS, columns=VARS).fillna(0.0)
        display_idx = [STATIONS.index(s) for s in DISPLAY_STATION_ORDER]
        imp_display = grid.to_numpy()[display_idx, :].clip(min=0)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        vmax = imp_display.max() or 1.0
        im = ax.imshow(imp_display, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(VARS))); ax.set_xticklabels(VARS, fontsize=11)
        ax.set_yticks(range(len(DISPLAY_STATION_ORDER)))
        ax.set_yticklabels([DISPLAY_STATION_LABELS[s] for s in DISPLAY_STATION_ORDER], fontsize=11)
        for si in range(len(DISPLAY_STATION_ORDER)):
            for vi in range(len(VARS)):
                val = imp_display[si, vi]
                ax.text(vi, si, f"{val:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if val > vmax * 0.5 else "black")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("RMSE increase after permutation — larger = more important", fontsize=9)
        ax.set_title(f"{model_label} Feature Importance (Permutation) — {TARGET_LABELS[tname]}",
                     fontsize=11.5)
        fig.tight_layout()
        out_png = out_dir / f"feature_importance_{tname}.png"
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"  {model_label} {tname}: {out_png}")


if __name__ == "__main__":
    MODEL_DIR = Path(PROJECT_ROOT, 'src', 'models')
    for key, label in [("cnn_lstm", "CNN-LSTM"), ("lstm", "LSTM"), ("gru", "GRU")]:
        replot_nn_importance(label, MODEL_DIR / key)
