# -*- coding: utf-8 -*-
"""
Agac tabanli modeller (XGBoost, CatBoost, LightGBM) icin ozellik onemi
PNG'si uretir. Bu modeller egitim sirasinda zaten 576 satirlik
(24 saat x 4 istasyon x 6 degisken) duz bir feature_importance_*.csv
yazar (train_xgboost.py/train_catboost.py/train_lightgbm.py); burada
YENI bir onem hesaplanmaz, sadece bu CSV'deki degerler saat boyutu
toplanarak (24 saatin tumunde bir (istasyon, degisken) ciftinin toplam
katkisi) 4x6'lik bir izgaraya indirgenip, CNN-LSTM/LSTM'in permutasyon
onem grafikleriyle AYNI gorsel stilde cizilir - boylece 6 model de
karsilastirilabilir tek bir formatta sunulur.

ONEMLI (dogru disclosure): her kutuphanenin varsayilan onem metrigi
FARKLIDIR - CSV'deki "gain" sutun adi tarihsel, gercek metrik
kutuphaneye gore degisir:
  - XGBoost: "gain" (bir ozelligin kullanildigi bolmelerin ortalama
    gain kazanci) - sklearn API varsayilani.
  - CatBoost: "PredictionValuesChange" (ozelligin degeri degistiginde
    tahminin ortalama ne kadar degistigi) - get_feature_importance()
    varsayilani.
  - LightGBM: "split" (ozelligin agac bolmelerinde kac kez kullanildigi,
    GAIN DEGIL) - LGBMRegressor varsayilani (importance_type='split').
Bu farklar heatmap basligina acikca yazilir, gizlenmez.
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

STATIONS_RAW = ["merkez", "inceburun", "airport", "wl"]
VARS = ["P", "T", "u", "v", "ws", "wd"]
DISPLAY_STATION_ORDER = ["inceburun", "airport", "merkez", "wl"]
DISPLAY_STATION_LABELS = {"inceburun": "Inceburun", "airport": "Airport",
                           "merkez": "Sinop", "wl": "OB"}
TARGET_LABELS = {"delta_ws": "Delta WS (ramp-up amplitude)", "delta_t": "Delta T (ramp-up duration)"}

_FEATURE_RE = re.compile(r"^h(\d{2})_([a-z]+)_([A-Za-z]+)$")

# Kutuphane basina gercek metrik adi (onem skorunun ANLAMI) - disclosure icin.
METRIC_LABEL = {
    "xgboost": "Gain (average split gain)",
    "catboost": "PredictionValuesChange",
    "lightgbm": "Split count (NOT gain)",
}


def _aggregate(csv_path):
    """576 satirlik (feature, gain) tablosunu 24 saat boyunca toplayip
    4x6'lik (istasyon x degisken) DataFrame'e indirger."""
    df = pd.read_csv(csv_path)
    df[["hour", "station", "var"]] = df["feature"].str.extract(_FEATURE_RE)
    grid = df.groupby(["station", "var"])["gain"].sum().unstack("var")
    grid = grid.reindex(index=STATIONS_RAW, columns=VARS).fillna(0.0)
    return grid


def render_tree_importance(model_key, model_label, out_dir):
    out_dir = Path(out_dir)
    metric_label = METRIC_LABEL.get(model_key, "Feature importance")
    for tname in ["delta_ws", "delta_t"]:
        csv_path = out_dir / f"feature_importance_{tname}.csv"
        grid = _aggregate(csv_path)
        display_idx = [STATIONS_RAW.index(s) for s in DISPLAY_STATION_ORDER]
        imp_display = grid.to_numpy()[display_idx, :]

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
        cbar.set_label(f"{metric_label} (summed over 24 hours)", fontsize=9)
        ax.set_title(f"{model_label} Feature Importance — {TARGET_LABELS[tname]}", fontsize=11.5)
        fig.tight_layout()
        out_png = out_dir / f"feature_importance_{tname}.png"
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"  {model_label} {tname}: {out_png}")
