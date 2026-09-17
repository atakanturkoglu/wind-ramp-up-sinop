# -*- coding: utf-8 -*-
"""
Figure 3 (24-Hour Pre-Event Matrix) icin okunakli versiyon: 24 saatin
tamamini 4x6 kucuk panel halinde sikistirmak yerine, 6 temsilci saati
(T-24, T-20, T-15, T-10, T-5, T-1) 3x2 buyuk panel halinde gosterir.

Word'e ~6.8 inc genislikte gomulecegi icin fontlar buna gore (Figure 1
ile ayni mantikla) buyutulmustur. Word dosyasinda daha UZUN (dikey) bir
alan kaplamasi icin 3 satir x 2 sutun duzeni tercih edilmistir (2x3'e
gore panel basina daha fazla genislik verir).

Veri kaynagi: matris/X.npy + meta.csv (build_matrices.py'nin ciktisi,
degistirilmedi, sadece OKUNUR).
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

BASE = Path(PROJECT_ROOT, 'src', 'feature_matrix')
OUT_PATH = BASE / "event4_matrix_6panel.png"

STATIONS = ["merkez", "inceburun", "airport", "wl"]
VARS = ["P", "T", "u", "v", "ws", "wd"]
N_HOURS = 24
DISPLAY_STATION_ORDER = ["inceburun", "airport", "merkez", "wl"]
DISPLAY_STATION_LABELS = {"inceburun": "Inceburun", "airport": "Airport",
                           "merkez": "Sinop", "wl": "WL"}
display_idx = [STATIONS.index(s) for s in DISPLAY_STATION_ORDER]
display_labels = [DISPLAY_STATION_LABELS[s] for s in DISPLAY_STATION_ORDER]

meta = pd.read_csv(BASE / "meta.csv", parse_dates=["t0", "t1", "pencere_bas", "pencere_son"])
X = np.load(BASE / "X.npy")

row = meta[meta["event_id"] == 4].iloc[0]
pos = meta.index[meta["event_id"] == 4][0]
matrix = X[pos]  # (24, 4, 6)
window_hours = pd.date_range(row["pencere_bas"], row["pencere_son"], freq="h")
assert len(window_hours) == N_HOURS

# --- 6 temsilci saat: T-24, T-20, T-15, T-10, T-5, T-1 ---
SELECTED_H = [0, 4, 9, 14, 19, 23]

norms = []
for vi in range(len(VARS)):
    vals = matrix[:, :, vi]
    finite = vals[~np.isnan(vals)]
    if finite.size == 0:
        norms.append((0.0, 1.0))
    else:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        norms.append((vmin, vmax))
cmap = plt.get_cmap("viridis")

fig, axes = plt.subplots(3, 2, figsize=(13, 15.5))
axes = axes.flatten()

for panel_i, h in enumerate(SELECTED_H):
    ax = axes[panel_i]
    cell_rgba = np.ones((len(VARS), len(DISPLAY_STATION_ORDER), 4))
    for vi in range(len(VARS)):
        vmin, vmax = norms[vi]
        for ci, si in enumerate(display_idx):
            val = matrix[h, si, vi]
            if np.isnan(val):
                cell_rgba[vi, ci] = (0.90, 0.90, 0.90, 1.0)
            else:
                nv = 0.5 if vmax <= vmin else min(max((val - vmin) / (vmax - vmin), 0.0), 1.0)
                cell_rgba[vi, ci] = cmap(nv)
    ax.imshow(cell_rgba, aspect="auto")
    ax.set_xticks(range(len(DISPLAY_STATION_ORDER)))
    ax.set_xticklabels(display_labels, fontsize=16, rotation=20)
    ax.set_yticks(range(len(VARS)))
    ax.set_yticklabels(VARS, fontsize=17)
    for vi in range(len(VARS)):
        for ci, si in enumerate(display_idx):
            val = matrix[h, si, vi]
            if np.isnan(val):
                ax.text(ci, vi, "\u2014", ha="center", va="center", fontsize=16, color="#555555")
                continue
            r, g, b, _ = cell_rgba[vi, ci]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.55 else "black"
            ax.text(ci, vi, f"{val:.1f}", ha="center", va="center",
                     fontsize=16, color=text_color, fontweight="medium")
    for edge in ("top", "right", "left", "bottom"):
        ax.spines[edge].set_visible(True)
        ax.spines[edge].set_color("white")
        ax.spines[edge].set_linewidth(2.0)
    ax.set_xticks(np.arange(-0.5, len(DISPLAY_STATION_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(VARS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    step_name = f"T-{N_HOURS - h}"
    label = pd.Timestamp(window_hours[h]).strftime("%d %b %H:%M")
    ax.set_title(f"{step_name}   ({label})", fontsize=19, pad=10)

fig.suptitle(
    f"Six Representative Hours of the 24-Hour Pre-Event Matrix\n"
    f"(\u0394v = +{row['delta_v']:.1f} m/s, duration = {row['duration_min']:.0f} min)",
    fontsize=20, fontweight="bold", y=0.998,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_PATH, dpi=130, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("Kaydedildi ->", OUT_PATH)
