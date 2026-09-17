# -*- coding: utf-8 -*-
"""
Figure 3 icin nihai versiyon: tum 24 saati veya 6 ornek saati degil,
TEK bir temsili saati (T-1, olay baslamadan hemen onceki saat) buyuk ve
net gosterir. Baslikta olaya OZGU sayilar (delta_v, duration) YOKTUR -
cunku bu matris yapisi TUM ramp-up olaylari icin genel/kapsayici olarak
kuruluyor, tek bir olayin buyuklugunu vurgulamak yanlis bir izlenim
verir. 18 Ekim 2023 (event_id=3) olayi somut bir ORNEK olarak kullanilir
- meta.csv'de doluluk_toplam=1.0 olan (Airport dahil 4 istasyonun tamami
dolu) tek olay oldugu icin secildi; boylece ornek panelde Airport
sutunu bomboş gorunmuyor.

Veri kaynagi: matris/X.npy + meta.csv (build_matrices.py ciktisi,
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
OUT_PATH = BASE / "event4_matrix_1panel.png"

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

row = meta[meta["event_id"] == 3].iloc[0]
pos = meta.index[meta["event_id"] == 3][0]
matrix = X[pos]  # (24, 4, 6)
window_hours = pd.date_range(row["pencere_bas"], row["pencere_son"], freq="h")
assert len(window_hours) == N_HOURS

H = 23  # T-1

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

fig, ax = plt.subplots(figsize=(7.5, 7.2))

cell_rgba = np.ones((len(VARS), len(DISPLAY_STATION_ORDER), 4))
for vi in range(len(VARS)):
    vmin, vmax = norms[vi]
    for ci, si in enumerate(display_idx):
        val = matrix[H, si, vi]
        if np.isnan(val):
            cell_rgba[vi, ci] = (0.90, 0.90, 0.90, 1.0)
        else:
            nv = 0.5 if vmax <= vmin else min(max((val - vmin) / (vmax - vmin), 0.0), 1.0)
            cell_rgba[vi, ci] = cmap(nv)
ax.imshow(cell_rgba, aspect="auto")
ax.set_xticks(range(len(DISPLAY_STATION_ORDER)))
ax.set_xticklabels(display_labels, fontsize=19, rotation=15)
ax.set_yticks(range(len(VARS)))
ax.set_yticklabels(VARS, fontsize=20)
for vi in range(len(VARS)):
    for ci, si in enumerate(display_idx):
        val = matrix[H, si, vi]
        if np.isnan(val):
            ax.text(ci, vi, "\u2014", ha="center", va="center", fontsize=19, color="#555555")
            continue
        r, g, b, _ = cell_rgba[vi, ci]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "white" if luminance < 0.55 else "black"
        ax.text(ci, vi, f"{val:.1f}", ha="center", va="center",
                 fontsize=19, color=text_color, fontweight="medium")
for edge in ("top", "right", "left", "bottom"):
    ax.spines[edge].set_visible(True)
    ax.spines[edge].set_color("white")
    ax.spines[edge].set_linewidth(2.5)
ax.set_xticks(np.arange(-0.5, len(DISPLAY_STATION_ORDER), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(VARS), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2.5)
ax.tick_params(which="minor", length=0)
ax.tick_params(which="major", length=0)

label = pd.Timestamp(window_hours[H]).strftime("%d %b %Y, %H:%M")
ax.set_title(f"T-1   ({label})", fontsize=22, pad=14)
fig.suptitle(
    "Example Hour from the 24-Hour Pre-Event Matrix",
    fontsize=17, fontweight="bold", y=1.045,
)

fig.savefig(OUT_PATH, dpi=130, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("Kaydedildi ->", OUT_PATH)
