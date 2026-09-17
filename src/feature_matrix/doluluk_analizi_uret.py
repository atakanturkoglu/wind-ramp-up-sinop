"""
Bulgular bolumu icin: oncul matrislerin tamlik (doluluk) analizi.
doluluk_raporu.csv ve meta.csv'den dogrudan okunur, yeni bir hesaplama
tanimlanmaz - sadece build_matrices.py'nin zaten urettigi sayilar
gorsellestirilir.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

MATRIS_DIR = Path(PROJECT_ROOT, 'src', 'feature_matrix')
OUT_PATH = MATRIS_DIR + r"\doluluk_analizi.png"

rapor = pd.read_csv(MATRIS_DIR + r"\doluluk_raporu.csv")
meta = pd.read_csv(MATRIS_DIR + r"\meta.csv")

DISPLAY_LABELS = {"merkez": "Sinop", "inceburun": "Inceburun",
                   "airport": "Airport", "wl": "WL"}
STATIONS = ["inceburun", "airport", "merkez", "wl"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# (a) Cumulative completeness thresholds
ax = axes[0]
ax.bar([f"\u2265{int(t*100)}%" for t in rapor["esik"]], rapor["olay_sayisi"],
       color="#1f4e79", edgecolor="white")
for i, (n, pct) in enumerate(zip(rapor["olay_sayisi"], rapor["yuzde"])):
    ax.text(i, n + 0.4, f"{n}\n({pct:.0f}%)", ha="center", fontsize=8)
ax.set_ylabel(f"Number of events (out of {len(meta)})")
ax.set_xlabel("Completeness threshold")
ax.set_title("(a) Events meeting completeness threshold", fontsize=10.5)
ax.set_ylim(0, len(meta) + 4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# (b) Per-station average completeness (mean across 29 events)
ax = axes[1]
means = [meta[f"doluluk_{s}"].mean() * 100 for s in STATIONS]
colors = ["#2b7bba" if m >= 80 else "#c00000" for m in means]
bars = ax.bar([DISPLAY_LABELS[s] for s in STATIONS], means, color=colors, edgecolor="white")
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, m + 1.5, f"{m:.1f}%", ha="center", fontsize=9)
ax.set_ylabel(f"Mean completeness across {len(meta)} events (%)")
ax.set_title("(b) Per-station average completeness", fontsize=10.5)
ax.set_ylim(0, 110)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300)
plt.close(fig)
print("Kaydedildi ->", OUT_PATH)
