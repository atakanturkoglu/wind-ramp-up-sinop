"""
Bulgular bolumu icin: ramp-up olaylarinin tanimlayici istatistikleri
(genlik dagilimi, sure dagilimi, aylik oruntu).

GUNCELLEME: Artik ham veriden tespit edilip tek tek gozle dogrulanmis ve
duzeltilmis 63 olayluk liste (ramp_up-son/ramp_up_events_HAM.csv)
kullaniliyor - onceki 46 olayluk temiz-veri listesi degil. Yeni bir esik
veya hesaplama tanimlanmaz, sadece veri kaynagi degisti.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

EVENTS_PATH = Path(PROJECT_ROOT, 'src', 'ramp_detection', 'ramp_up_events_HAM.csv')
OUT_PATH = Path(PROJECT_ROOT, 'src', 'ramp_detection', 'olay_istatistikleri.png')

df = pd.read_csv(EVENTS_PATH, parse_dates=["t0", "t1"])

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

# (a) Ramp amplitude distribution
ax = axes[0]
bins = np.arange(6, 15.5, 0.5)
ax.hist(df["delta_v"], bins=bins, color="#1f4e79", edgecolor="white")
ax.axvline(df["delta_v"].median(), color="#c00000", linestyle="--", linewidth=1.2,
           label=f"median = {df['delta_v'].median():.2f} m/s")
ax.set_xlabel("Ramp-up amplitude, \u0394v (m/s)")
ax.set_ylabel("Number of events")
ax.set_title(f"(a) Amplitude distribution (N={len(df)})", fontsize=10.5)
ax.legend(fontsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# (b) Duration distribution
ax = axes[1]
bins = np.arange(0.5, 12.5, 1)
ax.hist(df["duration_min"], bins=bins, color="#2b7bba", edgecolor="white")
ax.axvline(df["duration_min"].median(), color="#c00000", linestyle="--", linewidth=1.2,
           label=f"median = {df['duration_min'].median():.0f} min")
ax.set_xlabel("Ramp-up duration (min)")
ax.set_ylabel("Number of events")
ax.set_title(f"(b) Duration distribution (N={len(df)})", fontsize=10.5)
ax.legend(fontsize=8.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# (c) Monthly pattern (aggregated across 2023-2026)
ax = axes[2]
month_counts = df["t0"].dt.month.value_counts().reindex(range(1, 13), fill_value=0)
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                "Sep", "Oct", "Nov", "Dec"]
ax.bar(range(1, 13), month_counts.values, color="#4c8c4a", edgecolor="white")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels, fontsize=8, rotation=45)
ax.set_ylabel("Number of events")
ax.set_title("(c) Monthly pattern (2023\u20132026 combined)", fontsize=10.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300)
plt.close(fig)
print("Kaydedildi ->", OUT_PATH)
