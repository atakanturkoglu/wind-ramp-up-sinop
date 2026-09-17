# -*- coding: utf-8 -*-
"""
Rahan beyin #18 notu icin: 63 olayin TAMAMINI tek tek gostermek yerine,
en buyuk genlige (delta_v) sahip 6 olayi tek bir sayfada (2x3) gosterir.
Cizim mantigi replot_ham.py ile BIREBIR aynidir (context/highlight/renkler);
sadece hangi olaylarin secildigi farklidir.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_CSV = Path(PROJECT_ROOT, 'src', 'quality_control', 'wl_temizlenmis.csv')
EVENTS_CSV = os.path.join(BASE_DIR, "ramp_up_events_HAM.csv")
OUT_PATH = os.path.join(BASE_DIR, "top6_amplitude_events.png")
CONTEXT_MIN = 75

TOP_EVENT_IDS = [13, 4, 1, 20, 17, 54]  # delta_v'ye gore buyukten kucuge
# NOT: sirasiyla iki degisiklik yapildi -
#   1) orijinal 6.'ci en buyuk genlikli olay (event 28, 2024-09-23) veri
#      kalitesi acisindan saglikli degildi (donmus/duz cizgi segmentleri).
#   2) yerine konan event 5 (2023-11-11) ise isaretlenen bitis noktasinin
#      (t1) gercek yerel zirveyi tam yakalamadigi goruldu (olay bittikten
#      ~20 dk sonra ruzgar hizi isaretli degerden ~0.5 m/s daha yukselmis).
# Son olarak event 54 (2026-02-18) secildi: hem donma yok hem de isaretli
# bitis noktasi gercekten yerel zirveye denk geliyor.

df = pd.read_csv(CLEAN_CSV, parse_dates=["datetime"])
df["ws_temiz"] = df["ws_raw_kmh"] / 3.6
df_indexed = df.set_index("datetime")[["ws_temiz", "wd_deg"]]

events_df = pd.read_csv(EVENTS_CSV, parse_dates=["t0", "t1"])
events_df = events_df.set_index("event_id").loc[TOP_EVENT_IDS].reset_index()
has_note = "manual_note" in events_df.columns


def get_context(t_start, t_end):
    start = t_start - pd.Timedelta(minutes=CONTEXT_MIN)
    end = t_end + pd.Timedelta(minutes=CONTEXT_MIN)
    full_idx = pd.date_range(start, end, freq="min")
    return df_indexed.reindex(full_idx)


def draw_event(ax_speed, ax_dir, ev, rank):
    t0, t1 = ev["t0"], ev["t1"]
    ws0, ws1 = ev["ws0"], ev["ws1"]
    ctx = get_context(t0, t1)

    ax_speed.plot(ctx.index, ctx["ws_temiz"], color="#2b7bba", lw=1.3)
    seg = ctx.loc[t0:t1]
    ax_speed.axvspan(t0, t1, color="#e85d5d", alpha=0.15)
    ax_speed.plot(seg.index, seg["ws_temiz"], color="#d62728", lw=2.2)

    ax_speed.scatter([t0], [ws0], color="#2ca02c", s=55, zorder=5)
    ax_speed.scatter([t1], [ws1], color="#9467bd", s=55, zorder=5)

    title = (
        f"#{rank}  |  {t0:%Y-%m-%d}  |  "
        f"{ws0:.1f} to {ws1:.1f} m/s (+{ev['delta_v']:.1f} m/s)  |  "
        f"{ev['duration_min']:.0f} min"
    )
    ax_speed.set_title(title, fontsize=8.5, pad=8)
    y_lo, y_hi = ax_speed.get_ylim()
    ax_speed.set_ylim(y_lo, y_hi + 0.15 * (y_hi - y_lo))
    ax_speed.set_ylabel("Wind speed (m/s)", fontsize=7.5)
    ax_speed.tick_params(labelbottom=False, labelsize=7)
    ax_speed.grid(True, alpha=0.3)

    ax_dir.scatter(ctx.index, ctx["wd_deg"], color="#b8860b", marker="_", s=22)
    ax_dir.axvspan(t0, t1, color="#e85d5d", alpha=0.15)
    ax_dir.set_ylim(0, 360)
    ax_dir.set_yticks([0, 90, 180, 270, 360])
    ax_dir.set_yticklabels(["N", "E", "S", "W", "N"], fontsize=6.5)
    ax_dir.set_ylabel("Wind direction", fontsize=7.5)
    ax_dir.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
    for lbl in ax_dir.get_xticklabels():
        lbl.set_rotation(20)
        lbl.set_ha("right")
        lbl.set_fontsize(6.5)
    ax_dir.grid(True, alpha=0.3)


fig = plt.figure(figsize=(16, 12), dpi=150)
outer = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25)
for i, ev_id in enumerate(TOP_EVENT_IDS):
    row, col = divmod(i, 2)
    cell = outer[row, col]
    ev = events_df.iloc[i]
    inner = cell.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
    ax_speed = fig.add_subplot(inner[0])
    ax_dir = fig.add_subplot(inner[1], sharex=ax_speed)
    draw_event(ax_speed, ax_dir, ev, rank=i + 1)

fig.savefig(OUT_PATH, bbox_inches="tight")
plt.close(fig)
print("Kaydedildi ->", OUT_PATH)
