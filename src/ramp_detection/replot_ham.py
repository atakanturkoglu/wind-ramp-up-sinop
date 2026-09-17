# -*- coding: utf-8 -*-
"""
ramp_up_events_HAM.csv'yi (manuel eklemeler dahil, GUNCEL haliyle) okuyup
ham veri event sayfalarini yeniden cizer. detect_ramp_up_ham.py'nin cizim
kismiyla birebir ayni mantik - sadece veri kaynagi CSV.
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
PLOT_DIR = os.path.join(BASE_DIR, "events")
CONTEXT_MIN = 75
EVENTS_PER_PAGE = 6

df = pd.read_csv(CLEAN_CSV, parse_dates=["datetime"])
df["ws_temiz"] = df["ws_raw_kmh"] / 3.6
df_indexed = df.set_index("datetime")[["ws_temiz", "wd_deg"]]

events_df = pd.read_csv(EVENTS_CSV, parse_dates=["t0", "t1"])
has_note = "manual_note" in events_df.columns


def get_context(t_start, t_end):
    start = t_start - pd.Timedelta(minutes=CONTEXT_MIN)
    end = t_end + pd.Timedelta(minutes=CONTEXT_MIN)
    full_idx = pd.date_range(start, end, freq="min")
    return df_indexed.reindex(full_idx)


def draw_event(ax_speed, ax_dir, ev):
    t0, t1 = ev["t0"], ev["t1"]
    ws0, ws1 = ev["ws0"], ev["ws1"]
    ctx = get_context(t0, t1)

    ax_speed.plot(ctx.index, ctx["ws_temiz"], color="#2b7bba", lw=1.3)
    seg = ctx.loc[t0:t1]
    ax_speed.axvspan(t0, t1, color="#e85d5d", alpha=0.15)
    ax_speed.plot(seg.index, seg["ws_temiz"], color="#d62728", lw=2.2)

    ax_speed.scatter([t0], [ws0], color="#2ca02c", s=55, zorder=5)
    ax_speed.scatter([t1], [ws1], color="#9467bd", s=55, zorder=5)

    manual_tag = ""
    if has_note and isinstance(ev.get("manual_note"), str) and ev["manual_note"].strip():
        manual_tag = "  [manuel eklendi]"

    title = (
        f"HAM #{int(ev['event_id'])}  |  {t0:%Y-%m-%d}  |  "
        f"{ws0:.1f} to {ws1:.1f} m/s (+{ev['delta_v']:.1f} m/s)  |  "
        f"{ev['duration_min']:.0f} min{manual_tag}"
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


for _old in os.listdir(PLOT_DIR):
    if _old.endswith(".png"):
        os.remove(os.path.join(PLOT_DIR, _old))

n_all = len(events_df)
n_pages = max(1, math.ceil(n_all / EVENTS_PER_PAGE))
for page in range(n_pages):
    fig = plt.figure(figsize=(16, 12), dpi=150)
    outer = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25)
    start_i = page * EVENTS_PER_PAGE
    for i in range(EVENTS_PER_PAGE):
        row, col = divmod(i, 2)
        cell = outer[row, col]
        ev_idx = start_i + i
        if ev_idx < n_all:
            ev = events_df.iloc[ev_idx]
            inner = cell.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
            ax_speed = fig.add_subplot(inner[0])
            ax_dir = fig.add_subplot(inner[1], sharex=ax_speed)
            draw_event(ax_speed, ax_dir, ev)
        else:
            ax = fig.add_subplot(cell)
            ax.axis("off")
    out_path = os.path.join(PLOT_DIR, f"page_{page + 1:03d}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

print(f"{n_pages} sayfa yeniden cizildi ({n_all} olay) -> {PLOT_DIR}")
