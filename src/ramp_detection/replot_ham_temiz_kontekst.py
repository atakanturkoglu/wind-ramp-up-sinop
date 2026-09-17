# -*- coding: utf-8 -*-
"""
HAM veriden tespit edilen ramp-up olaylarini, HIBRIT bir gorunumle cizer:

  - [t0, t1] ARASI (rampanin kendisi)  -> HAM veri (temizlik uygulanmaz)
  - Geri kalan tum baglam (once/sonra) -> TEMIZ veri (wl_temizleme.py +
    manuel_duzeltmeler.py sonucu; isaretlenen noktalar bosluk olarak cikar)

Amac: "bu ham olay, etrafindaki arizali/donuk veri temizlenince hala
anlamli bir ramp-up gibi duruyor mu?" sorusunu gozle degerlendirebilmek.
Rampanin kendisi korunur ki olayin varligi tartisma disi kalsin; sadece
cevresi temizlenir.

Cikti: ramp_up-son/events_temiz_kontekst/  (saf ham grafikler
ramp_up-son/events/ altinda oldugu gibi kalir)
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_CSV = Path(PROJECT_ROOT, 'src', 'quality_control', 'wl_temizlenmis.csv')
EVENTS_CSV = os.path.join(BASE_DIR, "ramp_up_events_HAM.csv")
PLOT_DIR = os.path.join(BASE_DIR, "events_temiz_kontekst")
CONTEXT_MIN = 75
EVENTS_PER_PAGE = 6

os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(CLEAN_CSV, parse_dates=["datetime"])
df["ws_ham"] = df["ws_raw_kmh"] / 3.6          # hicbir temizlik uygulanmamis
df["ws_temiz"] = df["ws_clean_ms"]              # tum temizlik uygulanmis

# Olay baglamindaki donuk cizgi / testere disi bolumleri de gizle
# (olay_baglam_temizligi.py tarafindan isaretlenir; rampalarin kendisi
# o script tarafindan zaten korunuyor)
if "flag_baglam_temizlik" in df.columns:
    df.loc[df["flag_baglam_temizlik"].astype(bool), "ws_temiz"] = np.nan

df = df.sort_values("datetime").reset_index(drop=True)
df_indexed = df.set_index("datetime")[["ws_ham", "ws_temiz", "wd_deg"]]

events_df = pd.read_csv(EVENTS_CSV, parse_dates=["t0", "t1"])


def get_context(t_start, t_end):
    start = t_start - pd.Timedelta(minutes=CONTEXT_MIN)
    end = t_end + pd.Timedelta(minutes=CONTEXT_MIN)
    full_idx = pd.date_range(start, end, freq="min")
    return df_indexed.reindex(full_idx)


def draw_event(ax_speed, ax_dir, ev):
    t0, t1 = ev["t0"], ev["t1"]
    ws0, ws1 = ev["ws0"], ev["ws1"]
    ctx = get_context(t0, t1)

    # HIBRIT seri: [t0,t1] arasi ham, disi temiz
    in_event = (ctx.index >= t0) & (ctx.index <= t1)
    hybrid = np.where(in_event, ctx["ws_ham"].to_numpy(), ctx["ws_temiz"].to_numpy())
    hybrid = pd.Series(hybrid, index=ctx.index)

    ax_speed.plot(hybrid.index, hybrid.values, color="#2b7bba", lw=1.3)

    seg = hybrid.loc[t0:t1]
    ax_speed.axvspan(t0, t1, color="#e85d5d", alpha=0.15)
    ax_speed.plot(seg.index, seg.values, color="#d62728", lw=2.2)

    ax_speed.scatter([t0], [ws0], color="#2ca02c", s=55, zorder=5)
    ax_speed.scatter([t1], [ws1], color="#9467bd", s=55, zorder=5)

    title = (
        f"#{int(ev['event_id'])}  |  {t0:%Y-%m-%d}  |  "
        f"{ws0:.1f} to {ws1:.1f} m/s (+{ev['delta_v']:.1f} m/s)  |  "
        f"{ev['duration_min']:.0f} min"
    )
    ax_speed.set_title(title, fontsize=8.5, pad=8)
    y_lo, y_hi = ax_speed.get_ylim()
    ax_speed.set_ylim(y_lo, y_hi + 0.15 * (y_hi - y_lo))
    ax_speed.set_ylabel("Wind speed (m/s)", fontsize=7.5)
    ax_speed.tick_params(labelbottom=False, labelsize=7)
    ax_speed.grid(True, alpha=0.3)

    # yon: baglamda temizlenmis noktalarda da yon verisi anlamsiz olabilir,
    # ama yon ayrica temizlenmedigi icin oldugu gibi cizilir
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
    outer = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.25)
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

print(f"{n_pages} sayfa cizildi ({n_all} olay) -> {PLOT_DIR}")
