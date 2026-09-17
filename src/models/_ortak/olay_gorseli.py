# -*- coding: utf-8 -*-
"""
Herhangi bir modelin predictions.csv'sinden, her olay icin ayri bir panel
uretir: gercek zaman serisi + gercek t0/ws0 (yesil, oklu kutu) + gercek
t1/ws1 (mor, oklu kutu) + kirmizi ramp-up segmenti + modelin tahmin ettigi
(t0+delta_t_tahmin, ws0+delta_ws_tahmin) noktasi (siyah elmas, MAVI oklu
kutu).

Yerlesim teknigi onceki (arsivlenmis) test_event_plots_uret.py'den
BIREBIR alinmistir: kutular axes-fraction yerine VERI koordinatinda,
y eksenini dinamik olarak genisletip icine sigacak sekilde konumlanir -
bu, panel boyutundan bagimsiz olarak cakismayi engeller.

ONEMLI TERIM FARKI: Bu proje artik SABIT bir test seti kullanmiyor - her
olay, o olayi hic gormemis bir modelden 5 kat x 5 tekrar CV ile tahmin
ediliyor (out-of-fold, OOF). Bu yuzden kutularda "(test)" degil "(OOF)"
yaziyor - kavramsal olarak daha guclu bir dogrulama (her olay 5 kez,
farkli modellerden tahmin ediliyor), sabit bir test seti degil.

Baglam: rampanin kendisi ([t0,t1]) ham veriyle, disi (ramp_up-son/
events_temiz_kontekst ile AYNI mantik) donuk/testere temizligi uygulanmis
veriyle cizilir.
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

CLEAN_CSV = Path(PROJECT_ROOT, 'src', 'quality_control', 'wl_temizlenmis.csv')
CONTEXT_MIN = 75
EVENTS_PER_PAGE = 6

# TEST (OOF, hic gormedigi veri) - siyah elmas, GRI kutu
TEST_COLOR = "#555555"
TEST_FACE = "#e5e5e5"
TEST_MARKER = "black"

# TRAIN (ic-ornek, tum veriyle egitilmis final model) - MAVI elmas, MAVI kutu
TRAIN_COLOR = "#1f5fbf"
TRAIN_FACE = "#dbe9ff"
TRAIN_MARKER = "#1f5fbf"


def _load_context_series():
    df = pd.read_csv(CLEAN_CSV, parse_dates=["datetime"])
    df["ws_ham"] = df["ws_raw_kmh"] / 3.6
    df["ws_temiz"] = df["ws_clean_ms"]
    if "flag_baglam_temizlik" in df.columns:
        df.loc[df["flag_baglam_temizlik"].astype(bool), "ws_temiz"] = np.nan
    df = df.sort_values("datetime").reset_index(drop=True)
    return df.set_index("datetime")[["ws_ham", "ws_temiz", "wd_deg"]]


def _hybrid_segment(ctx, t0, t1):
    in_event = (ctx.index >= t0) & (ctx.index <= t1)
    hybrid = np.where(in_event, ctx["ws_ham"].to_numpy(), ctx["ws_temiz"].to_numpy())
    return pd.Series(hybrid, index=ctx.index)


def _info_box(ax, x, y, text, color, face, dx, dy, fontsize=6.8):
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=fontsize, ha="center", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=face, edgecolor=color,
                  alpha=0.95, linewidth=1.0),
    )


def render_event_pages(predictions_csv=None, model_label="", out_dir=".", note="",
                        events_per_page=EVENTS_PER_PAGE, pred_df=None, single_file=None):
    """predictions.csv (model/_ortak/degerlendirme.py formatinda) okuyup
    sayfalar halinde PNG uretir. out_dir icine 'page_NNN.png' olarak yazar.
    events_per_page=1 verilirse her olay tek basina, buyuk boyutta cizilir
    (yakinlastirilmis / 'zoom' gorunum).

    pred_df verilirse predictions_csv yerine dogrudan bu (onceden
    filtrelenmis) DataFrame kullanilir. single_file verilirse (ör. 'test_
    events_page.png') sayfalama yapilmaz, TEK bir dosya adiyla kaydedilir
    (butun olaylar tek sayfaya sigdirilmaya calisilir)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if single_file is None:
        for old in out_dir.glob("page_*.png"):
            old.unlink()

    if pred_df is not None:
        pred = pred_df.copy()
    else:
        pred = pd.read_csv(predictions_csv, parse_dates=["t0", "t1"])
    pred = pred.sort_values("t0").reset_index(drop=True)
    df_ctx = _load_context_series()

    n_all = len(pred)

    def draw(ax_speed, ax_dir, ev):
        t0, t1 = ev["t0"], ev["t1"]
        ws0, ws1 = ev["ws0"], ev["ws1"]

        # TEK, SABIT train/test bolmesine gore: bu olay YA train YA test -
        # sadece o role ait kutu ve elmas cizilir (ikisi birden degil).
        role = str(ev.get("split", "test"))
        is_train = role == "train"
        pred_t1 = t0 + pd.Timedelta(minutes=float(ev["delta_t_tahmin_split"]))
        pred_ws1 = ws0 + float(ev["delta_ws_tahmin_split"])

        # Baglam penceresi rampanin SURESINE gore uyarlamali (zoom): kisa
        # bir rampanin etrafinda 75 dk sabit baglam onu gorsel olarak
        # kucultup anlamsizlastiriyordu. Simdi pencere rampanin kendi
        # suresiyle olceklenir (5 kati, 12-30 dk arasinda sinirlanmis) ve
        # tahmin noktasini da icine alacak sekilde genisletilir.
        duration_min = float((t1 - t0).total_seconds() / 60.0)
        context_min = min(30, max(12, 5 * duration_min))
        win_start = min(t0, pred_t1) - pd.Timedelta(minutes=context_min)
        win_end = max(t1, pred_t1) + pd.Timedelta(minutes=context_min)
        ctx = df_ctx.reindex(pd.date_range(win_start, win_end, freq="min"))
        hybrid = _hybrid_segment(ctx, t0, t1)

        ax_speed.plot(hybrid.index, hybrid.values, color="#2b7bba", lw=1.3)
        seg = hybrid.loc[t0:t1]
        ax_speed.axvspan(t0, t1, color="#e85d5d", alpha=0.15)
        ax_speed.plot(seg.index, seg.values, color="#d62728", lw=2.2)
        color, face, marker = (TRAIN_COLOR, TRAIN_FACE, TRAIN_MARKER) if is_train \
            else (TEST_COLOR, TEST_FACE, TEST_MARKER)
        role_label = "train" if is_train else "test"

        # y eksenini, tahmin noktasini ve kutuyu icine alacak sekilde
        # DINAMIK olarak genislet (panel boyutundan bagimsiz, saglam yontem)
        y_lo, y_hi = ax_speed.get_ylim()
        y_hi = max(y_hi, pred_ws1 + 0.10 * (y_hi - y_lo))
        y_lo = min(y_lo, pred_ws1 - 0.10 * (y_hi - y_lo))
        span = y_hi - y_lo
        ax_speed.set_ylim(y_lo, y_hi + 0.42 * span)
        box_y = y_hi + 0.19 * span

        ax_speed.scatter([t0], [ws0], color="#2ca02c", s=55, zorder=5,
                          edgecolor="white", linewidth=0.6)
        ax_speed.scatter([t1], [ws1], color="#9467bd", s=55, zorder=5,
                          edgecolor="white", linewidth=0.6)
        ax_speed.scatter([pred_t1], [pred_ws1], color=marker, marker="D", s=60,
                          zorder=7, edgecolor="white", linewidth=0.7)

        _info_box(ax_speed, t0, ws0, f"t0={t0:%H:%M}\nws0={ws0:.2f}",
                  "#2ca02c", "#d4f7d4", -42, 18)
        _info_box(ax_speed, t1, ws1, f"t1={t1:%H:%M}\nws1={ws1:.2f}",
                  "#9467bd", "#ecd4f7", 40, -24)

        ax_speed.annotate(
            f"{model_label} ({role_label}): \u0394ws={ev['delta_ws_tahmin_split']:.2f} m/s, "
            f"\u0394t={ev['delta_t_tahmin_split']:.1f} min\n"
            f"({pred_t1:%H:%M}, {pred_ws1:.2f} m/s)",
            xy=(pred_t1, pred_ws1), xytext=(pred_t1, box_y), textcoords="data",
            fontsize=6.5, ha="center", va="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=face,
                      edgecolor=color, alpha=0.95, linewidth=1.0),
        )

        title = (f"#{int(ev['event_id'])} [{role_label.upper()}] | {t0:%Y-%m-%d} | "
                 f"actual: {ws0:.1f}\u2192{ws1:.1f} m/s (+{ev['delta_ws_gercek']:.2f}) | "
                 f"{ev['delta_t_gercek']:.0f} min")
        ax_speed.set_title(title, fontsize=8.2, pad=8)
        ax_speed.set_ylabel("Wind speed (m/s)", fontsize=7.5)
        ax_speed.tick_params(labelbottom=False, labelsize=7)
        ax_speed.grid(True, alpha=0.3)

        ax_dir.scatter(ctx.index, ctx["wd_deg"], color="#b8860b", marker="_", s=22)
        ax_dir.axvspan(t0, t1, color="#e85d5d", alpha=0.15)
        ax_dir.set_ylim(0, 360)
        ax_dir.set_yticks([0, 90, 180, 270, 360])
        ax_dir.set_yticklabels(["N", "E", "S", "W", "N"], fontsize=6.5)
        ax_dir.set_ylabel("Wind dir.", fontsize=7.5)
        ax_dir.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for lbl in ax_dir.get_xticklabels():
            lbl.set_rotation(20); lbl.set_ha("right"); lbl.set_fontsize(6.5)
        ax_dir.grid(True, alpha=0.3)

    if single_file is not None:
        events_per_page = n_all
        cols = min(3, n_all) or 1
        rows = math.ceil(n_all / cols)
        fig_w, fig_h = cols * 5.4, rows * 4.4
    else:
        grid_map = {1: (1, 1, 9, 7.5), 2: (2, 1, 9, 13), 4: (2, 2, 16, 13), 6: (3, 2, 16, 18)}
        rows, cols, fig_w, fig_h = grid_map.get(events_per_page, (3, 2, 16, 18))
    n_pages = max(1, math.ceil(n_all / events_per_page))

    for page in range(n_pages):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
        outer = fig.add_gridspec(rows, cols, hspace=0.55 if events_per_page > 1 else 0.25,
                                  wspace=0.25)
        start_i = page * events_per_page
        for i in range(events_per_page):
            row, col = divmod(i, cols)
            cell = outer[row, col]
            idx = start_i + i
            if idx < n_all:
                inner = cell.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
                ax_speed = fig.add_subplot(inner[0])
                ax_dir = fig.add_subplot(inner[1], sharex=ax_speed)
                draw(ax_speed, ax_dir, pred.iloc[idx])
            else:
                ax = fig.add_subplot(cell); ax.axis("off")
        if note:
            fig.suptitle(note, fontsize=12, fontweight="bold")
        fname = single_file if single_file is not None else f"page_{page + 1:03d}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)

    print(f"{n_pages} sayfa uretildi ({n_all} olay) -> {out_dir}")
