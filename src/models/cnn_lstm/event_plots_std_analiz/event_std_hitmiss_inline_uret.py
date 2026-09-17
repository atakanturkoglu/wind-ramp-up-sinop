# -*- coding: utf-8 -*-
"""
CNN-LSTM'in TEK/SABIT train-test bolmesinden gelen 9 test olayi icin:
tahmin edilen noktanin (siyah elmas) UZERINE, o tahmin seviyesinde
DOGAL olarak ne kadar dakikalik standart sapma oldugunu gosteren
inline bir kutu-biyik (box-whisker) isareti cizer.

GUNCELLEME: Kutu tekrar TAHMIN edilen degere (pred_ws1) merkezleniyor
- hem konum hem de dogal std'nin hangi ruzgar hizi SEVIYESINDEN
okunacagi (round(pred_ws1)) buna gore belirleniyor - ve GERCEK nokta
(mor) bu kutunun icine dusuyor mu diye bakiliyor. Bu, Yontem
bolumunde anlatilan yontemle (kutu tahmin edilen degerin etrafina
cizilir, gerceklesen deger kutunun icine duserse HIT sayilir) tutarli
olacak sekilde geri alinmistir.

Kutu = tahmin edilen zirve (pred_ws1) +/- o seviyedeki ORTALAMA std
(HIT esigi). Biyik uclari = +/- o seviyedeki 95. persentil std
(sadece referans). Gercek nokta (mor) bu kutunun icine duserse HIT
(yesil), disina duserse MISS (kirmizi).

Baglam: rampanin kendisi ([t0,t1]) ham veriyle, disi (ramp_up-son/
events_temiz_kontekst ile AYNI mantik) temiz veriyle cizilir - bkz.
model/_ortak/olay_gorseli.py:_load_context_series/_hybrid_segment.
std_analiz/std_distribution_general.csv (kayan 31 dk pencere, 15 sol +
1 merkez + 15 sag, kendi final temiz verimiz uzerinden) degismedi,
oldugu gibi kullanilir.

HIT kriteri: |gercek zirve - tahmin edilen zirve| <= tahmin seviyesindeki
ORTALAMA genel std.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_ortak"))
from olay_gorseli import _load_context_series, _hybrid_segment, _info_box  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

PRED_CSV = Path(PROJECT_ROOT, 'src', 'models', 'cnn_lstm', 'predictions.csv')
STD_DIST_CSV = Path(PROJECT_ROOT, 'src', 'std_analysis', 'std_distribution_general.csv')
OUT_PATH = Path(__file__).parent / "test_events_hit_miss_inline.png"

STYLE = dict(color="#222222", face="#e5e5e5", label="CNN-LSTM (test)")
COLOR_HIT = "#2e7d32"
COLOR_MISS = "#c00000"

pred = pd.read_csv(PRED_CSV, parse_dates=["t0", "t1"])
pred = pred[pred["split"] == "test"].sort_values("t0").reset_index(drop=True)
n_events = len(pred)
print(f"Test olay sayisi: {n_events}")

std_dist = pd.read_csv(STD_DIST_CSV).set_index("ws_ms")
df_ctx = _load_context_series()


def draw_event(ax_speed, ax_dir, ev):
    t0, t1 = ev["t0"], ev["t1"]
    ws0, ws1 = ev["ws0"], ev["ws1"]
    duration = float((t1 - t0).total_seconds() / 60.0)

    pred_time = t0 + pd.Timedelta(minutes=float(ev["delta_t_tahmin_split"]))
    pred_ws1 = ws0 + float(ev["delta_ws_tahmin_split"])

    # Dogal std, TAHMIN seviyesinden okunur - kutu tahmine merkezlendigi icin.
    ws_level = int(round(pred_ws1))
    ws_level = min(max(ws_level, std_dist.index.min()), std_dist.index.max())
    row = std_dist.loc[ws_level]
    error = abs(pred_ws1 - ws1)
    is_hit = error <= row["mean"]

    context_min = min(30, max(12, 5 * duration))
    win_start = min(t0, pred_time) - pd.Timedelta(minutes=context_min)
    win_end = max(t1, pred_time) + pd.Timedelta(minutes=context_min)
    ctx = df_ctx.reindex(pd.date_range(win_start, win_end, freq="min"))
    hybrid = _hybrid_segment(ctx, t0, t1)

    ax_speed.plot(hybrid.index, hybrid.values, color="#2b7bba", lw=1.3, zorder=2)
    seg = hybrid.loc[t0:t1]
    ax_speed.axvspan(t0, t1, color="#e85d5d", alpha=0.15)
    ax_speed.plot(seg.index, seg.values, color="#d62728", lw=2.2, zorder=3)

    ax_speed.scatter([t0], [ws0], color="#2ca02c", s=55, zorder=5, edgecolor="white", linewidth=0.6)
    ax_speed.scatter([t1], [ws1], color="#9467bd", s=70, zorder=8, edgecolor="white", linewidth=0.7)
    ax_speed.scatter([pred_time], [pred_ws1], color=STYLE["color"], marker="D", s=60,
                      zorder=7, edgecolor="white", linewidth=0.7)

    # --- Kucuk kutu-biyik isareti: TAHMIN noktasinin (siyah elmas) TAM
    # UZERINE, tahmin zamanina (pred_time) ve tahmin degerine (pred_ws1)
    # merkezlenerek, o seviyedeki dogal std'ye gore cizilir. GERCEK nokta
    # (mor) bu kutunun icine dusuyorsa HIT, disina dusuyorsa MISS sayilir. ---
    box_x = pred_time
    box_half_width = pd.Timedelta(minutes=max(1.0, context_min * 0.035))
    box_color = COLOR_HIT if is_hit else COLOR_MISS

    whisker_top = pred_ws1 + row["p95"]
    whisker_bot = pred_ws1 - row["p95"]
    box_top = pred_ws1 + row["mean"]
    box_bot = pred_ws1 - row["mean"]

    ax_speed.add_line(Line2D([box_x, box_x], [whisker_bot, whisker_top],
                              color=box_color, linewidth=1.3, zorder=6, alpha=0.9))
    ax_speed.add_line(Line2D([box_x - box_half_width, box_x + box_half_width], [whisker_top] * 2,
                              color=box_color, linewidth=1.6, zorder=6, alpha=0.9))
    ax_speed.add_line(Line2D([box_x - box_half_width, box_x + box_half_width], [whisker_bot] * 2,
                              color=box_color, linewidth=1.6, zorder=6, alpha=0.9))
    rect = Rectangle((mdates.date2num(box_x - box_half_width), box_bot),
                      mdates.date2num(box_x + box_half_width) - mdates.date2num(box_x - box_half_width),
                      box_top - box_bot, facecolor=box_color, edgecolor=box_color,
                      alpha=0.28, linewidth=1.3, zorder=5)
    ax_speed.add_patch(rect)
    ax_speed.add_line(Line2D([box_x - box_half_width, box_x + box_half_width], [pred_ws1] * 2,
                              color=box_color, linewidth=2.0, zorder=6))

    # Kutunun ust/alt sinirlarindan GERCEK noktanin (mor, t1/ws1) zamanina
    # kadar uzanan kesikli kilavuz cizgiler: goz, gercek noktanin bu iki
    # cizgi arasinda kalip kalmadigini DOGRUDAN kontrol edebilsin.
    guide_x_end = max(box_x, t1) + pd.Timedelta(minutes=1)
    guide_x_start = min(box_x, t1) - pd.Timedelta(minutes=0.5)
    ax_speed.add_line(Line2D([guide_x_start, guide_x_end], [box_top] * 2,
                              color=box_color, linewidth=1.0, linestyle=":", alpha=0.75, zorder=4))
    ax_speed.add_line(Line2D([guide_x_start, guide_x_end], [box_bot] * 2,
                              color=box_color, linewidth=1.0, linestyle=":", alpha=0.75, zorder=4))

    y_lo, y_hi = ax_speed.get_ylim()
    y_hi = max(y_hi, whisker_top + 0.10 * (y_hi - y_lo))
    y_lo = min(y_lo, whisker_bot - 0.10 * (y_hi - y_lo))
    span = y_hi - y_lo
    ax_speed.set_ylim(y_lo, y_hi + 0.40 * span)
    box_y = y_hi + 0.17 * span

    _info_box(ax_speed, t0, ws0, f"t0={t0:%H:%M}\nws0={ws0:.2f}", "#2ca02c", "#d4f7d4", -42, 18)
    _info_box(ax_speed, t1, ws1, f"t1={t1:%H:%M}\nws1={ws1:.2f}", "#9467bd", "#ecd4f7", 40, -24)

    ax_speed.annotate(
        f"pred: \u0394ws={ev['delta_ws_tahmin_split']:.2f} m/s\n({pred_time:%H:%M}, {pred_ws1:.2f} m/s)\n"
        f"box = \u00b1{row['mean']:.2f} m/s natural std @ {ws_level} m/s",
        xy=(pred_time, pred_ws1), xytext=(pred_time, box_y), textcoords="data",
        fontsize=6.2, ha="center", va="center", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=STYLE["color"], lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=STYLE["face"],
                  edgecolor=STYLE["color"], alpha=0.95, linewidth=1.0),
    )

    tag_color = COLOR_HIT if is_hit else COLOR_MISS
    tag_text = "HIT" if is_hit else "MISS"
    title = (f"#{int(ev['event_id'])} | {t0:%Y-%m-%d} | actual +{ev['delta_ws_gercek']:.2f} m/s   [{tag_text}]")
    ax_speed.set_title(title, fontsize=8.5, pad=8, color=tag_color, fontweight="bold")
    ax_speed.set_ylabel("Wind speed (m/s)", fontsize=7.5)
    ax_speed.tick_params(labelbottom=False, labelsize=6.8)
    ax_speed.grid(True, alpha=0.3)

    ax_dir.scatter(ctx.index, ctx["wd_deg"], color="#b8860b", marker="_", s=22)
    ax_dir.axvspan(t0, t1, color="#e85d5d", alpha=0.15)
    ax_dir.set_ylim(0, 360)
    ax_dir.set_yticks([0, 90, 180, 270, 360])
    ax_dir.set_yticklabels(["N", "E", "S", "W", "N"], fontsize=6.5)
    ax_dir.set_ylabel("Wind dir.", fontsize=7.5)
    ax_dir.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for lbl in ax_dir.get_xticklabels():
        lbl.set_rotation(20)
        lbl.set_ha("right")
        lbl.set_fontsize(6.5)
    ax_dir.grid(True, alpha=0.3)
    return is_hit


cols = 3
rows = -(-n_events // cols)  # ceil
fig = plt.figure(figsize=(6.2 * cols, 3.9 * rows), dpi=150)
outer = fig.add_gridspec(rows, cols, hspace=0.55, wspace=0.25)
n_hit = 0
for i in range(n_events):
    row_i, col_i = divmod(i, cols)
    ev = pred.iloc[i]
    inner = outer[row_i, col_i].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
    ax_speed = fig.add_subplot(inner[0])
    ax_dir = fig.add_subplot(inner[1], sharex=ax_speed)
    n_hit += int(draw_event(ax_speed, ax_dir, ev))

fig.suptitle(
    f"CNN-LSTM \u2014 Held-out Test Set Predictions with Inline Natural-Variability Band   \u2014   "
    f"{n_hit}/{n_events} HIT",
    fontsize=13, fontweight="bold",
)
fig.savefig(OUT_PATH, bbox_inches="tight")
plt.close(fig)
print(f"{n_hit}/{n_events} HIT")
print("Kaydedildi ->", OUT_PATH)
