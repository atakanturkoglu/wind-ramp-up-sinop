"""
KONTROL AMACLI DENEY: Su anki ramp-up tespit algoritmasi (detect_ramp_up.py
ile BIREBIR AYNI mantik - Kuang vd. 2020 iki asamali yontem + ayni
MIN_PEAK_WS/MIN_DELTA_V/MAX_WS0 esikleri), HIC TEMIZLENMEMIS HAM VERI
uzerinde calistirilir.

Amac: temizlik surecinin (flatline/outlier/step/periyotluluk testleri)
gercek bir ramp-up olayini yanlislikla silip silmedigini kontrol etmek.
Ham veride cikan ama temiz veride cikmayan olaylar, kullanici tarafindan
gozle/manuel incelenecek - bu script kendisi bir karar vermiyor, sadece
karsilastirma malzemesi uretiyor.

Girdi: wl_temizlenmis.csv'nin "ws_raw_kmh" sutunu (HICBIR flag/temizlik
UYGULANMADAN, sadece km/h -> m/s cevrimi yapilir) + "wd_deg" sutunu
(yon verisi ayrica temizlenmiyor zaten, degismedi).

Cikti bu script'in kendi klasorune (ramp_up-son/) yazilir - ana
ramp_up/ klasorundeki (temiz veriyle uretilmis) sonuclara DOKUNULMAZ.
"""

import math
import os
import warnings

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

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = Path(PROJECT_ROOT, 'src', 'quality_control', 'wl_temizlenmis.csv')
OUT_CSV = os.path.join(BASE_DIR, "ramp_up_events_HAM.csv")
PLOT_DIR = os.path.join(BASE_DIR, "events")
CONTEXT_MIN = 75
EVENTS_PER_PAGE = 6
CALIBRATION_PERCENTILE = 80

MIN_PEAK_WS = 10.00
MIN_DELTA_V = 6.00
MAX_WS0 = 7.00

os.makedirs(PLOT_DIR, exist_ok=True)


def find_all_extrema(ws, n):
    is_ext = np.zeros(n, dtype=bool)
    if n >= 3:
        d = np.diff(ws)
        is_ext[1:-1] = (d[:-1] * d[1:]) < 0
    is_ext[0] = True
    is_ext[n - 1] = True
    return np.flatnonzero(is_ext)


def stage1_compress(ws, dts, extrema, lam, beta):
    if len(extrema) == 0:
        return np.array([], dtype=int)
    Z = [int(extrema[0])]
    for j in range(len(extrema) - 1):
        a, b = int(extrema[j]), int(extrema[j + 1])
        dP = ws[b] - ws[a]
        dt = (dts[b] - dts[a]) / np.timedelta64(1, "m")
        rate = (dP / dt) if dt > 0 else 0.0
        if abs(dP) > lam and abs(rate) > beta:
            Z.append(b)
    return np.array(Z, dtype=int)


def stage2_up_runs(ws, dts, Z, beta):
    runs = []
    cur_start = None
    cur_end = None
    for q in range(len(Z) - 1):
        a, b = int(Z[q]), int(Z[q + 1])
        dP = ws[b] - ws[a]
        dt = (dts[b] - dts[a]) / np.timedelta64(1, "m")
        rate = (dP / dt) if dt > 0 else 0.0
        if rate > beta:
            if cur_start is None:
                cur_start = a
            cur_end = b
        else:
            if cur_start is not None:
                runs.append((cur_start, cur_end))
                cur_start, cur_end = None, None
    if cur_start is not None:
        runs.append((cur_start, cur_end))
    return runs


# ---------------------------------------------------------------------------
# Load HAM (raw, unclean) data
# ---------------------------------------------------------------------------
print("Reading:", INPUT_CSV, "(SADECE ws_raw_kmh kullanilacak, temizlik UYGULANMAYACAK)")
df = pd.read_csv(INPUT_CSV, parse_dates=["datetime"])
df = df.rename(columns={"datetime": "dt"})
df["ws_temiz"] = df["ws_raw_kmh"] / 3.6  # ham deger, sadece birim cevrimi
df = df.sort_values("dt").reset_index(drop=True)
print(f"  {len(df):,} rows loaded.")

df["is_valid"] = df["ws_temiz"].notna() & df["wd_deg"].notna()
df["block_break"] = (df["dt"].diff() != pd.Timedelta(minutes=1)) | (~df["is_valid"])
df["block_id"] = df["block_break"].cumsum()

valid = df[df["is_valid"]]
block_sizes = valid.groupby("block_id").size()
valid_block_ids = block_sizes[block_sizes >= 30].index
valid_blocks = valid[valid["block_id"].isin(valid_block_ids)]
print(f"  Valid blocks (>= 30 min uninterrupted): {len(valid_block_ids)}")

# ---------------------------------------------------------------------------
# Calibrate lambda / beta from THIS (raw) dataset's own extrema differences
# ---------------------------------------------------------------------------
amp_diffs = []
rate_diffs = []
block_cache = {}

for bid, block in valid_blocks.groupby("block_id"):
    block = block.sort_values("dt")
    n = len(block)
    ws = block["ws_temiz"].to_numpy(dtype=float)
    dts = block["dt"].to_numpy()
    extrema = find_all_extrema(ws, n)
    block_cache[bid] = (ws, dts, extrema)
    for j in range(len(extrema) - 1):
        a, b = int(extrema[j]), int(extrema[j + 1])
        dP = ws[b] - ws[a]
        dt = (dts[b] - dts[a]) / np.timedelta64(1, "m")
        if dt > 0:
            amp_diffs.append(abs(dP))
            rate_diffs.append(abs(dP / dt))

LAMBDA = float(np.percentile(amp_diffs, CALIBRATION_PERCENTILE))
BETA = float(np.percentile(rate_diffs, CALIBRATION_PERCENTILE))
print(f"  Calibrated amplitude threshold (HAM veriden): {LAMBDA:.4f} m/s")
print(f"  Calibrated rate threshold (HAM veriden): {BETA:.4f} m/s/min")

# ---------------------------------------------------------------------------
# Run the two-stage mechanism on every block
# ---------------------------------------------------------------------------
all_events = []
total_tsp = total_sp = total_runs = 0

for bid, (ws, dts, extrema) in block_cache.items():
    Z = stage1_compress(ws, dts, extrema, LAMBDA, BETA)
    up_runs = stage2_up_runs(ws, dts, Z, BETA)

    total_tsp += len(extrema)
    total_sp += len(Z)
    total_runs += len(up_runs)

    ws_roll3 = pd.Series(ws).rolling(3, center=True, min_periods=1).mean().to_numpy()

    for (s_idx, e_idx) in up_runs:
        ws0 = float(ws[s_idx])
        ws1 = float(ws[e_idx])
        delta_v1 = ws1 - ws0
        duration = float((dts[e_idx] - dts[s_idx]) / np.timedelta64(1, "m"))

        if duration <= 0:
            continue
        if ws1 < MIN_PEAK_WS:
            continue
        if delta_v1 < MIN_DELTA_V:
            continue
        if ws0 > MAX_WS0:
            continue

        avg_slope = delta_v1 / duration
        seg_roll3 = ws_roll3[s_idx:e_idx + 1]
        delta_roll3 = float(seg_roll3[-1] - seg_roll3[0])
        path_roll3 = float(np.sum(np.abs(np.diff(seg_roll3))))
        smoothness = (delta_roll3 / path_roll3) if path_roll3 > 0 else 1.0

        all_events.append({
            "block_id": bid,
            "t0": pd.Timestamp(dts[s_idx]),
            "t1": pd.Timestamp(dts[e_idx]),
            "ws0": round(ws0, 3),
            "ws1": round(ws1, 3),
            "delta_v": round(delta_v1, 3),
            "duration_min": round(duration, 2),
            "avg_slope": round(avg_slope, 4),
            "smoothness": round(smoothness, 4),
        })

print(f"  Total extreme points: {total_tsp:,}")
print(f"  Slope points after stage 1: {total_sp:,}")
print(f"  Uphill runs after stage 2: {total_runs:,}")

events_df = pd.DataFrame(all_events)
if not events_df.empty:
    events_df = events_df.sort_values("t0").reset_index(drop=True)
    events_df.insert(0, "event_id", range(1, len(events_df) + 1))

events_df.to_csv(OUT_CSV, index=False)
print(f"  Ramp-up events identified (HAM VERI): {len(events_df)}")
print("  Written ->", OUT_CSV)

# ---------------------------------------------------------------------------
# Plotting - English labels, same style as the clean-data version
# ---------------------------------------------------------------------------
for _old in os.listdir(PLOT_DIR):
    if _old.endswith(".png"):
        os.remove(os.path.join(PLOT_DIR, _old))

df_indexed = df.set_index("dt")[["ws_temiz", "wd_deg"]]


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

    title = (
        f"HAM #{int(ev['event_id'])}  |  {t0:%Y-%m-%d}  |  "
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


n_all = len(events_df)
if n_all == 0:
    print("No events to plot.")
else:
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
    print(f"{n_pages} page(s) generated ({n_all} events) -> {PLOT_DIR}")

print("\nDone.")
