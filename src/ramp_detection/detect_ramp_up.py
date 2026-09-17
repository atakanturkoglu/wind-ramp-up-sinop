"""
Ramp-up detection for the cleaned WL minute-level wind speed series.

Algorithm: Kuang et al. (2020, IEEE Access), "A New Definition Method of
Wind Power Ramp Sections", ported to wind-speed units. This is the same
two-stage extreme-point / slope-point mechanism reviewed and approved for
this project (see staj/ramp-up/ramp/t0_kuang_hibrit.py):

  1) Extreme point (TSP) extraction (Eq. 3): a point is a local extremum if
     the sign of the difference changes around it. Block start/end points
     are also included.
  2) Stage 1 - Y -> Z compression (Eq. 12): for every consecutive TSP pair,
     if BOTH the amplitude change (> lambda) AND the rate of change (> beta)
     exceed threshold, the point is kept as a slope point (SP); otherwise it
     is a stationary point (STP) and is dropped.
  3) Stage 2 - direction decision on Z (Eq. 14): for every consecutive SP
     pair, rate > +beta marks an uphill step, rate < -beta a downhill step,
     otherwise a flat/transition step. A ramp-up section is the maximal run
     of consecutive uphill steps. No beta_max ceiling is applied: in Kuang's
     paper beta_max is a power-grid interconnection limit tied to installed
     capacity, which has no counterpart for a single wind-speed sensor.

lambda and beta are calibrated from this dataset's own distribution of
consecutive-extreme-point differences (80th percentile), exactly as in
Kuang et al. Three additional thresholds are applied on top of the Kuang
mechanism, carried over unchanged from the reviewed script as explicit
engineering decisions (not derived from the literature):
  MIN_PEAK_WS = 10.00 m/s, MIN_DELTA_V = 6.00 m/s, MAX_WS0 = 7.00 m/s.
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
OUT_CSV = os.path.join(BASE_DIR, "ramp_up_events.csv")
PLOT_DIR = os.path.join(BASE_DIR, "events")
CONTEXT_MIN = 75
EVENTS_PER_PAGE = 6
CALIBRATION_PERCENTILE = 80

# Engineering decisions, carried over unchanged from the reviewed script.
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
# Load data and build valid blocks
# ---------------------------------------------------------------------------
print("Reading:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV, parse_dates=["datetime"])
df = df.rename(columns={"datetime": "dt", "ws_clean_ms": "ws_temiz"})
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
# Calibrate lambda / beta from this dataset's own extreme-point differences
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
print(f"  Calibrated amplitude threshold: {LAMBDA:.4f} m/s")
print(f"  Calibrated rate threshold: {BETA:.4f} m/s/min")

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
print(f"  Ramp-up events identified: {len(events_df)}")
print("  Written ->", OUT_CSV)

# ---------------------------------------------------------------------------
# Plotting - English labels only, publication style
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
        f"Event {int(ev['event_id'])}  |  {t0:%Y-%m-%d}  |  "
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
