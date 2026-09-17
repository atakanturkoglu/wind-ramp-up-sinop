"""
Kayan 31-dakikalik pencere (15 sol + 1 merkez + 15 sag) standart sapma
analizi - kendi FINAL temizlenmis verimiz uzerinde (wl_temizlenmis.csv,
ws_clean_ms sutunu, m/s). Yontem, staj/proje/std_analizi.py'deki kayan
pencere mantigiyla BIREBIR aynidir (kopyalanmadi, ayni mantik yeniden
yazildi); tek fark girdi kaynagi (bizim final temiz veri) ve birimdir
(km/h yerine m/s, projenin geri kalaniyla tutarli olsun diye).

Amac: her rüzgar hizi seviyesinde (m/s, tam sayiya yuvarlanmis), o
seviyenin etrafinda dakikalik olcumlerin standart sapmasinin ne kadar
oldugunu olcup bir referans tablosu (std_ozet.csv) uretmek. Bu tablo,
ileride CNN-LSTM tahminlerini bu standart sapma ile karsilastiran
bir HIT/MISS degerlendirmesinde kullanilacaktir.

Cikti klasoru: src/std_analysis/
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from numpy.lib.stride_tricks import sliding_window_view

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

INPUT_CSV = Path(PROJECT_ROOT, 'src', 'quality_control', 'wl_temizlenmis.csv')
OUT_DIR = Path(PROJECT_ROOT, 'src', 'std_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW = 31
HALF = 15
MIN_VALID = 30          # 31 dakikanin en az ~%97'si dolu olmali
MIN_GRUP_GUVENILIRLIK = 30
CHUNK_SIZE = 200_000

COLOR_GENERAL = "#1f4e79"
COLOR_LEFT = "#2a78d6"
COLOR_RIGHT = "#eb6834"
COLOR_TREND = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_TEXT = "#0b0b0b"
COLOR_SECONDARY = "#52514e"
COLOR_SURFACE = "#fcfcfb"

plt.rcParams["font.family"] = "sans-serif"

# ---------------------------------------------------------------------------
# 1. Veri yukleme (kendi final temiz verimiz, m/s)
# ---------------------------------------------------------------------------
print("[1/7] Loading:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV, usecols=["datetime", "ws_clean_ms"], parse_dates=["datetime"])
df = df.sort_values("datetime").drop_duplicates(subset="datetime").reset_index(drop=True)
print(f"      Rows: {len(df):,}  |  Range: {df['datetime'].min()} -> {df['datetime'].max()}")

full_index = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="1min")
s = df.set_index("datetime")["ws_clean_ms"].reindex(full_index)
print(f"[2/7] Grid length: {len(s):,}  |  Valid: {s.notna().sum():,}  |  NaN: {s.isna().sum():,}")

# ---------------------------------------------------------------------------
# 2. Kayan 31 dakikalik paket hesaplamasi
# ---------------------------------------------------------------------------
print("[3/7] Computing sliding 31-minute packets (15 left + 1 center + 15 right)...")
arr = s.to_numpy(dtype=np.float64)
n = len(arr)
n_centers = n - 2 * HALF

merkez_deger = np.full(n_centers, np.nan)
sol_std = np.full(n_centers, np.nan)
sag_std = np.full(n_centers, np.nan)
genel_std = np.full(n_centers, np.nan)
gecerli = np.zeros(n_centers, dtype=bool)

windows = sliding_window_view(arr, WINDOW)
center_vals = arr[HALF: n - HALF]

with np.errstate(invalid="ignore"):
    for start in range(0, n_centers, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_centers)
        w = windows[start:end]
        cvals = center_vals[start:end]

        valid_count = np.sum(~np.isnan(w), axis=1)
        mask = (valid_count >= MIN_VALID) & (~np.isnan(cvals))

        sol_w = w[:, :HALF]
        sag_w = w[:, HALF + 1:]

        sol_std[start:end] = np.nanstd(sol_w, axis=1, ddof=1)
        sag_std[start:end] = np.nanstd(sag_w, axis=1, ddof=1)
        genel_std[start:end] = np.nanstd(w, axis=1, ddof=1)
        merkez_deger[start:end] = cvals
        gecerli[start:end] = mask

paket_df = pd.DataFrame({
    "center_time": s.index[HALF: n - HALF],
    "center_value": merkez_deger,
    "left_std": sol_std,
    "right_std": sag_std,
    "packet_general_std": genel_std,
})
paket_df = paket_df.loc[gecerli].reset_index(drop=True)
print(f"      Valid packets: {len(paket_df):,}")

# ---------------------------------------------------------------------------
# 3. Merkez degerler ozeti
# ---------------------------------------------------------------------------
print("[4/7] Summarizing center-value series...")
seri = paket_df["center_value"]
ozet = {
    "count": int(seri.count()),
    "total": float(seri.sum()),
    "mean": float(seri.mean()),
    "std": float(seri.std(ddof=1)),
}
with open(os.path.join(OUT_DIR, "center_values_summary.txt"), "w", encoding="utf-8") as f:
    f.write("CENTER VALUE SERIES SUMMARY\n")
    f.write("=" * 40 + "\n")
    f.write(f"Valid packet (center value) count : {ozet['count']}\n")
    f.write(f"Total (m/s)                        : {ozet['total']:.4f}\n")
    f.write(f"Mean (m/s)                         : {ozet['mean']:.4f}\n")
    f.write(f"Standard deviation                 : {ozet['std']:.4f}\n")

# ---------------------------------------------------------------------------
# 4. m/s bazli gruplama (tam sayiya yuvarlanmis merkez degere gore)
# ---------------------------------------------------------------------------
print("[5/7] Grouping by rounded wind speed (m/s)...")


def kmh_gruplama(paket_df):
    d = paket_df.copy()
    d["ms_grup"] = d["center_value"].round().astype(int)
    grup = d.groupby("ms_grup").agg(
        left_std=("left_std", "mean"),
        right_std=("right_std", "mean"),
        general_std=("packet_general_std", "mean"),
        packet_count=("center_value", "size"),
    ).reset_index().rename(columns={"ms_grup": "ws_ms"})
    grup["left_right_avg_std"] = (grup["left_std"] + grup["right_std"]) / 2
    grup = grup[["ws_ms", "left_std", "right_std", "left_right_avg_std", "general_std", "packet_count"]]
    return grup.sort_values("ws_ms").reset_index(drop=True)


def dagilim(paket_df, kolon):
    d = paket_df[["center_value", kolon]].copy()
    d["ms_grup"] = d["center_value"].round().astype(int)
    kayit = []
    for grup, alt in d.groupby("ms_grup"):
        vals = alt[kolon].dropna().to_numpy()
        if len(vals) == 0:
            continue
        kayit.append({
            "ws_ms": int(grup), "min": float(np.min(vals)), "p25": float(np.percentile(vals, 25)),
            "median": float(np.percentile(vals, 50)), "mean": float(np.mean(vals)),
            "p75": float(np.percentile(vals, 75)), "p95": float(np.percentile(vals, 95)),
            "max": float(np.max(vals)), "packet_count": len(vals),
        })
    return pd.DataFrame(kayit).sort_values("ws_ms").reset_index(drop=True)


def guvenilir(d):
    return d[d["packet_count"] >= MIN_GRUP_GUVENILIRLIK].reset_index(drop=True)


grup_df = guvenilir(kmh_gruplama(paket_df))
grup_df.to_csv(os.path.join(OUT_DIR, "std_ozet.csv"), index=False, float_format="%.4f")
print(f"      std_ozet.csv written ({len(grup_df)} reliable ws groups)")

dagilim_left = guvenilir(dagilim(paket_df, "left_std"))
dagilim_right = guvenilir(dagilim(paket_df, "right_std"))
dagilim_general = guvenilir(dagilim(paket_df, "packet_general_std"))

dagilim_left.to_csv(os.path.join(OUT_DIR, "std_distribution_left.csv"), index=False, float_format="%.4f")
dagilim_right.to_csv(os.path.join(OUT_DIR, "std_distribution_right.csv"), index=False, float_format="%.4f")
dagilim_general.to_csv(os.path.join(OUT_DIR, "std_distribution_general.csv"), index=False, float_format="%.4f")
print("      distribution tables saved (std_distribution_*.csv)")


# ---------------------------------------------------------------------------
# 5. Gorsellestirme yardimcilari
# ---------------------------------------------------------------------------
def _clean_axes(ax):
    ax.set_facecolor(COLOR_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_SECONDARY)
    ax.tick_params(colors=COLOR_SECONDARY)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _draw_box(ax, x, row, color, width=0.6):
    half = width / 2
    whisker_half = width * 0.55 / 2
    p95_half = width * 0.40 / 2
    mean_half = width * 0.42 / 2
    ax.add_line(Line2D([x, x], [row["min"], row["max"]], color=color, linewidth=1.2, zorder=2))
    ax.add_line(Line2D([x - whisker_half, x + whisker_half], [row["min"]] * 2, color=color, linewidth=1.8, zorder=2))
    ax.add_line(Line2D([x - whisker_half, x + whisker_half], [row["max"]] * 2, color=color, linewidth=1.8, zorder=2))
    ax.add_line(Line2D([x - p95_half, x + p95_half], [row["p95"]] * 2, color=color, linewidth=1.3, alpha=0.85, zorder=2))
    rect = Rectangle((x - half, row["p25"]), width, row["p75"] - row["p25"],
                      facecolor=color, edgecolor=color, alpha=0.30, linewidth=1.3, zorder=1)
    ax.add_patch(rect)
    ax.add_line(Line2D([x - half, x + half], [row["median"]] * 2, color=color, linewidth=2.2, zorder=3))
    ax.add_line(Line2D([x - mean_half, x + mean_half], [row["mean"]] * 2, color=color, linewidth=1.1, zorder=3))


def _add_legend_note(ax):
    text = ("Box: 25th-75th percentile\nThick line: median\nThin line: mean\n"
            "Short tick: 95th percentile\nWhisker ends: min / max")
    ax.text(0.01, 0.98, text, transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            color=COLOR_SECONDARY, bbox=dict(boxstyle="round,pad=0.4", facecolor=COLOR_SURFACE, edgecolor=COLOR_GRID))


def _standalone_boxplot(dist_df, color, ylabel, title, out_name):
    fig, ax = plt.subplots(figsize=(20, 8), facecolor=COLOR_SURFACE)
    x = np.arange(len(dist_df))
    for i, row in dist_df.iterrows():
        _draw_box(ax, x[i], row, color, width=0.6)
    step = max(1, len(dist_df) // 34)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(dist_df["ws_ms"].iloc[::step], color=COLOR_TEXT, fontsize=9)
    ax.set_xlim(-1, len(dist_df))
    y_min, y_max = dist_df["min"].min(), dist_df["max"].max()
    pad = (y_max - y_min) * 0.05
    ax.set_ylim(max(0, y_min - pad), y_max + pad)
    ax.set_xlabel("Wind speed group (m/s)", color=COLOR_TEXT)
    ax.set_ylabel(ylabel, color=COLOR_TEXT)
    ax.set_title(title, color=COLOR_TEXT, fontsize=13, fontweight="bold")
    _add_legend_note(ax)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=300, facecolor=COLOR_SURFACE)
    plt.close(fig)
    print("      saved:", out_name)


print("[6/7] Rendering standalone boxplots (general / left / right)...")
_standalone_boxplot(dagilim_general, COLOR_GENERAL, "Packet general std (m/s)",
                     "Distribution of 31-Minute Packet General Std by Wind Speed",
                     "std_boxplot_general.png")
_standalone_boxplot(dagilim_left, COLOR_LEFT, "Left-neighbor std (m/s)",
                     "Distribution of Left-Neighbor (15 min before) Std by Wind Speed",
                     "std_boxplot_left.png")
_standalone_boxplot(dagilim_right, COLOR_RIGHT, "Right-neighbor std (m/s)",
                     "Distribution of Right-Neighbor (15 min after) Std by Wind Speed",
                     "std_boxplot_right.png")

# ---------------------------------------------------------------------------
# 6. Ruzgar hizi - std iliskisi (dogrusal trend)
# ---------------------------------------------------------------------------
print("[7/7] Rendering wind-speed vs std trend chart...")
x = grup_df["ws_ms"].to_numpy(dtype=float)
y = grup_df["general_std"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
r = np.corrcoef(x, y)[0, 1]
trend = slope * x + intercept

fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=COLOR_SURFACE)
ax.plot(x, y, color=COLOR_GENERAL, linewidth=2.2, marker="o", markersize=4,
        label="Mean packet general std", zorder=3)
ax.plot(x, trend, color=COLOR_TREND, linewidth=2, linestyle="--",
        label=f"Linear trend (slope={slope:.4f}, r={r:.4f})", zorder=2)
ax.set_xlabel("Wind speed (m/s)", color=COLOR_TEXT)
ax.set_ylabel("Mean standard deviation (m/s)", color=COLOR_TEXT)
ax.set_title("Relationship Between Wind Speed and Local Standard Deviation",
             color=COLOR_TEXT, fontsize=13, fontweight="bold")
ax.legend(frameon=False, labelcolor=COLOR_TEXT, loc="upper left")
_clean_axes(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "std_vs_windspeed_trend.png"), dpi=300, facecolor=COLOR_SURFACE)
plt.close(fig)

print(f"\nDone. slope={slope:.4f}, intercept={intercept:.4f}, r={r:.4f}")
print("Outputs ->", OUT_DIR)
