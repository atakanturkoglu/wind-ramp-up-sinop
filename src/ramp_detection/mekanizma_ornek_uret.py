"""
5.2 bolumu icin metodoloji seklidi: Kuang vd. (2020) iki asamali
mekanizmasinin (uc nokta cikarimi -> Asama 1 sikistirma -> Asama 2 yon
karari) 5 Kasim 2023 olayi (ramp_up-son/ramp_up_events_HAM.csv'de
event_id=4, block_id=57) uzerinde adim adim gorsellestirilmesi. Bu olay,
smoothness=1.0 degeriyle (tamamen monotonik yukselis) ve t0'in raw
seride de gercek yerel minimuma denk gelmesiyle, mekanizmayi "temiz"
(karisik ara sicramalar olmadan) gosteren bir ornek olarak secilmistir.

GUNCELLEME: Tespit artik ws_clean_ms (temizlenmis) degil, ws_raw_kmh/3.6
(HAM, temizlik uygulanmamis) seri uzerinde calisiyor - bkz.
ramp_up-son/detect_ramp_up_ham.py. Bu script o dosyayla BIREBIR ayni
blok ayirma ve kalibrasyon mantigini kullanir (kopyalanmadi, ayni mantik
yeniden calistirildi); burada hicbir yeni esik veya kural tanimlanmaz,
sadece ayni tespitin ic adimlari gorunur kilinir.
"""

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

INPUT_CSV = Path(PROJECT_ROOT, 'src', 'quality_control', 'wl_temizlenmis.csv')
OUT_PATH = Path(PROJECT_ROOT, 'src', 'ramp_detection', 'mekanizma_ornek_2023_11_05.png')
CALIBRATION_PERCENTILE = 80
TARGET_BLOCK_ID = 57
TARGET_T0 = pd.Timestamp("2023-11-05 03:30:00")
CONTEXT_MIN = 75


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
# Veri yukleme ve blok ayirma (detect_ramp_up.py ile birebir ayni)
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV, parse_dates=["datetime"])
df = df.rename(columns={"datetime": "dt"})
df["ws_temiz"] = df["ws_raw_kmh"] / 3.6  # ham deger, sadece birim cevrimi
df = df.sort_values("dt").reset_index(drop=True)

df["is_valid"] = df["ws_temiz"].notna() & df["wd_deg"].notna()
df["block_break"] = (df["dt"].diff() != pd.Timedelta(minutes=1)) | (~df["is_valid"])
df["block_id"] = df["block_break"].cumsum()

valid = df[df["is_valid"]]
block_sizes = valid.groupby("block_id").size()
valid_block_ids = block_sizes[block_sizes >= 30].index
valid_blocks = valid[valid["block_id"].isin(valid_block_ids)]

# ---------------------------------------------------------------------------
# lambda / beta kalibrasyonu (tum veri setinden, detect_ramp_up.py ile ayni)
# ---------------------------------------------------------------------------
amp_diffs, rate_diffs = [], []
for bid, block in valid_blocks.groupby("block_id"):
    block = block.sort_values("dt")
    ws = block["ws_temiz"].to_numpy(dtype=float)
    dts = block["dt"].to_numpy()
    extrema = find_all_extrema(ws, len(block))
    for j in range(len(extrema) - 1):
        a, b = int(extrema[j]), int(extrema[j + 1])
        dP = ws[b] - ws[a]
        dt = (dts[b] - dts[a]) / np.timedelta64(1, "m")
        if dt > 0:
            amp_diffs.append(abs(dP))
            rate_diffs.append(abs(dP / dt))

LAMBDA = float(np.percentile(amp_diffs, CALIBRATION_PERCENTILE))
BETA = float(np.percentile(rate_diffs, CALIBRATION_PERCENTILE))
print(f"LAMBDA={LAMBDA:.4f} m/s, BETA={BETA:.4f} m/s/min")

# ---------------------------------------------------------------------------
# Hedef blok uzerinde mekanizmayi calistir
# ---------------------------------------------------------------------------
block = valid_blocks[valid_blocks["block_id"] == TARGET_BLOCK_ID].sort_values("dt")
ws = block["ws_temiz"].to_numpy(dtype=float)
dts = block["dt"].to_numpy()
dt_index = pd.DatetimeIndex(dts)

extrema = find_all_extrema(ws, len(block))
Z = stage1_compress(ws, dts, extrema, LAMBDA, BETA)
up_runs = stage2_up_runs(ws, dts, Z, BETA)

# Bu blokta hedef olayi (TARGET_T0'dan baslayan yukselis blogu) sec
target_run = None
for (s_idx, e_idx) in up_runs:
    if pd.Timestamp(dts[s_idx]) == TARGET_T0:
        target_run = (s_idx, e_idx)
        break
assert target_run is not None, "Hedef olay bulunamadi"
s_idx, e_idx = target_run
t0, t1 = dt_index[s_idx], dt_index[e_idx]
ws0, ws1 = ws[s_idx], ws[e_idx]
print(f"Olay: {t0} -> {t1}, {ws0:.3f} -> {ws1:.3f} m/s")

# Gorsellestirme penceresi (context +-75 dk, detect_ramp_up.py ile ayni)
win_start = t0 - pd.Timedelta(minutes=CONTEXT_MIN)
win_end = t1 + pd.Timedelta(minutes=CONTEXT_MIN)
mask = (dt_index >= win_start) & (dt_index <= win_end)
win_idx = np.flatnonzero(mask)

extrema_in_win = np.array([i for i in extrema if i in win_idx])
Z_in_win = np.array([i for i in Z if i in win_idx])

# NOT: bu gorsel docx'e ~6.5-7 inc genislikte (JRAS tek sutun genisligi)
# gomulecek; 13 inclik tasarim genisligine gore ~yarim olcekte basilacagi
# icin tum yazi/cizgi/marker boyutlari bu kucultmeyi tolere edecek kadar
# (~1.8x) buyutulmustur - aksi halde eksen/legend yazilari okunmaz kaliyordu.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 18

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)

# --- Panel (a): extreme point (TSP) extraction and Stage-1 compression ----
axA.plot(dt_index[win_idx], ws[win_idx], color="#c9c9c9", lw=1.6, zorder=1)
axA.scatter(dt_index[extrema_in_win], ws[extrema_in_win], s=32, color="#9e9e9e",
            zorder=2, label=f"Extreme point (TSP), n={len(extrema_in_win)}")
axA.scatter(dt_index[Z_in_win], ws[Z_in_win], s=75, color="#1f4e79", marker="D",
            zorder=3, label=f"Slope point (SP), n={len(Z_in_win)}")
axA.set_title("(a) Turning Points and Slope-Point Filtering", fontsize=17)
axA.set_ylabel("Wind speed (m/s)", fontsize=18)
axA.legend(loc="upper left", fontsize=15, framealpha=0.9)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)
axA.tick_params(axis="both", labelsize=16)

# --- Panel (b): Stage 2 direction decision and detected ramp-up event -----
axB.plot(dt_index[win_idx], ws[win_idx], color="#2b7bba", lw=1.9, zorder=1)
axB.axvspan(t0, t1, color="#e85d5d", alpha=0.15, zorder=0)
seg_mask = (dt_index[win_idx] >= t0) & (dt_index[win_idx] <= t1)
axB.plot(dt_index[win_idx][seg_mask], ws[win_idx][seg_mask], color="#d62728", lw=3.2, zorder=2)
axB.scatter([t0], [ws0], color="#2ca02c", s=130, zorder=5, label=f"t0: {ws0:.2f} m/s")
axB.scatter([t1], [ws1], color="#9467bd", s=130, zorder=5, label=f"t1: {ws1:.2f} m/s")
axB.annotate(f"+{ws1 - ws0:.2f} m/s\n({(t1 - t0).seconds // 60} min)",
             xy=(t0 + (t1 - t0) / 2, (ws0 + ws1) / 2), xytext=(0.62, 0.30),
             textcoords="axes fraction", fontsize=17, ha="center", color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6))
axB.set_title("(b) Detected Ramp-Up Event", fontsize=17)
axB.legend(loc="upper left", fontsize=15, framealpha=0.9)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.tick_params(axis="both", labelsize=16)

for ax in (axA, axB):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(20)
        lbl.set_ha("right")
    ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.subplots_adjust(wspace=0.15, top=0.90)
fig.savefig(OUT_PATH, dpi=300)
plt.close(fig)
print("Kaydedildi ->", OUT_PATH)
