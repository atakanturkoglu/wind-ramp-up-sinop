"""
Ramp-up olaylari icin oncul (t-24..t-1 saat) matrislerin uretimi.

Tum istasyon verileri (merkez, inceburun, airport, wl) TEK kaynaktan,
birlesik_temiz.csv dosyasindan okunur. Ham tplink dosyalarindan ayrica
yeniden hesaplama YAPILMAZ.

Matris boyutu: (24 saat, 4 istasyon, 6 degisken)
  Istasyonlar : merkez, inceburun, airport, wl
  Degiskenler : P, T, u, v, ws, wd

Pencere: t0'dan once 24 saat (T-24 ... T-1), floor(t0) saatine hizali.
Eksik veri NaN olarak birakilir, sentetik doldurma YAPILMAZ.

Olay filtresi: yalnizca t0'i birlesik_temiz.csv'nin kapsadigi tarih
araliginda kalan ramp-up olaylari icin matris uretilir (df verileriyle
kesisen olaylar).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

STATIONS = ["merkez", "inceburun", "airport", "wl"]
VARS = ["P", "T", "u", "v", "ws", "wd"]
N_HOURS = 24
N_STATIONS = len(STATIONS)
N_VARS = len(VARS)

DISPLAY_STATION_ORDER = ["inceburun", "airport", "merkez", "wl"]
DISPLAY_STATION_LABELS = {"inceburun": "Inceburun", "airport": "Airport",
                           "merkez": "Sinop", "wl": "WL"}

# Olay listesi: ham (temizlenmemis) veri uzerinden tespit edilip elle
# incelenerek duzeltilmis 63 olayluk liste. Onceki surumde temiz veriden
# uretilen 46 olayluk liste kullaniliyordu; ham liste hem daha fazla
# gercek olay iceriyor hem de her olayi tek tek gozle dogrulanmis durumda.
EVENTS_PATH = Path(PROJECT_ROOT, 'src', 'ramp_detection', 'ramp_up_events_HAM.csv')
BIRLESIK_PATH = Path(PROJECT_ROOT, 'df veriler analiz', 'birlesik_temiz.csv')

OUT_DIR = Path(PROJECT_ROOT, 'src', 'feature_matrix')
SAMPLE_DIR = OUT_DIR / "ornekler"
os.makedirs(SAMPLE_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SAMPLE_PLOTS = 5

# ---------------------------------------------------------------------------
# 1. Girdi yukleme
# ---------------------------------------------------------------------------
print("[1/5] Girdi dosyalari yukleniyor...")

events = pd.read_csv(EVENTS_PATH, parse_dates=["t0", "t1"])
N_EVENTS_TOTAL = len(events)
print(f"  Ramp-up olay sayisi (tplink kaynakli): {N_EVENTS_TOTAL}")

birlesik = pd.read_csv(BIRLESIK_PATH, parse_dates=["datetime"])
birlesik = birlesik.drop_duplicates(subset="datetime", keep="first").set_index("datetime").sort_index()
birlesik_min, birlesik_max = birlesik.index.min(), birlesik.index.max()
print(f"  birlesik_temiz.csv kapsami: {birlesik_min} -> {birlesik_max} ({len(birlesik)} saat)")

# ---------------------------------------------------------------------------
# 2. 4 istasyon paneli (P,T,u,v,ws,wd) - hepsi birlesik_temiz.csv'den
# ---------------------------------------------------------------------------
print("[2/5] 4 istasyon icin saatlik panel kuruluyor (tek kaynak: birlesik_temiz.csv)...")

STATION_PREFIX = {"merkez": "merkez", "inceburun": "inceburun", "airport": "airport", "wl": "wl"}

station_frames = {}
for st in STATIONS:
    prefix = STATION_PREFIX[st]
    df_st = pd.DataFrame(index=birlesik.index)
    df_st["P"] = birlesik[f"{prefix}_basinc"]
    df_st["T"] = birlesik[f"{prefix}_sicaklik"]
    df_st["ws"] = birlesik[f"{prefix}_ws"]
    df_st["wd"] = birlesik[f"{prefix}_wd"]
    df_st["u"] = -df_st["ws"] * np.sin(np.radians(df_st["wd"]))
    df_st["v"] = -df_st["ws"] * np.cos(np.radians(df_st["wd"]))
    station_frames[st] = df_st[VARS]
    n_valid_ws = df_st["ws"].notna().sum()
    print(f"    {st:10s}: ws dolu saat = {n_valid_ws} ({df_st.index.min()} -> "
          f"{df_st['ws'].last_valid_index()})")

# ---------------------------------------------------------------------------
# 3. Yardimci fonksiyonlar
# ---------------------------------------------------------------------------


def build_window_matrix(t0):
    floor_t0 = pd.Timestamp(t0).floor("h")
    t_minus_1 = floor_t0 - pd.Timedelta(hours=1)
    t_minus_24 = floor_t0 - pd.Timedelta(hours=24)
    window_hours = pd.date_range(t_minus_24, t_minus_1, freq="h")
    assert len(window_hours) == N_HOURS

    matrix = np.full((N_HOURS, N_STATIONS, N_VARS), np.nan, dtype=np.float32)
    for si, st in enumerate(STATIONS):
        sub = station_frames[st].reindex(window_hours)[VARS]
        matrix[:, si, :] = sub.to_numpy(dtype=np.float32)
    return matrix, window_hours


def doluluk_metrikleri(matrix):
    notna = ~np.isnan(matrix)
    d = {"doluluk_toplam": notna.sum() / (N_HOURS * N_STATIONS * N_VARS)}
    for si, st in enumerate(STATIONS):
        d[f"doluluk_{st}"] = notna[:, si, :].mean()
    for vi, v in enumerate(VARS):
        d[f"doluluk_{v}"] = notna[:, :, vi].mean()
    return d


# ---------------------------------------------------------------------------
# 4. Olaylar icin matris uretimi (yalnizca birlesik_temiz araligiyla kesisenler)
# ---------------------------------------------------------------------------
print("[3/5] Ramp-up olaylari icin matrisler uretiliyor...")

pos_X, pos_M, pos_meta_rows, skipped_rows = [], [], [], []

for _, ev in events.iterrows():
    t0 = ev["t0"]
    if pd.isna(t0) or t0 < birlesik_min or t0 > birlesik_max:
        skipped_rows.append({"event_id": ev["event_id"], "t0": t0,
                              "gerekce": "df verileri tarih araligi disinda"})
        continue

    matrix, window_hours = build_window_matrix(t0)
    M = (~np.isnan(matrix)).astype(np.uint8)
    dol = doluluk_metrikleri(matrix)

    pos_X.append(matrix)
    pos_M.append(M)
    row = {
        "event_id": ev["event_id"], "block_id": ev["block_id"],
        "t0": t0, "t1": ev["t1"],
        "ws0": ev["ws0"], "ws1": ev["ws1"], "delta_v": ev["delta_v"],
        "duration_min": ev["duration_min"], "avg_slope": ev["avg_slope"],
        "smoothness": ev["smoothness"],
        "pencere_bas": window_hours[0], "pencere_son": window_hours[-1],
    }
    row.update(dol)
    pos_meta_rows.append(row)

N_POZ = len(pos_X)
X = np.stack(pos_X, axis=0).astype(np.float32) if N_POZ else np.zeros((0, N_HOURS, N_STATIONS, N_VARS), np.float32)
M = np.stack(pos_M, axis=0).astype(np.uint8) if N_POZ else np.zeros((0, N_HOURS, N_STATIONS, N_VARS), np.uint8)
meta = pd.DataFrame(pos_meta_rows)
skipped = pd.DataFrame(skipped_rows, columns=["event_id", "t0", "gerekce"])

print(f"  Matris uretilen olay sayisi : {N_POZ} / {N_EVENTS_TOTAL}")
print(f"  df tarih araligi disinda kalan (atlanan) olay sayisi: {len(skipped)}")

# ---------------------------------------------------------------------------
# 5. Ciktilarin kaydedilmesi
# ---------------------------------------------------------------------------
print("[4/5] Ciktilar kaydediliyor ->", OUT_DIR)

np.save(OUT_DIR / "X.npy", X)
np.save(OUT_DIR / "M.npy", M)
meta.to_csv(OUT_DIR / "meta.csv", index=False)
skipped.to_csv(OUT_DIR / "atlanan_olaylar.csv", index=False)

thresholds = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
rapor_rows = []
for th in thresholds:
    n_th = int((meta["doluluk_toplam"] >= th).sum()) if N_POZ else 0
    rapor_rows.append({"esik": th, "olay_sayisi": n_th,
                        "yuzde": round(100 * n_th / N_POZ, 2) if N_POZ else 0.0})
pd.DataFrame(rapor_rows).to_csv(OUT_DIR / "doluluk_raporu.csv", index=False)

with open(OUT_DIR / "matris_ozet.txt", "w", encoding="utf-8") as f:
    f.write("RAMP-UP MATRIS OZET RAPORU\n")
    f.write("=" * 30 + "\n\n")
    f.write(f"TOPLAM RAMP-UP OLAYI (tplink kaynakli): {N_EVENTS_TOTAL}\n")
    f.write(f"URETILEN MATRIS (df ile kesisen)      : {N_POZ}\n")
    f.write(f"ATLANAN OLAY (df tarih araligi disinda): {len(skipped)}\n\n")
    f.write("MATRIS BOYUTU: (24 saat, 4 istasyon, 6 degisken)\n")
    f.write("  Istasyonlar: merkez, inceburun, airport, wl\n")
    f.write("  Degiskenler: P, T, u, v, ws, wd\n\n")
    f.write("VERI KAYNAGI: birlesik_temiz.csv (4 istasyonun tamami, tek kaynak)\n")
    f.write("PENCERE: t0 oncesi 24 saat (T-24...T-1), floor(t0) saatine hizali.\n")
    f.write("Eksik veri NaN birakilir, sentetik doldurma yoktur.\n")

# ---------------------------------------------------------------------------
# Ornek matris gorsellestirme (estetik, okunakli versiyon)
# ---------------------------------------------------------------------------
print("[5/5] Ornek matris gorselleri olusturuluyor...")


def render_event_png(event_id, matrix, window_hours, delta_v, duration_min, out_path):
    norms = []
    for vi in range(N_VARS):
        vals = matrix[:, :, vi]
        finite = vals[~np.isnan(vals)]
        if finite.size == 0:
            norms.append((0.0, 1.0))
        else:
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5
            norms.append((vmin, vmax))
    cmap = plt.get_cmap("viridis")

    display_labels = [DISPLAY_STATION_LABELS[s] for s in DISPLAY_STATION_ORDER]
    display_idx = [STATIONS.index(s) for s in DISPLAY_STATION_ORDER]

    fig, axes = plt.subplots(4, 6, figsize=(32, 22))
    axes = axes.flatten()

    for h in range(N_HOURS):
        ax = axes[h]
        cell_rgba = np.ones((N_VARS, len(DISPLAY_STATION_ORDER), 4))
        for vi in range(N_VARS):
            vmin, vmax = norms[vi]
            for ci, si in enumerate(display_idx):
                val = matrix[h, si, vi]
                if np.isnan(val):
                    cell_rgba[vi, ci] = (0.90, 0.90, 0.90, 1.0)
                else:
                    nv = 0.5 if vmax <= vmin else min(max((val - vmin) / (vmax - vmin), 0.0), 1.0)
                    cell_rgba[vi, ci] = cmap(nv)
        ax.imshow(cell_rgba, aspect="auto")
        ax.set_xticks(range(len(DISPLAY_STATION_ORDER)))
        ax.set_xticklabels(display_labels, fontsize=9, rotation=25)
        ax.set_yticks(range(N_VARS))
        ax.set_yticklabels(VARS, fontsize=10)
        for vi in range(N_VARS):
            for ci, si in enumerate(display_idx):
                val = matrix[h, si, vi]
                if np.isnan(val):
                    ax.text(ci, vi, "\u2014", ha="center", va="center", fontsize=9, color="#555555")
                    continue
                r, g, b, _ = cell_rgba[vi, ci]
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "white" if luminance < 0.55 else "black"
                ax.text(ci, vi, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="medium")
        for edge in ("top", "right", "left", "bottom"):
            ax.spines[edge].set_visible(True)
            ax.spines[edge].set_color("white")
            ax.spines[edge].set_linewidth(1.5)
        ax.set_xticks(np.arange(-0.5, len(DISPLAY_STATION_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N_VARS, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0)
        step_name = f"T-{N_HOURS - h}"
        label = pd.Timestamp(window_hours[h]).strftime("%d %b %H:%M")
        ax.set_title(f"{step_name}   ({label})", fontsize=10.5, pad=6)

    fig.suptitle(
        f"Ramp-up Event {event_id}  \u2014  24-Hour Pre-Event Matrix   "
        f"(\u0394v = +{delta_v:.1f} m/s, duration = {duration_min:.0f} min)",
        fontsize=17, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)


if N_POZ > 0:
    for pos in range(N_POZ):
        ev_id = int(meta.iloc[pos]["event_id"])
        t0 = meta.iloc[pos]["t0"]
        floor_t0 = pd.Timestamp(t0).floor("h")
        window_hours = pd.date_range(floor_t0 - pd.Timedelta(hours=N_HOURS),
                                      floor_t0 - pd.Timedelta(hours=1), freq="h")
        out_path = SAMPLE_DIR / f"event_{ev_id:03d}_matrix.png"
        render_event_png(ev_id, X[pos], window_hours,
                          meta.iloc[pos]["delta_v"], meta.iloc[pos]["duration_min"],
                          out_path)
        print(f"  Kaydedildi: {out_path.name}")

print("\nTamamlandi.")
print(f"Cikti klasoru: {OUT_DIR}")
