# -*- coding: utf-8 -*-
"""
HAM veri listesinde manuel ws0 / ws1 duzeltmeleri.

Kuang algoritmasinin 1. asamasi (gurultu filtreleme), bazi olaylarda:
  (a) gercek firlamadan hemen once kucuk bir tumsek+cukur salinimi varsa
      bu salinimi "onemsiz" sayip atliyor -> ws0 gercek dipten degil,
      tumsegin tepesinden seciliyor;
  (b) yukselisin tepesi birkac dakika ayni seviyede platoya oturuyorsa,
      ws1 olarak platonun ILK degil daha sonraki bir noktasi secilebiliyor
      -> sure gereksiz uzuyor, ortalama egim dusuk gorunuyor.

Bu script, gozle incelenip onaylanmis olaylarda t0/t1'i gercek noktalara
elle tasir ve tum turev degerleri (delta_v, duration, avg_slope,
smoothness) yeniden hesaplar.

Calistirma sirasi:
  detect_ramp_up_ham.py -> manuel_eklemeler.py -> BU SCRIPT -> replot_ham.py
"""

import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

EVENTS_CSV = Path(PROJECT_ROOT, 'src', 'ramp_detection', 'ramp_up_events_HAM.csv')
CLEAN_CSV = Path(PROJECT_ROOT, 'src', 'quality_control', 'wl_temizlenmis.csv')

# (orijinal_t0, orijinal_t1, yeni_t0, yeni_t1, sebep)
# yeni_t0 / yeni_t1 None ise o alan degistirilmez.
MANUAL_FIXES = [
    (
        "2023-12-23 18:11:00", "2023-12-23 18:17:00",
        "2023-12-23 18:13:00", None,
        "Onceki t0=18:11 (ws0=6.64) aslinda ufak, onemsiz bir yerel tepe; "
        "gercek dip 18:13'te (5.92). Aradaki 0.72 m/s'lik inis Kuang 1. "
        "asamasinin lambda esigine takilmadigi icin atlanmis. "
        "(Temiz listedeki Event 11 ile ayni duzeltme.)",
    ),
    (
        "2024-04-20 16:28:00", "2024-04-20 16:36:00",
        "2024-04-20 16:34:00", None,
        "Onceki t0=16:28 (5.31) ile gercek firlama (16:34, 5.33) arasinda "
        "16:30'a kadar yukselip geri donen bir tumsek var; ws0 bu tumsegin "
        "oncesinde kalmis. Gercek firlamanin basladigi son dip 16:34. "
        "(Temiz listedeki Event 16 ile ayni duzeltme.)",
    ),
    (
        "2024-04-26 16:56:00", "2024-04-26 17:00:00",
        None, "2024-04-26 16:58:00",
        "Yukselis 16:58'de zirveye ulasip (12.22 m/s) platoya oturuyor "
        "(16:58=12.22, 16:59=12.22, 17:00=12.19, 17:01=12.22). Algoritma "
        "ws1 olarak platonun ILK noktasi yerine 17:00'i secmis - bu hem "
        "sureyi gereksiz uzatiyor (4 dk yerine 2 dk) hem de ws1'i platonun "
        "en yuksegi degil hafif altindaki bir degere sabitliyor.",
    ),
    (
        "2024-09-16 06:18:00", "2024-09-16 06:26:00",
        None, "2024-09-16 06:24:00",
        "Ayni plato sorunu: yukselis 06:24'te zirveye ulasiyor (10.583) ve "
        "orada kaliyor (06:24=10.583, 06:25=10.583, 06:26=10.528, "
        "06:27=10.583). Algoritma ws1'i platonun ilk noktasi yerine "
        "06:26'ya (10.528 - platonun en dusuk noktasi) koymus; sure 8 dk "
        "yerine 6 dk olmali.",
    ),
    (
        "2025-06-30 07:32:00", "2025-06-30 07:39:00",
        "2025-06-30 07:34:00", None,
        "Onceki t0=07:32 (4.81) ile gercek firlama arasinda 07:33'te 5.31'e "
        "cikip 07:34'te 4.89'a geri donen bir tumsek var; gercek yukselis "
        "07:34'ten (4.89) basliyor. "
        "(Temiz listedeki Event 28 ile ayni duzeltme.)",
    ),
    (
        "2026-05-13 03:42:00", "2026-05-13 03:47:00",
        None, "2026-05-13 03:46:00",
        "Ayni plato sorunu (daha hafif): yukselis 03:46'da platoya "
        "ulasiyor (03:46=10.333, 03:47=10.361, 03:48=10.333, 03:49=10.389, "
        "03:50=10.333 - hepsi ayni seviye). Algoritma 03:47'yi secmis; "
        "zirvenin ILK ulasildigi nokta 03:46, sure 5 dk yerine 4 dk olmali. "
        "Duzeltme sonrasi delta_v=6.139 olup MIN_DELTA_V=6.0 esigini hala "
        "gectigi kontrol edildi.",
    ),
]


def recompute(t0, t1, df):
    t0 = pd.Timestamp(t0)
    t1 = pd.Timestamp(t1)
    m = (df["dt"] >= t0) & (df["dt"] <= t1)
    seg = df.loc[m].sort_values("dt")
    ws = seg["ws_ham_ms"].to_numpy(dtype=float)

    ws0, ws1 = float(ws[0]), float(ws[-1])
    delta_v = ws1 - ws0
    duration = float((t1 - t0).total_seconds() / 60.0)
    avg_slope = delta_v / duration if duration > 0 else 0.0

    roll3 = pd.Series(ws).rolling(3, center=True, min_periods=1).mean().to_numpy()
    d_roll3 = float(roll3[-1] - roll3[0])
    path_roll3 = float(np.sum(np.abs(np.diff(roll3))))
    smoothness = (d_roll3 / path_roll3) if path_roll3 > 0 else 1.0

    return {
        "t0": t0, "t1": t1,
        "ws0": round(ws0, 3), "ws1": round(ws1, 3),
        "delta_v": round(delta_v, 3), "duration_min": round(duration, 2),
        "avg_slope": round(avg_slope, 4), "smoothness": round(smoothness, 4),
    }


df = pd.read_csv(CLEAN_CSV, parse_dates=["datetime"])
df = df.rename(columns={"datetime": "dt"})
df["ws_ham_ms"] = df["ws_raw_kmh"] / 3.6
df = df.sort_values("dt").reset_index(drop=True)

events = pd.read_csv(EVENTS_CSV, parse_dates=["t0", "t1"])
if "manual_fix_note" not in events.columns:
    events["manual_fix_note"] = ""

applied = 0
for orig_t0, orig_t1, new_t0, new_t1, reason in MANUAL_FIXES:
    orig_t0 = pd.Timestamp(orig_t0)
    orig_t1 = pd.Timestamp(orig_t1)
    mask = (events["t0"] == orig_t0) & (events["t1"] == orig_t1)
    if not mask.any():
        print(f"Bulunamadi (muhtemelen zaten duzeltilmis), atlaniyor: {orig_t0} -> {orig_t1}")
        continue

    target_t0 = pd.Timestamp(new_t0) if new_t0 else orig_t0
    target_t1 = pd.Timestamp(new_t1) if new_t1 else orig_t1
    old = events.loc[mask].iloc[0]
    vals = recompute(target_t0, target_t1, df)
    for k, v in vals.items():
        events.loc[mask, k] = v
    events.loc[mask, "manual_fix_note"] = reason
    applied += 1
    print(f"Duzeltildi: t0 {old['t0']} -> {vals['t0']}  |  t1 {old['t1']} -> {vals['t1']}")
    print(f"            ws0 {old['ws0']} -> {vals['ws0']}  ws1 {old['ws1']} -> {vals['ws1']}  "
          f"delta_v {old['delta_v']} -> {vals['delta_v']}  duration {old['duration_min']} -> {vals['duration_min']}")

events = events.sort_values("t0").reset_index(drop=True)
events["event_id"] = range(1, len(events) + 1)
events.to_csv(EVENTS_CSV, index=False)
print(f"\n{applied} duzeltme uygulandi. Toplam olay: {len(events)}")
print("Kaydedildi ->", EVENTS_CSV)
