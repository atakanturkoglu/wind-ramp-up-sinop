# -*- coding: utf-8 -*-
"""
HAM veri listesine manuel olay ekleme.

Ham (temizlenmemis) veri uzerinde calisan algoritma, bazi GERCEK ramp-up
olaylarini kaciriyor - cunku olayin hemen oncesindeki DONUK (arizali sensor)
veri, olayin suresini yapay olarak uzatip hizini esigin (BETA) altina
dusuruyor ve Stage 1 olayi hic aday olarak bile kabul etmiyor.

Bu script, gozle dogrulanmis boyle olaylari, gercek (fiziksel olarak
gozlemlenen) baslangic/bitis noktalariyla listeye elle ekler. Her ekleme
icin gerekce yazilidir.

Calistirma sirasi: detect_ramp_up_ham.py -> BU SCRIPT -> replot_ham.py
"""

import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

EVENTS_CSV = Path(PROJECT_ROOT, 'src', 'ramp_detection', 'ramp_up_events_HAM.csv')
CLEAN_CSV = Path(PROJECT_ROOT, 'src', 'quality_control', 'wl_temizlenmis.csv')

# (t0, t1, sebep) - degerler ham veriden (ws_raw_kmh / 3.6) hesaplanir
MANUAL_ADDITIONS = [
    (
        "2024-04-28 22:28:00", "2024-04-28 22:34:00",
        "Gercek ve net bir ramp-up (4.67 -> 13.64 m/s, 6 dk). Ham veride "
        "algoritma bunu kaciriyor cunku hemen oncesindeki 22:06-22:28 arasi "
        "23 dakikalik DONUK sensor verisi (sabit ~4.6-4.7 m/s) referans "
        "alininca sure 6 dk yerine 17 dk gibi hesaplaniyor ve hiz 0.53 "
        "m/s/dk'ya duserek BETA=0.6944 esiginin altinda kaliyor. Temiz "
        "veride bu donuk bolum silindigi icin olay dogru sekilde "
        "yakalaniyor (temiz listedeki Event 18, orada ws0=6.44@22:29 - "
        "cunku orada donuk veri silinmis durumda). "
        "NOT: HAM listede ws0, yukselisin gorsel olarak basladigi nokta "
        "olan 22:28'e (4.67 m/s, donuk platonun son noktasi) alindi - ham "
        "veri baglaminda temizlik uygulanmadigi icin yukselis bu seviyeden "
        "basliyor. Bu deger donuk sensor doneminden geldigi icin GERCEK "
        "ruzgar hizi olarak guvenilir degildir; temiz listedeki karsiligi "
        "(6.44) bu acidan daha dogrudur.",
    ),
]


def compute_event(t0, t1, df):
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
df["is_valid"] = df["ws_ham_ms"].notna() & df["wd_deg"].notna()
df["block_break"] = (df["dt"].diff() != pd.Timedelta(minutes=1)) | (~df["is_valid"])
df["block_id"] = df["block_break"].cumsum()

events = pd.read_csv(EVENTS_CSV, parse_dates=["t0", "t1"])
if "manual_note" not in events.columns:
    events["manual_note"] = ""

added = 0
for t0, t1, reason in MANUAL_ADDITIONS:
    t0_ts = pd.Timestamp(t0)
    if (events["t0"] == t0_ts).any():
        print(f"Zaten var, atlaniyor: {t0}")
        continue
    vals = compute_event(t0, t1, df)
    bid = int(df.loc[df["dt"] == t0_ts, "block_id"].iloc[0])
    row = {"event_id": -1, "block_id": bid, **vals, "manual_note": reason}
    events = pd.concat([events, pd.DataFrame([row])], ignore_index=True)
    added += 1
    print(f"Eklendi: {t0} -> {t1}  ws0={vals['ws0']}  ws1={vals['ws1']}  "
          f"delta_v={vals['delta_v']}  duration={vals['duration_min']}  slope={vals['avg_slope']}")

events = events.sort_values("t0").reset_index(drop=True)
events["event_id"] = range(1, len(events) + 1)
events.to_csv(EVENTS_CSV, index=False)
print(f"\n{added} olay eklendi. Toplam olay: {len(events)}")
print("Kaydedildi ->", EVENTS_CSV)
