# -*- coding: utf-8 -*-
"""
Manuel ws0 duzeltmeleri.

Kuang algoritmasinin 1. asamasi (gurultu filtreleme), bazi olaylarda,
gercek patlamadan hemen once ufak bir tumsek+cukur salinimi varsa, bu
salinimi "onemsiz" sayip atlayabiliyor - boylece ws0 gercek dipten degil,
bu tumsegin tepesinden secilmis oluyor (goruntude: yesil nokta hafif
yukselip neredeyse ayni yere geri dusen bir cizginin USTUNDE duruyor).

Bu, ALGORITMANIN GENELINI degistirmeden, TEK TEK incelenip onaylanmis
olaylar icin, ramp_up_events.csv uzerinde ws0/t0/delta_v/duration_min/
avg_slope/smoothness degerlerini gercek (goruntude gorulen) baslangic
noktasina gore elle duzeltir.

Calistirma sirasi: detect_ramp_up.py -> BU SCRIPT
"""

import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

EVENTS_CSV = Path(PROJECT_ROOT, 'src', 'ramp_detection', 'ramp_up_events.csv')
CLEAN_CSV = Path(PROJECT_ROOT, 'src', 'quality_control', 'wl_temizlenmis.csv')

# event_id -> (yeni_t0, sebep)
MANUAL_WS0_FIXES = {
    11: (
        "2023-12-23 18:13:00",
        "Onceki t0=18:11 (ws0=6.64) aslinda ufak, onemsiz bir yerel tepe; "
        "gercek dip 18:13'te (ws=5.92), aradaki 0.72 m/s'lik inis Kuang "
        "1. asamasinin esigine (lambda=1.1667) takilmadigi icin atlanmis.",
    ),
    16: (
        "2024-04-20 16:34:00",
        "Onceki t0=16:28 (ws0=5.31) ile gercek firlama (16:34, ws=5.33) arasinda "
        "16:30'a kadar hafif yukselip geri donen bir tumsek var; bu tumsegin "
        "genligi (1.61 m/s) esigi (lambda=1.1667) barely gecmesi yuzunden "
        "16:28 SP olarak kalmis, 16:34'teki gercek dip ise sonraki adimda "
        "(16:30->16:32->16:33->16:34, hepsi kucuk degisimler) elenmis.",
    ),
    28: (
        "2025-06-30 07:34:00",
        "Onceki t0=07:32 (ws0=4.81) ile gercek firlama (07:34, ws=4.89) arasinda "
        "07:33'e hafif yukselip geri donen bir tumsek var (genligi 0.5 m/s, "
        "esigin 1.1667 altinda kaldigi icin elenmis). Not: duzeltilmis "
        "delta_v (5.94) MIN_DELTA_V esiginin (6.0) az altinda kaliyor - "
        "kullanici onayiyla olay yine de tutuldu, sadece ws0/t0 duzeltildi.",
    ),
}


def recompute_event(ev_row, new_t0, clean_df):
    df = clean_df.rename(columns={"datetime": "dt", "ws_clean_ms": "ws_temiz"})
    df = df.sort_values("dt").reset_index(drop=True)
    df["is_valid"] = df["ws_temiz"].notna() & df["wd_deg"].notna()
    df["block_break"] = (df["dt"].diff() != pd.Timedelta(minutes=1)) | (~df["is_valid"])
    df["block_id"] = df["block_break"].cumsum()
    valid = df[df["is_valid"]]

    bid = int(ev_row["block_id"])
    block = valid[valid["block_id"] == bid].sort_values("dt").reset_index(drop=True)
    ws = block["ws_temiz"].to_numpy(dtype=float)
    dts = block["dt"].to_numpy()

    new_t0 = pd.Timestamp(new_t0)
    t1 = pd.Timestamp(ev_row["t1"])
    s_idx = int(block.index[block["dt"] == new_t0][0])
    e_idx = int(block.index[block["dt"] == t1][0])

    ws0 = float(ws[s_idx])
    ws1 = float(ws[e_idx])
    delta_v = ws1 - ws0
    duration = float((dts[e_idx] - dts[s_idx]) / np.timedelta64(1, "m"))
    avg_slope = delta_v / duration

    ws_roll3 = pd.Series(ws).rolling(3, center=True, min_periods=1).mean().to_numpy()
    seg = ws_roll3[s_idx:e_idx + 1]
    delta_roll3 = float(seg[-1] - seg[0])
    path_roll3 = float(np.sum(np.abs(np.diff(seg))))
    smoothness = (delta_roll3 / path_roll3) if path_roll3 > 0 else 1.0

    return {
        "t0": new_t0, "ws0": round(ws0, 3), "delta_v": round(delta_v, 3),
        "duration_min": round(duration, 2), "avg_slope": round(avg_slope, 4),
        "smoothness": round(smoothness, 4),
    }


events = pd.read_csv(EVENTS_CSV, parse_dates=["t0", "t1"])
clean = pd.read_csv(CLEAN_CSV, parse_dates=["datetime"])

if "manual_ws0_note" not in events.columns:
    events["manual_ws0_note"] = ""

for eid, (new_t0, reason) in MANUAL_WS0_FIXES.items():
    mask = events["event_id"] == eid
    if not mask.any():
        print(f"UYARI: event_id {eid} bulunamadi, atlaniyor (yeniden numaralanmis olabilir).")
        continue
    row = events.loc[mask].iloc[0]
    new_vals = recompute_event(row, new_t0, clean)
    for k, v in new_vals.items():
        events.loc[mask, k] = v
    events.loc[mask, "manual_ws0_note"] = reason
    print(f"Event {eid}: t0 {row['t0']} -> {new_vals['t0']}  ws0 {row['ws0']} -> {new_vals['ws0']}  "
          f"delta_v {row['delta_v']} -> {new_vals['delta_v']}  duration {row['duration_min']} -> {new_vals['duration_min']}  "
          f"avg_slope {row['avg_slope']} -> {new_vals['avg_slope']}")

events.to_csv(EVENTS_CSV, index=False)
print("\nKaydedildi ->", EVENTS_CSV)
