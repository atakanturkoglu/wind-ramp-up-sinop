import glob
import os

import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

DATA_DIR = Path(PROJECT_ROOT, 'data')
CSV_PATH = Path(PROJECT_ROOT, 'src', 'quality_control', 'wl_temizlenmis.csv')

DIR_COL = "Prevailing Wind Direction"

COMPASS_TO_DEG = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}

files = sorted(glob.glob(os.path.join(DATA_DIR, "tplink_*.csv")))

frames = []
for f in files:
    df = pd.read_csv(f, skiprows=5, encoding="latin1")
    df = df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%y %H:%M", errors="coerce")
    df = df.dropna(subset=["datetime"])
    df["wd_deg"] = df[DIR_COL].map(COMPASS_TO_DEG)
    frames.append(df[["datetime", "wd_deg"]])

all_dir = pd.concat(frames, ignore_index=True).drop_duplicates(subset="datetime")
all_dir = all_dir.sort_values("datetime").reset_index(drop=True)

n_total = len(all_dir)
n_valid = all_dir["wd_deg"].notna().sum()
print(f"Toplam yon kaydi: {n_total}, gecerli (taninan pusula degeri): {n_valid}")

clean = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
before_cols = clean.shape[1]

clean = clean.merge(all_dir, on="datetime", how="left")

print(f"wl_temizlenmis.csv satir sayisi: {len(clean)}")
print(f"wd_deg dolu satir sayisi: {clean['wd_deg'].notna().sum()}")
print(f"Sutun sayisi: {before_cols} -> {clean.shape[1]}")

clean.to_csv(CSV_PATH, index=False)
print(f"Guncellendi: {CSV_PATH}")
