# -*- coding: utf-8 -*-
"""
Manuel veri temizligi duzeltmeleri.

Bu script, otomatik testlerin (Adim 3-7, wl_temizleme.py) YAKALAYAMADIGI,
ama gorsel/manuel incelemede acikca donuk/arizali oldugu tespit edilen
KISA (20 dakikanin altinda, dolayisiyla Adim 4a'nin K_NORMAL=20 esigine
takilmayan) sensor donmalarini, TEK TEK, elle isaretleyip siler.

ONEMLI: Bu, genel algoritmayi (wl_temizleme.py) DEGISTIRMEZ - sadece
asagida acikca listelenen, incelenip onaylanmis zaman araliklarini
duzeltir. Her satirin nereden geldigi (hangi ramp-up olayinin hemen
sonrasinda goze carptigi) ve neden donuk sayildigi (once/sonrasindaki
gercek turbulansla karsilastirma) yorum olarak yazilidir.

Calistirma sirasi: wl_temizleme.py -> wl_yon_ekle.py -> BU SCRIPT ->
ramp_up/detect_ramp_up.py
"""

import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

CSV_PATH = Path(PROJECT_ROOT, 'src', 'quality_control', 'wl_temizlenmis.csv')

# (baslangic, bitis, sebep) - baslangic ve bitis dahil (inclusive), dakika cozunurlugunde
MANUAL_REMOVALS = [
    # --- Event 19 (2024-07-27, ramp-up 15:45-15:51) sonrasinda goze carpan donmalar ---
    ("2024-07-27 15:53:00", "2024-07-27 15:55:00",
     "Rampadan hemen sonra 3 dk boyunca 23.5-23.6 km/h'de donuk (once/sonrasi ~6-7 km/h std ile gercek turbulans)"),
    ("2024-07-27 16:04:00", "2024-07-27 16:11:00",
     "8 dk boyunca 42.9-43.2 km/h'de donuk, cok dar bant (K_NORMAL=20 esigine ulasamadigi icin otomatik testten kacmis)"),
    ("2024-07-27 16:16:00", "2024-07-27 16:19:00",
     "4 dk boyunca 40.8-40.9 km/h'de donuk; hemen ardindan zaten otomatik olarak isaretlenmis 16:21-16:45 donuk blogunun basi - ayni donma olayinin devami"),

    # --- Event 22 (2024-09-23, ramp-up 14:18-14:22) sonrasinda goze carpan donmalar ---
    ("2024-09-23 14:27:00", "2024-09-23 14:30:00",
     "4 dk boyunca tam 49.9 km/h'de donuk; hemen ardindan zaten otomatik isaretlenmis 14:31-14:54 blogunun basi - ayni donma olayinin devami"),
    ("2024-09-23 14:56:00", "2024-09-23 14:59:00",
     "4 dk boyunca 45.1-45.2 km/h'de donuk, cevresindeki gercek degisken veriyle net kontrast"),
    ("2024-09-23 15:01:00", "2024-09-23 15:12:00",
     "12 dk boyunca 34.5-34.6 km/h'de donuk - uzun ve cok dar bant, acik arizali sensor"),
    ("2024-09-23 15:17:00", "2024-09-23 15:24:00",
     "8 dk boyunca 23.6-23.9 km/h'de donuk"),
    ("2024-09-23 15:28:00", "2024-09-23 15:31:00",
     "4 dk boyunca 28.7-29.0 km/h'de donuk"),
    ("2024-09-23 15:32:00", "2024-09-23 15:36:00",
     "5 dk boyunca 31.3-31.5 km/h'de donuk - onceki blokla birlikte iki basamakli bir 'takilma' deseni"),

    # --- Event 17 (2024-04-20, ramp-up 17:48-17:52) sonrasinda goze carpan donmalar ---
    ("2024-04-20 17:53:00", "2024-04-20 17:59:00",
     "7 dk boyunca 35.8-36.0 km/h'de donuk (once rampanin kendi yukselisi ~6.7 km/h std, sonrasi ~1.4 std - net kontrast)"),
    ("2024-04-20 18:03:00", "2024-04-20 18:08:00",
     "6 dk boyunca 41.7-42.0 km/h'de donuk, hemen ardindan (18:09-18:10 gercek inis) ve daha sonra zaten otomatik isaretlenmis 18:11-18:48 blogu geliyor - ayni yuksek ruzgar donemindeki tekrarlanan kisa takilma deseni"),
]

df = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

if "flag_manual" not in df.columns:
    df["flag_manual"] = False
if "manual_note" not in df.columns:
    df["manual_note"] = ""

total_touched = 0
for start, end, reason in MANUAL_REMOVALS:
    mask = (df["datetime"] >= pd.Timestamp(start)) & (df["datetime"] <= pd.Timestamp(end))
    n = int(mask.sum())
    already_nan = int(df.loc[mask, "ws_clean_ms"].isna().sum())
    df.loc[mask, "ws_clean_ms"] = float("nan")
    df.loc[mask, "flag_manual"] = True
    df.loc[mask, "manual_note"] = reason
    total_touched += n
    print(f"{start} -> {end}  ({n} dk, {already_nan} zaten bostu)  :: {reason}")

print(f"\nToplam elle silinen nokta: {total_touched}")
df.to_csv(CSV_PATH, index=False)
print("Kaydedildi ->", CSV_PATH)
