# -*- coding: utf-8 -*-
"""
Ramp-up olaylarinin BAGLAM pencerelerinde (t0-75dk .. t1+75dk) kalan
donuk cizgi ve testere disi bolumlerinin temizligi.

GEREKCE
-------
Gorsel incelemede, olaylarin cevresinde iki tur bozuk desen goruldu:
  - duz cizgi  : deger dakikalarca neredeyse hic degismiyor
  - testere disi: deger cok dar bir bantta surekli asagi-yukari zipliyor
Ikisi de ayni fiziksel arizanin gorunumu: sensor bir seviyeye takilip
2-3 kuantalama adimi arasinda gidip geliyor. Bu yuzden ikisi de TEK bir
olcutle yakalanir: dar ARALIK (max-min).

Bu, QARTOD "Manual for Real-Time Quality Control of Wind Data" Flat Line
Test'in kendi tanimiyla ayni olcuttur ("range variation (MAX-MIN) value
that fails to exceed threshold values over a selected time period"). Ayni
el kitabi, sure ve esik degerlerinin evrensel olmadigini, her operatorun
KENDI verisinin istatistigine gore belirlemesi gerektigini soyler; asagidaki
degerler bu veri setinin kendi davranisina bakilarak secilmistir.

KAPSAM VE GUVENLIK SINIRLARI
----------------------------
  1. Sadece ramp-up olaylarinin baglam pencerelerinde calisir; veri
     setinin geri kalanina DOKUNULMAZ.
  2. Rampalarin KENDISI ([t0, t1] araliklari - hem ham hem temiz listedeki
     olaylar icin) tamamen korunur; oraya hicbir kosulda dokunulmaz.
  3. Bir bolumun silinmesi icin dar aralikta olmasi YETMEZ; cevresindeki
     +-15 dakikanin, o bolumden en az KONTRAST kati daha oynak olmasi da
     gerekir. Bu sart, gercekten sakin (ama kisa) donemlerin yanlislikla
     silinmesini engellemek icin vardir.

Calistirma sirasi:
  wl_temizleme.py -> wl_yon_ekle.py -> manuel_duzeltmeler.py -> BU SCRIPT
"""

import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

CSV_PATH = Path(PROJECT_ROOT, 'src', 'quality_control', 'wl_temizlenmis.csv')
EVENTS_HAM = Path(PROJECT_ROOT, 'src', 'ramp_detection', 'ramp_up_events_HAM.csv')
EVENTS_TEMIZ = Path(PROJECT_ROOT, 'src', 'ramp_detection', 'ramp_up_events.csv')

WIN = 4               # dakika - bir bolumun "dar aralik" sayilmasi icin gereken en kisa sure
RANGE_MAX = 0.40      # m/s  - bu sure boyunca max-min bu degerin altindaysa dar aralik
CONTEXT_MIN = 75      # dakika - olay baglam penceresi (grafiklerdekiyle ayni)
CTX_COMPARE = 15      # dakika - kontrast icin bakilan komsuluk
CONTRAST_MIN = 5.0    # cevre/bolum degiskenlik orani esigi
STD_FLOOR = 0.02      # m/s - sifira bolmeyi onlemek icin taban

df = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
ws = df["ws_clean_ms"].to_numpy()
dt = df["datetime"].to_numpy()
n = len(ws)
valid = ~np.isnan(ws)

# --- kapsam (baglam pencereleri) ve korunacak rampa araliklari ---------------
in_scope = np.zeros(n, dtype=bool)
ramp_zone = np.zeros(n, dtype=bool)

ev_ham = pd.read_csv(EVENTS_HAM, parse_dates=["t0", "t1"])
ev_temiz = pd.read_csv(EVENTS_TEMIZ, parse_dates=["t0", "t1"])

for _, e in ev_ham.iterrows():
    lo = e["t0"] - pd.Timedelta(minutes=CONTEXT_MIN)
    hi = e["t1"] + pd.Timedelta(minutes=CONTEXT_MIN)
    in_scope |= (dt >= np.datetime64(lo)) & (dt <= np.datetime64(hi))

for src in (ev_ham, ev_temiz):
    for _, e in src.iterrows():
        ramp_zone |= (dt >= np.datetime64(e["t0"])) & (dt <= np.datetime64(e["t1"]))

# --- dar aralikli bolumleri bul ---------------------------------------------
runs = []
i = 0
while i < n - WIN:
    if not (in_scope[i] and valid[i]) or ramp_zone[i]:
        i += 1
        continue
    w = ws[i:i + WIN]
    if np.isnan(w).any() or ramp_zone[i:i + WIN].any():
        i += 1
        continue
    if (np.nanmax(w) - np.nanmin(w)) <= RANGE_MAX:
        j = i + WIN
        while (j < n and valid[j] and not ramp_zone[j]
               and (np.nanmax(ws[i:j + 1]) - np.nanmin(ws[i:j + 1])) <= RANGE_MAX):
            j += 1
        runs.append((i, j - 1))
        i = j
    else:
        i += 1

# --- kontrast sarti ---------------------------------------------------------
kept = []
for a, b in runs:
    seg = ws[a:b + 1]
    seg = seg[~np.isnan(seg)]
    if len(seg) < 2:
        continue
    seg_std = max(float(np.std(seg)), STD_FLOOR)

    pre = ws[max(0, a - CTX_COMPARE):a]
    pre = pre[~np.isnan(pre)]
    post = ws[b + 1:b + 1 + CTX_COMPARE]
    post = post[~np.isnan(post)]

    stds = []
    if len(pre) >= 3:
        stds.append(float(np.std(pre)))
    if len(post) >= 3:
        stds.append(float(np.std(post)))
    if not stds:
        continue
    if max(stds) / seg_std >= CONTRAST_MIN:
        kept.append((a, b))

mask = np.zeros(n, dtype=bool)
for a, b in kept:
    mask[a:b + 1] = True

# --- ek adim: ZIRVE SONRASI PLATO ------------------------------------------
# Bazi olaylarda, ws1'e ulasildiktan sonra deger birkac dakika daha AYNI
# seviyede takiliyor (or. 2026-05-13'te 03:47-03:50 arasi 10.33-10.39).
# Bu, grafikte zirvenin tepesinde duz bir cizgi olarak gorunuyor ve yine
# ayni takilma arizasinin bir gorunumu. Yukaridaki pencere tabanli tarama
# bunlari kacirabiliyor (rampa koruma alani pencereyi bloke edebiliyor),
# bu yuzden ayrica ele alinir: t1'den sonra, degeri ws1'e TOL kadar yakin
# kalan ARDISIK noktalar (en az MIN_LEN tane) isaretlenir.
POST_PEAK_TOL = 0.30     # m/s - ws1'e bu kadar yakin sayilir
POST_PEAK_MIN_LEN = 2    # en az bu kadar ardisik nokta olmali

idx_of = pd.Series(np.arange(n), index=pd.DatetimeIndex(dt))
post_peak_total = 0
for _, e in ev_ham.iterrows():
    ws1 = float(e["ws1"])
    t = pd.Timestamp(e["t1"]) + pd.Timedelta(minutes=1)
    run_idx = []
    while t in idx_of.index:
        k = int(idx_of.loc[t])
        if np.isnan(ws[k]) or abs(ws[k] - ws1) > POST_PEAK_TOL:
            break
        run_idx.append(k)
        t += pd.Timedelta(minutes=1)
    if len(run_idx) >= POST_PEAK_MIN_LEN:
        for k in run_idx:
            if not ramp_zone[k]:
                mask[k] = True
                post_peak_total += 1

assert not (mask & ramp_zone).any(), "GUVENLIK IHLALI: rampa araligina denk gelen nokta var!"

# ONEMLI - SADECE ISARETLE, ws_clean_ms'i BOZMA
# ------------------------------------------------
# Bu noktalari ws_clean_ms'ten silmek denendi ve ANA PIPELINE'I KIRDIGI
# goruldu: silinen noktalar veri bloklarini parcaliyor, 30 dakikanin
# altina dusen bloklar detect_ramp_up.py tarafindan komple eleniyor ve
# temiz listedeki olay sayisi 46'dan 40'a dusuyor - rampalarin KENDI
# verisi hic bozulmadigi halde. Bu, olaylarin kotu olmasindan degil,
# tespit hattindaki "30 dakikadan kisa bloga bakma" kuralinin blok
# parcalanmasiyla etkilesiminden kaynaklaniyor.
#
# Bu yuzden burada yalnizca bir BAYRAK yazilir; ws_clean_ms'e
# dokunulmaz. Grafik script'leri bu bayragi kullanarak ilgili noktalari
# GORSELDE gizler, ana temizlik/tespit zinciri ise etkilenmez.
if "flag_baglam_temizlik" not in df.columns:
    df["flag_baglam_temizlik"] = False
df["flag_baglam_temizlik"] = mask

print(f"Bulunan donuk/testere bolum sayisi : {len(kept)}")
print(f"Zirve sonrasi plato noktasi        : {post_peak_total}")
print(f"Toplam isaretlenen nokta           : {int(mask.sum())}")
print(f"Rampa araligina denk gelen         : 0 (dogrulandi)")
print()
print("En uzun 10 bolum:")
for a, b in sorted(kept, key=lambda r: r[1] - r[0], reverse=True)[:10]:
    seg = ws[a:b + 1]
    print(f"  {pd.Timestamp(dt[a])} -> {pd.Timestamp(dt[b])}  ({b - a + 1} dk)  "
          f"aralik={np.nanmax(seg) - np.nanmin(seg):.3f} m/s  ort={np.nanmean(seg):.2f} m/s")

df.to_csv(CSV_PATH, index=False)
print(f"\nKaydedildi -> {CSV_PATH}")
