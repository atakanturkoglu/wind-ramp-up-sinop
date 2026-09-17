import glob
import os

import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

DATA_DIR = Path(PROJECT_ROOT, 'data')
OUT_DIR = Path(PROJECT_ROOT, 'src', 'quality_control')
os.makedirs(OUT_DIR, exist_ok=True)

WS_COL = "Avg Wind Speed - km/h"

# ---------------------------------------------------------------------------
# Ham veriyi yukle (Adim 1 ile ayni mantik)
# ---------------------------------------------------------------------------

files = sorted(glob.glob(os.path.join(DATA_DIR, "tplink_*.csv")))

frames = []
for f in files:
    df = pd.read_csv(f, skiprows=5, encoding="latin1")
    df = df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%y %H:%M", errors="coerce")
    df[WS_COL] = pd.to_numeric(df[WS_COL], errors="coerce")
    df = df.dropna(subset=["datetime"])
    frames.append(df[["datetime", WS_COL]])

all_ts = pd.concat(frames, ignore_index=True).drop_duplicates(subset="datetime")
all_ts = all_ts.sort_values("datetime").reset_index(drop=True)

start, end = all_ts["datetime"].min(), all_ts["datetime"].max()
full_range = pd.date_range(start, end, freq="min")
ws_full = all_ts.set_index("datetime")[WS_COL].reindex(full_range)
arr = ws_full.values
valid = ~np.isnan(arr)

print(f"Toplam dakika: {len(full_range)}, gecerli (dolu) dakika: {int(valid.sum())}")

# ---------------------------------------------------------------------------
# Adim 3 - Fiziksel sinir testi
# ---------------------------------------------------------------------------

LOWER_BOUND = 0.0
UPPER_BOUND = 284.4  # km/h, Davis Vantage Pro2 Plus anemometer sensor spec

flag_range = valid & ((arr < LOWER_BOUND) | (arr > UPPER_BOUND))

# ---------------------------------------------------------------------------
# Adim 4 - Persistence (sabit deger) testi
# ---------------------------------------------------------------------------

# EPS = 0.25 km/h: the anemometer's own reporting step is ~0.1 km/h, so a
# frozen reading can flicker by one step without ever being bit-identical
# for 20 straight minutes. Tolerance is set to bridge such flicker, in the
# same spirit as QARTOD's Flat Line Test (which uses an explicit tolerance,
# not bit-exact equality).
#
# DUZELTME (0.15 -> 0.25): 0.15 km/h, yalnizca TEK bir ~0.1 km/h adimlik
# titremeyi koprulemeye yetiyordu. Gercek donuk donemlerde (ör. 2024-09-30
# event 24: ~3 saat boyunca 22.9-23.1 km/h araliginda donen bir sensor)
# ardisik farkin 0.2 km/h'ye kadar ciktigi gozlemlendi - bu, 0.15'i asip
# "same" zincirini kirdigi icin uzun donuk donem, K_NORMAL=20'ye hic
# ulasamayan bir suru kisa parcaya bolunuyor ve hicbiri isaretlenmiyordu.
# 0.25, gozlemlenen en buyuk titremeyi (0.2) rahat bir payla kapsarken,
# gercek bir ramp-up'in dakikalik adimlarindan (tipik olarak birkac km/h)
# hala cok kucuk kaliyor, yani gercek bir artisi donuk sanma riski yok.
EPS = 0.25
NEAR_ZERO = 0.1
K_ZERO = 1440
K_NORMAL = 20

prev_valid = np.r_[False, valid[:-1]]
diff = np.abs(arr - np.r_[np.nan, arr[:-1]])
same = valid & prev_valid & (diff <= EPS)

same_series = pd.Series(same, index=ws_full.index)
run_id = (~same_series).cumsum()
run_len = same_series.groupby(run_id).cumcount().values + 1
run_len = np.where(valid, run_len, 0)

near_zero_mask = valid & (arr <= NEAR_ZERO)
K = np.where(near_zero_mask, K_ZERO, K_NORMAL)

# Once a block's run length reaches K anywhere, the WHOLE block (from its
# first point) is flagged, not just the tail from the K-th point onward.
_pers_df = pd.DataFrame({"run_id": run_id.values, "run_len": run_len, "valid": valid})
_pers_valid = _pers_df[_pers_df["valid"]]
_block_max_run = _pers_valid.groupby("run_id")["run_len"].transform("max")
_block_K = np.where(near_zero_mask[_pers_valid.index.values], K_ZERO, K_NORMAL)
_flag_block = _block_max_run.values >= _block_K
flag_persistence_exact = np.zeros(len(arr), dtype=bool)
flag_persistence_exact[_pers_valid.index.values] = _flag_block

# Adim 4b - dusuk varyans (dar-bant) testi: tam esitlik degil ama uzun sure
# cok dar bir bantta gezinen (gercek turbulansa gore anormal derecede duz)
# donemleri yakalar. Meek & Hatfield (1994) persistence testinin QARTOD Flat
# Line Test'teki tolerans (EPS) parametresiyle ayni ruhtaki genellemesidir.
#
# NOT: Pencere ortali (center=True) birakildi. Bir ara geriye-bakan
# (trailing) pencere denenmis, fakat bu sefer TERSI sorun ortaya cikmisti:
# donuk bir donemin hemen ardindan gelen GERCEK bir ramp-up'in ilk dakikalari,
# pencerenin buyuk kismi hala eski donuk degerlerle dolu oldugu icin
# "hala sabit" sayilip silinmeye basliyordu (2024-09-30 event 24'te
# dogrulandi: 14:50-14:55 arasindaki gercek yukselis bu sekilde kayboldu).
# Asil kok neden Adim 4a'daki EPS'in dar olmasiydi (asagida duzeltildi);
# o duzeltilince Adim 4a tek basina uzun donuk donemleri doguru sekilde
# (sizinti olmadan, tam donuk-bolgenin bittigi noktada) yakaliyor. Adim 4b/4c
# sadece Adim 4a'nin yakalayamadigi farkli bir deseni (birkac ayrik seviye
# arasinda git-gel) icin ek/yedek test olarak, orijinal (ortali) haliyle
# birakildi.
VAR_WINDOW = 60
VAR_MEAN_MIN = 1.0   # km/h, bu esigin altindaki (sakin) donemler haric tutulur
VAR_STD_MAX = 0.3   # km/h

s_var = pd.Series(arr)
roll_std = s_var.rolling(VAR_WINDOW, center=True, min_periods=VAR_WINDOW // 2).std().values
roll_mean = s_var.rolling(VAR_WINDOW, center=True, min_periods=VAR_WINDOW // 2).mean().values

flag_persistence_lowvar = valid & ~np.isnan(roll_std) & (roll_mean > VAR_MEAN_MIN) & (roll_std < VAR_STD_MAX)

# Adim 4c - dusuk benzersiz-deger sayisi testi: gercek turbulansta bir pencere
# icindeki degerlerin neredeyse tamami birbirinden farklidir. Sinyal, sabit
# tek bir degerde degil de birkac (ör. 2-3) ayrik seviye arasinda git-gel
# yapiyorsa (Adim 4a ve 4b'nin yakalayamadigi bir durum), bu pencerede cok az
# sayida benzersiz deger gorulur.
#
# NOT: Adim 4b ile ayni sebeple pencere ortali (center=True) birakildi -
# bkz. yukaridaki not.
UNIQ_WINDOW = 60
UNIQ_COUNT_MAX = 15


def _nunique(w):
    return len(np.unique(w[~np.isnan(w)]))


uniq_count = s_var.rolling(UNIQ_WINDOW, center=True, min_periods=UNIQ_WINDOW // 2).apply(_nunique, raw=True).values

flag_persistence_lowunique = valid & ~np.isnan(uniq_count) & (roll_mean > VAR_MEAN_MIN) & (uniq_count < UNIQ_COUNT_MAX)

flag_persistence = flag_persistence_exact | flag_persistence_lowvar | flag_persistence_lowunique

# NOT: Kisa sureli (4-19 dk), kontrast-tabanli bir donma testi denendi ve
# ramp-up tespiti uzerindeki etkisi olculdu (bkz. proje notlari). Genel
# kalibrasyonu (LAMBDA/BETA) ve blok bolunmelerini degistirerek 46 olaydan
# 41'e dusurdugu, bunlarin 9'unun kayboldugu (2'sinin de daha once elle
# duzeltilmis ws0'lari eski/yanlis haline dondurdugu) goruldugu icin GERI
# ALINDI. Kisa donma sorunu, bunun yerine sadece dogrudan etkilenen
# olaylarda elle (manuel_duzeltmeler.py ile) duzeltiliyor.

# ---------------------------------------------------------------------------
# Adim 5 - Ani sicrama (step) testi
#
# ONCEKI SURUM tek-tarafli ardisik farka (x[t]-x[t-1]) bakiyordu; bu, izole
# bir sicra-geri-don gurultusunu genis, gercek ve surekli bir rampanin
# ortasindaki buyuk ama TREND ILE TUTARLI bir adimdan ayirt edemiyordu -
# 2023-12-23 18:16 orneginde (referans pipeline ile karsilastirmada
# bulunmustur) +14.56 m/s'lik gercek bir ramp-up olayinin tam ortasindaki
# nokta bu yuzden yanlislikla silinip olayin tamami kayboluyordu.
#
# DUZELTME: Florita vd. (2013) "Swinging Door Algorithm" (SDA) mantigina
# gore, bir noktanin sicrama sayilip sayilmayacagina TEK BASINA rate ile
# degil, o noktanin KENDI IKI KOMSUSUNU BIRLESTIREN DOGRU CIZGIDEN ne kadar
# saptigina bakilarak karar veriliyor ("kapi genisligi" epsilon). Duzgun,
# sureli bir rampanin ortasindaki nokta bu dogruya yakin kalir (dusuk sapma,
# silinmez); izole bir sicrama ise dogrudan cok uzaklasir (yuksek sapma,
# silinir). Bu, ayni SDA/koridor fikri (Florita vd.), Kuang-hibrit
# script'inde (t0_kuang_hibrit.py) section birlestirme icin referans
# alinan kaynagin ta kendisidir - burada farkli bir amac (nokta-bazli
# sicrama testi) icin uygulanmistir.
# ---------------------------------------------------------------------------

K_STEP = 15

n_pts = len(arr)
valid_triplet = np.zeros(n_pts, dtype=bool)
valid_triplet[1:-1] = valid[:-2] & valid[1:-1] & valid[2:]
line_predicted = np.full(n_pts, np.nan)
line_predicted[1:-1] = (arr[:-2] + arr[2:]) / 2.0
line_deviation = arr - line_predicted

d = line_deviation[valid_triplet]
med_dev = np.median(d)
mad_dev = np.median(np.abs(d - med_dev))
scaled_mad_dev = 1.4826 * mad_dev
step_threshold = K_STEP * scaled_mad_dev

flag_step = valid_triplet & (np.abs(line_deviation - med_dev) > step_threshold)

# ---------------------------------------------------------------------------
# Adim 6 - Aykiri deger testi (Modified Z-score, yerel kayan pencere)
# ---------------------------------------------------------------------------

WINDOW = 61
MIN_PERIODS = 31
Z_THRESHOLD = 3.5

s_ws = pd.Series(arr)


def _local_mad(w):
    m = np.nanmedian(w)
    return np.nanmedian(np.abs(w - m))


local_median = s_ws.rolling(WINDOW, center=True, min_periods=MIN_PERIODS).median().values
local_mad = s_ws.rolling(WINDOW, center=True, min_periods=MIN_PERIODS).apply(_local_mad, raw=True).values
scaled_mad = 1.4826 * local_mad

departure = arr - local_median
denom_zero = scaled_mad == 0
denom_safe = np.where(denom_zero, 1.0, scaled_mad)
zmod = departure / denom_safe
zmod = np.where(denom_zero & (departure != 0), np.inf, zmod)
zmod = np.where(denom_zero & (departure == 0), 0.0, zmod)

valid_z = valid & ~np.isnan(local_median) & ~np.isnan(scaled_mad)
flag_outlier_finite = valid_z & np.isfinite(zmod) & (np.abs(zmod) > Z_THRESHOLD)

# ---------------------------------------------------------------------------
# Adim 7 - Testere disi (periyotluluk) testi
# NOT: Bu adim icin ruzgar/anemometre verisine ozel bir literatur kaynagi
# bulunamamistir. Yontem, baska muhendislik alanlarinda (ornegin takim
# tezgahi "chatter" tespiti) kullanilan otokorelasyon tabanli periyotluluk
# tespiti prensibinden esinlenerek, gercek ruzgarin fiziksel olarak pozitif
# kisa-vadeli otokorelasyona (persistence) sahip olmasi gerektigi mantigina
# dayanarak kurulmustur (kaynaksiz, genel prensip).
# ---------------------------------------------------------------------------

PERIOD_WINDOW = 21
R1_THRESHOLD = -0.6
AMPLITUDE_THRESHOLD = 1.0  # km/h


def _local_acf1(w):
    wc = w - np.mean(w)
    denom = np.sum(wc ** 2)
    if denom == 0:
        return 0.0
    return np.sum(wc[:-1] * wc[1:]) / denom


r1 = s_ws.rolling(PERIOD_WINDOW, center=True, min_periods=15).apply(_local_acf1, raw=True).values
local_range = (
    s_ws.rolling(PERIOD_WINDOW, center=True, min_periods=15).max()
    - s_ws.rolling(PERIOD_WINDOW, center=True, min_periods=15).min()
).values

flag_periodicity = valid & ~np.isnan(r1) & (r1 < R1_THRESHOLD) & (local_range > AMPLITUDE_THRESHOLD)

# ---------------------------------------------------------------------------
# Bayraklari birlestir ve temizle
# ---------------------------------------------------------------------------

flag_any = flag_range | flag_persistence | flag_step | flag_outlier_finite | flag_periodicity

print(f"\nAdim 3 (fiziksel sinir) ile isaretlenen: {int(flag_range.sum())}")
print(f"Adim 4a (persistence, tam esitlik) ile isaretlenen: {int(flag_persistence_exact.sum())}")
print(f"Adim 4b (persistence, dusuk varyans / dar-bant) ile isaretlenen: {int(flag_persistence_lowvar.sum())}")
print(f"Adim 4c (persistence, dusuk benzersiz-deger sayisi) ile isaretlenen: {int(flag_persistence_lowunique.sum())}")
print(f"Adim 4 (birlesik) ile isaretlenen: {int(flag_persistence.sum())}")
print(f"Adim 5 (step) ile isaretlenen: {int(flag_step.sum())}")
print(f"Adim 6 (aykiri deger, sonlu Zm) ile isaretlenen: {int(flag_outlier_finite.sum())}")
print(f"Adim 7 (periyotluluk / testere disi, kaynaksiz yontem) ile isaretlenen: {int(flag_periodicity.sum())}")
print(f"Toplam benzersiz isaretlenen (birlestirilmis): {int(flag_any.sum())}")

ws_clean_kmh = np.where(flag_any, np.nan, arr)
ws_clean_ms = ws_clean_kmh / 3.6

result = pd.DataFrame({
    "datetime": full_range,
    "ws_raw_kmh": arr,
    "flag_range": flag_range,
    "flag_persistence_exact": flag_persistence_exact,
    "flag_persistence_lowvar": flag_persistence_lowvar,
    "flag_persistence_lowunique": flag_persistence_lowunique,
    "flag_persistence": flag_persistence,
    "flag_step": flag_step,
    "flag_outlier": flag_outlier_finite,
    "flag_periodicity": flag_periodicity,
    "flag_any": flag_any,
    "ws_clean_ms": ws_clean_ms,
})

out_csv = os.path.join(OUT_DIR, "wl_temizlenmis.csv")
result.to_csv(out_csv, index=False)

n_before = int(valid.sum())
n_after = int((~np.isnan(ws_clean_ms)).sum())
print(f"\nTemizlik oncesi gecerli nokta: {n_before}")
print(f"Temizlik sonrasi gecerli nokta: {n_after}")
print(f"Cikarilan nokta: {n_before - n_after} ({100 * (n_before - n_after) / n_before:.3f}%)")
print(f"\nKaydedildi: {out_csv}")
