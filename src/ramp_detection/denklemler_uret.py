"""
5.2 bolumu icin denklem gorselleri. Bu surum, Kuang vd. (2020)'nin
PDF'inden (staj/ramp-up/A_New_Definition_Method_of_Wind_Power_Ramp_Section.pdf)
DOGRUDAN CIKARILAN Eq.(3), Eq.(12), Eq.(14) notasyonunu birebir esas alir:

  Eq.(3)  (orijinal makale, guc P uzerinden):
      P_{i-1} > P_i < P_{i+1}  -> minimum nokta
      P_{i-1} < P_i > P_{i+1}  -> maksimum nokta

  Eq.(12) (orijinal makale, guc P uzerinden, SP kriteri):
      |P_{j+1}-P_j| > lambda (kosul 1)  VE  beta_max > |rate| > beta (kosul 2)

  Eq.(14) (orijinal makale, guc P uzerinden, UP/DP kriteri):
      rate > beta   -> UP (uphill point)
      rate < -beta  -> DP (downhill point)

Bu calismada P (guc) yerine ws (ruzgar hizi) kullanilmis ve Eq.(12)'deki
beta_max ustsiniri KALDIRILMISTIR (sebep: beta_max sebeke baglanti
kapasitesine bagli bir kavramdir, tek bir ruzgar hizi sensorunde karsiligi
yoktur). Bu nedenle asagida hem ORIJINAL Eq.(12) (beta_max ile, P
notasyonuyla) hem de UYARLANMIS hali (beta_max'siz, ws notasyonuyla) AYRI
AYRI render edilir - degisiklik gizlenmeden gosterilsin diye.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

OUT_DIR = Path(PROJECT_ROOT, 'src', 'ramp_detection', 'denklemler')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["mathtext.fontset"] = "cm"

EQUATIONS = {
    # Eq. (3) - orijinal makale, TSP tanimi (P ile, birebir makaledeki gibi)
    "eq1a_min": r"$P_{i-1} > P_i < P_{i+1} \;\Rightarrow\; \mathrm{minimum\ point}$",
    "eq1b_max": r"$P_{i-1} < P_i > P_{i+1} \;\Rightarrow\; \mathrm{maximum\ point}$",

    # Eq. (12) - ORIJINAL makale (guc P uzerinden, beta_max ile)
    "eq2_kuang_original": r"$|\Delta P_{a,b}| > \lambda \quad \mathrm{and} \quad "
                           r"\beta_{max} > |\mathrm{rate}_{a,b}| > \beta$",

    # Bu calismada uyarlanmis hali (ruzgar hizi ws, beta_max KALDIRILMIS)
    "eq3_adapted": r"$|\Delta ws_{a,b}| > \lambda \quad \mathrm{and} \quad "
                   r"|\mathrm{rate}_{a,b}| > \beta$",

    # Eq. (14) - orijinal makale, UP / DP tanimi (ws notasyonuyla, degistirilmemis)
    "eq4a_up": r"$\mathrm{rate}_{p,q} > \beta \;\Rightarrow\; \mathrm{UP\ (uphill\ point)}$",
    "eq4b_down": r"$\mathrm{rate}_{p,q} < -\beta \;\Rightarrow\; \mathrm{DP\ (downhill\ point)}$",

    # Ek muhendislik esikleri - Kuang vd. (2020)'den GELMEZ, bu calismaya
    # ozgudur (detect_ramp_up.py: MAX_WS0, MIN_PEAK_WS, MIN_DELTA_V)
    "eq5a_ws0": r"$ws_0 \leq 7 \ \mathrm{m/s} \qquad (\mathrm{MAX\_WS0})$",
    "eq5b_ws1": r"$ws_1 \geq 10 \ \mathrm{m/s} \qquad (\mathrm{MIN\_PEAK\_WS})$",
    "eq5c_deltav": r"$\Delta v = ws_1 - ws_0 \geq 6 \ \mathrm{m/s} \qquad (\mathrm{MIN\_DELTA\_V})$",
}

for name, expr in EQUATIONS.items():
    fig = plt.figure(figsize=(6.8, 0.6))
    fig.patch.set_facecolor("white")
    fig.text(0.0, 0.5, expr, fontsize=16, ha="left", va="center", color="black")
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=300,
                facecolor="white", transparent=False, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("Kaydedildi:", name)

# Eski (yanlis notasyonlu) dosyalari temizle
for old in ["eq1_tsp.png", "eq4c_trans.png"]:
    p = os.path.join(OUT_DIR, old)
    if os.path.exists(p):
        os.remove(p)
        print("Silindi (eski/yanlis notasyon):", old)

print("Tamamlandi ->", OUT_DIR)
