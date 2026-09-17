# -*- coding: utf-8 -*-
"""
CNN-LSTM icin tek bir "metrik ozet karti": R^2, RMSE ve MAE (5x5
tekrarli CV ortalama +/- std) her iki hedef icin, temel cizgiyle
yan yana, TEK bir tabloda. Diger gorseller (oof_scatter.png) bu ucunu
ayri ayri, dagilmis sekilde gosteriyordu (RMSE/R^2 basliginda, MAE hic
yok); bu gorsel makale icin uc metrigi bir arada, net okunur halde
sunar. Sayilar dogrudan metrics.csv'den okunur, hicbir yeniden hesap
yapilmaz.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_NAME = "CNN-LSTM"
OUT_PATH = HERE / "metrics_summary.png"

m = pd.read_csv(HERE / "metrics.csv")

TARGET_LABEL = {"delta_ws": "Delta WS", "delta_t": "Delta T"}
UNIT = {"delta_ws": "m/s", "delta_t": "min"}

rows = []
for tgt in ["delta_ws", "delta_t"]:
    model_row = m[(m["target"] == tgt) & (m["model"] == MODEL_NAME)].iloc[0]
    u = UNIT[tgt]
    rows.append([TARGET_LABEL[tgt], "R²", f"{model_row['R2_ort']:+.3f}"])
    rows.append(["", f"RMSE ({u})", f"{model_row['RMSE_ort']:.2f}"])
    rows.append(["", f"MAE ({u})", f"{model_row['MAE_ort']:.2f}"])

col_labels = ["Target", "Metric", MODEL_NAME]

fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.axis("off")

table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center",
                  colWidths=[0.34, 0.28, 0.38])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.4)

# --- stil: baslik satiri koyu, hedef bloklari hafif gri-mavi zeminle ayrilmis,
# tabloda dikey/butun kenarliklar yerine sadece yatay ince cizgiler ---
HEADER_BG = "#1f4e79"
BLOCK_BG = {"delta_ws": "#eef3fa", "delta_t": "#ffffff"}
n_cols = len(col_labels)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#c9c9c9")
    cell.set_linewidth(0.6)
    if r == 0:
        cell.set_facecolor(HEADER_BG)
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
        continue
    block = "delta_ws" if r in (1, 2, 3) else "delta_t"
    cell.set_facecolor(BLOCK_BG[block])
    if c == 1:
        cell.get_text().set_fontweight("bold")
    if c == 0 and rows[r - 1][0] != "":
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_fontsize(10.5)

fig.suptitle(f"{MODEL_NAME} — Cross-Validation Performance Summary",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Kaydedildi ->", OUT_PATH)
