# -*- coding: utf-8 -*-
"""
XGBoost ile ramp-up oncul matrislerinden delta_ws ve delta_t tahmini.

EKSIK VERI: DOLDURULMAZ. XGBoost NaN'i dogrudan isler - her bolunmede
eksik degerlerin hangi dala gidecegini kendisi ogrenir. Yani "veri yok"
bilgisi modelin kullanabildigi gercek bir sinyal olarak kalir. Bu,
projenin "sentetik doldurma yapilmaz" kuralina birebir uyar.

Degerlendirme: model/_ortak/degerlendirme.py'deki ortak cerceve
(blok-farkinda 5 kat x 5 tekrar CV + temel cizgi). CNN-LSTM ile birebir
ayni bolmeler kullanilir, boylece karsilastirma adil olur.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_ortak"))
import degerlendirme as ev  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

MATRIS_DIR = Path(PROJECT_ROOT, 'src', 'feature_matrix')
OUT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "XGBoost"

# Kucuk veri icin bilincli olarak sig ve kuvvetli duzenlilestirilmis agaclar
PARAMS = dict(
    max_depth=2,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.3,
    min_child_weight=3,
    reg_lambda=5.0,
    reg_alpha=0.5,
    n_estimators=2000,
    objective="reg:squarederror",
    random_state=ev.SEED,
    n_jobs=4,
    early_stopping_rounds=50,
)

print("[1/4] Veri yukleniyor (NaN korunuyor)...")
X, y, groups, meta = ev.load_data(MATRIS_DIR)
N = len(y)
print(f"  X: {X.shape}   olay={N}   blok={len(np.unique(groups))}   NaN orani={np.isnan(X).mean():.1%}")

print(f"[2/4] Blok-farkinda {ev.N_FOLDS} kat x {ev.N_REPEATS} tekrar CV...")
oof = np.full((ev.N_REPEATS, N, 2), np.nan)
best_iters = {0: [], 1: []}

for rep, f, inner_tr, inner_va, te in ev.fold_iter(groups):
    for ti in range(2):
        model = xgb.XGBRegressor(**PARAMS)
        model.fit(X[inner_tr], y[inner_tr, ti],
                  eval_set=[(X[inner_va], y[inner_va, ti])],
                  verbose=False)
        best_iters[ti].append(model.best_iteration)
        oof[rep, te, ti] = model.predict(X[te])

    if f == ev.N_FOLDS - 1:
        from sklearn.metrics import r2_score
        print(f"    tekrar {rep + 1}/{ev.N_REPEATS}:  "
              f"R2(delta_ws)={r2_score(y[:, 0], oof[rep, :, 0]):+.3f}   "
              f"R2(delta_t)={r2_score(y[:, 1], oof[rep, :, 1]):+.3f}")

print("[3/4] Metrikler...")
metrics = ev.summarize(y, oof, MODEL_NAME, out_csv=OUT_DIR / "metrics.csv")
print(metrics.to_string(index=False))

pred_df = ev.prediction_frame(meta, y, oof)

# TRAIN (icinde-ornek) tahmini: tum 43 olayla egitilmis TEK bir model,
# AYNI olaylar uzerinde tahmin yapiyor - modelin gordugu veriyi ne kadar
# ezberleyebildigini gosterir (asiri ogrenme kontrolu icin OOF ile
# karsilastirilir). Ayni model, asagida ozellik onemi icin de kullanilir.
print("[3.5/4] Train (ic-ornek) tahmini icin tum veriyle final model...")
names = ev.feature_names()
train_pred = np.full((N, 2), np.nan)
for ti, tname in enumerate(ev.TARGETS):
    p = dict(PARAMS)
    p.pop("early_stopping_rounds")
    p["n_estimators"] = max(int(np.median(best_iters[ti])), 10)
    m = xgb.XGBRegressor(**p)
    m.fit(X, y[:, ti], verbose=False)
    train_pred[:, ti] = m.predict(X)
    imp = pd.DataFrame({"feature": names, "gain": m.feature_importances_})
    imp = imp.sort_values("gain", ascending=False)
    imp.to_csv(OUT_DIR / f"feature_importance_{tname}.csv", index=False)

pred_df["delta_ws_tahmin_train"] = train_pred[:, 0].round(3)
pred_df["delta_t_tahmin_train"] = train_pred[:, 1].round(3)

# TEK, SABIT train/test bolmesi (sadece olay-bazli gorsellestirme icin -
# her olay ACIKCA ya train ya test, iki kutu degil). Metrikler yukaridaki
# tekrarli CV'den geliyor, bu SADECE gorsel netlik icin ayri bir bolme.
print("[3.6/4] Tek train/test bolmesi (gorsellestirme icin)...")
train_mask, test_mask = ev.single_split(groups, meta, test_frac=0.20)
split_pred = np.full((N, 2), np.nan)
for ti in range(2):
    p = dict(PARAMS)
    p.pop("early_stopping_rounds")
    p["n_estimators"] = max(int(np.median(best_iters[ti])), 10)
    m = xgb.XGBRegressor(**p)
    m.fit(X[train_mask], y[train_mask, ti])
    split_pred[:, ti] = m.predict(X)
pred_df["split"] = np.where(train_mask, "train", "test")
pred_df["delta_ws_tahmin_split"] = split_pred[:, 0].round(3)
pred_df["delta_t_tahmin_split"] = split_pred[:, 1].round(3)
print(f"  train: {int(train_mask.sum())} olay   test: {int(test_mask.sum())} olay")

pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)

print("[4/4] Gorseller...")
ev.scatter_png(y, oof, metrics, MODEL_NAME, OUT_DIR / "oof_scatter.png")

with open(OUT_DIR / "split_info.json", "w", encoding="utf-8") as f:
    json.dump({"n_olay": int(N), "n_blok": int(len(np.unique(groups))),
               "n_fold": ev.N_FOLDS, "n_tekrar": ev.N_REPEATS,
               "eksik_veri": "DOLDURULMADI - XGBoost NaN'i dogrudan isler",
               "medyan_en_iyi_agac": {k: int(np.median(v)) for k, v in best_iters.items()},
               "params": {k: v for k, v in PARAMS.items() if k != "early_stopping_rounds"}},
              f, ensure_ascii=False, indent=2)

print(f"Tamamlandi -> {OUT_DIR}")
