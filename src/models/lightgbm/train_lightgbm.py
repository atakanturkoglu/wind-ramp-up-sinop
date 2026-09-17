# -*- coding: utf-8 -*-
"""
LightGBM ile ramp-up oncul matrislerinden delta_ws ve delta_t tahmini.

EKSIK VERI: DOLDURULMAZ. LightGBM NaN'i dogrudan isler (use_missing=True
varsayilan; eksik degerler bolunmede ayri bir yon olarak degerlendirilir).
Projenin "sentetik doldurma yapilmaz" kuralina uyar.

NOT (kucuk veri ayari): LightGBM'in varsayilan min_data_in_leaf=20 degeri
43 ornek icin fazla buyuk - hicbir bolunmeye izin vermez ve model sabit
tahmin uretir. Bu yuzden yaprak basina en az ornek sayisi 3'e cekildi,
yaprak sayisi 4 ile sinirlandi (derinlik ~2, diger agac modelleriyle ayni
kapasite duzeyi).

Degerlendirme: model/_ortak/degerlendirme.py'deki ortak cerceve - CNN-LSTM,
XGBoost ve CatBoost ile BIREBIR ayni bolmeler.
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_ortak"))
import degerlendirme as ev  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

MATRIS_DIR = Path(PROJECT_ROOT, 'src', 'feature_matrix')
OUT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "LightGBM"

PARAMS = dict(
    objective="regression",
    num_leaves=4,
    max_depth=2,
    learning_rate=0.03,
    min_child_samples=3,
    min_child_weight=1e-3,
    feature_fraction=0.3,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    lambda_l1=0.5,
    n_estimators=2000,
    random_state=ev.SEED,
    n_jobs=4,
    verbose=-1,
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
        model = lgb.LGBMRegressor(**PARAMS)
        model.fit(X[inner_tr], y[inner_tr, ti],
                  eval_set=[(X[inner_va], y[inner_va, ti])],
                  eval_metric="rmse",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        best_iters[ti].append(model.best_iteration_ or PARAMS["n_estimators"])
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
# AYNI olaylar uzerinde tahmin yapiyor - ezberleme kontrolu icin OOF ile
# karsilastirilir. Ayni model, ozellik onemi icin de kullanilir.
print("[3.5/4] Train (ic-ornek) tahmini icin tum veriyle final model...")
names = ev.feature_names()
train_pred = np.full((N, 2), np.nan)
for ti, tname in enumerate(ev.TARGETS):
    p = dict(PARAMS)
    p["n_estimators"] = max(int(np.median(best_iters[ti])), 10)
    m = lgb.LGBMRegressor(**p)
    m.fit(X, y[:, ti])
    train_pred[:, ti] = m.predict(X)
    imp = pd.DataFrame({"feature": names, "gain": m.feature_importances_})
    imp.sort_values("gain", ascending=False).to_csv(
        OUT_DIR / f"feature_importance_{tname}.csv", index=False)

pred_df["delta_ws_tahmin_train"] = train_pred[:, 0].round(3)
pred_df["delta_t_tahmin_train"] = train_pred[:, 1].round(3)

print("[3.6/4] Tek train/test bolmesi (gorsellestirme icin)...")
train_mask, test_mask = ev.single_split(groups, meta, test_frac=0.20)
split_pred = np.full((N, 2), np.nan)
for ti in range(2):
    p = dict(PARAMS)
    p["n_estimators"] = max(int(np.median(best_iters[ti])), 10)
    m = lgb.LGBMRegressor(**p)
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
               "eksik_veri": "DOLDURULMADI - LightGBM NaN'i dogrudan isler",
               "medyan_en_iyi_agac": {k: int(np.median(v)) for k, v in best_iters.items()},
               "params": PARAMS}, f, ensure_ascii=False, indent=2)

print(f"Tamamlandi -> {OUT_DIR}")
