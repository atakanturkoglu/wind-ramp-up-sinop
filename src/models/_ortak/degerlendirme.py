# -*- coding: utf-8 -*-
"""
Tum modeller icin ORTAK degerlendirme cercevesi.

Amac: CNN-LSTM, XGBoost, CatBoost vb. modellerin BIREBIR ayni kosullarda
karsilastirilabilmesi. Kullanilan protokol:

  - Blok-farkinda bolme (block_id). 43 olay yalnizca 20 blokta;
    ayni bloktaki olaylarin 24 saatlik oncul pencereleri buyuk olcude
    ortustugu icin rastgele bolme sizinti yaratir.
  - 5 kat x 5 tekrar GroupKFold. Her olay, onu hic gormemis bir model
    tarafindan 5 kez tahmin edilir (out-of-fold).
  - Her dis katta, erken durdurma icin dis egitim gruplarindan ayri bir
    ic dogrulama seti ayrilir; skorlanan dis kat durdurma kararina
    katilmaz.
  - Her metrigin yaninda "hep egitim ortalamasini tahmin et" temel
    cizgisi raporlanir.

EKSIK VERI: Bu cerceve NaN'lari DOLDURMAZ. Matrisler NaN'li olarak
verilir; her model kendi dogal davranisiyla ele alir (agac modelleri
NaN'i dogrudan isler). Doldurma gerektiren modeller (sinir aglari) bunu
kendi icinde, SADECE kendi egitim katinin istatistikleriyle yapmalidir
ve bu durum ilgili script'te acikca belirtilmelidir.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 42
N_FOLDS = 5
N_REPEATS = 5
INNER_VAL_FRAC = 0.25

TARGETS = {
    "delta_ws": {"col": "delta_v", "unit": "m/s", "label": "Delta WS (ramp-up amplitude)"},
    "delta_t": {"col": "duration_min", "unit": "min", "label": "Delta T (ramp-up duration)"},
}


def load_data(matris_dir):
    """X (N,24,4,6) -> (N, 576) duz; y (N,2); groups (N,). NaN KORUNUR."""
    X = np.load(matris_dir / "X.npy").astype(np.float64)
    meta = pd.read_csv(matris_dir / "meta.csv", parse_dates=["t0", "t1"])
    N = X.shape[0]
    X_flat = X.reshape(N, -1)
    y = np.stack([meta["delta_v"].to_numpy(dtype=np.float64),
                  meta["duration_min"].to_numpy(dtype=np.float64)], axis=1)
    groups = meta["block_id"].to_numpy()
    return X_flat, y, groups, meta


def feature_names():
    STATIONS = ["merkez", "inceburun", "airport", "wl"]
    VARS = ["P", "T", "u", "v", "ws", "wd"]
    return [f"h{h:02d}_{s}_{v}" for h in range(24) for s in STATIONS for v in VARS]


def single_split(groups, meta, test_frac=0.20):
    """TEK, sabit, kronolojik ve blok-farkinda train/test bolmesi.

    SADECE olay-bazli GORSELLESTIRME icin kullanilir (her olayin ACIKCA
    ya train ya test oldugu tek bir kutu gosterebilmek icin) - metrikler
    (metrics.csv, karsilastirma tablosu) HALA yukaridaki tekrarli CV'den
    gelir, çünkü 43 olayda tek bir ~9 olayluk test seti R2'yi cok
    gurultulu yapar. Bu fonksiyon o istatistiksel kaygiyi degistirmez,
    sadece "bu olay hangi rolde gosteriliyor" sorusuna tek/net bir cevap
    verir.
    """
    block_first_t0 = meta.groupby("block_id")["t0"].min().sort_values()
    order = block_first_t0.index.tolist()
    counts = pd.Series(groups).value_counts()
    target_n = round(test_frac * len(groups))
    test_blocks, n_acc = [], 0
    for b in reversed(order):
        if n_acc >= target_n:
            break
        test_blocks.append(b)
        n_acc += int(counts.get(b, 0))
    test_mask = np.isin(groups, test_blocks)
    return ~test_mask, test_mask


def fold_iter(groups, seed=SEED, n_folds=N_FOLDS, n_repeats=N_REPEATS,
              inner_val_frac=INNER_VAL_FRAC):
    """(rep, fold, inner_tr_mask, inner_va_mask, test_mask) uretir."""
    uniq = np.unique(groups)
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        shuffled = rng.permutation(uniq)
        fold_of = {g: i % n_folds for i, g in enumerate(shuffled)}
        fold_id = np.array([fold_of[g] for g in groups])

        for f in range(n_folds):
            te = fold_id == f
            tr = ~te
            if te.sum() == 0 or tr.sum() < 5:
                continue
            tr_groups = np.unique(groups[tr])
            n_val = max(1, int(round(inner_val_frac * len(tr_groups))))
            val_groups = set(rng.permutation(tr_groups)[:n_val].tolist())
            inner_va = tr & np.array([g in val_groups for g in groups])
            inner_tr = tr & ~inner_va
            if inner_tr.sum() < 4 or inner_va.sum() < 1:
                inner_tr, inner_va = tr, tr
            yield rep, f, inner_tr, inner_va, te


def summarize(y, oof_pred, model_name, out_csv=None):
    """oof_pred: (n_repeats, N, 2). Model + temel cizgi metriklerini dondurur."""
    n_repeats = oof_pred.shape[0]
    rows = []
    for ti, tname in enumerate(TARGETS):
        per_rep = []
        for rep in range(n_repeats):
            p = oof_pred[rep, :, ti]
            per_rep.append({
                "RMSE": float(np.sqrt(mean_squared_error(y[:, ti], p))),
                "MAE": float(mean_absolute_error(y[:, ti], p)),
                "R2": float(r2_score(y[:, ti], p)),
            })
        d = pd.DataFrame(per_rep)
        rows.append({
            "target": tname, "model": model_name,
            "RMSE_ort": d["RMSE"].mean(), "RMSE_std": d["RMSE"].std(),
            "MAE_ort": d["MAE"].mean(), "MAE_std": d["MAE"].std(),
            "R2_ort": d["R2"].mean(), "R2_std": d["R2"].std(),
            "R2_min": d["R2"].min(), "R2_max": d["R2"].max(),
        })
        base = np.full(len(y), y[:, ti].mean())
        rows.append({
            "target": tname, "model": "Temel cizgi (ortalama)",
            "RMSE_ort": float(np.sqrt(mean_squared_error(y[:, ti], base))), "RMSE_std": 0.0,
            "MAE_ort": float(mean_absolute_error(y[:, ti], base)), "MAE_std": 0.0,
            "R2_ort": 0.0, "R2_std": 0.0, "R2_min": 0.0, "R2_max": 0.0,
        })
    out = pd.DataFrame(rows).round(4)
    if out_csv is not None:
        out.to_csv(out_csv, index=False)
    return out


def prediction_frame(meta, y, oof_pred):
    pm = np.nanmean(oof_pred, axis=0)
    ps = np.nanstd(oof_pred, axis=0)
    return pd.DataFrame({
        "event_id": meta["event_id"], "block_id": meta["block_id"],
        "t0": meta["t0"], "t1": meta["t1"],
        "ws0": meta["ws0"], "ws1": meta["ws1"],
        "delta_ws_gercek": y[:, 0], "delta_ws_tahmin": pm[:, 0], "delta_ws_tahmin_std": ps[:, 0],
        "delta_t_gercek": y[:, 1], "delta_t_tahmin": pm[:, 1], "delta_t_tahmin_std": ps[:, 1],
    }).round(3)


def scatter_png(y, oof_pred, metrics, model_name, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pm = np.nanmean(oof_pred, axis=0)
    ps = np.nanstd(oof_pred, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ti, (tname, info) in enumerate(TARGETS.items()):
        ax = axes[ti]
        true, pred, err = y[:, ti], pm[:, ti], ps[:, ti]
        lo = min(true.min(), pred.min()) - 0.5
        hi = max(true.max(), pred.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
        ax.axhline(true.mean(), color="#d62728", ls=":", lw=1.2, label="Baseline (mean)")
        ax.errorbar(true, pred, yerr=err, fmt="o", ms=6, color="#1f77b4",
                    ecolor="#9ecae1", elinewidth=1, capsize=2, alpha=0.9)
        r = metrics[(metrics["target"] == tname) & (metrics["model"] == model_name)].iloc[0]
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Actual ({info['unit']})")
        ax.set_ylabel(f"Predicted ({info['unit']})")
        ax.set_title(f"{info['label']}\nR2 = {r['R2_ort']:+.3f} ± {r['R2_std']:.3f}   "
                     f"RMSE = {r['RMSE_ort']:.2f}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{model_name} — Out-of-fold predictions (5-fold x 5 repeats, block-aware)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
