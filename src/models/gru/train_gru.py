# -*- coding: utf-8 -*-
"""
GRU (Gated Recurrent Unit; Cho vd., 2014) ile delta_ws ve delta_t
tahmini - LSTM/CNN-LSTM'e ek, ucuncu bir tekrarlayan sinir agi varyanti.

Neden GRU: Bu oturumda (bkz. model/lstm ve model/cnn_lstm) 43 olayla
egitilen sinir aglarinda DAHA KUCUK kapasitenin daha iyi (ya da en azindan
daha az kotu) sonuc verdigi gozlemlendi (32 gizli boyutlu eski model
~9.600 parametreyle ezberliyordu; 8 gizli boyutlu LSTM/CNN-LSTM ~1.100
parametreyle cok daha iyiydi). GRU, LSTM'in ayni ailesinden, literatürde
yerlesik (Cho vd., 2014) ama yapisal olarak daha sade bir alternatiftir:
LSTM'in 4 kapisi (giris, unutma, cikis, aday hucre) yerine 3 kapisi
(guncelleme, sifirlama, aday) vardir ve ayri bir hucre durumu (cell
state) tasimaz. Ayni gizli boyutta (HIDDEN=8) LSTM'den ~%25 daha az
parametreye sahiptir (~830 vs ~1.106) - bu FARKI KUCULTMEK icin gizli
boyut buyutulmemistir; tam tersine, "kucuk kapasite kazaniyor" bulgusuyla
ayni yonde, dogal ve disclosed bir sonuctur.

EKSIK VERI: Sinir aglari NaN isleyemedigi icin doldurma ZORUNLU. Her dis
katta SADECE o katin ic egitim bolumunun ortalamasiyla doldurulur
(sizinti yok). Agac modellerinde ise hic doldurma yapilmiyor - bu fark
karsilastirma yorumlanirken goz onunde tutulmalidir.

Degerlendirme: model/_ortak/degerlendirme.py - diger tum modellerle
BIREBIR ayni bolmeler.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_ortak"))
import degerlendirme as ev  # noqa: E402

MATRIS_DIR = Path(PROJECT_ROOT, 'src', 'feature_matrix')
OUT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "GRU"

STATIONS = ["merkez", "inceburun", "airport", "wl"]
VARS = ["P", "T", "u", "v", "ws", "wd"]
N_HOURS, N_STATIONS, N_VARS = 24, 4, 6
N_CH = N_STATIONS * N_VARS
HIDDEN = 8
DROPOUT = 0.30
LR = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 8
MAX_EPOCHS = 300
PATIENCE = 25
DEVICE = torch.device("cpu")

torch.manual_seed(ev.SEED)
np.random.seed(ev.SEED)


class PlainGRU(nn.Module):
    def __init__(self, in_ch=N_CH, hidden=HIDDEN, dropout=DROPOUT, n_out=2):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_out)

    def forward(self, x):                 # (B, T, C)
        _, h_n = self.gru(x)              # GRU: tek h_n donuyor, (LSTM'deki
        return self.fc(self.drop(h_n[-1]))  # gibi ayri c_n yok)


def fit_stats(X):
    flat = X.reshape(-1, X.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1e-6
    mean[~np.isfinite(mean)] = 0.0
    return mean, std


def apply_stats(X, mean, std):
    return ((np.where(np.isnan(X), mean, X) - mean) / std).astype(np.float32)


def train_fold(Xtr, ytr, Xva, yva):
    model = PlainGRU().to(DEVICE)
    crit = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=min(BATCH_SIZE, len(Xtr)), shuffle=True,
                        generator=torch.Generator().manual_seed(ev.SEED))
    Xv, yv = torch.from_numpy(Xva).to(DEVICE), torch.from_numpy(yva).to(DEVICE)
    best, best_state, bad, best_epoch = np.inf, None, 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            p = model(xb)
            (crit(p[:, 0], yb[:, 0]) + crit(p[:, 1], yb[:, 1])).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv)
            v = (crit(pv[:, 0], yv[:, 0]) + crit(pv[:, 1], yv[:, 1])).item()
        if v < best - 1e-4:
            best, bad, best_epoch = v, 0, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


print("[1/3] Veri yukleniyor...")
X_flat, y, groups, meta = ev.load_data(MATRIS_DIR)
N = len(y)
X_seq = X_flat.reshape(N, N_HOURS, N_CH).astype(np.float64)
n_par = sum(p.numel() for p in PlainGRU().parameters())
print(f"  X: {X_seq.shape}   olay={N}   blok={len(np.unique(groups))}   "
      f"parametre={n_par:,}  (ornek basina {n_par / N:.0f})")

print(f"[2/3] Blok-farkinda {ev.N_FOLDS} kat x {ev.N_REPEATS} tekrar CV...")
oof = np.full((ev.N_REPEATS, N, 2), np.nan)
best_epochs = []

for rep, f, inner_tr, inner_va, te in ev.fold_iter(groups):
    fmean, fstd = fit_stats(X_seq[inner_tr])
    ymean, ystd = y[inner_tr].mean(axis=0), y[inner_tr].std(axis=0)
    ystd[ystd < 1e-6] = 1.0

    Xtr = apply_stats(X_seq[inner_tr], fmean, fstd)
    Xva = apply_stats(X_seq[inner_va], fmean, fstd)
    Xte = apply_stats(X_seq[te], fmean, fstd)
    ytr = ((y[inner_tr] - ymean) / ystd).astype(np.float32)
    yva = ((y[inner_va] - ymean) / ystd).astype(np.float32)

    model, best_ep = train_fold(Xtr, ytr, Xva, yva)
    best_epochs.append(best_ep)
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(Xte).to(DEVICE)).cpu().numpy()
    oof[rep, te, :] = p * ystd + ymean

    if f == ev.N_FOLDS - 1:
        from sklearn.metrics import r2_score
        print(f"    tekrar {rep + 1}/{ev.N_REPEATS}:  "
              f"R2(delta_ws)={r2_score(y[:, 0], oof[rep, :, 0]):+.3f}   "
              f"R2(delta_t)={r2_score(y[:, 1], oof[rep, :, 1]):+.3f}")

# TRAIN (ic-ornek) tahmini: tum 43 olayla egitilmis TEK final model.
print("[2.5/3] Train (ic-ornek) tahmini icin tum veriyle final model...")
final_epoch = max(int(np.median(best_epochs)), 5)
fmean_f, fstd_f = fit_stats(X_seq)
ymean_f, ystd_f = y.mean(axis=0), y.std(axis=0)
ystd_f[ystd_f < 1e-6] = 1.0
X_f = apply_stats(X_seq, fmean_f, fstd_f)
y_f = ((y - ymean_f) / ystd_f).astype(np.float32)

final_model = PlainGRU().to(DEVICE)
crit_f = nn.SmoothL1Loss()
opt_f = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loader_f = DataLoader(TensorDataset(torch.from_numpy(X_f), torch.from_numpy(y_f)),
                      batch_size=min(BATCH_SIZE, len(X_f)), shuffle=True,
                      generator=torch.Generator().manual_seed(ev.SEED))
for _ in range(final_epoch):
    final_model.train()
    for xb, yb in loader_f:
        opt_f.zero_grad()
        p = final_model(xb)
        (crit_f(p[:, 0], yb[:, 0]) + crit_f(p[:, 1], yb[:, 1])).backward()
        opt_f.step()
final_model.eval()
with torch.no_grad():
    train_pred_scaled = final_model(torch.from_numpy(X_f).to(DEVICE)).cpu().numpy()
train_pred = train_pred_scaled * ystd_f + ymean_f

# TEK, SABIT train/test bolmesi (sadece olay-bazli gorsellestirme icin).
print("[2.6/3] Tek train/test bolmesi (gorsellestirme icin)...")
train_mask, test_mask = ev.single_split(groups, meta, test_frac=0.20)
fmean_s, fstd_s = fit_stats(X_seq[train_mask])
ymean_s, ystd_s = y[train_mask].mean(axis=0), y[train_mask].std(axis=0)
ystd_s[ystd_s < 1e-6] = 1.0
X_s = apply_stats(X_seq[train_mask], fmean_s, fstd_s)
y_s = ((y[train_mask] - ymean_s) / ystd_s).astype(np.float32)

split_model = PlainGRU().to(DEVICE)
crit_s = nn.SmoothL1Loss()
opt_s = torch.optim.AdamW(split_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loader_s = DataLoader(TensorDataset(torch.from_numpy(X_s), torch.from_numpy(y_s)),
                      batch_size=min(BATCH_SIZE, len(X_s)), shuffle=True,
                      generator=torch.Generator().manual_seed(ev.SEED))
for _ in range(final_epoch):
    split_model.train()
    for xb, yb in loader_s:
        opt_s.zero_grad()
        p = split_model(xb)
        (crit_s(p[:, 0], yb[:, 0]) + crit_s(p[:, 1], yb[:, 1])).backward()
        opt_s.step()
split_model.eval()
X_all_s = apply_stats(X_seq, fmean_s, fstd_s)
with torch.no_grad():
    split_pred_scaled = split_model(torch.from_numpy(X_all_s).to(DEVICE)).cpu().numpy()
split_pred = split_pred_scaled * ystd_s + ymean_s
print(f"  train: {int(train_mask.sum())} olay   test: {int(test_mask.sum())} olay")

print("[3/3] Metrikler...")
metrics = ev.summarize(y, oof, MODEL_NAME, out_csv=OUT_DIR / "metrics.csv")
print(metrics.to_string(index=False))
pred_df = ev.prediction_frame(meta, y, oof)
pred_df["delta_ws_tahmin_train"] = train_pred[:, 0].round(3)
pred_df["delta_t_tahmin_train"] = train_pred[:, 1].round(3)
pred_df["split"] = np.where(train_mask, "train", "test")
pred_df["delta_ws_tahmin_split"] = np.round(split_pred[:, 0], 3)
pred_df["delta_t_tahmin_split"] = np.round(split_pred[:, 1], 3)
pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)
ev.scatter_png(y, oof, metrics, MODEL_NAME, OUT_DIR / "oof_scatter.png")

with open(OUT_DIR / "split_info.json", "w", encoding="utf-8") as f:
    json.dump({"n_olay": int(N), "n_blok": int(len(np.unique(groups))),
               "parametre_sayisi": int(n_par),
               "eksik_veri": "Dolduruldu (sadece ic egitim katinin ortalamasi) - "
                             "sinir agi NaN isleyemez",
               "mimari": {"hidden": HIDDEN, "dropout": DROPOUT,
                          "weight_decay": WEIGHT_DECAY, "hucre_tipi": "GRU"}},
              f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Permutasyon onem analizi: tum veriyle egitilmis final model kullanilir.
# ---------------------------------------------------------------------------
print("Permutasyon onem analizi hesaplaniyor...")
from sklearn.metrics import mean_squared_error as _mse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

DISPLAY_STATION_ORDER = ["inceburun", "airport", "merkez", "wl"]
DISPLAY_STATION_LABELS = {"inceburun": "Inceburun", "airport": "Airport",
                           "merkez": "Sinop", "wl": "OB"}

with torch.no_grad():
    base_pred = final_model(torch.from_numpy(X_f).to(DEVICE)).cpu().numpy()
base_p = base_pred * ystd_f + ymean_f
base_rmse = [float(np.sqrt(_mse(y[:, ti], base_p[:, ti]))) for ti in range(2)]

rng = np.random.default_rng(ev.SEED)
imp = np.zeros((2, N_CH))
for col in range(N_CH):
    Xp = X_f.copy()
    perm = rng.permutation(N)
    Xp[:, :, col] = Xp[perm, :, col]
    with torch.no_grad():
        p = final_model(torch.from_numpy(Xp).to(DEVICE)).cpu().numpy()
    p_real = p * ystd_f + ymean_f
    for ti in range(2):
        imp[ti, col] = float(np.sqrt(_mse(y[:, ti], p_real[:, ti]))) - base_rmse[ti]

display_idx = [STATIONS.index(s) for s in DISPLAY_STATION_ORDER]
target_labels = {"delta_ws": "Delta WS (ramp-up amplitude)", "delta_t": "Delta T (ramp-up duration)"}
for ti, tname in enumerate(["delta_ws", "delta_t"]):
    imp_grid = imp[ti].reshape(N_STATIONS, N_VARS)
    imp_display = np.clip(imp_grid[display_idx, :], 0, None)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    vmax = imp_display.max() or 1.0
    im = ax.imshow(imp_display, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(N_VARS)); ax.set_xticklabels(VARS, fontsize=11)
    ax.set_yticks(range(len(DISPLAY_STATION_ORDER)))
    ax.set_yticklabels([DISPLAY_STATION_LABELS[s] for s in DISPLAY_STATION_ORDER], fontsize=11)
    for si in range(len(DISPLAY_STATION_ORDER)):
        for vi in range(N_VARS):
            val = imp_display[si, vi]
            ax.text(vi, si, f"{val:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if val > vmax * 0.5 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("RMSE increase after permutation — larger = more important", fontsize=9)
    ax.set_title(f"GRU Feature Importance (Permutation) — {target_labels[tname]}\n"
                 f"(base RMSE={base_rmse[ti]:.3f})", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"feature_importance_{tname}.png", dpi=140)
    plt.close(fig)

    pd.DataFrame(imp[ti].reshape(N_STATIONS, N_VARS), index=STATIONS, columns=VARS
                 ).to_csv(OUT_DIR / f"feature_importance_{tname}.csv")

print(f"Tamamlandi -> {OUT_DIR}")
