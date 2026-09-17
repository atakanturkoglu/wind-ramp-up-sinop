# -*- coding: utf-8 -*-
"""
CNN-LSTM ile ramp-up oncul matrislerinden delta_ws (ramp-up genligi) ve
delta_t (ramp-up suresi) ES ZAMANLI tahmini.

VERI: 43 olay (ham veriden tespit edilip elle dogrulanmis 63 olayin,
saatlik 4-istasyon verisiyle kesisen kismi). Girdi (24 saat, 24 kanal
= 4 istasyon x 6 degisken).

TASARIM GEREKCESI (43 ornekle calismanin getirdigi kisitlar)
------------------------------------------------------------
1) KUCUK MODEL. Onceki surumde hidden_size=32 ve kanal koruyan bir CNN
   vardi; toplam ~9.600 parametre. 43 ornek icin bu ornek basina ~220
   parametre demek ve model egitim setini ezberliyordu (test R2 negatif).
   Bu surumde CNN kanallari 24 -> 8'e indiriyor, LSTM gizli boyutu 8;
   toplam ~1.200 parametre (ornek basina ~28).

2) BLOK-FARKINDA BOLME. 43 olay yalnizca 20 farkli veri blogundan
   geliyor (bazi bloklarda 6 olay var). Ayni bloktaki olaylarin 24
   saatlik oncul pencereleri buyuk olcude ortusuyor; rastgele bolunurse
   model egitimde gordugu pencereyi testte tekrar gorur (sizinti).
   Bu yuzden tum bolmeler block_id'ye gore yapilir.

3) TEKRARLI CAPRAZ DOGRULAMA, TEK BIR TEST SETI DEGIL. 43 olayda %20'lik
   bir test seti ~9 olay demek; tek bir olay R2'yi ucurabilir. Bunun
   yerine 5 kat x 5 tekrar GroupKFold uygulanir: her olay 5 kez, onu hic
   gormemis bir model tarafindan tahmin edilir. Raporlanan skor bu
   tahminlerin ortalamasi VE tekrarlar arasi yayilimidir.

4) ERKEN DURDURMA ICIN AYRI IC BOLME. Her dis katta, egitim
   gruplarindan bir kismi ic dogrulama olarak ayrilir ve epoch secimi
   SADECE onunla yapilir. Boylece skorlanan dis kat, durdurma kararina
   hicbir sekilde katkida bulunmaz.

5) TEMEL CIZGI. Her metrigin yaninda "hep egitim ortalamasini tahmin et"
   modeli raporlanir. Model bunu gecemiyorsa ogrenme yok demektir.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_ortak"))
from degerlendirme import single_split  # noqa: E402
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

STATIONS = ["merkez", "inceburun", "airport", "wl"]
VARS = ["P", "T", "u", "v", "ws", "wd"]
N_HOURS, N_STATIONS, N_VARS = 24, 4, 6
N_CH = N_STATIONS * N_VARS          # 24 kanal

MATRIS_DIR = Path(PROJECT_ROOT, 'src', 'feature_matrix')
OUT_DIR = Path(PROJECT_ROOT, 'src', 'models', 'cnn_lstm')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- mimari (kucultulmus) ---
CONV_OUT = 8          # CNN cikis kanali (24 -> 8 boyut indirgeme)
KERNEL = 3            # 3 saatlik yerel pencere
HIDDEN = 8            # LSTM gizli boyutu
DROPOUT = 0.30

# --- egitim ---
LR = 1e-3
WEIGHT_DECAY = 1e-3   # kucuk veride daha guclu duzenlilestirme
BATCH_SIZE = 8
MAX_EPOCHS = 300
PATIENCE = 25
DEVICE = torch.device("cpu")

# --- degerlendirme ---
N_FOLDS = 5
N_REPEATS = 5
INNER_VAL_FRAC = 0.25   # dis katin egitim gruplarinin bu kadari erken durdurma icin

TARGETS = {
    "delta_ws": {"col": "delta_v", "unit": "m/s", "label": "Delta WS (ramp-up amplitude)"},
    "delta_t": {"col": "duration_min", "unit": "min", "label": "Delta T (ramp-up duration)"},
}


# ---------------------------------------------------------------------------
# 1. Veri
# ---------------------------------------------------------------------------
print("[1/5] Veri yukleniyor...")
X_raw = np.load(MATRIS_DIR / "X.npy").astype(np.float32)      # (N,24,4,6)
meta = pd.read_csv(MATRIS_DIR / "meta.csv", parse_dates=["t0"])
N = X_raw.shape[0]
assert N == len(meta)

X_seq = X_raw.reshape(N, N_HOURS, N_CH)                        # (N,24,24)
y = np.stack([meta["delta_v"].to_numpy(dtype=np.float32),
              meta["duration_min"].to_numpy(dtype=np.float32)], axis=1)
groups = meta["block_id"].to_numpy()
uniq_groups = np.unique(groups)

print(f"  X: {X_seq.shape}   olay={N}   blok={len(uniq_groups)}   NaN orani={np.isnan(X_seq).mean():.1%}")
print(f"  delta_ws: ort={y[:,0].mean():.2f}  std={y[:,0].std():.2f}")
print(f"  delta_t : ort={y[:,1].mean():.2f}  std={y[:,1].std():.2f}")


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
class CNNLSTM(nn.Module):
    """Conv1d ile kanal indirgeme -> LSTM -> 2 cikis. ~1.200 parametre."""

    def __init__(self, in_ch=N_CH, conv_out=CONV_OUT, hidden=HIDDEN,
                 dropout=DROPOUT, kernel=KERNEL, n_out=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, conv_out, kernel_size=kernel, padding=kernel // 2)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(conv_out, hidden, num_layers=1, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_out)

    def forward(self, x):                    # x: (B, T, C)
        c = self.drop1(self.act(self.conv(x.transpose(1, 2))))
        _, (h_n, _) = self.lstm(c.transpose(1, 2))
        return self.fc(self.drop2(h_n[-1]))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 3. Yardimcilar
# ---------------------------------------------------------------------------
def fit_stats(X_arr):
    flat = X_arr.reshape(-1, X_arr.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1e-6
    mean[~np.isfinite(mean)] = 0.0
    return mean, std


def apply_stats(X_arr, mean, std):
    imputed = np.where(np.isnan(X_arr), mean, X_arr)
    return ((imputed - mean) / std).astype(np.float32)


def train_fold(X_tr, y_tr, X_va, y_va):
    """Bir modeli egitir; en iyi epoch'u SADECE ic dogrulama ile secer."""
    model = CNNLSTM().to(DEVICE)
    crit = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=min(BATCH_SIZE, len(X_tr)), shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    Xv = torch.from_numpy(X_va).to(DEVICE)
    yv = torch.from_numpy(y_va).to(DEVICE)

    best_loss, best_state, best_epoch, bad = np.inf, None, 0, 0
    hist = {"train": [], "val": []}
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad()
            p = model(xb)
            loss = crit(p[:, 0], yb[:, 0]) + crit(p[:, 1], yb[:, 1])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            pv = model(Xv)
            vloss = (crit(pv[:, 0], yv[:, 0]) + crit(pv[:, 1], yv[:, 1])).item()
        hist["train"].append(float(np.mean(losses)))
        hist["val"].append(vloss)
        if vloss < best_loss - 1e-4:
            best_loss, best_epoch, bad = vloss, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_epoch


# ---------------------------------------------------------------------------
# 4. Tekrarli, blok-farkinda capraz dogrulama
# ---------------------------------------------------------------------------
print(f"[2/5] {N_FOLDS} kat x {N_REPEATS} tekrar, blok-farkinda CV basliyor "
      f"({N_FOLDS * N_REPEATS} model egitilecek)...")
print(f"  Model parametre sayisi: {n_params(CNNLSTM()):,}  (ornek basina "
      f"{n_params(CNNLSTM()) / N:.0f})")

oof_pred = np.full((N_REPEATS, N, 2), np.nan, dtype=np.float32)
all_hists, best_epochs = [], []

for rep in range(N_REPEATS):
    rng = np.random.default_rng(SEED + rep)
    shuffled = rng.permutation(uniq_groups)
    fold_of_group = {g: i % N_FOLDS for i, g in enumerate(shuffled)}
    fold_id = np.array([fold_of_group[g] for g in groups])

    for f in range(N_FOLDS):
        te_mask = fold_id == f
        tr_mask = ~te_mask
        if te_mask.sum() == 0 or tr_mask.sum() < 5:
            continue

        # ic dogrulama: dis egitim gruplarinin bir kismi (erken durdurma icin)
        tr_groups = np.unique(groups[tr_mask])
        n_val_g = max(1, int(round(INNER_VAL_FRAC * len(tr_groups))))
        val_groups = set(rng.permutation(tr_groups)[:n_val_g].tolist())
        inner_va = tr_mask & np.array([g in val_groups for g in groups])
        inner_tr = tr_mask & ~inner_va
        if inner_tr.sum() < 4 or inner_va.sum() < 1:
            inner_tr, inner_va = tr_mask, tr_mask   # cok kucukse ayni seti kullan

        fmean, fstd = fit_stats(X_seq[inner_tr])
        ymean = y[inner_tr].mean(axis=0)
        ystd = y[inner_tr].std(axis=0)
        ystd[ystd < 1e-6] = 1.0

        Xtr = apply_stats(X_seq[inner_tr], fmean, fstd)
        Xva = apply_stats(X_seq[inner_va], fmean, fstd)
        Xte = apply_stats(X_seq[te_mask], fmean, fstd)
        ytr = ((y[inner_tr] - ymean) / ystd).astype(np.float32)
        yva = ((y[inner_va] - ymean) / ystd).astype(np.float32)

        model, hist, best_ep = train_fold(Xtr, ytr, Xva, yva)
        best_epochs.append(best_ep)
        all_hists.append(hist)

        model.eval()
        with torch.no_grad():
            pte = model(torch.from_numpy(Xte).to(DEVICE)).cpu().numpy()
        oof_pred[rep, te_mask, :] = pte * ystd + ymean

    r2ws = r2_score(y[:, 0], oof_pred[rep, :, 0])
    r2t = r2_score(y[:, 1], oof_pred[rep, :, 1])
    print(f"    tekrar {rep + 1}/{N_REPEATS}:  R2(delta_ws)={r2ws:+.3f}   R2(delta_t)={r2t:+.3f}")


# ---------------------------------------------------------------------------
# 4.5. TRAIN (ic-ornek) tahmini: tum 43 olayla egitilmis TEK final model
#
# Bu model TUM olaylari egitim sirasinda gormustur; ayni olaylar uzerindeki
# tahmini ("train"), OOF tahminiyle ("test") karsilastirilarak ezberleme
# miktari gorsellestirilebilir. Epoch sayisi, CV'nin medyan en-iyi-epoch
# degeridir (yukaridaki ile ayni mantik).
# ---------------------------------------------------------------------------
print("[2.5/5] Train (ic-ornek) tahmini icin tum veriyle final model egitiliyor...")
final_epoch = max(int(np.median(best_epochs)), 5)
fmean_f, fstd_f = fit_stats(X_seq)
ymean_f, ystd_f = y.mean(axis=0), y.std(axis=0)
ystd_f[ystd_f < 1e-6] = 1.0
X_f = apply_stats(X_seq, fmean_f, fstd_f)
y_f = ((y - ymean_f) / ystd_f).astype(np.float32)

final_model = CNNLSTM().to(DEVICE)
crit_f = nn.SmoothL1Loss()
opt_f = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loader_f = DataLoader(TensorDataset(torch.from_numpy(X_f), torch.from_numpy(y_f)),
                      batch_size=min(BATCH_SIZE, len(X_f)), shuffle=True,
                      generator=torch.Generator().manual_seed(SEED))
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

# ---------------------------------------------------------------------------
# 4.6. TEK, SABIT train/test bolmesi (sadece olay-bazli gorsellestirme icin)
#
# Yukaridaki CV, metrikler (R2 tablosu) icin kullanilir - 43 olayda tek bir
# ~9 olayluk test seti cok gurultulu olur. Ama grafikte HER olayin ACIKCA
# ya train ya test oldugunu gostermek icin, burada AYRICA tek/sabit bir
# bolme yapilip SADECE o modelin tahmini kaydedilir (iki kutu degil, bir).
# ---------------------------------------------------------------------------
print("[2.6/5] Tek train/test bolmesi (gorsellestirme icin)...")
train_mask, test_mask = single_split(groups, meta, test_frac=0.20)

fmean_s, fstd_s = fit_stats(X_seq[train_mask])
ymean_s, ystd_s = y[train_mask].mean(axis=0), y[train_mask].std(axis=0)
ystd_s[ystd_s < 1e-6] = 1.0
X_s = apply_stats(X_seq[train_mask], fmean_s, fstd_s)
y_s = ((y[train_mask] - ymean_s) / ystd_s).astype(np.float32)

split_model = CNNLSTM().to(DEVICE)
crit_s = nn.SmoothL1Loss()
opt_s = torch.optim.AdamW(split_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loader_s = DataLoader(TensorDataset(torch.from_numpy(X_s), torch.from_numpy(y_s)),
                      batch_size=min(BATCH_SIZE, len(X_s)), shuffle=True,
                      generator=torch.Generator().manual_seed(SEED))
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

print("[3/5] Metrikler hesaplaniyor...")

rows = []
for ti, tname in enumerate(TARGETS):
    per_rep = []
    for rep in range(N_REPEATS):
        p = oof_pred[rep, :, ti]
        per_rep.append({
            "RMSE": float(np.sqrt(mean_squared_error(y[:, ti], p))),
            "MAE": float(mean_absolute_error(y[:, ti], p)),
            "R2": float(r2_score(y[:, ti], p)),
        })
    dfp = pd.DataFrame(per_rep)

    # temel cizgi: hep ortalamayi tahmin et (ayni CV cercevesinde)
    base_pred = np.full(N, y[:, ti].mean(), dtype=np.float32)
    rows.append({
        "target": tname, "model": "CNN-LSTM",
        "RMSE_ort": dfp["RMSE"].mean(), "RMSE_std": dfp["RMSE"].std(),
        "MAE_ort": dfp["MAE"].mean(), "MAE_std": dfp["MAE"].std(),
        "R2_ort": dfp["R2"].mean(), "R2_std": dfp["R2"].std(),
        "R2_min": dfp["R2"].min(), "R2_max": dfp["R2"].max(),
    })
    rows.append({
        "target": tname, "model": "Temel cizgi (ortalama)",
        "RMSE_ort": float(np.sqrt(mean_squared_error(y[:, ti], base_pred))),
        "RMSE_std": 0.0,
        "MAE_ort": float(mean_absolute_error(y[:, ti], base_pred)), "MAE_std": 0.0,
        "R2_ort": 0.0, "R2_std": 0.0, "R2_min": 0.0, "R2_max": 0.0,
    })

metrics = pd.DataFrame(rows).round(4)
metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
print(metrics.to_string(index=False))

# olay bazli tahminler (tekrarlar arasi ortalama)
pred_mean = np.nanmean(oof_pred, axis=0)
pred_std = np.nanstd(oof_pred, axis=0)
pred_df = pd.DataFrame({
    "event_id": meta["event_id"], "block_id": meta["block_id"],
    "t0": meta["t0"], "t1": meta["t1"],
    "ws0": meta["ws0"], "ws1": meta["ws1"],
    "delta_ws_gercek": y[:, 0], "delta_ws_tahmin": pred_mean[:, 0],
    "delta_ws_tahmin_std": pred_std[:, 0],
    "delta_t_gercek": y[:, 1], "delta_t_tahmin": pred_mean[:, 1],
    "delta_t_tahmin_std": pred_std[:, 1],
    "delta_ws_tahmin_train": train_pred[:, 0], "delta_t_tahmin_train": train_pred[:, 1],
}).round(3)
pred_df["split"] = np.where(train_mask, "train", "test")
pred_df["delta_ws_tahmin_split"] = np.round(split_pred[:, 0], 3)
pred_df["delta_t_tahmin_split"] = np.round(split_pred[:, 1], 3)
pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)

with open(OUT_DIR / "split_info.json", "w", encoding="utf-8") as f:
    json.dump({
        "n_olay": int(N), "n_blok": int(len(uniq_groups)),
        "n_fold": N_FOLDS, "n_tekrar": N_REPEATS,
        "parametre_sayisi": int(n_params(CNNLSTM())),
        "medyan_en_iyi_epoch": int(np.median(best_epochs)),
        "mimari": {"conv_out": CONV_OUT, "kernel": KERNEL, "hidden": HIDDEN,
                    "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY},
    }, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 6. Gorseller
# ---------------------------------------------------------------------------
print("[4/5] Gorseller olusturuluyor...")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ti, (tname, info) in enumerate(TARGETS.items()):
    ax = axes[ti]
    true = y[:, ti]
    pred = pred_mean[:, ti]
    err = pred_std[:, ti]
    lo = min(true.min(), pred.min()) - 0.5
    hi = max(true.max(), pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.axhline(true.mean(), color="#d62728", ls=":", lw=1.2,
               label="Baseline (train mean)")
    ax.errorbar(true, pred, yerr=err, fmt="o", ms=6, color="#1f77b4",
                ecolor="#9ecae1", elinewidth=1, capsize=2, alpha=0.9)
    row = metrics[(metrics["target"] == tname) & (metrics["model"] == "CNN-LSTM")].iloc[0]
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Actual ({info['unit']})")
    ax.set_ylabel(f"Predicted ({info['unit']})")
    ax.set_title(f"{info['label']}\n"
                 f"R2 = {row['R2_ort']:+.3f} ± {row['R2_std']:.3f}   "
                 f"RMSE = {row['RMSE_ort']:.2f}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("CNN-LSTM — Out-of-fold predictions (5-fold x 5 repeats, block-aware)",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "oof_scatter.png", dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.5))
for h in all_hists[:N_FOLDS]:
    ax.plot(h["train"], color="#1f77b4", lw=1, ls="--", alpha=0.6)
    ax.plot(h["val"], color="#d62728", lw=1.4, alpha=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Huber loss (both targets)")
ax.set_title("CNN-LSTM learning curves (first repeat, 5 folds)\n"
             "dashed = train, solid = inner validation")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "learning_curve.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# Permutasyon onem analizi: tum veriyle egitilmis final model (4.5. bolum)
# kullanilir. Bir kanal (istasyon x degisken, 24 saatin tamaminda birden)
# olaylar arasinda karistirilir, RMSE'nin ne kadar arttigina bakilir.
# ---------------------------------------------------------------------------
print("  Permutasyon onem analizi hesaplaniyor...")
DISPLAY_STATION_ORDER = ["inceburun", "airport", "merkez", "wl"]
DISPLAY_STATION_LABELS = {"inceburun": "Inceburun", "airport": "Airport",
                           "merkez": "Sinop", "wl": "OB"}

with torch.no_grad():
    base_pred = final_model(torch.from_numpy(X_f).to(DEVICE)).cpu().numpy()
base_p = base_pred * ystd_f + ymean_f
base_rmse = [float(np.sqrt(mean_squared_error(y[:, ti], base_p[:, ti]))) for ti in range(2)]

rng = np.random.default_rng(SEED)
imp = np.zeros((2, N_CH))
for col in range(N_CH):
    Xp = X_f.copy()
    perm = rng.permutation(N)
    Xp[:, :, col] = Xp[perm, :, col]
    with torch.no_grad():
        p = final_model(torch.from_numpy(Xp).to(DEVICE)).cpu().numpy()
    p_real = p * ystd_f + ymean_f
    for ti in range(2):
        imp[ti, col] = float(np.sqrt(mean_squared_error(y[:, ti], p_real[:, ti]))) - base_rmse[ti]

display_idx = [STATIONS.index(s) for s in DISPLAY_STATION_ORDER]
for ti, tname in enumerate(TARGETS):
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
    ax.set_title(f"CNN-LSTM Feature Importance (Permutation) — {TARGETS[tname]['label']}\n"
                 f"(base RMSE={base_rmse[ti]:.3f})", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"feature_importance_{tname}.png", dpi=140)
    plt.close(fig)

    pd.DataFrame(imp[ti].reshape(N_STATIONS, N_VARS), index=STATIONS, columns=VARS
                 ).to_csv(OUT_DIR / f"feature_importance_{tname}.csv")

print(f"[5/5] Tamamlandi -> {OUT_DIR}")
