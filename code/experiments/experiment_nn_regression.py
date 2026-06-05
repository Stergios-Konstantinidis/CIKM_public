"""
experiment_nn_regression.py
============================
Experiment: MLP Neural Network to predict OCR accuracy (WER & CER).

Dataset
-------
Both baselines (easyocr + tesseract) are pooled into one dataset (~1200 samples).
An extra binary feature `engine_flag` (0=easyocr, 1=tesseract) is appended.
Split: 60% train (used for training + internal 10% val for early stopping)
       40% test  (held out, evaluated once at the end)

Architecture
------------
  Input → BN → [512 → BN → GELU → Drop(0.3)] → [256 → BN → GELU → Drop(0.3)]
       → [128 → BN → GELU → Drop(0.2)] → head_wer(1) + head_cer(1)

Usage
-----
    python code/experiment_nn_regression.py [--no-embeddings] [--epochs 300] [--seed 42]
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.regression_features import (
    build_dataset, SURFACE_FEATURE_NAMES, ALL_FEATURE_NAMES, METADATA_FEATURE_NAMES,
    load_metadata_lookup, load_confidence_lookup, enrich_records
)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
GROUNDTRUTH_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation_dataset" / "groundtruth.json"
BASELINE_FILES = {
    "easyocr":   RESULTS_DIR / "baselines/baseline_easyocr.json",
    "tesseract": RESULTS_DIR / "baselines/baseline_tesseract.json",
}
ENGINE_FLAG = {"easyocr": 0.0, "tesseract": 1.0}


# ─── model ───────────────────────────────────────────────────────────────────

class OCRRegressionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(512, 256, 128), dropouts=(0.3, 0.3, 0.2)):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(in_dim)
        layers = []
        prev = in_dim
        for h, d in zip(hidden_dims, dropouts):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(d)]
            prev = h
        self.backbone  = nn.Sequential(*layers)
        self.head_wer  = nn.Linear(prev, 1)
        self.head_cer  = nn.Linear(prev, 1)

    def forward(self, x):
        h   = self.backbone(self.input_norm(x))
        return self.head_wer(h).squeeze(1), self.head_cer(h).squeeze(1)


# ─── helpers ─────────────────────────────────────────────────────────────────

def load_all_records() -> list[dict]:
    """Pool both baselines and enrich with groundtruth metadata and confidence."""
    records = []
    for engine, path in BASELINE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Baseline not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        valid = [r for r in data if r["wer"] <= 5.0 and r["raw_ocr"].strip()]
        for r in valid:
            r["engine"] = engine
        records.extend(valid)
        print(f"  [{engine}] {len(valid)} valid samples")
    meta_lookup = load_metadata_lookup(GROUNDTRUTH_PATH)
    for engine, path in BASELINE_FILES.items():
        conf_lookup = load_confidence_lookup(engine, RESULTS_DIR)
        if conf_lookup:
            print(f"  [{engine}] confidence lookup: {len(conf_lookup)} entries")
        engine_recs = [r for r in records if r["engine"] == engine]
        enrich_records(engine_recs, meta_lookup, confidence_lookup=conf_lookup)
    print(f"  Combined pool: {len(records)} samples")
    return records


def stratified_split(records, test_size=0.4, random_state=42):
    """
    60/40 split keyed on FILENAME so the same document always lands in the
    same partition for both OCR engines.
    Stratification is on the per-filename mean WER quartile.
    """
    from collections import defaultdict
    by_fname: dict[str, list] = defaultdict(list)
    for r in records:
        by_fname[r["filename"]].append(r)

    fnames    = sorted(by_fname.keys())
    mean_wer  = np.array([np.mean([r["wer"] for r in by_fname[f]]) for f in fnames])
    quartiles = np.digitize(mean_wer, np.percentile(mean_wer, [25, 50, 75]))

    tr_fnames, te_fnames = train_test_split(
        fnames, test_size=test_size, stratify=quartiles, random_state=random_state
    )
    tr_set, te_set = set(tr_fnames), set(te_fnames)
    return [r for r in records if r["filename"] in tr_set], \
           [r for r in records if r["filename"] in te_set]



def evaluate(y_true, y_pred, label=""):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r, p = pearsonr(y_true, y_pred)
    print(f"    {label:5s}  R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  r={r:.4f} (p={p:.2e})")
    return {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse),
            "pearson_r": float(r), "pearson_p": float(p)}


def build_X(recs, use_embeddings):
    X, y_wer, y_cer = build_dataset(recs, use_embeddings=use_embeddings, verbose=False)
    engine_col = np.array(
        [ENGINE_FLAG[r["engine"]] for r in recs], dtype=np.float32
    ).reshape(-1, 1)
    return np.concatenate([X, engine_col], axis=1), y_wer, y_cer


# ─── permutation importance ───────────────────────────────────────────────────

def permutation_importance(model, X_t, y_wer_t, y_cer_t, feat_names, device, n_rep=8, top_k=15):
    model.eval()
    with torch.no_grad():
        pw, pc = model(X_t)
    base = (mean_squared_error(y_wer_t.cpu(), pw.cpu().numpy())
            + mean_squared_error(y_cer_t.cpu(), pc.cpu().numpy()))

    rng    = np.random.default_rng(42)
    X_np   = X_t.cpu().numpy().copy()
    imps   = np.zeros(X_t.shape[1])

    for fi in range(X_t.shape[1]):
        deltas = []
        for _ in range(n_rep):
            Xp = X_np.copy(); rng.shuffle(Xp[:, fi])
            with torch.no_grad():
                pw2, pc2 = model(torch.tensor(Xp, dtype=torch.float32, device=device))
            deltas.append(
                mean_squared_error(y_wer_t.cpu(), pw2.cpu().numpy())
                + mean_squared_error(y_cer_t.cpu(), pc2.cpu().numpy())
                - base
            )
        imps[fi] = np.mean(deltas)

    top_idx = np.argsort(imps)[::-1][:top_k]
    print(f"\n  Top-{top_k} features (permutation importance):")
    result = []
    for rank, i in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {feat_names[i]:35s}  Δmse={imps[i]:+.6f}")
        result.append((feat_names[i], float(imps[i])))
    return result


# ─── training loop ────────────────────────────────────────────────────────────

def train_model(model, X_tr_t, yw_tr, yc_tr, X_v_t, yw_v, yc_v,
                device, epochs=300, lr=3e-4, patience=40):
    opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn  = nn.MSELoss()
    loader   = DataLoader(TensorDataset(X_tr_t, yw_tr, yc_tr), batch_size=32, shuffle=True)

    history, best_val, best_state, no_imp = [], float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yw_b, yc_b in loader:
            Xb, yw_b, yc_b = Xb.to(device), yw_b.to(device), yc_b.to(device)
            opt.zero_grad()
            pw, pc = model(Xb)
            loss   = loss_fn(pw, yw_b) + loss_fn(pc, yc_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(Xb)
        tr_loss /= len(loader.dataset)
        sched.step()

        model.eval()
        with torch.no_grad():
            pw_v, pc_v = model(X_v_t)
            val_loss = (loss_fn(pw_v, yw_v) + loss_fn(pc_v, yc_v)).item()

        history.append({"epoch": epoch, "train": tr_loss, "val": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"    Early stop at epoch {epoch}")
                break

        if epoch % 50 == 0 or epoch == 1:
            print(f"    Epoch {epoch:4d}  train={tr_loss:.5f}  val={val_loss:.5f}")

    if best_state:
        model.load_state_dict(best_state)
    return history


# ─── plotting ─────────────────────────────────────────────────────────────────

def plot_results(history, y_true_wer, y_pred_wer, y_true_cer, y_pred_cer,
                 top_feat, n_train, n_test, use_embeddings, out_path):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)
    scatter_kw = dict(alpha=0.4, s=14, edgecolors="none")
    diag_kw    = dict(color="#ff6b6b", lw=1.5, ls="--")

    # loss curve
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.set_facecolor("#f5f5f5")
    ep = [h["epoch"] for h in history]
    ax0.plot(ep, [h["train"] for h in history], color="#4ecdc4", lw=1.5, label="Train")
    ax0.plot(ep, [h["val"]   for h in history], color="#f7d794", lw=1.5, label="Val")
    ax0.set_xlabel("Epoch", color="black"); ax0.set_ylabel("MSE Loss", color="black")
    ax0.set_title("Training Curve", color="black", fontweight="bold")
    ax0.legend(facecolor="#f5f5f5", labelcolor="black", edgecolor="#333")
    ax0.tick_params(colors="white"); ax0.spines[:].set_color("#333")
    ax0.set_yscale("log")

    for col, (y_true, y_pred, color, metric) in enumerate([
        (y_true_wer, y_pred_wer, "#4ecdc4", "WER"),
        (y_true_cer, y_pred_cer, "#f7d794", "CER"),
    ], start=1):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#f5f5f5")
        ax.scatter(y_true, y_pred, c=color, **scatter_kw)
        lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
        ax.plot(lim, lim, **diag_kw)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"True {metric}", color="black")
        ax.set_ylabel(f"Predicted {metric}", color="black")
        ax.set_title(f"{metric}: True vs Predicted", color="black", fontweight="bold")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#333")

        ax2 = fig.add_subplot(gs[1, col])
        ax2.set_facecolor("#f5f5f5")
        ax2.scatter(y_pred, y_pred - y_true, c=color, **scatter_kw)
        ax2.axhline(0, **diag_kw)
        ax2.set_xlabel(f"Predicted {metric}", color="black")
        ax2.set_ylabel("Residual", color="black")
        ax2.set_title(f"{metric} Residuals", color="black", fontweight="bold")
        ax2.tick_params(colors="white"); ax2.spines[:].set_color("#333")

    # permutation importance
    ax5 = fig.add_subplot(gs[:, 3])
    ax5.set_facecolor("#f5f5f5")
    names = [f[:22] for f, _ in top_feat[:12]]
    vals  = [v for _, v in top_feat[:12]]
    ax5.barh(range(len(names)), vals,
             color=["#a29bfe" if v >= 0 else "#fd79a8" for v in vals],
             edgecolor="none")
    ax5.set_yticks(range(len(names)))
    ax5.set_yticklabels(names, color="black", fontsize=7.5)
    ax5.set_xlabel("Δ MSE (WER+CER)", color="black")
    ax5.set_title("Permutation Importance", color="black", fontweight="bold")
    ax5.tick_params(colors="white"); ax5.spines[:].set_color("#333")
    ax5.invert_yaxis()

    emb_label = "surface+embeddings" if use_embeddings else "surface features"
    fig.suptitle(
        f"MLP Regression — OCR Accuracy Prediction  "
        f"[pooled easyocr+tesseract | {emb_label} | train={n_train} test={n_test}]",
        color="black", fontsize=12, fontweight="bold", y=1.01
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


# ─── main experiment ──────────────────────────────────────────────────────────

def run_experiment(use_embeddings: bool, max_epochs: int, seed: int):
    print(f"\n{'='*60}")
    print(f"MLP Regression  |  Embeddings: {use_embeddings}  |  Seed: {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    records = load_all_records()

    # 60/40 primary split
    train_recs, test_recs = stratified_split(records, test_size=0.40, random_state=seed)

    # Internal val: 15% of train (~9% of total) for early stopping
    # Shuffle before carving val to avoid ordering bias
    rng_shuffle = np.random.default_rng(seed)
    rng_shuffle.shuffle(train_recs)
    val_n      = max(int(0.15 * len(train_recs)), 10)
    val_recs   = train_recs[-val_n:]
    train_recs = train_recs[:-val_n]
    print(f"  Train: {len(train_recs)}  |  Val: {len(val_recs)}  |  Test: {len(test_recs)}")

    feat_names_base = ALL_FEATURE_NAMES if use_embeddings else (SURFACE_FEATURE_NAMES + METADATA_FEATURE_NAMES)
    feat_names      = feat_names_base + ["engine_flag"]

    print("\n  Building features …")
    X_tr, yw_tr, yc_tr = build_X(train_recs, use_embeddings)
    X_v,  yw_v,  yc_v  = build_X(val_recs,   use_embeddings)
    X_te, yw_te, yc_te = build_X(test_recs,  use_embeddings)

    scaler    = StandardScaler()
    X_tr_s    = scaler.fit_transform(X_tr).astype(np.float32)
    X_v_s     = scaler.transform(X_v).astype(np.float32)
    X_te_s    = scaler.transform(X_te).astype(np.float32)

    def t(a): return torch.tensor(a, dtype=torch.float32, device=device)

    model = OCRRegressionMLP(X_tr_s.shape[1]).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}  |  Features: {X_tr_s.shape[1]}")

    print(f"\n  Training (max {max_epochs} epochs, patience=40) …")
    history = train_model(
        model, t(X_tr_s), t(yw_tr), t(yc_tr),
        t(X_v_s),  t(yw_v),  t(yc_v),
        device=device, epochs=max_epochs, patience=40
    )

    model.eval()
    with torch.no_grad():
        pw_tr, pc_tr = model(t(X_tr_s))
        pw_te, pc_te = model(t(X_te_s))

    print("\n  === WER ===")
    tr_wer = evaluate(yw_tr, pw_tr.cpu().numpy(), "train")
    te_wer = evaluate(yw_te, pw_te.cpu().numpy(), "test ")

    print("\n  === CER ===")
    tr_cer = evaluate(yc_tr, pc_tr.cpu().numpy(), "train")
    te_cer = evaluate(yc_te, pc_te.cpu().numpy(), "test ")

    print("\n  Computing permutation importance …")
    top_feat = permutation_importance(
        model, t(X_te_s), t(yw_te), t(yc_te), feat_names, device
    )

    suffix    = "emb" if use_embeddings else "noemb"
    plot_path = RESULTS_DIR / f"nn_regression_pooled_{suffix}.png"
    plot_results(
        history,
        yw_te, pw_te.cpu().numpy(),
        yc_te, pc_te.cpu().numpy(),
        top_feat, len(train_recs), len(test_recs), use_embeddings, plot_path
    )

    out = RESULTS_DIR / f"nn_regression_pooled_{suffix}_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "nn_regression_pooled",
            "use_embeddings": use_embeddings,
            "n_train": len(train_recs),
            "n_val":   len(val_recs),
            "n_test":  len(test_recs),
            "n_features": X_tr_s.shape[1],
            "epochs_trained": len(history),
            "best_val_loss": min(h["val"] for h in history),
            "train_wer": tr_wer, "test_wer": te_wer,
            "train_cer": tr_cer, "test_cer": te_cer,
            "top_features_permutation": top_feat,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Results saved → {out}")

    # ── Per-document predictions (ALL records: train + val + test) ─────────────
    # Saved as a flat JSON list so routing threshold sweeps need zero retraining.
    # Join with LLM correction results by filename to build the routing table.
    print("  Saving per-document predictions (10-Fold CV on Tesseract) …")
    from sklearn.model_selection import KFold
    all_recs  = train_recs + val_recs + test_recs
    tesseract_recs = [r for r in all_recs if r["engine"] == "tesseract"]
    split_tag = ["cv_10fold"] * len(tesseract_recs)

    X_all, yw_all, yc_all = build_X(tesseract_recs, use_embeddings)
    # We must fit a fresh scaler for the entire CV process? Standard is to scale per fold.
    # To keep it simple and robust, we will just scale per fold inside the loop.
    
    pw_all = np.zeros_like(yw_all)
    pc_all = np.zeros_like(yc_all)

    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        # Scale per fold
        fold_scaler = StandardScaler()
        X_tr_fold = fold_scaler.fit_transform(X_all[train_idx]).astype(np.float32)
        X_te_fold = fold_scaler.transform(X_all[test_idx]).astype(np.float32)
        
        fold_model = OCRRegressionMLP(X_all.shape[1]).to(device)
        # Fast training (150 epochs max)
        train_model(
            fold_model, t(X_tr_fold), t(yw_all[train_idx]), t(yc_all[train_idx]),
            t(X_te_fold), t(yw_all[test_idx]), t(yc_all[test_idx]),
            device=device, epochs=150, patience=20
        )
        fold_model.eval()
        with torch.no_grad():
            pw, pc = fold_model(t(X_te_fold))
        pw_all[test_idx] = pw.cpu().numpy()
        pc_all[test_idx] = pc.cpu().numpy()
        print(f"    Fold {fold+1}/10 complete.")

    per_doc = []
    for rec, sp, pw, pc, aw, ac in zip(
        tesseract_recs, split_tag, pw_all, pc_all, yw_all, yc_all
    ):
        per_doc.append({
            "filename":   rec["filename"],
            "engine":     rec["engine"],
            "split":      sp,
            "actual_wer": round(float(aw), 6),
            "actual_cer": round(float(ac), 6),
            "pred_wer":   round(float(pw), 6),
            "pred_cer":   round(float(pc), 6),
        })

    pred_out = RESULTS_DIR / f"predictions_nn_{suffix}.json"
    with open(pred_out, "w", encoding="utf-8") as f:
        json.dump(per_doc, f, indent=2, ensure_ascii=False)
    print(f"  Per-doc predictions saved → {pred_out}  ({len(per_doc)} records)")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_experiment(use_embeddings=not args.no_embeddings,
                       max_epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
