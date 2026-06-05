"""
experiment_linear_regression.py
================================
Experiment: Ridge Regression to predict OCR accuracy (WER & CER).

Dataset
-------
Both baselines (easyocr + tesseract) are pooled into one dataset.
An extra binary feature `engine_flag` (0=easyocr, 1=tesseract) is appended.
A single 60/40 stratified split is applied on the combined ~1200 samples.

Features (40 surface + 1 engine_flag + 384 embeddings = 425 total)
---------------------------------------------------------------------
See regression_features.py for the full feature list.

Usage
-----
    python code/experiment_linear_regression.py [--no-embeddings]
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

from sklearn.linear_model import RidgeCV
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


# ─── data loading ─────────────────────────────────────────────────────────────

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
            r["groundtruth"] = r.get("groundtruth", "")
        records.extend(valid)
        print(f"  [{engine}] {len(valid)} valid samples")
    # Enrich with publication year, newspaper, groundtruth length, and avg_confidence
    meta_lookup = load_metadata_lookup(GROUNDTRUTH_PATH)
    # Build a combined confidence lookup across both engines (keyed by filename;
    # records already carry their engine, so per-engine lookup is applied below)
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
    # Group records by filename
    by_fname: dict[str, list] = defaultdict(list)
    for r in records:
        by_fname[r["filename"]].append(r)

    fnames = sorted(by_fname.keys())          # deterministic order
    mean_wer = np.array([np.mean([r["wer"] for r in by_fname[f]]) for f in fnames])
    quartiles = np.digitize(mean_wer, np.percentile(mean_wer, [25, 50, 75]))

    tr_fnames, te_fnames = train_test_split(
        fnames, test_size=test_size, stratify=quartiles, random_state=random_state
    )
    tr_set, te_set = set(tr_fnames), set(te_fnames)

    train = [r for r in records if r["filename"] in tr_set]
    test  = [r for r in records if r["filename"] in te_set]
    return train, test



# ─── evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, label=""):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r, p = pearsonr(y_true, y_pred)
    print(f"    {label:5s}  R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  r={r:.4f} (p={p:.2e})")
    return {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse),
            "pearson_r": float(r), "pearson_p": float(p)}


def top_features(coef, feature_names, k=15, label=""):
    abs_coef = np.abs(coef)
    top_idx  = np.argsort(abs_coef)[::-1][:k]
    print(f"\n  Top-{k} features ({label}):")
    for rank, i in enumerate(top_idx, 1):
        print(f"    {rank:2d}. {feature_names[i]:35s}  coef={coef[i]:+.5f}")
    return [(feature_names[i], float(coef[i])) for i in top_idx]


# ─── plotting ─────────────────────────────────────────────────────────────────

def plot_results(
    y_true_wer, y_pred_wer, y_true_cer, y_pred_cer,
    top_wer, top_cer, engine_colors,
    n_train, n_test, out_path
):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    scatter_kw = dict(alpha=0.4, s=16, edgecolors="none")
    diag_kw    = dict(color="#ff6b6b", lw=1.5, ls="--")

    for row, (y_true, y_pred, color, metric) in enumerate([
        (y_true_wer, y_pred_wer, "#4ecdc4", "WER"),
        (y_true_cer, y_pred_cer, "#f7d794", "CER"),
    ]):
        # scatter
        ax = fig.add_subplot(gs[row, 0])
        ax.set_facecolor("#f5f5f5")
        ax.scatter(y_true, y_pred, c=color, **scatter_kw)
        lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
        ax.plot(lim, lim, **diag_kw)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"True {metric}", color="black")
        ax.set_ylabel(f"Predicted {metric}", color="black")
        ax.set_title(f"{metric}: True vs Predicted", color="black", fontweight="bold")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#333")

        # residuals
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.set_facecolor("#f5f5f5")
        ax2.scatter(y_pred, y_pred - y_true, c=color, **scatter_kw)
        ax2.axhline(0, **diag_kw)
        ax2.set_xlabel(f"Predicted {metric}", color="black")
        ax2.set_ylabel("Residual", color="black")
        ax2.set_title(f"{metric} Residuals", color="black", fontweight="bold")
        ax2.tick_params(colors="white"); ax2.spines[:].set_color("#333")

    # top features WER
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#f5f5f5")
    names_w = [f[:22] for f, _ in top_wer[:10]]
    vals_w  = [v for _, v in top_wer[:10]]
    ax3.barh(range(len(names_w)), vals_w,
             color=["#4ecdc4" if v >= 0 else "#ff6b6b" for v in vals_w],
             edgecolor="none")
    ax3.set_yticks(range(len(names_w)))
    ax3.set_yticklabels(names_w, color="black", fontsize=7)
    ax3.set_xlabel("Coefficient", color="black")
    ax3.set_title("Top Features (WER)", color="black", fontweight="bold")
    ax3.tick_params(colors="white"); ax3.spines[:].set_color("#333")
    ax3.invert_yaxis()

    # top features CER
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor("#f5f5f5")
    names_c = [f[:22] for f, _ in top_cer[:10]]
    vals_c  = [v for _, v in top_cer[:10]]
    ax4.barh(range(len(names_c)), vals_c,
             color=["#f7d794" if v >= 0 else "#ff6b6b" for v in vals_c],
             edgecolor="none")
    ax4.set_yticks(range(len(names_c)))
    ax4.set_yticklabels(names_c, color="black", fontsize=7)
    ax4.set_xlabel("Coefficient", color="black")
    ax4.set_title("Top Features (CER)", color="black", fontweight="bold")
    ax4.tick_params(colors="white"); ax4.spines[:].set_color("#333")
    ax4.invert_yaxis()

    emb_label = "surface+embeddings" if len(top_wer) > 40 else "surface features"
    fig.suptitle(
        f"Ridge Regression — OCR Accuracy Prediction  "
        f"[pooled easyocr+tesseract | {emb_label} | train={n_train} test={n_test}]",
        color="black", fontsize=12, fontweight="bold", y=1.01
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


# ─── main experiment ──────────────────────────────────────────────────────────

def run_experiment(use_embeddings: bool):
    print(f"\n{'='*60}")
    print(f"Linear Regression  |  Embeddings: {use_embeddings}")
    print(f"{'='*60}")

    records = load_all_records()
    train_recs, test_recs = stratified_split(records, test_size=0.40)
    print(f"  Train: {len(train_recs)}  |  Test: {len(test_recs)}")

    feat_names_base = ALL_FEATURE_NAMES if use_embeddings else (SURFACE_FEATURE_NAMES + METADATA_FEATURE_NAMES)
    feat_names = feat_names_base + ["engine_flag"]

    def build_X(recs, use_emb):
        X, y_wer, y_cer = build_dataset(recs, use_embeddings=use_emb, verbose=True)
        engine_col = np.array(
            [ENGINE_FLAG[r["engine"]] for r in recs], dtype=np.float32
        ).reshape(-1, 1)
        return np.concatenate([X, engine_col], axis=1), y_wer, y_cer

    print("\n  [TRAIN] building features …")
    X_train, y_wer_train, y_cer_train = build_X(train_recs, use_embeddings)
    print("  [TEST]  building features …")
    X_test,  y_wer_test,  y_cer_test  = build_X(test_recs, use_embeddings)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    alphas = np.logspace(-3, 5, 30)

    print("\n  Fitting Ridge (WER) …")
    ridge_wer = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    ridge_wer.fit(X_train_s, y_wer_train)
    print(f"    Best α (WER): {ridge_wer.alpha_:.4f}")

    print("  Fitting Ridge (CER) …")
    ridge_cer = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    ridge_cer.fit(X_train_s, y_cer_train)
    print(f"    Best α (CER): {ridge_cer.alpha_:.4f}")

    # 5-fold CV R² on the train set (generalisation check)
    from sklearn.model_selection import cross_val_score
    cv_r2_wer = cross_val_score(
        RidgeCV(alphas=alphas, cv=5), X_train_s, y_wer_train,
        cv=5, scoring="r2"
    )
    cv_r2_cer = cross_val_score(
        RidgeCV(alphas=alphas, cv=5), X_train_s, y_cer_train,
        cv=5, scoring="r2"
    )
    print(f"    CV R² WER: {cv_r2_wer.mean():.4f} ± {cv_r2_wer.std():.4f}")
    print(f"    CV R² CER: {cv_r2_cer.mean():.4f} ± {cv_r2_cer.std():.4f}")


    print("\n  === WER ===")
    train_wer_m = evaluate_model(y_wer_train, ridge_wer.predict(X_train_s), "train")
    test_wer_m  = evaluate_model(y_wer_test,  ridge_wer.predict(X_test_s),  "test ")

    print("\n  === CER ===")
    train_cer_m = evaluate_model(y_cer_train, ridge_cer.predict(X_train_s), "train")
    test_cer_m  = evaluate_model(y_cer_test,  ridge_cer.predict(X_test_s),  "test ")

    coef_wer = ridge_wer.coef_ / scaler.scale_
    coef_cer = ridge_cer.coef_ / scaler.scale_
    top_wer  = top_features(coef_wer, feat_names, k=15, label="WER")
    top_cer  = top_features(coef_cer, feat_names, k=15, label="CER")

    suffix   = "emb" if use_embeddings else "noemb"
    plot_path = RESULTS_DIR / f"linear_regression_pooled_{suffix}.png"
    plot_results(
        y_wer_test,  ridge_wer.predict(X_test_s),
        y_cer_test,  ridge_cer.predict(X_test_s),
        top_wer, top_cer, {},
        len(train_recs), len(test_recs), plot_path
    )

    out = RESULTS_DIR / f"linear_regression_pooled_{suffix}_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "linear_regression_pooled",
            "use_embeddings": use_embeddings,
            "n_train": len(train_recs),
            "n_test":  len(test_recs),
            "n_features": X_train.shape[1],
            "ridge_alpha_wer": float(ridge_wer.alpha_),
            "ridge_alpha_cer": float(ridge_cer.alpha_),
            "cv_r2_wer_mean": float(cv_r2_wer.mean()),
            "cv_r2_wer_std":  float(cv_r2_wer.std()),
            "cv_r2_cer_mean": float(cv_r2_cer.mean()),
            "cv_r2_cer_std":  float(cv_r2_cer.std()),
            "train_wer": train_wer_m, "test_wer": test_wer_m,
            "train_cer": train_cer_m, "test_cer": test_cer_m,
            "top_features_wer": top_wer,
            "top_features_cer": top_cer,
        }, f, indent=2, ensure_ascii=False)

    print(f"  Results saved → {out}")

    # ── Per-document predictions (ALL records, train + test) ──────────────────
    # This file is the foundation for threshold sweep analysis without retraining.
    # Join with LLM correction results (by filename) to get corrected_wer later.
    print("  Saving per-document predictions (Strict Leave-One-Out CV) …")
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import Pipeline
    
    all_recs  = train_recs + test_recs
    tesseract_recs = [r for r in all_recs if r["engine"] == "tesseract"]
    split_tag = ["cv_loo"] * len(tesseract_recs)

    X_all, y_wer_all, y_cer_all = build_X(tesseract_recs, use_embeddings)
    
    # Proper out-of-sample prediction via LOO pipeline
    pipeline_wer = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 50)))])
    pipeline_cer = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 50)))])
    
    print("    Running LOO for WER...")
    pred_wer = cross_val_predict(pipeline_wer, X_all, y_wer_all, cv=LeaveOneOut(), n_jobs=-1)
    print("    Running LOO for CER...")
    pred_cer = cross_val_predict(pipeline_cer, X_all, y_cer_all, cv=LeaveOneOut(), n_jobs=-1)

    per_doc = []
    for rec, split, pw, pc, aw, ac in zip(
        tesseract_recs, split_tag, pred_wer, pred_cer, y_wer_all, y_cer_all
    ):
        per_doc.append({
            "filename":    rec["filename"],
            "engine":      rec["engine"],
            "split":       split,
            "actual_wer":  round(float(aw), 6),
            "actual_cer":  round(float(ac), 6),
            "pred_wer":    round(float(pw), 6),
            "pred_cer":    round(float(pc), 6),
        })

    pred_out = RESULTS_DIR / f"predictions_ridge_{suffix}.json"
    with open(pred_out, "w", encoding="utf-8") as f:
        json.dump(per_doc, f, indent=2, ensure_ascii=False)
    print(f"  Per-doc predictions saved → {pred_out}  ({len(per_doc)} records)")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embeddings", action="store_true")
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_experiment(use_embeddings=not args.no_embeddings)


if __name__ == "__main__":
    main()
