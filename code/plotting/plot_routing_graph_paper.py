"""
Generate the paper-specific routing frontier graph (1×2: WER + CER).

Four curves per panel:
  (a) Oracle — perfect hindsight routing
  (b) Orthographic Correction — non-LLM spellcheck baseline (flat line)
  (c) ConfBERT — confidence-aware BERT model (Hemmer et al.-style)
  (d) Our Approach — LogReg classifier with 54-d features + regression preds

X-axis: % of Documents Corrected
Secondary x-axis: cumulative token cost (thousands of tokens processed)

Output: results/routing_graph_paper.png
"""
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections,
    build_features,
)
from utils.regression_features import load_confidence_lookup
from experiments.confbert_router import train_confbert_router

# ── Orthographic baseline (Tesseract, Full_Orthographic) ──
ORTHO_BASELINES = {"wer": 0.2284, "cer": 0.0675}

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
IMAGES = BASE / "data" / "evaluation_dataset" / "images"


def train_delta_regressions(X, records, corrections):
    delta_wer = np.array([
        float(r["wer"]) - corrections.get(r["filename"], {}).get("wer", float(r["wer"]))
        for r in records
    ], dtype=np.float32)
    delta_cer = np.array([
        float(r["cer"]) - corrections.get(r["filename"], {}).get("cer", float(r["cer"]))
        for r in records
    ], dtype=np.float32)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    ridge_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 50)))
    ])
    from sklearn.neural_network import MLPRegressor
    mlp_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=2000,
            solver='lbfgs', random_state=42, verbose=False
        ))
    ])

    pred_ridge_delta_wer = cross_val_predict(ridge_pipe, X, delta_wer, cv=cv)
    pred_ridge_delta_cer = cross_val_predict(ridge_pipe, X, delta_cer, cv=cv)
    pred_mlp_delta_wer   = cross_val_predict(mlp_pipe, X, delta_wer, cv=cv)
    pred_mlp_delta_cer   = cross_val_predict(mlp_pipe, X, delta_cer, cv=cv)

    predictions = {}
    for i, r in enumerate(records):
        predictions[r["filename"]] = {
            "pred_ridge_delta_wer": float(pred_ridge_delta_wer[i]),
            "pred_ridge_delta_cer": float(pred_ridge_delta_cer[i]),
            "pred_mlp_delta_wer":   float(pred_mlp_delta_wer[i]),
            "pred_mlp_delta_cer":   float(pred_mlp_delta_cer[i]),
        }
    return predictions


def train_our_classifier(X_stacked, delta_target, min_delta):
    y = (delta_target > min_delta).astype(int)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced',
                                   max_iter=1000, random_state=42))
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probas = cross_val_predict(clf, X_stacked, y, cv=cv, n_jobs=-1,
                                   method='predict_proba')[:, 1]
    return probas


def compute_routing_curves(sort_idx, base_vals, corr_vals, N, scores, threshold):
    """Compute the full and capped frontiers for a given document ordering."""
    avg_base = np.mean(base_vals)
    pct_list = [0.0]
    val_list_full = [avg_base]
    val_list_capped = [avg_base]
    current_sum_full = np.sum(base_vals)
    current_sum_capped = np.sum(base_vals)
    for k in range(N):
        idx = sort_idx[k]
        
        current_sum_full -= base_vals[idx]
        current_sum_full += corr_vals[idx]
        
        if scores[idx] >= threshold:
            current_sum_capped -= base_vals[idx]
            current_sum_capped += corr_vals[idx]
            
        pct_list.append((k + 1) / N * 100)
        val_list_full.append(current_sum_full / N)
        val_list_capped.append(current_sum_capped / N)
    return pct_list, val_list_full, val_list_capped


def compute_token_axis(sort_idx, token_counts, N):
    """Compute cumulative tokens processed at each routing step."""
    cum_tokens = [0]
    running = 0
    for k in range(N):
        running += token_counts[sort_idx[k]]
        cum_tokens.append(running)
    return cum_tokens


def plot_subplot(ax, display_metric, records, corrections,
                 our_probas, confbert_probas, token_counts):
    base_vals = np.array([float(r[display_metric]) for r in records])
    corr_vals = np.array([
        corrections.get(r["filename"], {}).get(display_metric, float(r[display_metric]))
        for r in records
    ])
    deltas = base_vals - corr_vals
    N = len(records)
    avg_base = np.mean(base_vals)

    # (a) Oracle — sort by true delta, largest first
    oracle_idx = np.argsort(deltas)[::-1]
    oracle_pct, oracle_full, oracle_capped = compute_routing_curves(oracle_idx, base_vals, corr_vals, N, deltas, 0.0)
    ax.plot(oracle_pct, oracle_full, color='#7f7f7f', linestyle=':', linewidth=1.8, alpha=0.3)
    ax.plot(oracle_pct, oracle_capped, color='#7f7f7f', linestyle=':', linewidth=1.8,
            label='Oracle (Perfect Routing)')

    # (b) Orthographic Correction — flat horizontal line
    ortho_val = ORTHO_BASELINES.get(display_metric)
    if ortho_val is not None:
        ax.axhline(ortho_val, color='#2ca02c', linestyle='-.', linewidth=1.8,
                   label='Spell-check')

    # (c) ConfBERT — confidence-aware BERT routing
    confbert_idx = np.argsort(confbert_probas)[::-1]
    confbert_pct, confbert_full, confbert_capped = compute_routing_curves(confbert_idx, base_vals, corr_vals, N, confbert_probas, 0.5)
    ax.plot(confbert_pct, confbert_full, color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.3)
    ax.plot(confbert_pct, confbert_capped, color='#ff7f0e', linestyle='--', linewidth=2,
            label='ConfBERT (Hemmer et al.)')

    # (d) Our Approach — LogReg with enriched features
    our_idx = np.argsort(our_probas)[::-1]
    our_pct, our_full, our_capped = compute_routing_curves(our_idx, base_vals, corr_vals, N, our_probas, 0.5)
    ax.plot(our_pct, our_full, color='#d62728', linewidth=2.5, alpha=0.3)
    ax.plot(our_pct, our_capped, color='#d62728', linewidth=2.5,
            label='Our Approach (LogReg + Δ-pred)')

    # Baseline reference line
    ax.axhline(avg_base, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
               label=f'Baseline OCR ({avg_base:.4f})')

    ax.set_xlabel('% of Documents Corrected', fontsize=11)
    ax.set_ylabel(display_metric.upper(), fontsize=11)
    ax.grid(True, alpha=0.2)

    # ── Secondary x-axis: cumulative tokens (thousands) ──
    our_cum_tokens = compute_token_axis(our_idx, token_counts, N)
    ax2 = ax.twiny()
    tick_pcts = [0, 20, 40, 60, 80, 100]
    tick_tokens = []
    for p in tick_pcts:
        idx_pos = int(round(p / 100 * N))
        tick_tokens.append(our_cum_tokens[idx_pos])
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_pcts)
    ax2.set_xticklabels([f'{t/1000:.1f}k' for t in tick_tokens], fontsize=8)
    ax2.set_xlabel('Cumulative Tokens Processed', fontsize=9, labelpad=6)


def main():
    print("Loading data...")
    records = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Advanced_5_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)
    X = build_features(records)

    # Token counts per document
    token_counts = np.array([
        len(r.get("raw_ocr", "").split()) for r in records
    ], dtype=np.int32)
    print(f"  Total tokens: {np.sum(token_counts):,}")

    print("Training delta regressions (Ridge + MLP via 10-fold CV)...")
    delta_preds = train_delta_regressions(X, records, corrections)

    # Precompute ground-truth deltas for CER
    base_cer = np.array([float(r["cer"]) for r in records])
    corr_cer = np.array([
        corrections.get(r["filename"], {}).get("cer", float(r["cer"]))
        for r in records
    ])
    delta_cer_gt = base_cer - corr_cer

    # Stacked feature matrix (base features + regression predictions)
    pred_ridge_dw = np.array([delta_preds[r["filename"]]["pred_ridge_delta_wer"] for r in records])
    pred_ridge_dc = np.array([delta_preds[r["filename"]]["pred_ridge_delta_cer"] for r in records])
    pred_mlp_dw   = np.array([delta_preds[r["filename"]]["pred_mlp_delta_wer"]   for r in records])
    pred_mlp_dc   = np.array([delta_preds[r["filename"]]["pred_mlp_delta_cer"]   for r in records])
    X_stacked = np.column_stack([X, pred_ridge_dw, pred_ridge_dc, pred_mlp_dw, pred_mlp_dc])

    min_delta = 0.0  # δ > 0pp
    print("Training our classifier (LogReg on δ-CER > 0.00)...")
    our_probas = train_our_classifier(X_stacked, delta_cer_gt, min_delta)

    # ── ConfBERT ──
    print("Training ConfBERT (10-fold CV)...")
    confbert_cache = RESULTS / "ml_models/confbert_probas.npy"
    if confbert_cache.exists():
        confbert_probas = np.load(confbert_cache)
        print(f"  Loaded cached ConfBERT probas: {len(confbert_probas)}")
        if len(confbert_probas) != len(records):
            print(f"  Cache mismatch ({len(confbert_probas)} vs {len(records)}), retraining...")
            confbert_probas = train_confbert_router(
                records, corrections, str(RESULTS), str(IMAGES),
                metric="cer", min_delta=min_delta
            )
            np.save(confbert_cache, confbert_probas)
    else:
        confbert_probas = train_confbert_router(
            records, corrections, str(RESULTS), str(IMAGES),
            metric="cer", min_delta=min_delta
        )
        np.save(confbert_cache, confbert_probas)
        print(f"  Cached ConfBERT probas to {confbert_cache}")

    # ── 1×2 figure ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_subplot(axes[0], "wer", records, corrections,
                 our_probas, confbert_probas, token_counts)

    axes[0].legend(loc='upper right', fontsize=8, framealpha=0.9)

    plot_subplot(axes[1], "cer", records, corrections,
                 our_probas, confbert_probas, token_counts)

    axes[1].legend(loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out_path = "results/routing_graph_paper.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
