"""
Generate routing frontier graph WITH overcorrection guard.

Same 4 strategies as routing_graph_paper.png, but each routing curve
now includes a binary "harm guard": documents predicted to be harmed
by correction (Δ < 0) are skipped — their raw OCR is kept instead.

Five curves per panel:
  (a) Oracle — perfect hindsight routing (unchanged)
  (b) Spell-check — orthographic baseline (flat line, unchanged)
  (c) ConfBERT — with overcorrection guard
  (d) Our Approach — with overcorrection guard
  (e) Our Approach (no guard) — for comparison

Output: results/routing_graph_guarded.png
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

# ── Orthographic baseline (Tesseract, Full_Orthographic) ──
ORTHO_BASELINES = {"wer": 0.2900, "cer": 0.0907}

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


def train_classifier(X_stacked, delta_target, min_delta):
    """Train LogReg routing classifier. Returns P(Δ > min_delta)."""
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


def train_harm_guard(X_stacked, delta_target):
    """
    Train a binary overcorrection guard.
    Returns P(Δ >= 0) — probability that correction does NOT harm.
    Documents with low probability should NOT be corrected.
    """
    y = (delta_target >= 0).astype(int)  # 1 = safe to correct, 0 = correction harms
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


def compute_routing_curve(sort_idx, base_vals, corr_vals, N):
    """Standard routing curve: correct documents in sort_idx order."""
    avg_base = np.mean(base_vals)
    pct_list, val_list = [0.0], [avg_base]
    current_sum = np.sum(base_vals)
    for k in range(N):
        idx = sort_idx[k]
        current_sum -= base_vals[idx]
        current_sum += corr_vals[idx]
        pct_list.append((k + 1) / N * 100)
        val_list.append(current_sum / N)
    return pct_list, val_list


def compute_guarded_routing_curve(sort_idx, base_vals, corr_vals, guard_safe, N):
    """
    Routing curve with overcorrection guard.
    When a document is routed, if guard_safe[idx] < 0.5 (predicted harmful),
    keep the raw OCR instead of applying the correction.
    """
    avg_base = np.mean(base_vals)
    pct_list, val_list = [0.0], [avg_base]
    current_sum = np.sum(base_vals)
    for k in range(N):
        idx = sort_idx[k]
        if guard_safe[idx] >= 0.5:
            # Safe to correct → apply LLM correction
            current_sum -= base_vals[idx]
            current_sum += corr_vals[idx]
        # else: predicted harmful → keep raw OCR (no change to current_sum)
        pct_list.append((k + 1) / N * 100)
        val_list.append(current_sum / N)
    return pct_list, val_list


def compute_token_axis(sort_idx, token_counts, N):
    cum_tokens = [0]
    running = 0
    for k in range(N):
        running += token_counts[sort_idx[k]]
        cum_tokens.append(running)
    return cum_tokens


def plot_subplot(ax, display_metric, records, corrections,
                 our_probas, confbert_probas, guard_safe, token_counts):
    base_vals = np.array([float(r[display_metric]) for r in records])
    corr_vals = np.array([
        corrections.get(r["filename"], {}).get(display_metric, float(r[display_metric]))
        for r in records
    ])
    deltas = base_vals - corr_vals
    N = len(records)
    avg_base = np.mean(base_vals)

    # Count overcorrection stats
    n_harmful = np.sum(deltas < 0)
    guard_blocks = np.sum(guard_safe < 0.5)
    true_neg = np.sum((deltas < 0) & (guard_safe < 0.5))
    print(f"  [{display_metric.upper()}] Harmful docs: {n_harmful}/{N} "
          f"({100*n_harmful/N:.1f}%), Guard blocks: {guard_blocks}, "
          f"True negatives caught: {true_neg}")

    # (a) Oracle — sort by true delta, largest first
    oracle_idx = np.argsort(deltas)[::-1]
    oracle_pct, oracle_val = compute_routing_curve(oracle_idx, base_vals, corr_vals, N)
    ax.plot(oracle_pct, oracle_val, color='#7f7f7f', linestyle=':', linewidth=1.8,
            label='Oracle (Perfect Routing)')

    # (b) Orthographic Correction — flat horizontal line
    ortho_val = ORTHO_BASELINES.get(display_metric)
    if ortho_val is not None:
        ax.axhline(ortho_val, color='#2ca02c', linestyle='-.', linewidth=1.8,
                   label=f'Spell-check ({ortho_val:.4f})')

    # (c) ConfBERT with guard
    confbert_idx = np.argsort(confbert_probas)[::-1]
    confbert_pct, confbert_val = compute_guarded_routing_curve(
        confbert_idx, base_vals, corr_vals, guard_safe, N)
    ax.plot(confbert_pct, confbert_val, color='#ff7f0e', linestyle='--', linewidth=2,
            label='ConfBERT + Guard')

    # (d) Our Approach WITHOUT guard (faded, for comparison)
    our_idx = np.argsort(our_probas)[::-1]
    our_pct_noguard, our_val_noguard = compute_routing_curve(
        our_idx, base_vals, corr_vals, N)
    ax.plot(our_pct_noguard, our_val_noguard, color='#d62728', linewidth=1.2,
            alpha=0.35, linestyle='--', label='Ours (no guard)')

    # (e) Our Approach WITH guard
    our_pct_guarded, our_val_guarded = compute_guarded_routing_curve(
        our_idx, base_vals, corr_vals, guard_safe, N)
    ax.plot(our_pct_guarded, our_val_guarded, color='#d62728', linewidth=2.5,
            label='Ours + Guard')

    # Baseline reference line
    ax.axhline(avg_base, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
               label=f'Baseline OCR ({avg_base:.4f})')

    ax.set_xlabel('% of Documents Corrected', fontsize=11)
    ax.set_ylabel(display_metric.upper(), fontsize=11)
    ax.grid(True, alpha=0.2)

    # Secondary x-axis: cumulative tokens
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

    # Stacked feature matrix
    pred_ridge_dw = np.array([delta_preds[r["filename"]]["pred_ridge_delta_wer"] for r in records])
    pred_ridge_dc = np.array([delta_preds[r["filename"]]["pred_ridge_delta_cer"] for r in records])
    pred_mlp_dw   = np.array([delta_preds[r["filename"]]["pred_mlp_delta_wer"]   for r in records])
    pred_mlp_dc   = np.array([delta_preds[r["filename"]]["pred_mlp_delta_cer"]   for r in records])
    X_stacked = np.column_stack([X, pred_ridge_dw, pred_ridge_dc, pred_mlp_dw, pred_mlp_dc])

    # ── Train routing classifier ──
    min_delta = 0.0
    print("Training routing classifier (LogReg on δ-CER > 0.00)...")
    our_probas = train_classifier(X_stacked, delta_cer_gt, min_delta)

    # ── Train overcorrection guard ──
    print("Training overcorrection guard (P(Δ ≥ 0))...")
    guard_safe = train_harm_guard(X_stacked, delta_cer_gt)

    # Report guard accuracy
    truly_harmful = (delta_cer_gt < 0)
    guard_blocks = (guard_safe < 0.5)
    tp = np.sum(truly_harmful & guard_blocks)
    fp = np.sum(~truly_harmful & guard_blocks)
    fn = np.sum(truly_harmful & ~guard_blocks)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"  Guard stats: harmful={truly_harmful.sum()}/{len(delta_cer_gt)}, "
          f"blocked={guard_blocks.sum()}, "
          f"precision={precision:.3f}, recall={recall:.3f}")

    # ── Load ConfBERT (cached) ──
    confbert_cache = RESULTS / "ml_models/confbert_probas.npy"
    if confbert_cache.exists():
        confbert_probas = np.load(confbert_cache)
        print(f"  Loaded cached ConfBERT probas: {len(confbert_probas)}")
    else:
        print("  ERROR: ConfBERT cache not found. Run plot_routing_graph_paper.py first.")
        return

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_subplot(axes[0], "wer", records, corrections,
                 our_probas, confbert_probas, guard_safe, token_counts)
    axes[0].set_title("Word Error Rate (WER)", fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=7.5, framealpha=0.9)

    plot_subplot(axes[1], "cer", records, corrections,
                 our_probas, confbert_probas, guard_safe, token_counts)
    axes[1].set_title("Character Error Rate (CER)", fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    out_path = "results/routing_graph_guarded.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
