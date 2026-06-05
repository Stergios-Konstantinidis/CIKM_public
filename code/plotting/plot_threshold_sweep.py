"""
Threshold sweep: train our LogReg router at different min-delta thresholds.

For each x in a range, the classifier is trained on:
    y = 1  iff  (WER_raw - WER_corrected) > x

This explores whether requiring a larger minimum improvement before
classifying a document as "worth correcting" improves routing quality.

Outputs one graph per threshold value to results/threshold_sweep/
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

# Selective ortho correction: apply spell-check only to docs with avg_conf < 0.73
ORTHO_BASELINES = {"wer": 0.2254, "cer": 0.0649}
BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
SWEEP_DIR = RESULTS / "threshold_sweep"


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
    y = (delta_target > min_delta).astype(int)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos < 10 or n_neg < 10:
        return None, n_pos, n_neg
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
    return probas, n_pos, n_neg


def compute_routing_curves(sort_idx, base_vals, corr_vals, N, scores, threshold):
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
    cum_tokens = [0]
    running = 0
    for k in range(N):
        running += token_counts[sort_idx[k]]
        cum_tokens.append(running)
    return cum_tokens


def _auc(pct_list, val_list):
    """Area under the routing curve, normalised to [0,1] x-range."""
    return np.trapezoid(val_list, pct_list) / 100


def _annotate_curve(ax, pct_list, val_list, label, color,
                    frac=0.35, offset=(10, -15)):
    """Place an inline annotation on a curve at a given fractional position."""
    idx = max(1, min(int(frac * len(pct_list)), len(pct_list) - 1))
    x, y = pct_list[idx], val_list[idx]
    ax.annotate(
        label, xy=(x, y), xytext=offset,
        textcoords='offset points', fontsize=7.5, fontweight='bold',
        color=color, ha='left', va='center',
        arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color,
                  alpha=0.85, lw=0.8),
    )


def plot_for_threshold(records, corrections, X_stacked,
                       delta_wer_gt, delta_cer_gt,
                       confbert_probas, token_counts, threshold, out_path):
    """Generate one 1×2 (WER + CER) graph for a given threshold."""

    result = train_classifier(X_stacked, delta_wer_gt, threshold)
    if result[0] is None:
        print(f"  SKIP x={threshold:.3f}: too few positives ({result[1]}) or negatives ({result[2]})")
        return False
    probas, n_pos, n_neg = result

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, metric in enumerate(["wer", "cer"]):
        ax = axes[ax_idx]
        base_vals = np.array([float(r[metric]) for r in records])
        corr_vals = np.array([
            corrections.get(r["filename"], {}).get(metric, float(r[metric]))
            for r in records
        ])
        deltas = base_vals - corr_vals
        N = len(records)
        avg_base = np.mean(base_vals)

        # Oracle
        oracle_idx = np.argsort(deltas)[::-1]
        oracle_pct, oracle_full, oracle_capped = compute_routing_curves(oracle_idx, base_vals, corr_vals, N, deltas, 0.0)
        oracle_auc = _auc(oracle_pct, oracle_capped)
        ax.plot(oracle_pct, oracle_full, color='#7f7f7f', linestyle=':', linewidth=1.8, alpha=0.3)
        ax.plot(oracle_pct, oracle_capped, color='#7f7f7f', linestyle=':', linewidth=1.8)
        _annotate_curve(ax, oracle_pct, oracle_capped,
                        f'Oracle (AUC={oracle_auc:.4f})',
                        '#7f7f7f', frac=0.55, offset=(8, -18))

        # Spell-check
        ortho_val = ORTHO_BASELINES.get(metric)
        if ortho_val is not None:
            ax.axhline(ortho_val, color='#2ca02c', linestyle='-.', linewidth=1.8)
            ax.annotate(
                f'Spell-check',
                xy=(75, ortho_val), xytext=(0, -6),
                textcoords='offset points', fontsize=7.5, fontweight='bold',
                color='#2ca02c', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#2ca02c',
                          alpha=0.85, lw=0.8),
            )

        # ConfBERT
        confbert_idx = np.argsort(confbert_probas)[::-1]
        confbert_pct, confbert_full, confbert_capped = compute_routing_curves(confbert_idx, base_vals, corr_vals, N, confbert_probas, threshold)
        confbert_auc = _auc(confbert_pct, confbert_capped)
        ax.plot(confbert_pct, confbert_full, color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.3)
        ax.plot(confbert_pct, confbert_capped, color='#ff7f0e', linestyle='--', linewidth=2)
        _annotate_curve(ax, confbert_pct, confbert_capped,
                        f'ConfBERT (AUC={confbert_auc:.4f})',
                        '#ff7f0e', frac=0.45, offset=(10, 15))

        # Our approach at this threshold
        our_idx = np.argsort(probas)[::-1]
        our_pct, our_full, our_capped = compute_routing_curves(our_idx, base_vals, corr_vals, N, probas, threshold)
        our_auc = _auc(our_pct, our_capped)
        ax.plot(our_pct, our_full, color='#d62728', linewidth=2.5, alpha=0.3)
        ax.plot(our_pct, our_capped, color='#d62728', linewidth=2.5)
        _annotate_curve(ax, our_pct, our_capped,
                        f'Ours (AUC={our_auc:.4f})',
                        '#d62728', frac=0.25, offset=(-10, 20))

        # Baseline
        ax.axhline(avg_base, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.annotate(
            f'Baseline',
            xy=(85, avg_base), xytext=(0, -6),
            textcoords='offset points', fontsize=7.5, fontweight='bold',
            color='black', ha='center', va='top', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black',
                      alpha=0.6, lw=0.8),
        )

        ax.set_xlabel('% of Documents Corrected', fontsize=11)
        ax.set_ylabel(metric.upper(), fontsize=11)

        ax.grid(True, alpha=0.2)

        # Token axis
        our_cum_tokens = compute_token_axis(our_idx, token_counts, N)
        ax2 = ax.twiny()
        tick_pcts = [0, 20, 40, 60, 80, 100]
        tick_tokens = [our_cum_tokens[int(round(p / 100 * N))] for p in tick_pcts]
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(tick_pcts)
        ax2.set_xticklabels([f'{t/1000:.1f}k' for t in tick_tokens], fontsize=8)
        ax2.set_xlabel('Cumulative Tokens', fontsize=9, labelpad=6)

        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches='tight')
    plt.close(fig)
    return True



def main():
    print("Loading data...")
    records = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)
    X = build_features(records)

    token_counts = np.array([len(r.get("raw_ocr", "").split()) for r in records], dtype=np.int32)

    print("Training delta regressions...")
    delta_preds = train_delta_regressions(X, records, corrections)

    # Ground-truth deltas
    base_wer = np.array([float(r["wer"]) for r in records])
    corr_wer = np.array([corrections.get(r["filename"], {}).get("wer", float(r["wer"])) for r in records])
    delta_wer_gt = base_wer - corr_wer

    base_cer = np.array([float(r["cer"]) for r in records])
    corr_cer = np.array([corrections.get(r["filename"], {}).get("cer", float(r["cer"])) for r in records])
    delta_cer_gt = base_cer - corr_cer

    # Stacked features
    pred_ridge_dw = np.array([delta_preds[r["filename"]]["pred_ridge_delta_wer"] for r in records])
    pred_ridge_dc = np.array([delta_preds[r["filename"]]["pred_ridge_delta_cer"] for r in records])
    pred_mlp_dw   = np.array([delta_preds[r["filename"]]["pred_mlp_delta_wer"]   for r in records])
    pred_mlp_dc   = np.array([delta_preds[r["filename"]]["pred_mlp_delta_cer"]   for r in records])
    X_stacked = np.column_stack([X, pred_ridge_dw, pred_ridge_dc, pred_mlp_dw, pred_mlp_dc])

    # Load ConfBERT
    confbert_cache = RESULTS / "ml_models/confbert_probas.npy"
    confbert_probas = np.load(confbert_cache)
    print(f"  Loaded ConfBERT probas: {len(confbert_probas)}")

    # Threshold sweep values
    thresholds = [-0.10, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]

    # Print delta distribution for reference
    print(f"\n  Delta WER distribution:")
    for pct in [10, 25, 50, 75, 90]:
        print(f"    P{pct}: {np.percentile(delta_wer_gt, pct):.4f}")
    print(f"    Mean: {delta_wer_gt.mean():.4f}, Std: {delta_wer_gt.std():.4f}")
    print(f"    Negative (harmful): {(delta_wer_gt < 0).sum()}/{len(delta_wer_gt)}")

    SWEEP_DIR.mkdir(exist_ok=True)

    print(f"\nSweeping {len(thresholds)} thresholds...")
    for x in thresholds:
        # Filename encoding
        if x < 0:
            fname = f"routing_x_neg{abs(x):.3f}.png"
        elif x == 0:
            fname = "routing_x_0.000.png"
        else:
            fname = f"routing_x_pos{x:.3f}.png"

        out_path = SWEEP_DIR / fname
        print(f"  x={x:+.3f} → {fname}", end=" ... ")
        ok = plot_for_threshold(records, corrections, X_stacked,
                                delta_wer_gt, delta_cer_gt,
                                confbert_probas, token_counts, x, out_path)
        if ok:
            print("done")

    print(f"\nAll graphs saved to {SWEEP_DIR}/")


if __name__ == "__main__":
    main()
