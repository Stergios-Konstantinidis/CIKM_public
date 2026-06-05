"""
Generate per-model routing frontier graphs (1×2: WER + CER).

Usage:
  python plot_routing_per_model.py <model_name>

  model_name: ridge_mlp | elasticnetcv | lassocv | lassolarscv

Each model's graph is saved to results/figures/routing_frontier/<model_name>/routing_graph_paper.png
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
from sklearn.linear_model import (
    RidgeCV, LogisticRegression, ElasticNetCV, LassoCV, LassoLarsCV,
)
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections,
    build_features,
)
from utils.regression_features import load_confidence_lookup
# Lazy import — only needed if cache is missing
def train_confbert_router(*args, **kwargs):
    from experiments.confbert_router import train_confbert_router as _impl
    return _impl(*args, **kwargs)

# ── Orthographic baseline (Tesseract, best threshold) ──
ORTHO_BASELINES = {"wer": 0.2284, "cer": 0.0675}

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
IMAGES = BASE / "data" / "evaluation_dataset" / "images"


# ── Model factories ──
def get_regressors(model_name):
    """Return a list of (name, sklearn_pipeline) for the chosen model."""
    if model_name == "ridge_mlp":
        return [
            ("ridge", Pipeline([
                ('scaler', StandardScaler()),
                ('reg', RidgeCV(alphas=np.logspace(-3, 3, 50)))
            ])),
            ("mlp", Pipeline([
                ('scaler', StandardScaler()),
                ('reg', MLPRegressor(
                    hidden_layer_sizes=(128, 64), max_iter=2000,
                    solver='lbfgs', random_state=42, verbose=False
                ))
            ])),
        ]
    elif model_name == "elasticnetcv":
        return [
            ("elasticnetcv", Pipeline([
                ('scaler', StandardScaler()),
                ('reg', ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                                     cv=5, max_iter=5000, random_state=42))
            ])),
        ]
    elif model_name == "lassocv":
        return [
            ("lassocv", Pipeline([
                ('scaler', StandardScaler()),
                ('reg', LassoCV(cv=5, max_iter=5000, random_state=42))
            ])),
        ]
    elif model_name == "lassolarscv":
        return [
            ("lassolarscv", Pipeline([
                ('scaler', StandardScaler()),
                ('reg', LassoLarsCV(cv=5, max_iter=5000))
            ])),
        ]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_delta_regressions(X, records, corrections, model_name):
    """Train delta regressions using the chosen model(s)."""
    delta_wer = np.array([
        float(r["wer"]) - corrections.get(r["filename"], {}).get("wer", float(r["wer"]))
        for r in records
    ], dtype=np.float32)
    delta_cer = np.array([
        float(r["cer"]) - corrections.get(r["filename"], {}).get("cer", float(r["cer"]))
        for r in records
    ], dtype=np.float32)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    regressors = get_regressors(model_name)

    predictions = {r["filename"]: {} for r in records}
    all_pred_arrays = []

    for reg_name, pipe in regressors:
        print(f"  Training {reg_name} for delta WER/CER ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_dw = cross_val_predict(pipe, X, delta_wer, cv=cv)
            pred_dc = cross_val_predict(pipe, X, delta_cer, cv=cv)

        for i, r in enumerate(records):
            predictions[r["filename"]][f"pred_{reg_name}_delta_wer"] = float(pred_dw[i])
            predictions[r["filename"]][f"pred_{reg_name}_delta_cer"] = float(pred_dc[i])

        all_pred_arrays.extend([pred_dw, pred_dc])

    return predictions, all_pred_arrays


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


def plot_subplot(ax, display_metric, records, corrections,
                 our_probas, confbert_probas, token_counts, model_label):
    base_vals = np.array([float(r[display_metric]) for r in records])
    corr_vals = np.array([
        corrections.get(r["filename"], {}).get(display_metric, float(r[display_metric]))
        for r in records
    ])
    deltas = base_vals - corr_vals
    N = len(records)
    avg_base = np.mean(base_vals)

    def _auc(pct, val):
        if hasattr(np, 'trapezoid'):
            return np.trapezoid(val, pct) / 100
        else:
            return np.trapz(val, pct) / 100

    def _annotate_curve(ax, pct_list, val_list, label, color,
                        frac=0.35, offset=(10, -15)):
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

    # (a) Oracle
    oracle_idx = np.argsort(deltas)[::-1]
    oracle_pct, oracle_full, _ = compute_routing_curves(
        oracle_idx, base_vals, corr_vals, N, deltas, 0.0)
    oracle_auc = _auc(oracle_pct, oracle_full)
    ax.plot(oracle_pct, oracle_full, color='#7f7f7f', linestyle=':', linewidth=1.8,
            label='Oracle (Perfect Routing)')
    _annotate_curve(ax, oracle_pct, oracle_full,
                    f'Oracle (AUC = {oracle_auc * 100:.2f})',
                    '#7f7f7f', frac=0.55, offset=(8, -18))

    # (b) Orthographic Correction
    ortho_val = ORTHO_BASELINES.get(display_metric)
    if ortho_val is not None:
        ax.axhline(ortho_val, color='#2ca02c', linestyle='-.', linewidth=1.8)
        ax.annotate(
            'Spell-check',
            xy=(75, ortho_val), xytext=(0, -6),
            textcoords='offset points', fontsize=7.5, fontweight='bold',
            color='#2ca02c', ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#2ca02c',
                      alpha=0.85, lw=0.8),
        )

    # (c) ConfBERT
    confbert_idx = np.argsort(confbert_probas)[::-1]
    confbert_pct, confbert_full, _ = compute_routing_curves(
        confbert_idx, base_vals, corr_vals, N, confbert_probas, 0.5)
    confbert_auc = _auc(confbert_pct, confbert_full)
    ax.plot(confbert_pct, confbert_full, color='#ff7f0e', linestyle='--', linewidth=2,
            label='ConfBERT')
    _annotate_curve(ax, confbert_pct, confbert_full,
                    f'ConfBERT (AUC = {confbert_auc * 100:.2f})',
                    '#ff7f0e', frac=0.45, offset=(10, 15))

    # (d) Our Approach
    our_idx = np.argsort(our_probas)[::-1]
    our_pct, our_full, _ = compute_routing_curves(
        our_idx, base_vals, corr_vals, N, our_probas, 0.5)
    our_auc = _auc(our_pct, our_full)
    ax.plot(our_pct, our_full, color='#d62728', linewidth=2.5,
            label=f'Our Approach ({model_label} + LogReg)')
    _annotate_curve(ax, our_pct, our_full,
                    f'Ours (AUC = {our_auc * 100:.2f})',
                    '#d62728', frac=0.25, offset=(-10, 20))

    # Baseline
    ax.axhline(avg_base, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.annotate(
        'Baseline',
        xy=(85, avg_base), xytext=(0, -6),
        textcoords='offset points', fontsize=7.5, fontweight='bold',
        color='black', ha='center', va='top', alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black',
                  alpha=0.6, lw=0.8),
    )

    ax.set_xlabel('% of Documents Corrected', fontsize=11)
    ax.set_ylabel(display_metric.upper(), fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)

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


# ── Pretty labels ──
MODEL_LABELS = {
    "ridge_mlp": "Ridge + MLP",
    "elasticnetcv": "ElasticNetCV",
    "lassocv": "LassoCV",
    "lassolarscv": "LassoLarsCV",
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_name>")
        print(f"  model_name: {' | '.join(MODEL_LABELS.keys())}")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in MODEL_LABELS:
        print(f"Unknown model: {model_name}")
        print(f"Choose from: {' | '.join(MODEL_LABELS.keys())}")
        sys.exit(1)

    model_label = MODEL_LABELS[model_name]
    print(f"\n{'='*60}")
    print(f"  Routing Frontier — {model_label}")
    print(f"{'='*60}")

    print("Loading data...")
    records = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)
    X = build_features(records)

    # Token counts per document
    token_counts = np.array([
        len(r.get("raw_ocr", "").split()) for r in records
    ], dtype=np.int32)

    print(f"Training delta regressions ({model_label}) via 10-fold CV...")
    delta_preds, pred_arrays = train_delta_regressions(X, records, corrections, model_name)

    # Ground-truth deltas for CER
    base_cer = np.array([float(r["cer"]) for r in records])
    corr_cer = np.array([
        corrections.get(r["filename"], {}).get("cer", float(r["cer"]))
        for r in records
    ])
    delta_cer_gt = base_cer - corr_cer

    # Stacked feature matrix (base features + regression predictions)
    X_stacked = np.column_stack([X] + pred_arrays)
    print(f"  Stacked feature matrix: {X_stacked.shape}")

    min_delta = 0.0
    print(f"Training classifier (LogReg on δ-CER > {min_delta})...")
    our_probas = train_our_classifier(X_stacked, delta_cer_gt, min_delta)

    # ConfBERT
    print("Loading ConfBERT probas...")
    confbert_cache = RESULTS / "ml_models/confbert_probas.npy"
    if confbert_cache.exists():
        confbert_probas = np.load(confbert_cache)
        if len(confbert_probas) != len(records):
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

    # ── 1×2 figure ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_subplot(axes[0], "wer", records, corrections,
                 our_probas, confbert_probas, token_counts, model_label)

    plot_subplot(axes[1], "cer", records, corrections,
                 our_probas, confbert_probas, token_counts, model_label)

    fig.suptitle(f"Routing Frontier — {model_label}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save to per-model folder
    out_dir = RESULTS / "figures" / "routing_frontier" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "routing_graph_paper.png"
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
