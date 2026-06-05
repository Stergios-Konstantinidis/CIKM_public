"""
Routing frontier sweep for LassoCV — clean line-curve style with ConfBERT.
Generates 16 plots (δ > 0pp to δ > 15pp), each 1×2 (WER + CER).
Output: results/figures/routing_frontier/lassocv/routing_frontier_{pp}pp.png
"""
import sys
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections,
    build_features,
)

# Lazy import for ConfBERT (avoids transformers dependency when cache exists)
def train_confbert_router(*args, **kwargs):
    from experiments.confbert_router import train_confbert_router as _impl
    return _impl(*args, **kwargs)

ORTHO_BASELINES = {"wer": 0.2284, "cer": 0.0675}
DELTA_MULT = 1.0  # adjusted via --delta-multiplier CLI arg

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
IMAGES = BASE / "data" / "evaluation_dataset" / "images"


def train_lassocv_delta(X, records, corrections):
    delta_wer = np.array([
        float(r["wer"]) - corrections.get(r["filename"], {}).get("wer", float(r["wer"]))
        for r in records
    ], dtype=np.float32)
    delta_cer = np.array([
        float(r["cer"]) - corrections.get(r["filename"], {}).get("cer", float(r["cer"]))
        for r in records
    ], dtype=np.float32)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', LassoCV(cv=5, max_iter=5000, random_state=42))
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_dw = cross_val_predict(pipe, X, delta_wer, cv=cv)
        pred_dc = cross_val_predict(pipe, X, delta_cer, cv=cv)

    return pred_dw, pred_dc





def compute_routing_curves(sort_idx, base_vals, corr_vals, N, scores, threshold):
    """Compute routing frontier curves.
    
    Full curve: applies correction to every document (shows overcorrection).
    Capped curve: applies correction only when scores[idx] >= threshold.
    """
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
                 pred_deltas, confbert_probas, token_counts, min_delta=0.0):
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

    # Oracle
    oracle_idx = np.argsort(deltas)[::-1]
    oracle_pct, oracle_full, _ = compute_routing_curves(
        oracle_idx, base_vals, corr_vals, N, deltas, 0.0)
    oracle_auc = _auc(oracle_pct, oracle_full)
    ax.plot(oracle_pct, oracle_full, color='#7f7f7f', linestyle=':', linewidth=1.8,
            label='Oracle (Perfect Routing)')
    _annotate_curve(ax, oracle_pct, oracle_full,
                    f'Oracle (AUC = {oracle_auc * 100:.2f})',
                    '#7f7f7f', frac=0.55, offset=(8, -18))

    # Spell-check
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

    # ConfBERT
    confbert_idx = np.argsort(confbert_probas)[::-1]
    confbert_pct, confbert_full, _ = compute_routing_curves(
        confbert_idx, base_vals, corr_vals, N, confbert_probas, 0.5)
    confbert_auc = _auc(confbert_pct, confbert_full)
    ax.plot(confbert_pct, confbert_full, color='#ff7f0e', linestyle='--', linewidth=2,
            label='ConfBERT')
    _annotate_curve(ax, confbert_pct, confbert_full,
                    f'ConfBERT (AUC = {confbert_auc * 100:.2f})',
                    '#ff7f0e', frac=0.45, offset=(10, 15))

    # Our Approach (LassoCV) — sort by predicted Δ
    our_idx = np.argsort(pred_deltas)[::-1]
    our_pct, our_full, _ = compute_routing_curves(
        our_idx, base_vals, corr_vals, N, pred_deltas, min_delta)
    our_auc = _auc(our_pct, our_full)
    ax.plot(our_pct, our_full, color='#d62728', linewidth=2.5,
            label='Our Approach (LassoCV)')
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


def main():
    print("Loading data...")
    records = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)
    X = build_features(records)

    token_counts = np.array([
        len(r.get("raw_ocr", "").split()) for r in records
    ], dtype=np.int32)

    print("Training LassoCV delta regressions (10-fold CV)...")
    pred_dw, pred_dc = train_lassocv_delta(X, records, corrections)
    print(f"  pred_ΔWER range: [{pred_dw.min():.4f}, {pred_dw.max():.4f}]")
    print(f"  pred_ΔCER range: [{pred_dc.min():.4f}, {pred_dc.max():.4f}]")

    # ConfBERT (load once)
    print("Loading ConfBERT probas...")
    confbert_cache = RESULTS / "ml_models/confbert_probas.npy"
    if confbert_cache.exists():
        confbert_probas = np.load(confbert_cache)
        if len(confbert_probas) != len(records):
            confbert_probas = train_confbert_router(
                records, corrections, str(RESULTS), str(IMAGES),
                metric="cer", min_delta=0.0)
            np.save(confbert_cache, confbert_probas)
    else:
        confbert_probas = train_confbert_router(
            records, corrections, str(RESULTS), str(IMAGES),
            metric="cer", min_delta=0.0)
        np.save(confbert_cache, confbert_probas)

    out_dir = RESULTS / "figures" / "routing_frontier" / "lassocv"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pp in range(0, 16):
        min_delta = pp / 100.0
        print(f"\nGenerating δ > {pp}pp (min_delta={min_delta:.2f})...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        plot_subplot(axes[0], "wer", records, corrections,
                     pred_dw, confbert_probas, token_counts, min_delta=min_delta)

        plot_subplot(axes[1], "cer", records, corrections,
                     pred_dc, confbert_probas, token_counts, min_delta=min_delta)

        fig.suptitle(f'Routing Frontier — LassoCV — δ > {pp}pp',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        out_path = out_dir / f"routing_frontier_{pp}pp.png"
        plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
