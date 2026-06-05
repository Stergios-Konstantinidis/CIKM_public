"""
Routing frontier sweep for LassoCV regressor.
Generates 16 plots (δ > 0pp to δ > 15pp), one 2×2 grid each.
Output: results/figures/routing_frontier/lassolarscv/routing_frontier_{pp}pp.png
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
from sklearn.linear_model import LassoCV, LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections,
    build_features,
)

# ── Orthographic baseline (best threshold) ──
ORTHO_BASELINES = {"wer": 0.2284, "cer": 0.0675}

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"


def train_lassocv_delta(X, records, corrections):
    """Train LassoCV to predict delta_wer and delta_cer via 10-fold CV."""
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


def train_classifier(X_stacked, delta_target, min_delta):
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


def plot_subplot(ax, display_metric, records, corrections, probas):
    base_vals = np.array([float(r[display_metric]) for r in records])
    corr_vals = np.array([
        corrections.get(r["filename"], {}).get(display_metric, float(r[display_metric]))
        for r in records
    ])
    deltas = base_vals - corr_vals
    N = len(records)
    avg_base = np.mean(base_vals)

    ax.axhline(avg_base, color='black', linestyle='--', linewidth=1.5,
               label=f'Baseline ({avg_base:.4f})')

    # Orthographic Correction
    ortho_val = ORTHO_BASELINES.get(display_metric)
    if ortho_val is not None:
        ax.axhline(ortho_val, color='green', linestyle='-.', linewidth=1.5,
                   label='Spell-check')

    # Oracle
    sort_idx = np.argsort(deltas)[::-1]
    oracle_pct, oracle_val = [0.0], [avg_base]
    current_sum = np.sum(base_vals)
    for i in range(N):
        current_sum -= deltas[sort_idx[i]]
        oracle_pct.append((i + 1) / N * 100)
        oracle_val.append(current_sum / N)
    ax.plot(oracle_pct, oracle_val, color='black', linestyle=':', label='Oracle', linewidth=1.5)

    # LogReg + LassoCV curve with stacked areas
    sort_idx = np.argsort(probas)[::-1]
    frontier_pct, frontier_val = [0.0], [avg_base]
    y_unrouted = [avg_base]
    y_missed = [0.0]
    y_over = [0.0]

    sum_unrouted = np.sum(base_vals)
    sum_missed = 0.0
    sum_over = 0.0

    for k in range(N):
        idx = sort_idx[k]
        b = base_vals[idx]
        c = corr_vals[idx]

        sum_unrouted -= b
        sum_missed += min(b, c)
        sum_over += max(0.0, c - b)

        frontier_pct.append((k + 1) / N * 100)
        y_unrouted.append(sum_unrouted / N)
        y_missed.append(sum_missed / N)
        y_over.append(sum_over / N)
        frontier_val.append((sum_unrouted + sum_missed + sum_over) / N)

    ax.plot(frontier_pct, frontier_val, color='black', linewidth=2,
            label='LogReg + LassoCV')

    t1 = np.array(y_unrouted)
    t2 = t1 + np.array(y_missed)
    t3 = t2 + np.array(y_over)

    ax.fill_between(frontier_pct, 0, t1, color='lightcoral', alpha=0.7,
                    label='OCR Errors (Uncorrected)')
    ax.fill_between(frontier_pct, t1, t2, color='gold', alpha=0.7,
                    label='Elements not corrected by the LLM')
    ax.fill_between(frontier_pct, t2, t3, color='skyblue', alpha=0.7,
                    label='Over-correction (LLM introduced)')

    ax.set_xlabel('% of Documents Corrected')
    ax.set_ylabel(f'{display_metric.upper()}')
    ax.grid(True, alpha=0.3)


def main():
    print("Loading data...")
    records = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)
    X = build_features(records)

    print("Training LassoCV delta regressions (10-fold CV)...")
    pred_dw, pred_dc = train_lassocv_delta(X, records, corrections)

    # Stacked features
    X_stacked = np.column_stack([X, pred_dw, pred_dc])
    print(f"  Stacked feature matrix: {X_stacked.shape}")

    # Ground-truth deltas
    base_wer = np.array([float(r["wer"]) for r in records])
    corr_wer = np.array([corrections.get(r["filename"], {}).get("wer", float(r["wer"])) for r in records])
    delta_wer_gt = base_wer - corr_wer

    base_cer = np.array([float(r["cer"]) for r in records])
    corr_cer = np.array([corrections.get(r["filename"], {}).get("cer", float(r["cer"])) for r in records])
    delta_cer_gt = base_cer - corr_cer

    out_dir = RESULTS / "figures" / "routing_frontier" / "lassolarscv"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pp in range(0, 16):
        min_delta = pp / 100.0
        print(f"\nGenerating δ > {pp}pp (min_delta={min_delta:.2f})...")

        # Train classifiers for WER and CER targets
        probas_wer = train_classifier(X_stacked, delta_wer_gt, min_delta)
        probas_cer = train_classifier(X_stacked, delta_cer_gt, min_delta)

        # 2×2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Top row: classifiers trained on δ-WER
        plot_subplot(axes[0, 0], "wer", records, corrections, probas_wer)
        axes[0, 0].set_title("WER Performance (classifier trained on δ-WER)")
        axes[0, 0].legend(loc='upper right', fontsize=7, framealpha=0.9)

        plot_subplot(axes[0, 1], "cer", records, corrections, probas_wer)
        axes[0, 1].set_title("CER Performance (classifier trained on δ-WER)")
        axes[0, 1].legend(loc='upper right', fontsize=7, framealpha=0.9)

        # Bottom row: classifiers trained on δ-CER
        plot_subplot(axes[1, 0], "wer", records, corrections, probas_cer)
        axes[1, 0].set_title("WER Performance (classifier trained on δ-CER)")
        axes[1, 0].legend(loc='upper right', fontsize=7, framealpha=0.9)

        plot_subplot(axes[1, 1], "cer", records, corrections, probas_cer)
        axes[1, 1].set_title("CER Performance (classifier trained on δ-CER)")
        axes[1, 1].legend(loc='upper right', fontsize=7, framealpha=0.9)

        fig.suptitle(f'Routing Frontier — LassoCV — δ > {pp}pp', fontsize=16, y=1.01)
        plt.tight_layout()
        out_path = out_dir / f"routing_frontier_{pp}pp.png"
        plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
