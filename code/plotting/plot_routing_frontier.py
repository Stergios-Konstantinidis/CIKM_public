import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections,
    build_features,
    build_labels,
    GBT_PARAMS
)

# Orthographic correction baselines (from ortho_summary.json, Tesseract Full_Orthographic)
ORTHO_BASELINES = {"wer": 0.2900, "cer": 0.0907}


# ─── Delta regression: train Ridge & MLP to predict delta (baseline - corrected)
def train_delta_regressions(X, records, corrections):
    """Train Ridge and MLP regressors to predict delta_wer and delta_cer.
    
    Uses 10-fold cross-validation to produce out-of-sample predictions
    for every record, avoiding data leakage.
    
    Returns dict: {filename: {pred_ridge_delta_wer, pred_ridge_delta_cer,
                               pred_mlp_delta_wer, pred_mlp_delta_cer}}
    """
    # Compute ground-truth deltas
    delta_wer = np.array([
        float(r["wer"]) - corrections.get(r["filename"], {}).get("wer", float(r["wer"]))
        for r in records
    ], dtype=np.float32)
    delta_cer = np.array([
        float(r["cer"]) - corrections.get(r["filename"], {}).get("cer", float(r["cer"]))
        for r in records
    ], dtype=np.float32)
    
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Ridge regression for delta
    ridge_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 50)))
    ])
    pred_ridge_delta_wer = cross_val_predict(ridge_pipe, X, delta_wer, cv=cv)
    pred_ridge_delta_cer = cross_val_predict(ridge_pipe, X, delta_cer, cv=cv)
    
    # MLP regression for delta (using L-BFGS which is better for small datasets)
    mlp_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=2000,
            solver='lbfgs', random_state=42, verbose=False
        ))
    ])
    pred_mlp_delta_wer = cross_val_predict(mlp_pipe, X, delta_wer, cv=cv)
    pred_mlp_delta_cer = cross_val_predict(mlp_pipe, X, delta_cer, cv=cv)
    
    predictions = {}
    for i, r in enumerate(records):
        predictions[r["filename"]] = {
            "pred_ridge_delta_wer": float(pred_ridge_delta_wer[i]),
            "pred_ridge_delta_cer": float(pred_ridge_delta_cer[i]),
            "pred_mlp_delta_wer": float(pred_mlp_delta_wer[i]),
            "pred_mlp_delta_cer": float(pred_mlp_delta_cer[i]),
        }
    
    return predictions


# ─── Classifier definitions ──────────────────────────────────────────────────
def get_models():
    return {
        "LogReg": LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=42,
        ),
    }


COLORS = ['red', 'green', 'orange', 'cyan', 'brown', 'navy', 'purple', 'olive']
MARKERS = ['o', 's', '^', 'D', 'P', 'v', 'X', '*']


# ─── Train classifiers ONCE for a given target metric ────────────────────────
def train_classifiers(X_stacked, delta_target, min_delta):
    """Train all classifiers via 10-fold CV on delta_target > min_delta.
    
    Returns dict: {model_name: probas} where probas[i] is the predicted 
    probability of the positive class (routing document i).
    """
    y = (delta_target > min_delta).astype(int)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    models = get_models()
    
    routing_probas = {}
    for name, clf in models.items():
        try:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', clf)
            ])
            # Use method='predict_proba' to get continuous probabilities
            probas = cross_val_predict(pipeline, X_stacked, y, cv=cv, n_jobs=-1, method='predict_proba')[:, 1]
            routing_probas[name] = probas
        except Exception as e:
            print(f"    ⚠ {name} failed: {e}")
            routing_probas[name] = None
    
    return routing_probas


# ─── Plot a single subplot ───────────────────────────────────────────────────
def plot_subplot(ax, display_metric, records, corrections, delta_preds, routing_probas):
    """Plot one frontier subplot showing `display_metric` performance using 
    pre-computed routing probabilities from classifiers trained on some target metric.
    """
    base_vals = np.array([float(r[display_metric]) for r in records])
    corr_vals = np.array([corrections.get(r["filename"], {}).get(display_metric, float(r[display_metric])) for r in records])
    deltas = base_vals - corr_vals  # ground-truth delta for display_metric
    N = len(records)
    
    avg_base = np.mean(base_vals)
    ax.axhline(avg_base, color='black', linestyle='--', linewidth=1.5,
               label=f'Baseline ({avg_base:.4f})')
    
    # Orthographic Correction (Full) baseline — horizontal line
    ortho_val = ORTHO_BASELINES.get(display_metric)
    if ortho_val is not None:
        ax.axhline(ortho_val, color='green', linestyle='-.', linewidth=1.5,
                   label=f'Orthographic Correction ({ortho_val:.4f})')
    
    # Oracle Frontier (sort by actual delta of display_metric, highest first)
    sort_idx = np.argsort(deltas)[::-1]
    oracle_pct, oracle_val = [0.0], [avg_base]
    current_sum = np.sum(base_vals)
    for i in range(N):
        current_sum -= deltas[sort_idx[i]]
        oracle_pct.append((i + 1) / N * 100)
        oracle_val.append(current_sum / N)
    ax.plot(oracle_pct, oracle_val, color='black', linestyle=':', label='Oracle', linewidth=1.5)

    # LogReg Curve & Stacked Areas
    if "LogReg" in routing_probas and routing_probas["LogReg"] is not None:
        probas = routing_probas["LogReg"]
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
            
        ax.plot(frontier_pct, frontier_val, color='black', linewidth=2, label='LogReg Routing Curve')
        
        t1 = np.array(y_unrouted)
        t2 = t1 + np.array(y_missed)
        t3 = t2 + np.array(y_over)
        
        ax.fill_between(frontier_pct, 0, t1, color='lightcoral', alpha=0.7, label='OCR Errors (Uncorrected)')
        ax.fill_between(frontier_pct, t1, t2, color='gold', alpha=0.7, label='Elements not corrected by the LLM')
        ax.fill_between(frontier_pct, t2, t3, color='skyblue', alpha=0.7, label='Over-correction (LLM introduced)')
    ax.set_xlabel('% of Documents Corrected')
    ax.set_ylabel(f'{display_metric.upper()} per Token')
    ax.grid(True, alpha=0.3)


def main():
    print("Loading data...")
    records = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Advanced_5_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)
    X = build_features(records)
    
    # Train delta regressions (once, shared across all pp values)
    print("Training delta regressions (Ridge + MLP via 10-fold CV)...")
    delta_preds = train_delta_regressions(X, records, corrections)
    print(f"  Delta predictions ready for {len(delta_preds)} documents")
    
    # Precompute arrays needed for classifier training
    base_wer = np.array([float(r["wer"]) for r in records])
    corr_wer = np.array([corrections.get(r["filename"], {}).get("wer", float(r["wer"])) for r in records])
    delta_wer_gt = base_wer - corr_wer
    
    base_cer = np.array([float(r["cer"]) for r in records])
    corr_cer = np.array([corrections.get(r["filename"], {}).get("cer", float(r["cer"])) for r in records])
    delta_cer_gt = base_cer - corr_cer
    
    # Stacked feature matrix (shared)
    pred_ridge_delta_wer = np.array([delta_preds[r["filename"]]["pred_ridge_delta_wer"] for r in records])
    pred_ridge_delta_cer = np.array([delta_preds[r["filename"]]["pred_ridge_delta_cer"] for r in records])
    pred_mlp_delta_wer = np.array([delta_preds[r["filename"]]["pred_mlp_delta_wer"] for r in records])
    pred_mlp_delta_cer = np.array([delta_preds[r["filename"]]["pred_mlp_delta_cer"] for r in records])
    X_stacked = np.column_stack([
        X, pred_ridge_delta_wer, pred_ridge_delta_cer,
        pred_mlp_delta_wer, pred_mlp_delta_cer
    ])
    
    for pp in range(0, 16):
        min_delta = pp / 100.0
        out_path = f"results/figures/routing_frontier/routing_frontier_{pp}pp.png"
        # Force regeneration to pick up the new orthographic baseline
        if Path(out_path).exists():
            Path(out_path).unlink()
        print(f"\nGenerating plots for δ > {pp}pp (min_delta={min_delta:.2f})...")
        
        # Train classifiers ONCE per target metric
        print(f"  Training classifiers on δ-WER > {min_delta:.2f}...")
        masks_wer = train_classifiers(X_stacked, delta_wer_gt, min_delta)
        print(f"  Training classifiers on δ-CER > {min_delta:.2f}...")
        masks_cer = train_classifiers(X_stacked, delta_cer_gt, min_delta)
        
        # 2×2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Top row: classifiers trained on δ-WER
        plot_subplot(axes[0, 0], "wer", records, corrections, delta_preds, masks_wer)
        axes[0, 0].set_title("WER Performance (classifiers trained on δ-WER)")
        plot_subplot(axes[0, 1], "cer", records, corrections, delta_preds, masks_wer)
        axes[0, 1].set_title("CER Performance (classifiers trained on δ-WER)")
        
        # Bottom row: classifiers trained on δ-CER
        plot_subplot(axes[1, 0], "wer", records, corrections, delta_preds, masks_cer)
        axes[1, 0].set_title("WER Performance (classifiers trained on δ-CER)")
        plot_subplot(axes[1, 1], "cer", records, corrections, delta_preds, masks_cer)
        axes[1, 1].set_title("CER Performance (classifiers trained on δ-CER)")
        
        fig.suptitle(f'Routing Frontier — δ > {pp}pp', fontsize=16, y=1.01)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")

if __name__ == "__main__":
    main()
