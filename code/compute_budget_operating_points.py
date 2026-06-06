"""
Compute CER, WER, token savings, and docs corrected at fixed budget
operating points (40% and 60% of documents corrected), using the same
LassoCV routing frontier logic as the paper.
"""
import sys
import warnings
import numpy as np
from pathlib import Path

sys.path.insert(0, "code")
from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections,
    build_features,
)
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"


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

    return pred_dw, pred_dc, delta_wer, delta_cer


def compute_operating_point(records, corrections, pred_deltas_cer, pred_deltas_wer, budget_frac):
    """Compute metrics when correcting exactly budget_frac of documents,
    sorted by predicted delta (descending)."""
    N = len(records)
    k = int(round(budget_frac * N))  # number of docs to correct

    base_wer = np.array([float(r["wer"]) for r in records])
    base_cer = np.array([float(r["cer"]) for r in records])
    corr_wer = np.array([
        corrections.get(r["filename"], {}).get("wer", float(r["wer"]))
        for r in records
    ])
    corr_cer = np.array([
        corrections.get(r["filename"], {}).get("cer", float(r["cer"]))
        for r in records
    ])

    # Sort by predicted CER delta (descending) — same as the paper's approach
    sort_idx = np.argsort(pred_deltas_cer)[::-1]
    routed_set = set(sort_idx[:k])

    # Compute resulting CER and WER
    result_cer = np.array([
        corr_cer[i] if i in routed_set else base_cer[i]
        for i in range(N)
    ])
    result_wer = np.array([
        corr_wer[i] if i in routed_set else base_wer[i]
        for i in range(N)
    ])

    avg_cer = float(np.mean(result_cer))
    avg_wer = float(np.mean(result_wer))

    # Token savings: compute token counts
    token_counts = np.array([
        len(r.get("raw_ocr", "").split()) for r in records
    ], dtype=np.int32)
    total_tokens = token_counts.sum()
    routed_tokens = sum(token_counts[i] for i in routed_set)
    token_savings = 1.0 - (routed_tokens / total_tokens)

    # CER improvement over baseline
    baseline_cer = float(np.mean(base_cer))
    cer_improvement = (baseline_cer - avg_cer) / baseline_cer * 100

    return {
        "budget_frac": budget_frac,
        "k": k,
        "N": N,
        "avg_cer": avg_cer,
        "avg_wer": avg_wer,
        "token_savings": token_savings,
        "cer_improvement": cer_improvement,
        "baseline_cer": baseline_cer,
        "baseline_wer": float(np.mean(base_wer)),
    }


def main():
    print("Loading data...")
    records = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)
    X = build_features(records)

    print(f"Total records: {len(records)}")

    print("Training LassoCV delta regressions (10-fold CV)...")
    pred_dw, pred_dc, actual_dw, actual_dc = train_lassocv_delta(X, records, corrections)

    # Also compute the Δ≥0.03 point for reference
    print("\n" + "=" * 70)
    print("Operating Points for Budget-Based Routing")
    print("=" * 70)

    budgets = [0.40, 0.60]

    for b in budgets:
        result = compute_operating_point(records, corrections, pred_dc, pred_dw, b)
        print(f"\n--- Budget: {b*100:.0f}% of documents corrected ---")
        print(f"  Docs corrected: {result['k']}/{result['N']}")
        print(f"  CER: {result['avg_cer']:.4f}")
        print(f"  WER: {result['avg_wer']:.4f}")
        print(f"  Token savings: {result['token_savings']*100:.1f}%")
        print(f"  CER improvement: +{result['cer_improvement']:.1f}%")
        print(f"  Baseline CER: {result['baseline_cer']:.4f}")
        print(f"  Baseline WER: {result['baseline_wer']:.4f}")

    # Also compute with the Δ≥0.03 threshold for cross-check
    print("\n--- Reference: Δ≥0.03 threshold ---")
    routed_mask = pred_dc >= 0.03
    N = len(records)
    k_delta = int(routed_mask.sum())

    base_cer = np.array([float(r["cer"]) for r in records])
    base_wer = np.array([float(r["wer"]) for r in records])
    corr_cer = np.array([
        corrections.get(r["filename"], {}).get("cer", float(r["cer"]))
        for r in records
    ])
    corr_wer = np.array([
        corrections.get(r["filename"], {}).get("wer", float(r["wer"]))
        for r in records
    ])

    result_cer = np.where(routed_mask, corr_cer, base_cer)
    result_wer = np.where(routed_mask, corr_wer, base_wer)

    token_counts = np.array([len(r.get("raw_ocr", "").split()) for r in records])
    total_tokens = token_counts.sum()
    routed_tokens = token_counts[routed_mask].sum()
    token_savings = 1.0 - (routed_tokens / total_tokens)

    print(f"  Docs corrected: {k_delta}/{N} ({k_delta/N*100:.1f}%)")
    print(f"  CER: {np.mean(result_cer):.4f}")
    print(f"  WER: {np.mean(result_wer):.4f}")
    print(f"  Token savings: {token_savings*100:.1f}%")


if __name__ == "__main__":
    main()
