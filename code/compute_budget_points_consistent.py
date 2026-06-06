"""
Compute budget-based operating points (40%, 60%) using the EXACT same
prediction pipeline as the paper's routing frontier figure.

This reuses plot_routing_frontier_lassocv_clean.py's train_lassocv_delta()
and the same data loading, so predictions are identical to the paper's
existing results.
"""
import sys
import warnings
import numpy as np
from pathlib import Path

sys.path.insert(0, "code")
# Import the same functions the frontier plot uses
from plotting.plot_routing_frontier_lassocv_clean import train_lassocv_delta, RESULTS
from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections,
    build_features,
)

def main():
    print("Loading data (same path as routing frontier)...")
    records = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)
    X = build_features(records)
    N = len(records)
    print(f"Total records: {N}")

    print("Training LassoCV (same as routing frontier)...")
    pred_dw, pred_dc = train_lassocv_delta(X, records, corrections)

    # Base and corrected values
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
    delta_cer = base_cer - corr_cer
    delta_wer = base_wer - corr_wer
    token_counts = np.array([len(r.get("raw_ocr", "").split()) for r in records], dtype=np.int32)
    total_tokens = int(token_counts.sum())

    # Sort by predicted CER delta (descending) — same as frontier plot
    sort_idx_cer = np.argsort(pred_dc)[::-1]
    # Sort by predicted WER delta (descending)
    sort_idx_wer = np.argsort(pred_dw)[::-1]

    print(f"\nBaseline CER: {np.mean(base_cer):.4f}")
    print(f"Baseline WER: {np.mean(base_wer):.4f}")
    print(f"Full correction CER: {np.mean(corr_cer):.4f}")
    print(f"Full correction WER: {np.mean(corr_wer):.4f}")
    full_cer_improv = (np.mean(base_cer) - np.mean(corr_cer)) / np.mean(base_cer) * 100
    print(f"Full correction CER improvement: +{full_cer_improv:.1f}%")

    # --- Δ≥0.03 threshold (paper's approach) ---
    print("\n" + "="*70)
    print("Reference: Δ_CER ≥ 0.03 threshold")
    print("="*70)
    mask_delta = pred_dc >= 0.03
    k_delta = int(mask_delta.sum())
    result_cer_delta = np.where(mask_delta, corr_cer, base_cer)
    result_wer_delta = np.where(mask_delta, corr_wer, base_wer)
    routed_tokens_delta = int(token_counts[mask_delta].sum())
    savings_delta = 1.0 - routed_tokens_delta / total_tokens
    avg_cer_delta = float(np.mean(result_cer_delta))
    avg_wer_delta = float(np.mean(result_wer_delta))
    cer_improv_delta = (np.mean(base_cer) - avg_cer_delta) / np.mean(base_cer) * 100
    print(f"  Docs corrected: {k_delta}/{N} ({k_delta/N*100:.1f}%)")
    print(f"  CER: {avg_cer_delta:.4f}")
    print(f"  WER: {avg_wer_delta:.4f}")
    print(f"  Token savings: {savings_delta*100:.1f}%")
    print(f"  CER improvement: +{cer_improv_delta:.1f}%")

    # --- Budget-based points: top-k by predicted CER delta ---
    for budget_pct in [0.20, 0.40, 0.60, 0.80, 1.00]:
        k = int(round(budget_pct * N))
        print(f"\n{'='*70}")
        print(f"Budget: top {budget_pct*100:.0f}% ({k}/{N} docs)")
        print("="*70)

        # Select top-k by predicted CER delta
        routed_indices = set(sort_idx_cer[:k])

        result_cer = np.array([
            corr_cer[i] if i in routed_indices else base_cer[i]
            for i in range(N)
        ])
        result_wer = np.array([
            corr_wer[i] if i in routed_indices else base_wer[i]
            for i in range(N)
        ])

        routed_tokens = sum(token_counts[i] for i in routed_indices)
        savings = 1.0 - routed_tokens / total_tokens
        avg_cer = float(np.mean(result_cer))
        avg_wer = float(np.mean(result_wer))
        cer_improv = (np.mean(base_cer) - avg_cer) / np.mean(base_cer) * 100

        print(f"  CER: {avg_cer:.4f}")
        print(f"  WER: {avg_wer:.4f}")
        print(f"  Token savings: {savings*100:.1f}%")
        print(f"  CER improvement: +{cer_improv:.1f}%")
        print(f"  % of full CER improvement recovered: {cer_improv/full_cer_improv*100:.1f}%")

    # --- Also compute: what threshold gives ~40% and ~60%? ---
    print(f"\n{'='*70}")
    print("Threshold finder: what Δ_CER threshold gives ~40% and ~60%?")
    print("="*70)
    sorted_pred_dc = np.sort(pred_dc)[::-1]
    for target_pct in [0.40, 0.60]:
        k = int(round(target_pct * N))
        if k < N:
            threshold_at_k = sorted_pred_dc[k-1]
            print(f"  Top {target_pct*100:.0f}% ({k} docs): Δ_CER threshold = {threshold_at_k:.4f}")


if __name__ == "__main__":
    main()
