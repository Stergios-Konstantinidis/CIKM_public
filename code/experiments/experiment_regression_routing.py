"""
experiment_regression_routing.py
=================================
Offline routing threshold analysis — NO LLM API calls, NO retraining.

Workflow
--------
1. Load per-doc prediction files produced by experiment_linear/nn_regression.py
   (results/predictions_ridge_noemb.json, results/predictions_nn_noemb.json)
2. Load ALL LLM correction result files in results/ for a given engine+model
3. Build a "routing table" JSON: one row per (document, engine) containing
       filename | engine | split | actual_wer | actual_cer
       pred_wer | pred_cer | corrected_wer | corrected_cer | strategy
4. Sweep routing thresholds T ∈ [0.05 … 0.95] over predicted WER:
       route if pred_wer >= T  →  use corrected_wer, else keep actual_wer
5. Report: savings (1−fraction_routed), avg WER after routing, WER reduction
6. Compare against oracle threshold (route if actual_wer >= T)
7. Save routing table + threshold sweep results + figure

Outputs
-------
  results/routing_table.json                — full per-doc table (all thresholds)
  results/routing_threshold_sweep.json      — sweep summary (one row per threshold)
  results/regression_routing.png            — cost–quality frontier figure
  paper/figures/regression_routing.png
"""

import json
import glob
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
FIGS    = BASE / "paper" / "figures"
FIGS.mkdir(exist_ok=True)

# ── styling ──────────────────────────────────────────────────────────────────
DARK, PANEL, GRID, WHITE = "white", "#f5f5f5", "#d0d0d0", "black"
TEAL, VIOLET, CORAL, AMBER, GREEN = "#4ecdc4", "#a29bfe", "#ff6b6b", "#f7d794", "#55efc4"


# ─── Step 1: load prediction files ───────────────────────────────────────────

def load_predictions(suffix: str = "noemb") -> dict[tuple, dict]:
    """
    Load predictions from regression experiments.
    Returns dict keyed by (filename, engine) → record dict.
    Merges Ridge and NN predictions into a single lookup.
    """
    lookup: dict[tuple, dict] = {}

    for model_tag, fname in [
        ("ridge", f"predictions_ridge_{suffix}.json"),
        ("nn",    f"predictions_nn_{suffix}.json"),
    ]:
        path = RESULTS / "ml_models" / fname
        if not path.exists():
            print(f"  ⚠  {fname} not found — run the regression experiments first.")
            continue
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            key = (r["filename"], r["engine"])
            if key not in lookup:
                lookup[key] = {
                    "filename":   r["filename"],
                    "engine":     r["engine"],
                    "split":      r["split"],
                    "actual_wer": r["actual_wer"],
                    "actual_cer": r["actual_cer"],
                }
            # Clip predictions to [0, 1]: Ridge has no non-negativity
            # constraint so can produce pred_wer < 0 (unphysical), causing
            # those docs to be silently never routed at any positive threshold.
            lookup[key][f"pred_wer_{model_tag}"] = max(0.0, min(1.0, r["pred_wer"]))
            lookup[key][f"pred_cer_{model_tag}"] = max(0.0, min(1.0, r["pred_cer"]))

    print(f"  Prediction records loaded: {len(lookup)}")
    return lookup


# ─── Step 2: load all correction results ─────────────────────────────────────

def load_all_correction_results(
    engine: str,
    model_substr: str = "gemini-3-flash",
) -> dict[str, dict]:
    """
    Scan results/ for correction files matching engine + model, across all
    strategies and prompt levels.
    Returns dict: filename → {strategy, prompt_level, corrected_wer, corrected_cer}
    We keep the result with the BEST (lowest) corrected WER per filename,
    so the routing table reflects the best available correction.
    """
    pattern = str(RESULTS / "corrections" / engine / f"{engine}_*_{model_substr}*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  ⚠  No correction files found matching: {pattern}")
        return {}

    print(f"  Found {len(files)} correction result files for {engine}/{model_substr}")

    best: dict[str, dict] = {}      # filename → best correction record
    all_corrections: dict[str, list] = {}  # filename → [list of all corrections]

    for fpath in files:
        fname = Path(fpath).stem
        parts = fname.split("_")
        # Parse: engine_Strategy_thrXX_PromptName_level_model
        try:
            strategy = parts[1]  # Selective / FullText / Conditional
        except IndexError:
            strategy = "unknown"

        with open(fpath, encoding="utf-8") as f:
            records = json.load(f)

        for r in records:
            key = r["filename"]
            cw  = float(r.get("wer", 1.0))
            cc  = float(r.get("cer", 1.0))
            entry = {
                "corrected_wer":  cw,
                "corrected_cer":  cc,
                "strategy":       strategy,
                "source_file":    fname,
            }
            if key not in best or cw < best[key]["corrected_wer"]:
                best[key] = entry
            all_corrections.setdefault(key, []).append(entry)

    print(f"  Unique filenames with correction data: {len(best)}")
    return best, all_corrections


# ─── Step 3: build routing table ─────────────────────────────────────────────

def build_routing_table(
    predictions: dict,
    best_corrections: dict,
    all_corrections: dict,
) -> list[dict]:
    """
    Join predictions with correction results by filename.
    One row per (filename, engine).
    corrected_wer = best available LLM correction for that filename;
                    = actual_wer if no correction exists (worst case).
    """
    rows = []
    n_missing = 0
    for key, pred in predictions.items():
        filename, engine = key
        corr = best_corrections.get(filename)
        row = dict(pred)
        if corr:
            row["corrected_wer"] = corr["corrected_wer"]
            row["corrected_cer"] = corr["corrected_cer"]
            row["has_correction"] = True
            row["best_strategy"]  = corr["strategy"]
            row["n_corrections_available"] = len(all_corrections.get(filename, []))
        else:
            # No LLM correction available — fall back to actual (no improvement)
            row["corrected_wer"] = pred["actual_wer"]
            row["corrected_cer"] = pred["actual_cer"]
            row["has_correction"] = False
            row["best_strategy"]  = None
            row["n_corrections_available"] = 0
            n_missing += 1
        rows.append(row)

    print(f"  Routing table: {len(rows)} rows | {n_missing} without correction data")
    return rows


# ─── Step 4: threshold sweep ──────────────────────────────────────────────────

def sweep_thresholds(
    rows: list[dict],
    pred_key: str,          # e.g. "pred_wer_ridge" or "pred_wer_nn"
    thresholds: list[float],
    label: str,
    use_oracle: bool = False,
    test_only: bool = False,
    engine_filter: str = "tesseract",   # only evaluate rows for this engine
) -> list[dict]:
    """
    For each threshold T:
      - Documents with pred_wer (or actual_wer if oracle) >= T → corrected
      - Compute aggregate WER after routing, savings, WER reduction

    engine_filter: restrict analysis to rows from this OCR engine only.
    Corrections are engine-specific (e.g. tesseract corrections should not
    be applied to evaluate EasyOCR routing quality and vice versa).
    """
    subset = [
        r for r in rows
        if (not test_only or r.get("split") == "test")
        and (not engine_filter or r.get("engine") == engine_filter)
        and r.get("has_correction", False)   # only rows with real corrections
    ]
    if not subset:
        print(f"  ⚠  No rows match engine_filter={engine_filter!r} with corrections — skipping.")
        return []

    actual    = np.array([r["actual_wer"]    for r in subset], dtype=np.float64)
    corrected = np.array([r["corrected_wer"] for r in subset], dtype=np.float64)
    signal    = actual if use_oracle else np.array(
        [r.get(pred_key, r["actual_wer"]) for r in subset]
    )
    baseline_avg = float(actual.mean())


    results = []
    for T in thresholds:
        mask         = signal >= T
        n_routed     = int(mask.sum())
        frac_routed  = float(mask.mean())

        wer_after       = np.where(mask, corrected, actual)
        avg_wer_after   = float(wer_after.mean())
        wer_reduction   = baseline_avg - avg_wer_after
        savings         = 1.0 - frac_routed

        # Coverage: fraction of truly-bad docs that get routed
        bad_mask        = actual >= 0.30   # "high error" documents
        if bad_mask.sum() > 0:
            recall = float((mask & bad_mask).sum() / bad_mask.sum())
        else:
            recall = 0.0

        results.append({
            "threshold":     round(T, 3),
            "label":         label,
            "n_test":        len(subset),
            "n_routed":      n_routed,
            "frac_routed":   round(frac_routed, 4),
            "avg_wer_after": round(avg_wer_after, 6),
            "wer_reduction": round(wer_reduction, 6),
            "savings":       round(savings, 4),
            "recall_bad":    round(recall, 4),
            "baseline_wer":  round(baseline_avg, 6),
        })
    return results


# ─── Step 5: plot ─────────────────────────────────────────────────────────────

def plot_sweep(sweep_groups: list[tuple[str, list, str, str]], out_path: Path):
    """sweep_groups: [(label, results_list, color, linestyle)]"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK)

    titles  = [
        "Cost–Quality Frontier\n(Savings vs WER Reduction)",
        "Avg WER After Routing vs Coverage",
    ]
    xlabels = [
        "Savings (fraction NOT corrected)",
        "Fraction of Documents Routed",
    ]
    ylabels = [
        "WER Reduction (baseline − after)",
        "Average WER After Routing",
    ]

    for ax, title, xl, yl in zip(axes, titles, xlabels, ylabels):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=WHITE)
        ax.grid(True, color=GRID, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        # ax.set_title removed — caption in LaTeX
        ax.set_xlabel(xl, color=WHITE, fontsize=9)
        ax.set_ylabel(yl, color=WHITE, fontsize=9)

    baseline_wer = sweep_groups[0][1][0]["baseline_wer"]

    for label, results, color, ls in sweep_groups:
        xs0 = [r["savings"]       for r in results]
        ys0 = [r["wer_reduction"] for r in results]
        xs1 = [r["frac_routed"]   for r in results]
        ys1 = [r["avg_wer_after"] for r in results]

        axes[0].plot(xs0, ys0, color=color, linestyle=ls, linewidth=2,
                     marker="o", markersize=4, label=label)
        for r in results[::3]:
            axes[0].annotate(f"T={r['threshold']:.2f}", (r["savings"], r["wer_reduction"]),
                             textcoords="offset points", xytext=(3, 3),
                             color=color, fontsize=6.5, alpha=0.9)

        axes[1].plot(xs1, ys1, color=color, linestyle=ls, linewidth=2,
                     marker="o", markersize=4, label=label)

    # Baseline reference line on plot 1
    axes[1].axhline(baseline_wer, color=CORAL, linestyle=":",
                    linewidth=1.5, label=f"Baseline WER ({baseline_wer:.3f})")

    for ax in axes:
        ax.legend(facecolor="white", labelcolor="black", edgecolor="#cccccc", fontsize=7.5)

    # suptitle removed — caption in LaTeX
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"  ✓ {out_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main(engine: str = "tesseract", model_substr: str = "gemini-3-flash"):
    print("\n" + "=" * 60)
    print("Regression Routing Threshold Analysis (offline)")
    print("=" * 60)

    # 1. Predictions
    predictions = load_predictions(suffix="noemb")
    if not predictions:
        print("ERROR: No prediction files found. Run regression experiments first.")
        return

    # 2. Correction results
    best_corrections, all_corrections = load_all_correction_results(engine, model_substr)

    # 3. Routing table
    routing_table = build_routing_table(predictions, best_corrections, all_corrections)

    # Save full routing table
    rt_path = RESULTS / "routing/routing_table.json"
    with open(rt_path, "w", encoding="utf-8") as f:
        json.dump(routing_table, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Routing table saved → {rt_path}  ({len(routing_table)} rows)")

    # Quick stats
    test_rows = [r for r in routing_table if r.get("split") == "test"]
    covered   = sum(1 for r in test_rows if r["has_correction"])
    print(f"  Test rows: {len(test_rows)}  |  With correction: {covered}")

    # 4. Threshold sweep
    thresholds = [round(t, 3) for t in np.arange(0.05, 0.96, 0.05)]

    sweep_configs = [
        ("True WER",        "pred_wer_ridge", True,  False),
        ("Ridge WER Prediction",        "pred_wer_ridge", False, False),
        ("NN WER Prediction",           "pred_wer_nn",    False, False),
    ]

    all_sweeps = []
    for label, pred_key, oracle, _ in sweep_configs:
        if not oracle and not any(pred_key in r for r in routing_table[:1]):
            print(f"  ⚠  Skipping {label}: {pred_key} not in routing table")
            continue
        sweep = sweep_thresholds(routing_table, pred_key, thresholds, label, use_oracle=oracle)
        all_sweeps.append((label, sweep))
        all_sweeps[-1][1]  # keep

    # Print comparison table for T=0.3 and T=0.5
    print(f"\n  {'Label':30s}  {'T':4s}  {'Routed%':8s}  {'WER↓':8s}  {'Savings':8s}  {'Recall':7s}")
    print("  " + "-" * 78)
    for label, sweep in all_sweeps:
        for row in sweep:
            if row["threshold"] in (0.3, 0.5, 0.7):
                print(f"  {label:30s}  {row['threshold']:.2f}  "
                      f"{row['frac_routed']:8.3f}  {row['wer_reduction']:8.4f}  "
                      f"{row['savings']:8.3f}  {row['recall_bad']:7.3f}")

    # Find optimal threshold for each model (max WER reduction at >50% savings)
    print("\n  Optimal thresholds (max WER reduction with savings≥0.50):")
    for label, sweep in all_sweeps:
        candidates = [r for r in sweep if r["savings"] >= 0.50]
        if candidates:
            best = max(candidates, key=lambda r: r["wer_reduction"])
            print(f"    {label:30s}  T={best['threshold']:.2f}  "
                  f"ΔWER={best['wer_reduction']:.4f}  savings={best['savings']:.3f}")

    # Save sweep results
    sweep_path = RESULTS / "routing/routing_threshold_sweep.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump([
            {"label": label, "rows": sweep}
            for label, sweep in all_sweeps
        ], f, indent=2, ensure_ascii=False)
    print(f"  ✓ Sweep saved → {sweep_path}")

    # 5. Plot
    colors  = [CORAL, TEAL, VIOLET, AMBER]
    styles  = ["--",  "-",   "-.",   ":"]
    groups  = [(label, sweep, colors[i], styles[i])
               for i, (label, sweep) in enumerate(all_sweeps)]
    out_png = RESULTS / "figures/regression/regression_routing.png"
    plot_sweep(groups, out_png)
    shutil.copy(out_png, FIGS / "regression_routing.png")
    print(f"  ✓ Copied to paper/figures/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine",       default="tesseract")
    parser.add_argument("--model-substr", default="gemini-3-flash")
    args = parser.parse_args()
    main(args.engine, args.model_substr)
