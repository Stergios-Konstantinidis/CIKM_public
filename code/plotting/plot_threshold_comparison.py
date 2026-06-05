"""
plot_threshold_comparison.py
────────────────────────────
Generates the threshold-vs-quality figure for the paper (light mode).

X-axis : routing threshold T
Y-axis : average WER (left panel) and average CER (right panel) on the
         Tesseract test set AFTER routing:
           - documents with signal >= T  → use LLM-corrected version
           - documents with signal <  T  → keep raw OCR

Three routing signal families are shown:
  1. Oracle        – threshold applied to TRUE WER (upper bound)
  2. Ridge pred    – threshold applied to Ridge-predicted WER
  3. MLP pred      – threshold applied to MLP-predicted WER
  4. Confidence    – threshold applied to (1 − avg_conf), so high (1−conf)
                     means low confidence → correction triggered.
                     Only the two available discrete values are shown as
                     scatter markers with dashed connector.

Horizontal baselines:
  • "Raw OCR"         – no correction at all
  • "Correct All"     – all documents corrected (full LLM pass)

Saves to results/ and copies to paper/figures/.
"""

from __future__ import annotations
import json, glob, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import shutil

ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Load the offline routing sweep (WER + CER per threshold per model)
# ──────────────────────────────────────────────────────────────────────────────
with open(RESULTS / "routing/routing_threshold_sweep.json") as f:
    sweep = json.load(f)

# Index by label
sweep_by_label: dict[str, list[dict]] = {}
for s in sweep:
    sweep_by_label[s["label"]] = sorted(s["rows"], key=lambda r: r["threshold"])

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Load per-document predictions + routing table for CER sweep
#     (routing_threshold_sweep only stores WER; we compute CER offline here)
# ──────────────────────────────────────────────────────────────────────────────
with open(RESULTS / "routing/routing_table.json") as f:
    routing_table = json.load(f)

# Tesseract test rows with corrections
tess_test = [
    r for r in routing_table
    if r.get("split") == "test"
    and r.get("engine") == "tesseract"
    and r.get("has_correction", False)
]

# Load predictions (clipped to [0,1])
with open(RESULTS / "ml_models/predictions_ridge_noemb.json") as f:
    ridge_preds = {(r["filename"], r["engine"]): r for r in json.load(f)}
with open(RESULTS / "ml_models/predictions_nn_noemb.json") as f:
    nn_preds    = {(r["filename"], r["engine"]): r for r in json.load(f)}

# Attach pred_wer from ridge/nn to each test row
for row in tess_test:
    key = (row["filename"], row["engine"])
    row["pred_wer_ridge"] = max(0.0, min(1.0, ridge_preds[key]["pred_wer"])) if key in ridge_preds else row["actual_wer"]
    row["pred_wer_nn"]    = max(0.0, min(1.0, nn_preds[key]["pred_wer"]))    if key in nn_preds    else row["actual_wer"]

actual_wer    = np.array([r["actual_wer"]    for r in tess_test])
corrected_wer = np.array([r["corrected_wer"] for r in tess_test])
actual_cer    = np.array([r["actual_cer"]    for r in tess_test])
corrected_cer = np.array([r.get("corrected_cer", r["actual_cer"]) for r in tess_test])

pred_ridge = np.array([r["pred_wer_ridge"] for r in tess_test])
pred_nn    = np.array([r["pred_wer_nn"]    for r in tess_test])

baseline_wer    = float(actual_wer.mean())
baseline_cer    = float(actual_cer.mean())
correct_all_wer = float(corrected_wer.mean())
correct_all_cer = float(corrected_cer.mean())

def sweep_metric(signal: np.ndarray, actual: np.ndarray,
                 corrected: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Average metric after routing: route if signal >= T, else keep raw."""
    results = []
    for T in thresholds:
        mask = signal >= T
        after = np.where(mask, corrected, actual)
        results.append(float(after.mean()))
    return np.array(results)

thresholds = np.arange(0.0, 1.01, 0.02)

oracle_wer = sweep_metric(actual_wer,  actual_wer, corrected_wer, thresholds)
ridge_wer  = sweep_metric(pred_ridge,  actual_wer, corrected_wer, thresholds)
nn_wer     = sweep_metric(pred_nn,     actual_wer, corrected_wer, thresholds)

oracle_cer = sweep_metric(actual_wer,  actual_cer, corrected_cer, thresholds)
ridge_cer  = sweep_metric(pred_ridge,  actual_cer, corrected_cer, thresholds)
nn_cer     = sweep_metric(pred_nn,     actual_cer, corrected_cer, thresholds)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Confidence-based routing — continuous sweep
#     Route if avg_confidence < T (low-confidence docs get corrected)
# ──────────────────────────────────────────────────────────────────────────────
with open(RESULTS / "confidence_data/low_confidence_words_90_tesseract.json") as f:
    conf_raw = json.load(f)  # dict: filename -> {avg_confidence: float, ...}

# Attach avg_confidence to each test row
conf_vals = []
for row in tess_test:
    entry = conf_raw.get(row["filename"])
    conf_vals.append(entry["avg_confidence"] if isinstance(entry, dict) else None)

actual_wer_arr    = actual_wer
corrected_wer_arr = corrected_wer
actual_cer_arr    = actual_cer
corrected_cer_arr = corrected_cer

def conf_sweep_metric(conf_list, actual, corrected, thresholds):
    """Average metric after confidence routing: route if (1 - avg_conf) >= T.

    Signal = 1 - avg_confidence, so the threshold axis matches WER-based routing.
    """
    results = []
    for T in thresholds:
        after = [
            corrected[i] if (conf_list[i] is not None and (1.0 - conf_list[i]) >= T) else actual[i]
            for i in range(len(conf_list))
        ]
        results.append(float(np.mean(after)))
    return np.array(results)

conf_wer = conf_sweep_metric(conf_vals, actual_wer, corrected_wer, thresholds)
conf_cer = conf_sweep_metric(conf_vals, actual_cer, corrected_cer, thresholds)

print(f"Baseline WER={baseline_wer:.4f}  CER={baseline_cer:.4f}")
print(f"Correct-all WER={correct_all_wer:.4f}  CER={correct_all_cer:.4f}")
print(f"Ground-truth T=0.20 WER={oracle_wer[10]:.4f}")
print(f"Ridge  T=0.20 WER={ridge_wer[10]:.4f}")
print(f"Conf  T=0.90 WER={conf_wer[45]:.4f}  (idx for T=0.90)")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Light-mode palette
# ──────────────────────────────────────────────────────────────────────────────
BG       = "white"
PANEL_BG = "#f5f5f5"
GRID_CLR = "#d0d0d0"
TEXT_CLR = "black"

C_ORACLE = "#e6a817"   # amber
C_RIDGE  = "#1565c0"   # navy blue
C_NN     = "#b71c1c"   # dark red
C_CONF   = "#7b2d8b"   # purple  – confidence routing
C_BASE   = "#444444"   # mid grey

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   "#aaaaaa",
    "axes.labelcolor":  TEXT_CLR,
    "xtick.color":      TEXT_CLR,
    "ytick.color":      TEXT_CLR,
    "text.color":       TEXT_CLR,
    "grid.color":       GRID_CLR,
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "font.family":      "DejaVu Sans",
    "font.size":        9,
})

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Draw
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)

for ax, metric, oracle_m, ridge_m, nn_m, conf_m, ylabel in [
    (axes[0], "WER", oracle_wer, ridge_wer, nn_wer, conf_wer, "Average WER after Routing"),
    (axes[1], "CER", oracle_cer, ridge_cer, nn_cer, conf_cer, "Average CER after Routing"),
]:
    # Continuous curves
    ax.plot(thresholds, oracle_m, color=C_ORACLE, lw=2.2, label=f"True {metric}", zorder=3)
    ax.plot(thresholds, ridge_m,  color=C_RIDGE,  lw=2.0, label="Ridge pred-WER",    zorder=3)
    ax.plot(thresholds, nn_m,     color=C_NN,     lw=2.0, label="MLP pred-WER",      zorder=3, linestyle="--")
    ax.plot(thresholds, conf_m,   color=C_CONF,   lw=2.0, label="Confidence routing", zorder=3, linestyle="-.")

    # Baselines
    ax.axhline(baseline_wer if metric == "WER" else baseline_cer,
               color=C_BASE, lw=1.3, linestyle=":", label="Raw OCR (no correction)", alpha=0.9)
    ax.axhline(correct_all_wer if metric == "WER" else correct_all_cer,
               color="#00695c", lw=1.3, linestyle="--", label="Correct All (100%)", alpha=0.9)

    ax.set_xlabel("Routing Threshold T  (route if signal \u2265 T)", color="black", fontsize=9)
    ax.set_ylabel(ylabel, color="black", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.6, linewidth=0.6)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.95, edgecolor="#cccccc")

plt.tight_layout()
out_path = RESULTS / "figures/threshold_comparison.png"
fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"\u2713 Saved \u2192 {out_path}")

FIGURES.mkdir(parents=True, exist_ok=True)
dest = FIGURES / "figures/threshold_comparison.png"
shutil.copy(out_path, dest)
print(f"\u2713 Copied \u2192 {dest}")
