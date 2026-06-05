import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

with open(_ROOT / "results" / "ml_models/gbt_classifier_results.json") as f:
    results = json.load(f)

print("=" * 100)
print(f"  {'Metric':<5}  {'Threshold':<20}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'AUC':>6}  {'%Routed':>8}  {'δ':>7}")
print("  " + "-" * 98)

for m in results:
    sim = m["routing_at_05"]
    metric = m["metric"]
    delta_val = sim[f"{metric}_reduction"]
    print(
        f"  {metric.upper():<5}  {m['threshold_label']:<20}  "
        f"{m['accuracy']:>6.3f}  {m['precision_at_05']:>6.3f}  {m['recall_at_05']:>6.3f}  "
        f"{m['f1_at_05']:>6.3f}  {m['auc_roc']:>6.3f}  "
        f"{sim['pct_routed']:>7.1f}%  {delta_val:>7.4f}"
    )
print("=" * 100)
