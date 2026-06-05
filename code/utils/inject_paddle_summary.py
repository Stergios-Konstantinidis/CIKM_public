"""
inject_paddle_summary.py
========================
Reads all paddle_*.json result files and injects the paddle engine stats
into results/summary.json and results/leaderboard.json.

Also adds the paddle baseline to the baseline_no_llm entry.

Run from the repo root:
    python code/inject_paddle_summary.py
"""

import json
import re
import numpy as np
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results"

# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  ✓  Saved → {path.name}")


# ── 1.  Build a lookup: (strategy, llm_model) → {wer, cer, n} from paddle files
# ─────────────────────────────────────────────────────────────────────────────

# Filename pattern:  paddle_{strategy}_{prompt_num}_{llm_model}.json
# e.g. paddle_Full_Advanced_5_google__gemini-3-flash-preview.json
#      paddle_Selective_thr80_Basic_1_google__gemini-3-flash-preview.json

paddle_stats = {}   # key: (strategy_canonical, llm_model_canonical) → dict

for pf in sorted((RESULTS_DIR / "corrections" / "paddle").glob("paddle_*.json")):
    stem = pf.stem   # e.g. paddle_Full_Advanced_5_google__gemini-3-flash-preview
    # Strip leading "paddle_"
    rest = stem[len("paddle_"):]

    # The LLM model part starts at the first "google__" / "openai__" / "meta-llama__" etc.
    # Strategy + prompt_id come before that.
    # We split on the last occurrence of a digit followed by underscore then a vendor string.
    # A safe heuristic: split at _<digits>_ where the next token looks like a vendor prefix.
    vendor_prefixes = ("google__", "openai__", "meta-llama__", "mistralai__",
                       "qwen__", "anthropic__")

    split_idx = None
    for vp in vendor_prefixes:
        idx = rest.find("_" + vp)
        if idx != -1:
            split_idx = idx
            break

    if split_idx is None:
        print(f"  WARN: Cannot parse vendor from {pf.name}, skipping.")
        continue

    strategy_part = rest[:split_idx]   # e.g. "Full_Advanced_5"
    llm_raw       = rest[split_idx+1:] # e.g. "google__gemini-3-flash-preview"

    # Convert llm_raw back to OpenRouter-style: google__gemini-3-flash-preview → google/gemini-3-flash-preview
    llm_model = llm_raw.replace("__", "/", 1)   # only first __ is the slash

    # Strip trailing prompt number from strategy to get canonical strategy name
    # e.g. "Full_Advanced_5" → strip "_5" → "Full_Advanced"
    strategy_no_num = re.sub(r"_\d+$", "", strategy_part)

    data = load_json(pf)
    if not data:
        continue

    wers = [r["wer"] for r in data if "wer" in r]
    cers = [r["cer"] for r in data if "cer" in r]
    if not wers:
        continue

    key = (strategy_no_num, llm_model)
    paddle_stats[key] = {
        "wer": float(np.mean(wers)),
        "cer": float(np.mean(cers)),
        "num_items": len(wers),
    }
    print(f"  Parsed  {pf.name}  →  strategy={strategy_no_num!r}  llm={llm_model!r}"
          f"  WER={np.mean(wers):.4f}  n={len(wers)}")

print(f"\nTotal paddle (strategy, llm) pairs parsed: {len(paddle_stats)}\n")


# ── 2.  Inject into summary.json ─────────────────────────────────────────────

summary = load_json(RESULTS_DIR / "summaries/summary.json")

updated_count = 0
for entry in summary:
    strategy  = entry.get("strategy", "")
    llm_model = entry.get("llm_model", "")

    if strategy == "baseline_no_llm":
        continue  # handled separately below

    key = (strategy, llm_model)
    if key in paddle_stats:
        entry.setdefault("by_ocr_engine", {})["paddle"] = paddle_stats[key]

        # Recalculate overall_average_wer / cer across all present engines
        engines = entry["by_ocr_engine"]
        all_wers = [v["wer"] for v in engines.values() if "wer" in v]
        all_cers = [v["cer"] for v in engines.values() if "cer" in v]
        entry["overall_average_wer"] = float(np.mean(all_wers))
        entry["overall_average_cer"] = float(np.mean(all_cers))
        updated_count += 1
    else:
        print(f"  WARN: No paddle file found for ({strategy!r}, {llm_model!r})")

# ── Baseline entry: add paddle stats from baseline_paddle.json ────────────────
bp = load_json(RESULTS_DIR / "baselines/baseline_paddle.json")
bp_wers = [r["wer"] for r in bp]
bp_cers = [r["cer"] for r in bp]
for entry in summary:
    if entry.get("strategy") == "baseline_no_llm":
        entry.setdefault("by_ocr_engine", {})["paddle"] = {
            "wer": float(np.mean(bp_wers)),
            "cer": float(np.mean(bp_cers)),
            "num_items": len(bp_wers),
        }
        engines = entry["by_ocr_engine"]
        all_wers = [v["wer"] for v in engines.values()]
        all_cers = [v["cer"] for v in engines.values()]
        entry["overall_average_wer"] = float(np.mean(all_wers))
        entry["overall_average_cer"] = float(np.mean(all_cers))
        print(f"  Baseline paddle: WER={np.mean(bp_wers):.4f}  CER={np.mean(bp_cers):.4f}")

print(f"  Updated {updated_count} summary entries with paddle stats.")
save_json(RESULTS_DIR / "summaries/summary.json", summary)


# ── 3.  Rebuild leaderboard.json ──────────────────────────────────────────────

leaderboard = []
rank = 1

# Sort by overall_average_wer, skip baseline
non_baseline = [e for e in summary if e.get("strategy") != "baseline_no_llm"]
non_baseline_sorted = sorted(non_baseline, key=lambda x: x.get("overall_average_wer", 1.0))

for entry in non_baseline_sorted:
    by_eng = entry.get("by_ocr_engine", {})
    row = {
        "rank":          rank,
        "strategy":      entry.get("strategy", ""),
        "llm_model":     entry.get("llm_model", ""),
        "overall_wer":   entry.get("overall_average_wer", 0.0),
        "overall_cer":   entry.get("overall_average_cer", 0.0),
        "total_cost":    entry.get("cost", 0.0),
    }
    # Per-engine WER / CER columns
    for eng in ("tesseract", "easyocr", "paddle"):
        eng_data = by_eng.get(eng, {})
        row[f"wer_{eng}"] = eng_data.get("wer", None)
        row[f"cer_{eng}"] = eng_data.get("cer", None)

    leaderboard.append(row)
    rank += 1

save_json(RESULTS_DIR / "summaries/leaderboard.json", leaderboard)
print(f"  Leaderboard rebuilt: {len(leaderboard)} entries.")


# ── 4.  Print top 10 ──────────────────────────────────────────────────────────
print("\n=== TOP 10 (by overall WER, 3 engines) ===")
print(f"{'RK':>3}  {'STRATEGY':<30}  {'LLM':<30}  {'WER_all':>7}  "
      f"{'WER_tess':>8}  {'WER_easy':>8}  {'WER_paddle':>10}")
for row in leaderboard[:10]:
    def _f(v): return f"{v:.4f}" if v is not None else "  N/A  "
    print(f"{row['rank']:>3}  {row['strategy']:<30}  {row['llm_model']:<30}  "
          f"{_f(row['overall_wer']):>7}  {_f(row['wer_tesseract']):>8}  "
          f"{_f(row['wer_easyocr']):>8}  {_f(row['wer_paddle']):>10}")

# Baseline
baseline = next((e for e in summary if e.get("strategy") == "baseline_no_llm"), None)
if baseline:
    by_eng = baseline.get("by_ocr_engine", {})
    print("\n--- Baseline (no LLM) ---")
    for eng in ("tesseract", "easyocr", "paddle"):
        d = by_eng.get(eng, {})
        print(f"  {eng:12s}  WER={d.get('wer', 0):.4f}  CER={d.get('cer', 0):.4f}")

print("\nDone ✓")
