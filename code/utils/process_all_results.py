import pandas as pd
import json
from pathlib import Path

def load_json(p):
    if not p.exists(): return []
    try:
        with open(p, "r") as f:
            return json.load(f)
    except:
        return []

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    results_dir = base_dir / "results"
    
    # 1. Load Standard results
    std_data = load_json(results_dir / "summaries/summary.json")
    # Normalize standard results
    std_rows = []
    baseline_wer = 0.5439 # fallback
    for entry in std_data:
        if entry["strategy"] == "baseline_no_llm":
            baseline_wer = entry["overall_average_wer"]
            continue
        std_rows.append({
            "strategy": entry["strategy"],
            "llm_model": entry["llm_model"],
            "wer": entry["overall_average_wer"],
            "cer": entry["overall_average_cer"],
            "cost": entry.get("cost", 0.0)
        })
    
    # 2. Load Conditional results
    cond_data = load_json(results_dir / "summaries/summary_conditional.json")
    cond_rows = []
    for entry in cond_data:
        cond_rows.append({
            "strategy": entry["strategy"],
            "llm_model": entry["llm_model"],
            "wer": entry["average_wer"],
            "cer": entry["average_cer"],
            "cost": entry.get("cost", 0.0)
        })
        
    # 3. Load Ortho results
    ortho_data = load_json(results_dir / "summaries/ortho_summary.json")
    ortho_rows = []
    for entry in ortho_data:
        ortho_rows.append({
            "strategy": entry["strategy"],
            "llm_model": entry["llm_model"],
            "wer": entry["average_wer"],
            "cer": entry["average_cer"],
            "cost": entry.get("cost", 0.0)
        })
        
    # Combine
    all_rows = std_rows + cond_rows + ortho_rows
    df = pd.DataFrame(all_rows)
    
    if df.empty:
        print("No results found.")
        return

    # Improvement
    df["improvement_pct"] = ((baseline_wer - df["wer"]) / baseline_wer) * 100
    
    # Strategy categorization
    def categorize(s):
        if "ConditionalFull" in s: return "Hybrid (Full if Low Conf)"
        if "Orthographic" in s: return "Orthographic (Spellcheck)"
        if "SelectiveNoContext" in s: return "Selective (No Context)"
        if "Selective" in s: return "Selective (Context)"
        if "Full" in s: return "Full LLM"
        return "Other"
    
    df["category"] = df["strategy"].apply(categorize)
    
    # Sort by WER
    df = df.sort_values("wer")
    
    cols = ["strategy", "llm_model", "wer", "improvement_pct", "cost", "category"]
    print(f"BASELINE WER: {baseline_wer:.4f}\n")
    print("Combined Leaderboard (Top 30):")
    print(df[cols].head(30).to_string(index=False))
    
    print("\nBest per category:")
    best = df.sort_values("wer").groupby("category").head(3)
    print(best[cols].to_string(index=False))

if __name__ == "__main__":
    main()
