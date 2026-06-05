import pandas as pd
import json

# Load the summary data
with open("../../results/summary.json", "r") as f:
    data = json.load(f)

# Extract baseline
baseline_wer = next(x for x in data if x["strategy"] == "baseline_no_llm")["overall_average_wer"]

# Convert to DataFrame
df = pd.DataFrame(data)
df = df[df["strategy"] != "baseline_no_llm"]

# Calculate improvement
df["improvement_pct"] = ((baseline_wer - df["overall_average_wer"]) / baseline_wer) * 100

# Group by Strategy Type
def get_strategy_type(s):
    thr = "80" if "thr" not in s else s.split("_thr")[1][:2]
    base = "Other"
    if "Full" in s: base = "Full"
    elif "SelectiveNoContext" in s: base = "Selective (No Context)"
    elif "Selective" in s: base = "Selective (Context)"
    
    if base == "Full": return base
    return f"{base} @ {thr}%"

df["strategy_type"] = df["strategy"].apply(get_strategy_type)

# Sort and display clean version
cols = ["strategy", "llm_model", "overall_average_wer", "improvement_pct", "cost", "strategy_type"]
top_performers = df.sort_values("overall_average_wer").head(30)

print(f"BASELINE WER: {baseline_wer:.4f}")
print("\nTOP RESULTS (Overall Average WER):")
print(top_performers[cols].to_string(index=False))

# Show best per strategy type
print("\nBEST PER STRATEGY TYPE:")
best_per_type = df.sort_values("overall_average_wer").groupby("strategy_type").head(5)
print(best_per_type[cols].to_string(index=False))
