import json
from pathlib import Path
import numpy as np

def main():
    results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    summary = []
    
    # Files pattern: corrections/{engine}/{engine}_ConditionalFull_thr{threshold}_{prompt_id}_{model}.json
    files = list((results_dir / "corrections").glob("*/*_ConditionalFull_*.json"))
    
    for f in files:
        with open(f, "r") as src:
            data = json.load(src)
            # data is a list of results for each document
            wers = [r["wer"] for r in data if "wer" in r]
            cers = [r["cer"] for r in data if "cer" in r]
            costs = [r.get("cost", 0) for r in data if "cost" in r]
            
            # Extract metadata from filename
            # engine_ConditionalFull_thr80_5_google__gemini-3-flash-preview.json
            parts = f.stem.split("_")
            engine = parts[0]
            thr = parts[3] # thr80
            prompt_id = parts[4]
            model = "_".join(parts[5:]).replace("__", "/")
            
            summary.append({
                "ocr_engine": engine,
                "strategy": f"ConditionalFull_{thr}_{prompt_id}",
                "llm_model": model,
                "average_wer": np.mean(wers) if wers else 0,
                "average_cer": np.mean(cers) if cers else 0,
                "cost": np.sum(costs) if costs else 0
            })
            
    with open(results_dir / "summaries/summary_conditional.json", "w") as out:
        json.dump(summary, out, indent=2)
    print(f"Generated summary_conditional.json with {len(summary)} entries.")

if __name__ == "__main__":
    main()
