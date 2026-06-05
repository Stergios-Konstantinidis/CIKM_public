import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Fix paths to allow importing from experiments
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections
)

def main():
    print("Loading baseline data...")
    try:
        records = load_tesseract_records()
    except FileNotFoundError:
        print("Baseline data not found. Please ensure 'results/baselines/baseline_tesseract.json' exists.")
        return

    # We want to plot oracle curves for different prompting strategies.
    # The prompt levels typically found in the files:
    # Basic, Advanced, Expert_Robuste, Ultimate_Master
    # Let's search the corrections directory for a specific model (e.g., gemini-3-flash-preview)
    # and strategy (e.g., Full) to isolate the prompting strategy differences.
    
    corrections_dir = Path("results/corrections/tesseract")
    if not corrections_dir.exists():
        print(f"Corrections directory not found: {corrections_dir}")
        print("Please ensure the LLM correction JSON files are present.")
        return

    # Find relevant correction files
    # E.g. tesseract_Full_Basic_1_google__gemini-3-flash-preview.json
    # We will group them by the PromptLevel part of the filename.
    files = list(corrections_dir.glob("tesseract_Full_*_google__gemini-3-flash-preview.json"))
    if not files:
        # Fallback to any model if gemini is not found
        files = list(corrections_dir.glob("tesseract_Full_*.json"))
        
    if not files:
        print("No correction files found for the 'Full' strategy.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    display_metrics = ["wer", "cer"]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'cyan', 'magenta']

    # For each metric, plot the oracle curves
    for m_idx, metric in enumerate(display_metrics):
        ax = ax1 if metric == "wer" else ax2
        
        base_vals = np.array([float(r[metric]) for r in records])
        avg_base = np.mean(base_vals)
        ax.axhline(avg_base, color='black', linestyle='--', linewidth=1.5, label=f'Baseline ({avg_base:.4f})')
        N = len(records)

        # Plot each prompting strategy
        for f_idx, file_path in enumerate(sorted(files)):
            # Extract prompting strategy name from filename
            # filename format: tesseract_Full_{PromptLevel}_{PromptID}_{LLMModel}.json
            parts = file_path.stem.split("_")
            prompt_strategy = parts[2]
            # Sometimes it's Expert_Robuste so we can just join parts[2] and [3] if [3] is not a digit
            if len(parts) > 3 and not parts[3].isdigit():
                prompt_strategy = f"{parts[2]}_{parts[3]}"
                
            print(f"Processing {prompt_strategy} for {metric.upper()}...")
            
            corrections = load_llm_corrections(f"corrections/tesseract/{file_path.name}")
            
            corr_vals = np.array([corrections.get(r["filename"], {}).get(metric, float(r[metric])) for r in records])
            deltas = base_vals - corr_vals
            
            # Oracle Frontier
            sort_idx = np.argsort(deltas)[::-1]
            oracle_pct, oracle_val = [0.0], [avg_base]
            current_sum = np.sum(base_vals)
            
            for i in range(N):
                current_sum -= deltas[sort_idx[i]]
                oracle_pct.append((i + 1) / N * 100)
                oracle_val.append(current_sum / N)
                
            color = colors[f_idx % len(colors)]
            ax.plot(oracle_pct, oracle_val, label=f'Oracle ({prompt_strategy})', color=color, linewidth=2)

        ax.set_xlabel('% of Documents Corrected')
        ax.set_ylabel(f'{metric.upper()} per Token')
        ax.set_title(f'Oracle Curves for Prompting Strategies ({metric.upper()})')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Save figure
    out_dir = Path("results/figures/validationprompts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "oracle_prompting_strategies.png"
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()
