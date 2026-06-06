"""Complete corrections for models with partial data."""
import sys, os, json, logging
from pathlib import Path

sys.path.insert(0, "code")
from dotenv import load_dotenv
load_dotenv()

from evaluation.run_evaluations import run_single_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

eval_dir = Path("data/evaluation_dataset")
data_dir = Path("data")
results_dir = Path("results")

with open(eval_dir / "groundtruth.json", "r") as f:
    groundtruth_data = json.load(f)
print(f"Loaded {len(groundtruth_data)} groundtruth items")

with open(data_dir / "sample_prompts.json", "r") as f:
    cfg = json.load(f)
    prompts = cfg["prompts"]
    templates = cfg.get("templates", {})

# Find prompt 8 (Expert Robuste)
prompt_8 = None
for p in prompts:
    if p["id"] == 8:
        prompt_8 = p
        break
if not prompt_8:
    print("ERROR: Prompt 8 not found!")
    print("Available:", [p["id"] for p in prompts])
    sys.exit(1)
print(f"Using prompt ID={prompt_8['id']} level={prompt_8.get('level','')}")

# OCR cache
with open(data_dir / "raw_ocr_results.json", "r") as f:
    ocr_cache = json.load(f)
print(f"OCR cache engines: {list(ocr_cache.keys())}")

models_to_complete = [
    "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-4o",
    "qwen/qwen-2.5-72b-instruct",
    "mistralai/mistral-small-3.1-24b-instruct",
]

for llm_model in models_to_complete:
    print(f"\n=== Running {llm_model} ===")
    result = run_single_experiment(
        ocr_name="tesseract",
        prompt=prompt_8,
        llm_model=llm_model,
        groundtruth_data=groundtruth_data,
        ocr_cache=ocr_cache,
        results_dir=results_dir,
        strategy="Full_Expert_Robuste",
        dry_run=False,
        is_selective=False,
        templates=templates,
    )
    cached = result.get("cached", False)
    new = result.get("new_items", 0)
    print(f"  Done: WER={result['average_wer']:.4f} CER={result['average_cer']:.4f} items={result['num_items']} cached={cached} new={new}")

print("\nAll models completed!")
