"""Complete corrections for models with batch size 1 to avoid index-shifting bug, parallelized across models."""
import sys, os, json, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
    sys.exit(1)
print(f"Using prompt ID={prompt_8['id']} level={prompt_8.get('level','')}")

# OCR cache
with open(data_dir / "raw_ocr_results.json", "r") as f:
    ocr_cache = json.load(f)
print(f"OCR cache engines: {list(ocr_cache.keys())}")

models_to_complete = [
    "google/gemma-4-31b-it",
]

# Force batch size = 1
run_single_experiment._batch_size = 1
print("Set run_single_experiment._batch_size = 1")

def run_model(llm_model):
    target_filename = f"tesseract_Full_Expert_Robuste_8_{llm_model.replace('/', '__')}.json"
    target_path = results_dir / "corrections" / "tesseract" / target_filename
    
    if target_path.exists():
        print(f"Deleting existing cached file: {target_path}")
        target_path.unlink()
        
    print(f"Starting {llm_model}...")
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
    msg = f"Completed {llm_model}: WER={result['average_wer']:.4f} CER={result['average_cer']:.4f} items={result['num_items']} cached={cached} new={new}"
    print(msg)
    return msg

print("\nStarting parallel run for all 4 models...")
with ThreadPoolExecutor(max_workers=len(models_to_complete)) as executor:
    results = list(executor.map(run_model, models_to_complete))

print("\nAll tasks completed:")
for r in results:
    print(" ", r)
