"""
run_evaluations_conditional.py
OCR Evaluation Pipeline — Conditional Full Correction Strategy

Key logic:
  - Corrects the full text ONLY if the image's average OCR confidence is below a threshold.
  - Otherwise, uses the raw OCR output (no LLM call).
  - This is a cost-optimisation strategy to target documents with likely higher error rates.
"""

import json
import os
import re
import sys
import time
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── third-party imports (with user-friendly errors) ────────────────────────
try:
    import jiwer
except ImportError:
    sys.exit("Please install jiwer: pip install jiwer")

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  
        def __init__(self, iterable=None, **kw):
            self._iter = iterable
        def __iter__(self):
            return iter(self._iter)
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            pass
        def set_postfix(self, **kw):
            pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── configuration ──────────────────────────────────────────────────────────
DEFAULT_LLM_MODELS = [
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
]

ENGINES_TO_EVAL = ["paddle", "easyocr", "tesseract"]
MAX_WORKERS = 3
BATCH_SIZE = 15
MAX_RETRIES = 4
RETRY_BASE_DELAY = 5   # seconds
RETRY_MAX_DELAY = 60   # seconds


# ── text normalisation ──────────────────────────────────────────────────────
def apply_annotator_rules(text: str) -> str:
    """Apply topological normalisation rules matching the human annotator spec."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.replace("E'", "É").replace("E`", "É")
    text = text.replace("&z", "&")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+([;.,!?:])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    import string
    text = text.strip(string.punctuation + " ")
    return text.strip()


# ── metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(groundtruth: str, hypothesis: str):
    """Return (WER, CER) after applying annotator normalisation."""
    try:
        gt_norm = apply_annotator_rules(groundtruth) or "[EMPTY]"
        hyp_norm = apply_annotator_rules(hypothesis) or "[EMPTY]"
        wer = jiwer.wer(gt_norm, hyp_norm)
        cer = jiwer.cer(gt_norm, hyp_norm)
        return wer, cer
    except Exception as exc:
        log.warning("compute_metrics error: %s", exc)
        return 1.0, 1.0


# ── OCR helpers ──────────────────────────────────────────────────────────────
def setup_ocr_engines() -> dict:
    """Load all available OCR engines (each initialised once)."""
    engines = {}

    try:
        import pytesseract
        engines["tesseract"] = lambda img: pytesseract.image_to_string(img, lang="fra")
    except Exception as exc:
        log.warning("Tesseract not available: %s", exc)

    try:
        import easyocr
        reader = easyocr.Reader(["fr"], gpu=False)
        engines["easyocr"] = lambda img: "\n".join(reader.readtext(img, detail=0))
    except Exception as exc:
        log.warning("EasyOCR not available: %s", exc)

    try:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ocr = PaddleOCR(use_textline_orientation=True, lang="fr")
            except TypeError:
                ocr = PaddleOCR(use_angle_cls=True, lang="fr")

        def run_paddle(img):
            res = ocr.ocr(img, cls=True)
            if not res or not res[0]: return ""
            return "\n".join(line[1][0] for line in res[0])
        engines["paddle"] = run_paddle
    except Exception as exc:
        log.warning("PaddleOCR not available: %s", exc)

    return engines


def load_or_run_ocr(eval_dir: Path, ocr_engines: dict, groundtruth_data: list, cache_path: Path) -> dict:
    ocr_cache: dict = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            ocr_cache = json.load(f)

    changes_made = False
    for engine_name, engine_func in ocr_engines.items():
        if engine_name not in ocr_cache: ocr_cache[engine_name] = {}
        missing = [item["filename"] for item in groundtruth_data if item["filename"] not in ocr_cache[engine_name]]
        if not missing: continue

        for fname in tqdm(missing, desc=f"OCR/{engine_name}", unit="img"):
            img_path = eval_dir / "images" / fname
            if not img_path.exists(): continue
            try:
                ocr_cache[engine_name][fname] = engine_func(str(img_path))
                changes_made = True
            except Exception as exc:
                ocr_cache[engine_name][fname] = ""
                changes_made = True

    if changes_made:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(ocr_cache, f, ensure_ascii=False, indent=2)
    return ocr_cache


def extract_ocr_confidence_data(img_path, engine="paddle", threshold=0.8, cached_reader=None) -> dict:
    results = []
    lines, confs = [], []
    if engine == "paddle":
        try:
            res = cached_reader.ocr(str(img_path), cls=True)
            if res and res[0]:
                lines = [l[1][0] for l in res[0]]
                confs = [float(l[1][1]) / 100.0 if float(l[1][1]) > 1.0 else float(l[1][1]) for l in res[0]]
        except: pass
    elif engine == "easyocr":
        try:
            res = cached_reader.readtext(str(img_path))
            lines = [r[1] for r in res]
            confs = [float(r[2]) for r in res]
        except: pass
    elif engine == "tesseract":
        try:
            import pytesseract
            import pandas as pd
            from io import StringIO
            data = pytesseract.image_to_data(str(img_path), lang="fra")
            df = pd.read_csv(StringIO(data), sep="\t", quoting=3)
            df = df[df["conf"] != -1]
            line_groups = df.groupby(["block_num", "par_num", "line_num"])
            for _, group in line_groups:
                text = " ".join([str(x) for x in group["text"].tolist() if str(x).strip()])
                if not text: continue
                avg_conf = group["conf"].mean() / 100.0
                lines.append(text); confs.append(avg_conf)
        except: pass

    avg_image_conf = float(np.mean(confs)) if confs else 1.0
    for i, (text, conf) in enumerate(zip(lines, confs)):
        if conf < threshold:
            results.append({"index": i, "text": text, "confidence": conf, "prev_context": lines[max(0, i-3):i], "next_context": lines[i+1:min(len(lines), i+4)]})
    return {"avg_confidence": avg_image_conf, "low_confidence_lines": results}


# ── LLM interface ────────────────────────────────────────────────────────────
def get_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("openrouter_api_key")
    if not api_key: raise EnvironmentError("OPENROUTER_API_KEY not set.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=120)

def _extract_corrected_text(raw_response: str) -> str:
    if not raw_response: return ""
    text = raw_response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "corrected_text" in obj: return obj["corrected_text"]
    except: pass
    json_match = re.search(r'\{[^{}]*"corrected_text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', text, re.DOTALL)
    if json_match:
        try: return json_match.group(1).encode("raw_unicode_escape").decode("unicode_escape")
        except: return json_match.group(1).replace("\\n", "\n").replace('\\"', '"')
    return raw_response

def invoke_llm_batch(base_prompt_text: str, ocr_texts: list[str], full_text_template: str, llm_model: str, dry_run: bool = False) -> tuple[list[str], int, int]:
    if dry_run: return [""] * len(ocr_texts), 0, 0
    numbered = "\n\n".join(f"[{i}]\n{text}" for i, text in enumerate(ocr_texts))
    batch_suffix = "\n\n### FORMAT DE RÉPONSE OBLIGATOIRE ###\n" + f"Tu reçois {len(ocr_texts)} textes OCR numérotés de [0] à [{len(ocr_texts)-1}].\n" + "Renvoie UNIQUEMENT un objet JSON valide. Chaque clé est l'index, chaque valeur est le texte corrigé.\n" + json.dumps({str(i): f"<texte {i} corrigé>" for i in range(len(ocr_texts))}, ensure_ascii=False)
    
    base = full_text_template.replace("{base_prompt}", base_prompt_text)
    base = re.sub(r"### FORMAT DE RÉPONSE OBLIGATOIRE ###.*$", "", base, flags=re.DOTALL).rstrip()
    prompt = base.replace("{ocr_text}", numbered) + batch_suffix

    total_pt, total_ct = 0, 0
    corrections = [None] * len(ocr_texts)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = get_client()
            response = client.chat.completions.create(model=llm_model, messages=[{"role": "user", "content": prompt}])
            total_pt += response.usage.prompt_tokens; total_ct += response.usage.completion_tokens
            content = response.choices[0].message.content or ""
            corrections = _parse_batch_response(content, len(ocr_texts))
            break
        except Exception as exc:
            if attempt < MAX_RETRIES: time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            else: log.error("Batch LLM failed: %s", exc)

    missing = [i for i, v in enumerate(corrections) if v is None]
    if missing:
        for idx in missing:
            single_prompt = base.replace("{ocr_text}", ocr_texts[idx]) + '\n\n### FORMAT DE RÉPONSE OBLIGATOIRE ###\nRéponds UNIQUEMENT avec {"corrected_text": "<texte corrigé>"}'
            text, pt, ct = invoke_llm(single_prompt, llm_model, dry_run)
            total_pt += pt; total_ct += ct; corrections[idx] = text

    return [v if v is not None else "" for v in corrections], total_pt, total_ct

def _parse_batch_response(content, expected_count):
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        parsed = json.loads(m.group(0)) if m else None
    
    if isinstance(parsed, dict):
        return [parsed.get(str(i), parsed.get(i, None)) for i in range(expected_count)]
    return [None] * expected_count

def invoke_llm(full_prompt, llm_model, dry_run=False):
    if dry_run: return "", 0, 0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = get_client()
            response = client.chat.completions.create(model=llm_model, messages=[{"role": "user", "content": full_prompt}])
            return _extract_corrected_text(response.choices[0].message.content), response.usage.prompt_tokens, response.usage.completion_tokens
        except:
            if attempt < MAX_RETRIES: time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    return "", 0, 0


# ── experiment runner ────────────────────────────────────────────────────────
COST_RATES = {
    "google/gemini-3": (0.075, 0.30),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "meta-llama/llama-3.3-70b-instruct": (0.13, 0.40),
}
def estimate_cost(llm_model, pt, ct):
    for k, rates in COST_RATES.items():
        if k in llm_model: return (pt / 1e6 * rates[0]) + (ct / 1e6 * rates[1])
    return (pt / 1e6 * 0.5) + (ct / 1e6 * 1.5)

def get_result_filename(ocr_name, strategy, prompt_id, llm_model, thr):
    return f"{ocr_name}_ConditionalFull_thr{int(thr*100)}_{prompt_id}_{llm_model.replace('/', '__')}.json"

def run_single_experiment(*, ocr_name, prompt, llm_model, groundtruth_data, ocr_cache, results_dir, confidence_threshold, templates, dry_run=False) -> dict:
    prompt_id = prompt["id"]
    result_filename = get_result_filename(ocr_name, "ConditionalFull", prompt_id, llm_model, confidence_threshold)
    result_path = results_dir / "corrections" / ocr_name / result_filename
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        with open(result_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        wers = [x["wer"] for x in res if x["wer"] is not None]
        cers = [x["cer"] for x in res if x["cer"] is not None]
        return {"ocr_engine": ocr_name, "strategy": f"ConditionalFull_thr{int(confidence_threshold*100)}", "prompt_id": prompt_id, "llm_model": llm_model, "average_wer": float(np.mean(wers)) if wers else 1.0, "average_cer": float(np.mean(cers)) if cers else 1.0, "num_items": len(wers), "cached": True}

    # Load confidence data
    thr_val = int(confidence_threshold * 100)
    lcf = results_dir / f"confidence_data/low_confidence_words_{thr_val}_{ocr_name}.json"
    if not lcf.exists(): lcf = results_dir / f"confidence_data/low_confidence_words_{thr_val}.json"
    if not lcf.exists(): raise FileNotFoundError(f"Missing confidence data: {lcf}")
    with open(lcf, "r") as f: conf_data = json.load(f)

    experiment_results = []
    items_to_correct = []
    
    per_image_data = [(item["filename"], item["groundtruth_text"], ocr_cache[ocr_name].get(item["filename"], "")) for item in groundtruth_data if ocr_cache[ocr_name].get(item["filename"], "")]

    for fname, gt, raw in per_image_data:
        entry = conf_data.get(fname, {})
        avg_conf = entry.get("avg_confidence", 1.0) if isinstance(entry, dict) else 1.0
        
        item = {"filename": fname, "groundtruth": gt, "raw_ocr": raw, "corrected_ocr": raw, "wer": None, "cer": None}
        if avg_conf < confidence_threshold:
            items_to_correct.append(len(experiment_results))
        experiment_results.append(item)

    total_pt, total_ct = 0, 0
    full_tmpl = templates.get("full_text", "{base_prompt}\n\n{ocr_text}")
    
    for i in range(0, len(items_to_correct), BATCH_SIZE):
        batch_indices = items_to_correct[i : i + BATCH_SIZE]
        batch_texts = [experiment_results[idx]["raw_ocr"] for idx in batch_indices]
        corrs, pt, ct = invoke_llm_batch(prompt["prompt_text"], batch_texts, full_tmpl, llm_model, dry_run)
        total_pt += pt; total_ct += ct
        for local_idx, global_idx in enumerate(batch_indices):
            experiment_results[global_idx]["corrected_ocr"] = corrs[local_idx]

    cost = estimate_cost(llm_model, total_pt, total_ct)
    for item in experiment_results:
        w, c = compute_metrics(item["groundtruth"], item["corrected_ocr"])
        item["wer"], item["cer"] = w, c
        item["cost"] = cost / len(items_to_correct) if items_to_correct else 0

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, ensure_ascii=False, indent=2)

    wers = [x["wer"] for x in experiment_results]
    cers = [x["cer"] for x in experiment_results]
    return {"ocr_engine": ocr_name, "strategy": f"ConditionalFull_thr{int(confidence_threshold*100)}", "prompt_id": prompt_id, "llm_model": llm_model, "average_wer": np.mean(wers), "average_cer": np.mean(cers), "num_items": len(wers), "cost": cost, "cached": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    
    with open(data_dir / "evaluation_dataset/groundtruth.json", "r") as f:
        gt_data = json.load(f)
    if args.limit > 0: gt_data = gt_data[:args.limit]
    
    with open(data_dir / "sample_prompts.json", "r") as f:
        cfg = json.load(f); sample_prompts = cfg["prompts"]; templates = cfg.get("templates", {})

    ocr_cache = json.load(open(data_dir / "raw_ocr_results.json"))

    experiments = []
    for model in DEFAULT_LLM_MODELS:
        for prompt in sample_prompts:
            if prompt["id"] not in [5, 10]: continue # Only run on Advanced and Ultimate Master for speed
            for engine in ENGINES_TO_EVAL:
                if engine not in ocr_cache: continue
                for thr in [0.8, 0.9]:
                    experiments.append({"ocr_name": engine, "prompt": prompt, "llm_model": model, "groundtruth_data": gt_data, "ocr_cache": ocr_cache, "results_dir": results_dir, "confidence_threshold": thr, "templates": templates, "dry_run": args.dry_run})

    summaries = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exec:
        futures = {exec.submit(run_single_experiment, **exp): exp for exp in experiments}
        for f in tqdm(as_completed(futures), total=len(experiments)):
            summaries.append(f.result())

    # Sort and print
    summaries.sort(key=lambda x: x["average_wer"])
    print("\nLEADERBOARD (Conditional Full Correction)")
    print(f"{'STRATEGY':<30} {'MODEL':<25} {'WER':>6} {'CER':>6}")
    for s in summaries[:10]:
        print(f"{s['strategy']:<30} {s['llm_model'].split('/')[-1]:<25} {s['average_wer']:.4f} {s['average_cer']:.4f}")

if __name__ == "__main__":
    main()
