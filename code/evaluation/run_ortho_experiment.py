import json
import re
import jiwer
import numpy as np
from pathlib import Path
from tqdm import tqdm
try:
    from spellchecker import SpellChecker
except ImportError:
    import sys
    sys.exit("Please install pyspellchecker: pip install pyspellchecker")

def apply_annotator_rules(text: str) -> str:
    """Apply topological normalisation rules identical to run_evaluations."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.replace("E'", "É").replace("E`", "É")
    text = text.replace("&z", "&")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+([;.,!?:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    import string
    return text.strip(string.punctuation + " ")

def compute_metrics(groundtruth: str, hypothesis: str):
    try:
        gt_norm = apply_annotator_rules(groundtruth) or "[EMPTY]"
        hyp_norm = apply_annotator_rules(hypothesis) or "[EMPTY]"
        return jiwer.wer(gt_norm, hyp_norm), jiwer.cer(gt_norm, hyp_norm)
    except Exception:
        return 1.0, 1.0

def _spell_correct(text: str, spell: SpellChecker) -> str:
    """Corrects a string word by word, preserving punctuation/spacing."""
    tokens = re.split(r'(\W+)', text)
    res = []
    for t in tokens:
        # Pyspellchecker works best on lowercased words but we want to retain capitalization if possible
        if t.isalpha():
            c = spell.correction(t.lower())
            if c:
                # Naive capitalization restore
                if t.isupper():
                    res.append(c.upper())
                elif t.istitle():
                    res.append(c.title())
                else:
                    res.append(c)
            else:
                res.append(t)
        else:
            res.append(t)
    return "".join(res)

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data"
    eval_dir = data_dir / "evaluation_dataset"
    results_dir = base_dir / "results"
    
    with open(eval_dir / "groundtruth.json", "r", encoding="utf-8") as f:
        groundtruth_data = json.load(f)
    
    ocr_cache_path = data_dir / "raw_ocr_results.json"
    ocr_cache = {}
    if ocr_cache_path.exists():
        with open(ocr_cache_path, "r", encoding="utf-8") as f:
            ocr_cache = json.load(f)
            
    if not ocr_cache:
        print("No OCR cache found. Please run run_evaluations.py first.")
        return

    print("Initializing French SpellChecker...")
    spell = SpellChecker(language='fr')
    
    summary_data = []

    for engine, results in ocr_cache.items():
        print(f"\nProcessing engine: {engine}")
        
        items = []
        for gt in groundtruth_data:
            fname = gt["filename"]
            if fname in results and results[fname].strip():
                items.append({
                    "filename": fname,
                    "groundtruth": gt["groundtruth_text"],
                    "raw_ocr": results[fname]
                })

        if not items:
            continue

        # Strategy 1: Full Text (correcting every word in the document)
        print(" -> Running Full Text Correction (Every Text)")
        wers_full, cers_full = [], []
        for idx in tqdm(range(len(items)), desc="Full Text"):
            item = items[idx]
            corrected = _spell_correct(item["raw_ocr"], spell)
            wer, cer = compute_metrics(item["groundtruth"], corrected)
            wers_full.append(wer)
            cers_full.append(cer)
            
        summary_data.append({
            "ocr_engine": engine,
            "strategy": "Full_Orthographic", # maps to "full text", "every text"
            "llm_model": "pyspellchecker",
            "average_wer": float(np.mean(wers_full)),
            "average_cer": float(np.mean(cers_full)),
            "cost": 0.0,
            "num_items": len(wers_full)
        })

        # Strategies: Selective (Low Confidence Words/Lines)
        for thr in [80, 90]:
            lcf_file = results_dir / f"confidence_data/low_confidence_words_{thr}_{engine}.json"
            if not lcf_file.exists():
                lcf_file = results_dir / f"confidence_data/low_confidence_words_{thr}.json"
            
            if lcf_file.exists():
                print(f" -> Running Selective Correction (Threshold {thr})")
                with open(lcf_file, "r", encoding="utf-8") as f:
                    sel_data = json.load(f)
                
                wers_sel, cers_sel = [], []
                for idx in tqdm(range(len(items)), desc=f"Sel {thr}"):
                    item = items[idx]
                    fname = item["filename"]
                    
                    lines = item["raw_ocr"].split("\n")
                    if fname in sel_data:
                        entry = sel_data[fname]
                        # Handle both old format (list) and new format (dict with "low_confidence_lines")
                        lc_items = entry if isinstance(entry, list) else entry.get("low_confidence_lines", [])
                        for target in lc_items:
                            line_idx = target.get("index", -1)
                            if 0 <= line_idx < len(lines):
                                lines[line_idx] = _spell_correct(lines[line_idx], spell)
                    
                    corrected = "\n".join(lines)
                    wer, cer = compute_metrics(item["groundtruth"], corrected)
                    wers_sel.append(wer)
                    cers_sel.append(cer)

                summary_data.append({
                    "ocr_engine": engine,
                    "strategy": f"SelectiveNoContext_Orthographic_thr{thr}", # maps to "just the words with low confidence"
                    "llm_model": "pyspellchecker",
                    "average_wer": float(np.mean(wers_sel)),
                    "average_cer": float(np.mean(cers_sel)),
                    "cost": 0.0,
                    "num_items": len(wers_sel)
                })

    # Update summary.json if we want it on the leaderboard
    summary_file = results_dir / "summaries/ortho_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
    print(f"\nExperiment complete! Saved results to {summary_file}")
    
    # Quick display
    print("\n--- Orthographic Corrector Results ---")
    for row in summary_data:
        print(f"Engine: {row['ocr_engine']:10} | Strategy: {row['strategy']:40} | WER: {row['average_wer']:.4f} | CER: {row['average_cer']:.4f}")

if __name__ == "__main__":
    main()
