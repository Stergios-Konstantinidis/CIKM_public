import sys
import re

with open("code/evaluation/run_evaluations.py", "r") as f:
    code = f.read()

# 1. Update invoke_llm to track & return prompt/completion tokens
old_llm_def = """def invoke_llm(
    prompt_text: str,
    ocr_text: str,
    llm_model: str,
    dry_run: bool = False,
) -> str:
    \"\"\"
    Send OCR text to the LLM with the given prompt and return the correction.
    Retries on transient errors with exponential back-off + jitter.
    In dry-run mode returns the raw OCR text unchanged (no API call).
    \"\"\"
    if dry_run:
        return ocr_text

    full_prompt = f"{prompt_text}\\n\\nTexte OCR à corriger :\\n{ocr_text}\""""

new_llm_def = """def invoke_llm(
    prompt_text: str,
    ocr_text: str,
    llm_model: str,
    dry_run: bool = False,
) -> tuple[str, int, int]:
    \"\"\"
    Returns (corrected_text, prompt_tokens, completion_tokens).
    \"\"\"
    if dry_run:
        return ocr_text, 10, 5

    full_prompt = f"{prompt_text}\\n\\nTexte OCR à corriger :\\n{ocr_text}" if ocr_text else prompt_text"""
code = code.replace(old_llm_def, new_llm_def)

# update return inside invoke_llm
old_return = """            return response.choices[0].message.content"""
new_return = """            pt = response.usage.prompt_tokens if hasattr(response, "usage") and response.usage else 0
            ct = response.usage.completion_tokens if hasattr(response, "usage") and response.usage else 0
            return response.choices[0].message.content, pt, ct"""
code = code.replace(old_return, new_return)
code = code.replace("return ocr_text  # fallback", "return ocr_text, 0, 0  # fallback")
code = code.replace("return ocr_text  # unreachable", "return ocr_text, 0, 0")

# 2. Add COST_RATES estimation
cost_logic = """
COST_RATES = {
    "google/gemini-2.5-pro-preview": (1.25, 5.00),
    "google/gemini-2.5-flash-preview": (0.075, 0.30),
    "google/gemini-3.1-pro-preview": (1.25, 5.00),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "meta-llama/llama-3.3-70b-instruct": (0.13, 0.40)
}
def estimate_cost(llm_model: str, pt: int, ct: int) -> float:
    for k, rates in COST_RATES.items():
        if k in llm_model: return (pt / 1e6 * rates[0]) + (ct / 1e6 * rates[1])
    return (pt / 1e6 * 0.5) + (ct / 1e6 * 1.5) # default fallback
"""
code = code.replace("def run_single_experiment(", cost_logic + "\\ndef run_single_experiment(")

# 3. Update run_single_experiment
old_experiment = """    for fname, gt_text, raw_ocr in per_image_data:
        corrected_ocr = invoke_llm(
            prompt["prompt_text"], raw_ocr, llm_model, dry_run=dry_run
        )
        wer, cer = compute_metrics(gt_text, corrected_ocr)
        experiment_results.append(
            {
                "filename": fname,
                "groundtruth": gt_text,
                "raw_ocr": raw_ocr,
                "corrected_ocr": corrected_ocr,
                "wer": wer,
                "cer": cer,
            }
        )

    wers = [x["wer"] for x in experiment_results]
    cers = [x["cer"] for x in experiment_results]"""

new_experiment = """    total_pt = 0
    total_ct = 0

    # Load 3-neighbor selective dataset if needed
    is_selective = strategy == "Selective_3_Neighbors"
    selective_data = {}
    if is_selective:
        lcf = results_dir / f"confidence_data/low_confidence_words_80.json"
        if lcf.exists():
            with open(lcf, "r") as f: selective_data = json.load(f)

    for fname, gt_text, raw_ocr in per_image_data:
        if is_selective:
            # We skip tesseract as the low_conf file was generated for blocks
            # But we apply the logic on the raw lines of the current engine.
            lines = raw_ocr.split("\\n")
            img_low_conf = selective_data.get(fname, [])
            img_pt, img_ct = 0, 0
            for lc in img_low_conf:
                idx = lc.get("index", -1)
                if 0 <= idx < len(lines):
                    prev = "\\n".join(lc.get("prev_context", []))
                    nxt = "\\n".join(lc.get("next_context", []))
                    p = f"Tu es un expert OCR. Corrige UNIQUEMENT la ligne OCR à confiance faible.\\n\\nContexte avant:\\n{prev}\\n\\nContexte après:\\n{nxt}\\n\\nLigne OCR à corriger:\\n{lc['text']}\\n\\nRenvoie la ligne corrigée sans fioritures ni markdown ni commentaires."
                    c_line, pt, ct = invoke_llm(p, "", llm_model, dry_run=dry_run)
                    lines[idx] = c_line
                    img_pt += pt
                    img_ct += ct
            corrected_ocr = "\\n".join(lines)
            total_pt += img_pt
            total_ct += img_ct
        else:
            corrected_ocr, pt, ct = invoke_llm(
                prompt["prompt_text"], raw_ocr, llm_model, dry_run=dry_run
            )
            total_pt += pt
            total_ct += ct
            
        wer, cer = compute_metrics(gt_text, corrected_ocr)
        experiment_results.append({
            "filename": fname,
            "groundtruth": gt_text, "raw_ocr": raw_ocr, "corrected_ocr": corrected_ocr,
            "wer": wer, "cer": cer,
        })
        
    cost = estimate_cost(llm_model, total_pt, total_ct)
    wers = [x["wer"] for x in experiment_results]
    cers = [x["cer"] for x in experiment_results]"""
code = code.replace(old_experiment, new_experiment)

code = code.replace('"cached": False,', '"cached": False,\n        "cost": cost,\n        "prompt_tokens": total_pt,\n        "comp_tokens": total_ct,')
code = code.replace('"cached": True,', '"cached": True,\n            "cost": float(np.sum([x.get("cost", 0) for x in experiment_results])) if "cost" in experiment_results[0] else 0.0,')

# 4. Baseline
code = code.replace('"num_items": len(wers),', '"num_items": len(wers),\n                "cost": 0.0,\n                "prompt_tokens": 0,\n                "comp_tokens": 0,')

# 5. Extract low conf
old_low_conf = """def extract_low_confidence_words(
    img_path, engine="paddle", threshold=0.8, cached_reader=None
) -> list:
    \"\"\"
    Return words with confidence < threshold plus their neighbours.
    Pass `cached_reader` to avoid re-initialising the engine for every image.
    \"\"\"
    results = []
    if engine == "paddle":
        try:
            from paddleocr import PaddleOCR
            ocr = cached_reader or PaddleOCR(
                use_angle_cls=True, lang="fr", show_log=False
            )
            res = ocr.ocr(str(img_path), cls=True)
            if not res or not res[0]:
                return []
            lines = res[0]
            for i, line in enumerate(lines):
                text, conf = line[1][0], float(line[1][1])
                if conf < threshold:
                    results.append(
                        {
                            "word": text,
                            "confidence": conf,
                            "prev_neighbor": lines[i - 1][1][0] if i > 0 else None,
                            "next_neighbor": (
                                lines[i + 1][1][0] if i < len(lines) - 1 else None
                            ),
                        }
                    )
        except Exception as exc:
            log.error("Paddle low-conf extraction error: %s", exc)

    elif engine == "easyocr":
        try:
            import easyocr
            reader = cached_reader or easyocr.Reader(["fr"])
            res = reader.readtext(str(img_path))
            for i, item in enumerate(res):
                text, conf = item[1], float(item[2])
                if conf < threshold:
                    results.append(
                        {
                            "word": text,
                            "confidence": conf,
                            "prev_neighbor": res[i - 1][1] if i > 0 else None,
                            "next_neighbor": (
                                res[i + 1][1] if i < len(res) - 1 else None
                            ),
                        }
                    )
        except Exception as exc:
            log.error("EasyOCR low-conf extraction error: %s", exc)

    return results"""

new_low_conf = """def extract_low_confidence_words(
    img_path, engine="paddle", threshold=0.8, cached_reader=None
) -> list:
    results = []
    lines, confs = [], []
    if engine == "paddle":
        try:
            from paddleocr import PaddleOCR
            ocr = cached_reader
            res = ocr.ocr(str(img_path), cls=True)
            if res and res[0]:
                lines = [l[1][0] for l in res[0]]
                confs = [float(l[1][1]) for l in res[0]]
        except Exception as exc: log.error("Paddle error: %s", exc)
    elif engine == "easyocr":
        try:
            res = cached_reader.readtext(str(img_path))
            lines = [r[1] for r in res]
            confs = [float(r[2]) for r in res]
        except Exception as exc: log.error("EasyOCR error: %s", exc)

    for i, (text, conf) in enumerate(zip(lines, confs)):
        if conf < threshold:
            prev_c = lines[max(0, i-3):i]
            next_c = lines[i+1:min(len(lines), i+4)]
            results.append({
                "index": i,
                "text": text,
                "confidence": conf,
                "prev_context": prev_c,
                "next_context": next_c
            })
    return results"""
code = code.replace(old_low_conf, new_low_conf)

# 6. Summary grouping for cost tracking
old_group = """        grouped[key]["_count"] += 1

    result = []
    for v in grouped.values():
        count = v.pop("_count")
        v["overall_average_wer"] = v.pop("_wer_sum") / count if count else 0.0
        v["overall_average_cer"] = v.pop("_cer_sum") / count if count else 0.0
        result.append(v)"""

new_group = """        grouped[key]["_count"] += 1
        grouped[key]["cost"] = grouped[key].get("cost", 0.0) + entry.get("cost", 0.0)

    result = []
    for v in grouped.values():
        count = v.pop("_count")
        v["overall_average_wer"] = v.pop("_wer_sum") / count if count else 0.0
        v["overall_average_cer"] = v.pop("_cer_sum") / count if count else 0.0
        result.append(v)"""
code = code.replace(old_group, new_group)

old_lb = """                "overall_cer": entry["overall_average_cer"],"""
new_lb = """                "overall_cer": entry["overall_average_cer"],
                "total_cost": entry.get("cost", 0.0),"""
code = code.replace(old_lb, new_lb)

# 7. Print cost in LB
code = code.replace("""    print(f"{'RANK':>4}  {'STRATEGY':<30}  {'MODEL':<25}  {'WER':>6}  {'CER':>6}")""", """    print(f"{'RANK':>4}  {'STRATEGY':<25}  {'MODEL':<25}  {'WER':>6}  {'CER':>6}  {'COST ($)':>8}")""")
code = code.replace("""            f"  {row['overall_wer']:.4f}  {row['overall_cer']:.4f}"
        )""", """            f"  {row['overall_wer']:.4f}  {row['overall_cer']:.4f}  ${row.get('total_cost', 0):.4f}"
        )""")


# 8. Add virtual prompt for selective
old_main_load = """    with open(data_dir / "sample_prompts.json", "r", encoding="utf-8") as f:
        sample_prompts = json.load(f)["prompts"]
    log.info("Loaded %d prompts.", len(sample_prompts))"""
new_main_load = """    with open(data_dir / "sample_prompts.json", "r", encoding="utf-8") as f:
        sample_prompts = json.load(f)["prompts"]
        
    sample_prompts.append({
        "id": "selective_3_neighbors",
        "level": "Selective_3_Neighbors",
        "name": "Correction 3 Voisins",
        "description": "Corrige unqiuement les mots de faible confiances avec +- 3 lignes de contexte",
        "prompt_text": "Virtual prompt" # the actual string is built dynamically
    })
    log.info("Loaded %d prompts (including 1 virtual for selective 3-neighbors).", len(sample_prompts))"""
code = code.replace(old_main_load, new_main_load)

with open("code/evaluation/run_evaluations.py", "w") as f:
    f.write(code)
    
print("Updated successfully")
