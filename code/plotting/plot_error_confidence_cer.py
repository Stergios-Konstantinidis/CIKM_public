import os, json, logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
try:
    import jiwer
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "jiwer"])
    import jiwer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def align_and_extract_character_confidences(gt_text, ocr_text, ocr_char_confs):
    """
    Aligns GT and OCR text at character level, returning two lists:
    correct_confs: confidences of characters correctly recognized
    error_confs: confidences of characters incorrectly recognized (substitutions/insertions)
    """
    out_correct = []
    out_error = []
    
    if len(ocr_text) == 0:
        return [], []
        
    try:
        # Jiwer process_characters for alignment
        alignment = jiwer.process_characters(gt_text, ocr_text).alignments[0]
        
        ocr_idx = 0
        for op in alignment:
            if op.type == 'equal':
                # Exact match
                if ocr_idx < len(ocr_char_confs):
                    out_correct.append(ocr_char_confs[ocr_idx])
                ocr_idx += 1
            elif op.type in ['substitute', 'insert']:
                # Mismatch or hallucination
                if ocr_idx < len(ocr_char_confs):
                    out_error.append(ocr_char_confs[ocr_idx])
                ocr_idx += 1
            elif op.type == 'delete':
                # Character missed in OCR, ignore as no confidence value exists
                pass
    except Exception as e:
        log.error(f"Alignment error: {e}")
        # Fallback simplistic alignment
        for i, char in enumerate(ocr_text):
            if i < len(ocr_char_confs):
                if i < len(gt_text) and char == gt_text[i]:
                    out_correct.append(ocr_char_confs[i])
                else:
                    out_error.append(ocr_char_confs[i])
                    
    return out_correct, out_error

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    eval_dir = base_dir / "data" / "evaluation_dataset"
    img_dir = eval_dir / "images"
    results_dir = base_dir / "results"
    
    gt_file = eval_dir / "groundtruth.json"
    with open(gt_file, "r") as f:
        gt_data = json.load(f)
        
    gt_dict = {item["filename"]: item["groundtruth_text"] for item in gt_data}
    filenames = list(gt_dict.keys())
    
    import pytesseract
    import easyocr
    import warnings
    warnings.simplefilter("ignore")
    
    engines = {
        "tesseract": True,
        "easyocr": easyocr.Reader(["fr"], gpu=False, verbose=False),
    }
    
    try:
        import paddleocr
        engines["paddle"] = paddleocr.PaddleOCR(use_angle_cls=True, lang="fr")
    except Exception as e:
        log.warning(f"Skipping PaddleOCR due to import error: {e}")
    
    records = []
    
    for eng_name, eng_reader in engines.items():
        log.info(f"Processing {eng_name} for CER analysis...")
        for fname in tqdm(filenames, desc=eng_name):
            img_path = img_dir / fname
            if not img_path.exists(): continue
            
            gt_text = gt_dict.get(fname, "")
            
            ocr_text = ""
            ocr_char_confs = []
            
            if eng_name == "tesseract":
                from io import StringIO
                try:
                    data = pytesseract.image_to_data(str(img_path), lang="fra")
                    df = pd.read_csv(StringIO(data), sep="\t", quoting=3)
                    df = df[df["conf"] != -1]
                    if not df.empty:
                        # Reconstruct text and project word confidence to characters
                        parts = []
                        for _, row in df.iterrows():
                            text = str(row['text'])
                            if pd.notna(text): # could be space or just empty
                                # Treat word as a block to preserve spacing
                                conf = row['conf'] / 100.0
                                parts.append((text, conf))
                        
                        # Build the full string with spaces
                        for i, (p_text, p_conf) in enumerate(parts):
                            ocr_text += p_text
                            ocr_char_confs.extend([p_conf] * len(p_text))
                            if i < len(parts) - 1:
                                ocr_text += " "
                                ocr_char_confs.append(p_conf) # Assign current word confidence to trailing space
                except: pass
            elif eng_name == "easyocr":
                try:
                    res = eng_reader.readtext(str(img_path))
                    for i, r in enumerate(res):
                        text = r[1]
                        conf = float(r[2])
                        ocr_text += text
                        ocr_char_confs.extend([conf] * len(text))
                        if i < len(res) - 1:
                            ocr_text += " "
                            ocr_char_confs.append(conf)
                except: pass
            elif eng_name == "paddle":
                try:
                    res = eng_reader.ocr(str(img_path), cls=True)
                    if res and res[0]:
                        for i, l in enumerate(res[0]):
                            text = l[1][0]
                            conf = float(l[1][1])
                            if conf > 1.0: conf /= 100.0
                            ocr_text += text
                            ocr_char_confs.extend([conf] * len(text))
                            if i < len(res[0]) - 1:
                                ocr_text += " "
                                ocr_char_confs.append(conf)
                except: pass
            
            c_confs, e_confs = align_and_extract_character_confidences(gt_text, ocr_text, ocr_char_confs)
            
            for c in c_confs:
                records.append({"engine": eng_name, "confidence": c, "status": "Correct"})
            for c in e_confs:
                records.append({"engine": eng_name, "confidence": c, "status": "Error"})
                
    df = pd.DataFrame(records)
    if df.empty:
        log.warning("No data extracted!")
        return
        
    engines_present = df["engine"].unique()
    n = len(engines_present)
    
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]
    
    sns.set_theme(style="whitegrid")
    color_map = {"Correct": "#2ca02c", "Error": "#d62728"}
    
    for ax, eng in zip(axes, engines_present):
        for status, color in color_map.items():
            subset = df[(df["engine"] == eng) & (df["status"] == status)]["confidence"]
            if len(subset) > 5:
                sns.kdeplot(
                    subset, ax=ax, fill=True, alpha=0.4, linewidth=2.5,
                    color=color, label=status, warn_singular=False
                )
        
        ax.axvline(0.80, color='#555555', linestyle='--', linewidth=1.5, label='Threshold 0.80')
        ax.axvline(0.90, color='#555555', linestyle='dashdot', linewidth=1.5, label='Threshold 0.90')
        ax.set_title(f"{eng.capitalize()}", fontsize=14, fontweight='bold')
        ax.set_xlabel("OCR Confidence Score", fontsize=12)
        ax.set_xlim(0, 1.0)
    
    axes[0].set_ylabel("Density", fontsize=12)
    
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='#2ca02c', alpha=0.6, label='Correct chars'),
        mpatches.Patch(color='#d62728', alpha=0.6, label='Erroneous chars'),
        Line2D([0], [0], color='#555555', linestyle='--', linewidth=1.5, label='Threshold 0.80'),
        Line2D([0], [0], color='#555555', linestyle='dashdot', linewidth=1.5, label='Threshold 0.90'),
    ]
    fig.legend(handles=legend_handles, loc='upper right',
               bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=11)
    
    fig.suptitle("Character-level OCR Confidence: Correct vs. Erroneous recognitions",
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    plot_path = results_dir / "figures/confidence/error_confidence_dist_cer.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    log.info(f"CER Error Distribution plot saved to {plot_path}")

    # Console stats
    print("\n" + "="*40)
    print("ERROR CONFIDENCE STATS (Character Level)")
    print("="*40)
    stats = df.groupby(["engine", "status"])["confidence"].describe()
    print(stats)
    print("="*40)

if __name__ == "__main__":
    main()
