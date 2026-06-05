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

def align_and_extract_confidences(gt_text, ocr_text, ocr_confs):
    """
    Aligns GT and OCR text, returning two lists:
    correct_confs: confidences of words correctly recognized
    error_confs: confidences of words incorrectly recognized (substitutions/insertions)
    Note: ocr_confs should ideally match the length of ocr_text split by spaces.
    """
    out_correct = []
    out_error = []
    
    # Simple split (if ocr_confs length differs, we handle gracefully)
    gt_words = gt_text.split()
    ocr_words = ocr_text.split()
    
    if len(ocr_words) == 0:
        return [], []
        
    # Align using Jiwer
    try:
        alignment = jiwer.process_words(gt_text, ocr_text).alignments[0]
        # Iterate through alignment operations
        ocr_idx = 0
        for op in alignment:
            if op.type == 'equal':
                # Exact match
                if ocr_idx < len(ocr_confs):
                    out_correct.append(ocr_confs[ocr_idx])
                ocr_idx += 1
            elif op.type == 'substitute':
                # Incorrect match
                if ocr_idx < len(ocr_confs):
                    out_error.append(ocr_confs[ocr_idx])
                ocr_idx += 1
            elif op.type == 'insert':
                # Extra word
                if ocr_idx < len(ocr_confs):
                    out_error.append(ocr_confs[ocr_idx])
                ocr_idx += 1
            elif op.type == 'delete':
                # Missed word (no OCR confidence to assign, ignore)
                pass
    except Exception as e:
        # Fallback to simple matching if alignment throws weird errors
        for i, w in enumerate(ocr_words):
            if i < len(ocr_confs):
                if i < len(gt_words) and w == gt_words[i]:
                    out_correct.append(ocr_confs[i])
                else:
                    out_error.append(ocr_confs[i])
    return out_correct, out_error

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    eval_dir = base_dir / "data" / "evaluation_dataset"
    img_dir = eval_dir / "images"
    results_dir = base_dir / "results"
    
    gt_file = eval_dir / "groundtruth.json"
    with open(gt_file, "r") as f:
        gt_data = json.load(f)
        
    filenames = [item["filename"] for item in gt_data]
    gt_dict = {item["filename"]: item["groundtruth_text"] for item in gt_data}
    
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
        log.info(f"Processing {eng_name}...")
        for fname in tqdm(filenames, desc=eng_name):
            img_path = img_dir / fname
            if not img_path.exists(): continue
            
            gt_text = gt_dict.get(fname, "")
            
            ocr_words = []
            ocr_confs = []
            
            if eng_name == "tesseract":
                from io import StringIO
                try:
                    data = pytesseract.image_to_data(str(img_path), lang="fra")
                    df = pd.read_csv(StringIO(data), sep="\t", quoting=3)
                    df = df[df["conf"] != -1]
                    if not df.empty:
                        for _, row in df.iterrows():
                            text = str(row['text'])
                            if pd.notna(text) and text.strip():
                                ocr_words.append(text.strip())
                                ocr_confs.append(row['conf']/100.0)
                except: pass
            elif eng_name == "easyocr":
                try:
                    res = eng_reader.readtext(str(img_path))
                    for r in res:
                        ws = r[1].split()
                        c = float(r[2])
                        ocr_words.extend(ws)
                        ocr_confs.extend([c] * len(ws))
                except: pass
            elif eng_name == "paddle":
                try:
                    res = eng_reader.ocr(str(img_path), cls=True)
                    if res and res[0]:
                        for l in res[0]:
                            ws = l[1][0].split()
                            c = float(l[1][1])
                            if c > 1.0: c /= 100.0
                            ocr_words.extend(ws)
                            ocr_confs.extend([c] * len(ws))
                except: pass
            
            ocr_text = " ".join(ocr_words)
            c_confs, e_confs = align_and_extract_confidences(gt_text, ocr_text, ocr_confs)
            
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
            if len(subset) > 1:
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
    
    # Build shared legend on the last axis
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='#2ca02c', alpha=0.6, label='Correct words'),
        mpatches.Patch(color='#d62728', alpha=0.6, label='Erroneous words'),
        Line2D([0], [0], color='#555555', linestyle='--', linewidth=1.5, label='Threshold 0.80'),
        Line2D([0], [0], color='#555555', linestyle='dashdot', linewidth=1.5, label='Threshold 0.90'),
    ]
    fig.legend(handles=legend_handles, loc='upper right',
               bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=11)
    
    fig.suptitle("Word-level OCR Confidence: Correct vs. Erroneous Recognitions",
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    plot_path = results_dir / "figures/confidence/error_confidence_dist.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    log.info(f"Error Distribution plot saved to {plot_path}")

    # Display some summary stats to console
    print("\n" + "="*40)
    print("ERROR CONFIDENCE STATS (Word Level)")
    print("="*40)
    # Average conf of errors vs corrects
    stats = df.groupby(["engine", "status"])["confidence"].describe()
    print(stats)
    print("="*40)

if __name__ == "__main__":
    main()
