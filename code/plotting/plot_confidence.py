import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def setup_engines():
    engines = {}
    
    # Tesseract
    try:
        import pytesseract
        engines["tesseract"] = True
        log.info("Tesseract detected.")
    except Exception as e:
        log.warning(f"Tesseract setup failed: {e}")
    
    # EasyOCR
    try:
        import easyocr
        engines["easyocr"] = easyocr.Reader(["fr"], gpu=False)
        log.info("EasyOCR detected.")
    except Exception as e:
        log.warning(f"EasyOCR setup failed: {e}")
    
    # PaddleOCR
    try:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                engines["paddle"] = PaddleOCR(use_textline_orientation=True, lang="fr")
            except TypeError:
                engines["paddle"] = PaddleOCR(use_angle_cls=True, lang="fr")
        log.info("PaddleOCR detected.")
    except Exception as e:
        log.warning(f"PaddleOCR setup failed: {e}")
        import traceback
        traceback.print_exc()
    
    return engines

def get_confidences(img_path, eng_name, eng_reader):
    confs = []
    if eng_name == "tesseract":
        try:
            import pytesseract
            from io import StringIO
            data = pytesseract.image_to_data(str(img_path), lang="fra")
            df = pd.read_csv(StringIO(data), sep="\t", quoting=3)
            df = df[df["conf"] != -1]
            # Word-level confidence directly extracted from Tesseract
            if not df.empty:
                confs = (df["conf"] / 100.0).tolist()
        except: pass
    elif eng_name == "easyocr":
        try:
            res = eng_reader.readtext(str(img_path))
            for r in res:
                words = r[1].split()
                c = float(r[2])
                confs.extend([c] * max(1, len(words)))
        except: pass
    elif eng_name == "paddle":
        try:
            res = eng_reader.ocr(str(img_path), cls=True)
            if res and res[0]:
                for l in res[0]:
                    words = l[1][0].split()
                    c = float(l[1][1])
                    if c > 1.0: c /= 100.0
                    confs.extend([c] * max(1, len(words)))
        except: pass
    return confs

def main():
    # Paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    eval_dir = base_dir / "data" / "evaluation_dataset"
    img_dir = eval_dir / "images"
    results_dir = base_dir / "results"
    
    # Load groundtruth to get filenames
    gt_file = eval_dir / "groundtruth.json"
    if not gt_file.exists():
        log.error("Groundtruth file not found.")
        return
        
    with open(gt_file, "r") as f:
        gt_data = json.load(f)
    
    filenames = [item["filename"] for item in gt_data]
    
    engines = setup_engines()
    all_confs = []
    
    for eng_name, eng_reader in engines.items():
        log.info(f"Processing {eng_name}...")
        for fname in tqdm(filenames, desc=eng_name):
            img_path = img_dir / fname
            if img_path.exists():
                confs = get_confidences(img_path, eng_name, eng_reader)
                for c in confs:
                    all_confs.append({"engine": eng_name, "confidence": c})
    
    if not all_confs:
        log.error("No confidence data collected.")
        return
        
    df = pd.DataFrame(all_confs)
    
    # ── Plot 1: Histogram/KDE Distribution ─────────────────────────────────
    log.info("Generating distribution plot...")
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="white")
    
    # Custom palette with clear names for legend
    palette = {"tesseract": "#1f77b4", "easyocr": "#ff7f0e", "paddle": "#2ca02c"}
    
    sns.histplot(
        data=df, 
        x="confidence", 
        hue="engine", 
        element="poly", 
        kde=True,
        palette=palette,
        alpha=0.4,
        bins=40
    )
    
    plt.title("OCR Confidence Score Distribution (Word-level)", fontsize=18, fontweight='bold', pad=25)
    plt.xlabel("Confidence Score (0.0 → 1.0)", fontsize=14, labelpad=10)
    plt.ylabel("Word Count", fontsize=14, labelpad=10)
    plt.xlim(0, 1.0)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    handles = [
        mpatches.Patch(color='#1f77b4', label='tesseract', alpha=0.4),
        mpatches.Patch(color='#ff7f0e', label='easyocr', alpha=0.4),
        mpatches.Patch(color='#2ca02c', label='paddle', alpha=0.4),
        Line2D([0], [0], color='#1f77b4', linewidth=2, label='Tesseract smoothed'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2, label='easyOCR smoothed'),
        Line2D([0], [0], color='#2ca02c', linewidth=2, label='PaddleOCR smoothed'),
        Line2D([0], [0], color='#d62728', linestyle='--', linewidth=2, label='Selection Threshold (0.80)'),
        Line2D([0], [0], color='#d62728', linestyle='dashdot', linewidth=2, label='Selection Threshold (0.90)')
    ]
    plt.legend(handles=handles, title="OCR Methodology", loc="upper left", frameon=True, shadow=True)
    plt.axvline(0.80, color='#d62728', linestyle='--', linewidth=2, label="Selection Threshold (0.80)")
    plt.axvline(0.90, color='#d62728', linestyle='dashdot', linewidth=2, label="Selection Threshold (0.90)")
    
    plot_path = results_dir / "figures/confidence/confidence_distribution.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    log.info(f"Distribution plot saved to {plot_path}")

    # ── Plot 2: Cumulative Distribution (ECDF) ─────────────────────────────
    log.info("Generating cumulative plot...")
    plt.figure(figsize=(12, 7))
    
    sns.ecdfplot(
        data=df, 
        x="confidence", 
        hue="engine", 
        palette=palette,
        linewidth=3
    )
    
    plt.title("Cumulative Confidence Distribution", fontsize=18, fontweight='bold', pad=25)
    plt.xlabel("Confidence Score (0.0 → 1.0)", fontsize=14, labelpad=10)
    plt.ylabel("Accumulated Percentage (%)", fontsize=14, labelpad=10)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.1)
    
    # Format y-axis as percentage
    import matplotlib.ticker as mtick
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.grid(True, which='both', alpha=0.3, linestyle='--')
    
    handles = [
        Line2D([0], [0], color='#1f77b4', linewidth=3, label='tesseract'),
        Line2D([0], [0], color='#ff7f0e', linewidth=3, label='easyocr'),
        Line2D([0], [0], color='#2ca02c', linewidth=3, label='paddle'),
        Line2D([0], [0], color='#d62728', linestyle='--', linewidth=2, label='Selection Threshold (0.80)'),
        Line2D([0], [0], color='#d62728', linestyle='dashdot', linewidth=2, label='Selection Threshold (0.90)')
    ]
    plt.legend(handles=handles, title="OCR Methodology", loc="upper left", frameon=True, shadow=True)
    plt.axvline(0.80, color='#d62728', linestyle='--', linewidth=2, label="Selection Threshold (0.80)")
    plt.axvline(0.90, color='#d62728', linestyle='dashdot', linewidth=2, label="Selection Threshold (0.90)")
    cum_plot_path = results_dir / "figures/confidence/confidence_cumulative.png"
    plt.tight_layout()
    plt.savefig(cum_plot_path, dpi=300)
    log.info(f"Cumulative plot saved to {cum_plot_path}")
    
    
    # Show summary stats
    print("\n" + "="*40)
    print("CONFIDENCE SUMMARY STATISTICS")
    print("="*40)
    print(df.groupby("engine")["confidence"].describe())
    print("="*40)

if __name__ == "__main__":
    main()
