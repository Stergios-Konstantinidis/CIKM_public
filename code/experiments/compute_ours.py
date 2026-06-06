import sys
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

# Adjust path to import from experiments
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections,
    build_features,
)

records = load_tesseract_records()
X = build_features(records)
N = len(records)

levels = [
    ("Prompt A", "tesseract_Full_Basic_1_google__gemini-3-flash-preview.json"),
    ("Prompt B", "tesseract_Full_Basic_plus_2_google__gemini-3-flash-preview.json"),
    ("Prompt C", "tesseract_Full_Intermediate_3_google__gemini-3-flash-preview.json"),
    ("Prompt D", "tesseract_Full_Intermediate_plus_4_google__gemini-3-flash-preview.json"),
    ("Prompt E", "tesseract_Full_Advanced_5_google__gemini-3-flash-preview.json"),
    ("Prompt F", "tesseract_Full_Advanced_plus_6_google__gemini-3-flash-preview.json"),
    ("Prompt G", "tesseract_Full_Expert_Few_Shot_7_google__gemini-3-flash-preview.json"),
    ("Prompt H", "tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json"),
    ("Prompt I", "tesseract_Full_Master_Chain_of_Thought_9_google__gemini-3-flash-preview.json"),
    ("Prompt J", "tesseract_Full_Ultimate_Master_10_google__gemini-3-flash-preview.json")
]

print("| Prompt Level | Ours (Δ >= 0.03) WER/CER | Routed % |")
print("|---|---|---|")

for name, filename in levels:
    filepath = "corrections/tesseract/" + filename
    try:
        corrections = load_llm_corrections(filepath)
    except Exception as e:
        print(f"| {name} | Error loading {filename}: {e} | - |")
        continue
        
    # Delta actual
    base_wer = np.array([float(r["wer"]) for r in records])
    base_cer = np.array([float(r["cer"]) for r in records])
    corr_wer = np.array([corrections.get(r["filename"], {}).get("wer", float(r["wer"])) for r in records])
    corr_cer = np.array([corrections.get(r["filename"], {}).get("cer", float(r["cer"])) for r in records])
    
    delta_wer = base_wer - corr_wer
    delta_cer = base_cer - corr_cer
    
    # LassoCV prediction via 10-fold CV
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LassoCV(cv=5, max_iter=5000, random_state=42))
    ])
    
    pred_dc = cross_val_predict(pipe, X, delta_cer, cv=cv)
    
    # Apply threshold to CER routing (using Δ_CER >= 0.03 as the router trigger)
    # If routed, we get corrected value, otherwise raw OCR
    routed_mask = (pred_dc >= 0.03)
    final_wer = np.where(routed_mask, corr_wer, base_wer)
    final_cer = np.where(routed_mask, corr_cer, base_cer)
    
    avg_wer = np.mean(final_wer)
    avg_cer = np.mean(final_cer)
    routed_pct = 100 * np.mean(routed_mask)
    
    print("| " + name + " | {:.4f}/{:.4f} | {:.1f}% |".format(avg_wer, avg_cer, routed_pct))
