"""Analyze per-document orthographic correction and find the optimal confidence threshold."""
import json, re, sys, numpy as np
from pathlib import Path

base = Path(__file__).resolve().parent.parent.parent
eval_dir = base / 'data' / 'evaluation_dataset'
results_dir = base / 'results'

with open(eval_dir / 'groundtruth.json') as f:
    gt_data = json.load(f)

with open(base / 'data' / 'raw_ocr_results.json') as f:
    ocr_cache = json.load(f)

with open(results_dir / 'confidence_data' / 'word_confidences_tesseract.json') as f:
    conf_data = json.load(f)

from spellchecker import SpellChecker
import jiwer

def apply_annotator_rules(text):
    if not isinstance(text, str) or not text.strip(): return ''
    text = text.replace("E'", 'É').replace('E`', 'É')
    text = text.replace('&z', '&')
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+([;.,!?:])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    import string
    return text.strip(string.punctuation + ' ')

def spell_correct(text, spell):
    tokens = re.split(r'(\W+)', text)
    res = []
    for t in tokens:
        if t.isalpha():
            c = spell.correction(t.lower())
            if c:
                if t.isupper(): res.append(c.upper())
                elif t.istitle(): res.append(c.title())
                else: res.append(c)
            else: res.append(t)
        else: res.append(t)
    return ''.join(res)

spell = SpellChecker(language='fr')
engine = 'tesseract'
results = ocr_cache[engine]

data = []
for gt in gt_data:
    fname = gt['filename']
    if fname not in results or not results[fname].strip():
        continue
    raw = results[fname]
    corrected = spell_correct(raw, spell)

    gt_norm = apply_annotator_rules(gt['groundtruth_text']) or '[EMPTY]'
    raw_norm = apply_annotator_rules(raw) or '[EMPTY]'
    corr_norm = apply_annotator_rules(corrected) or '[EMPTY]'

    wer_raw = jiwer.wer(gt_norm, raw_norm)
    wer_corr = jiwer.wer(gt_norm, corr_norm)
    cer_raw = jiwer.cer(gt_norm, raw_norm)
    cer_corr = jiwer.cer(gt_norm, corr_norm)

    avg_conf = conf_data.get(fname, {}).get('avg_confidence', None)

    data.append({
        'filename': fname,
        'wer_raw': wer_raw, 'wer_ortho': wer_corr,
        'cer_raw': cer_raw, 'cer_ortho': cer_corr,
        'delta_wer': wer_raw - wer_corr,
        'delta_cer': cer_raw - cer_corr,
        'avg_confidence': avg_conf
    })

deltas_wer = np.array([d['delta_wer'] for d in data])
deltas_cer = np.array([d['delta_cer'] for d in data])

print(f'Total documents: {len(data)}')
print(f'\n--- Orthographic Correction Delta WER ---')
print(f'  Helped (delta>0): {(deltas_wer > 0).sum()} ({100*(deltas_wer > 0).mean():.1f}%)')
print(f'  Neutral (delta=0): {(deltas_wer == 0).sum()} ({100*(deltas_wer == 0).mean():.1f}%)')
print(f'  Harmed (delta<0): {(deltas_wer < 0).sum()} ({100*(deltas_wer < 0).mean():.1f}%)')
print(f'  Mean delta: {deltas_wer.mean():.4f}')

print(f'\n--- Orthographic Correction Delta CER ---')
print(f'  Helped (delta>0): {(deltas_cer > 0).sum()} ({100*(deltas_cer > 0).mean():.1f}%)')
print(f'  Neutral (delta=0): {(deltas_cer == 0).sum()} ({100*(deltas_cer == 0).mean():.1f}%)')
print(f'  Harmed (delta<0): {(deltas_cer < 0).sum()} ({100*(deltas_cer < 0).mean():.1f}%)')
print(f'  Mean delta: {deltas_cer.mean():.4f}')

raw_wers = np.array([d['wer_raw'] for d in data])
raw_cers = np.array([d['cer_raw'] for d in data])
ortho_wers = np.array([d['wer_ortho'] for d in data])
ortho_cers = np.array([d['cer_ortho'] for d in data])
confs_all = np.array([d['avg_confidence'] if d['avg_confidence'] is not None else 100 for d in data])

print(f'\n--- Optimal Confidence Threshold for Ortho Correction ---')
print(f'  (Apply ortho only to docs with avg_confidence < T)')
header = f'  {"T":>5} | {"WER":>8} | {"CER":>8} | {"#corr":>5} | {"#skip":>5}'
print(header)
print('  ' + '-' * len(header))
print(f'  {"none":>5} | {raw_wers.mean():>8.4f} | {raw_cers.mean():>8.4f} | {"0":>5} | {len(data):>5}  (baseline)')

for T in [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
    mask = confs_all < T
    result_wers = np.where(mask, ortho_wers, raw_wers)
    result_cers = np.where(mask, ortho_cers, raw_cers)
    n_corr = int(mask.sum())
    n_skip = int((~mask).sum())
    print(f'  {T:>5} | {result_wers.mean():>8.4f} | {result_cers.mean():>8.4f} | {n_corr:>5} | {n_skip:>5}')

print(f'  {"all":>5} | {ortho_wers.mean():>8.4f} | {ortho_cers.mean():>8.4f} | {len(data):>5} | {"0":>5}  (full ortho)')

# Fine-grained sweep
best_wer_t, best_wer = None, raw_wers.mean()
best_cer_t, best_cer = None, raw_cers.mean()
for T in range(40, 101):
    mask = confs_all < T
    wer = np.where(mask, ortho_wers, raw_wers).mean()
    cer = np.where(mask, ortho_cers, raw_cers).mean()
    if wer < best_wer:
        best_wer = wer
        best_wer_t = T
    if cer < best_cer:
        best_cer = cer
        best_cer_t = T

print(f'\n  Best WER threshold: T={best_wer_t} -> WER={best_wer:.4f} (vs baseline {raw_wers.mean():.4f})')
print(f'  Best CER threshold: T={best_cer_t} -> CER={best_cer:.4f} (vs baseline {raw_cers.mean():.4f})')

# Per-confidence-band analysis
print(f'\n--- Per-Confidence-Band Analysis ---')
bands = [(0, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
for lo, hi in bands:
    mask = (confs_all >= lo) & (confs_all < hi)
    n = mask.sum()
    if n == 0:
        continue
    d_wer = (raw_wers[mask] - ortho_wers[mask]).mean()
    d_cer = (raw_cers[mask] - ortho_cers[mask]).mean()
    helped = (raw_wers[mask] > ortho_wers[mask]).sum()
    harmed = (raw_wers[mask] < ortho_wers[mask]).sum()
    print(f'  Conf [{lo:>2}-{hi:>2}): n={n:>3}, avg_delta_WER={d_wer:+.4f}, '
          f'avg_delta_CER={d_cer:+.4f}, helped={helped}, harmed={harmed}')
