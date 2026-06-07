"""
Generate 10 supplementary figures for the DocEng 2026 paper.
Output: results/figures/sample_extra_graphs/
"""

import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

OUT = "results/figures/sample_extra_graphs"
os.makedirs(OUT, exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────────────
C = dict(
    oracle="#2ecc71",
    ours="#e74c3c",
    confbert="#f39c12",
    spell="#3498db",
    baseline="#7f8c8d",
    pad="#9b59b6",
    easy="#1abc9c",
    tess="#e67e22",
)

STYLE = dict(figure_dpi=150, font_family="DejaVu Sans")
plt.rcParams.update({"font.family": STYLE["font_family"], "axes.spines.top": False,
                     "axes.spines.right": False})

# ── helpers ─────────────────────────────────────────────────────────────────

def load(path):
    with open(path) as f:
        return json.load(f)


def save(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, dpi=STYLE["figure_dpi"], bbox_inches="tight")
    plt.close()
    print(f"  ✓  {path}")


# ── newspaper metadata ───────────────────────────────────────────────────────
gt = load("data/evaluation_dataset/groundtruth.json")
fn2news = {r["filename"]: r["newspaper"] for r in gt}

# Maps groundtruth newspaper codes → display labels (sorted chronologically)
NEWS_SHORT = {
    "ME":                       "Mercure Suisse\n(1733–38)",
    "Feuille d'Avis de Lausanne": "Feuille d'Avis\n(1762–1841)",
    "Nouvelliste Vaudois":       "Nouvelliste Vd.\n(1822–40)",
    "ACI":                      "Almanach\n(1832)",
    "esta":                     "Estafette\n(1862)",
    "RL":                       "La Revue\n(1875–1945)",
    "TL":                       "Tribune de Lsn.\n(1912)",
    "LP":                       "Lausanne Art.\n(1926)",
    "RLP":                      "Petite Revue\n(1943)",
}

PROMPT_LEVELS = [
    (1,  "Basic"),
    (2,  "Basic+"),
    (3,  "Intermed."),
    (4,  "Intermed.+"),
    (5,  "Advanced"),
    (6,  "Advanced+"),
    (7,  "Expert\n(Few-Shot)"),
    (8,  "Expert\n(Robuste)"),
    (9,  "Master\n(CoT)"),
    (10, "Ult. Master"),
]

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1 – Prompt Complexity vs WER (line per engine, Gemini 3 Flash)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 1 – Prompt complexity vs WER …")

lb = load("results/summaries/leaderboard.json")
full_gemini = [r for r in lb if r["strategy"].startswith("Full_") and
               r["llm_model"] == "google/gemini-3-flash-preview"]

# build prompt_id → entry map
summary = load("results/summaries/summary.json")
pid2entry = {}
for r in summary:
    if r["strategy"].startswith("Full_") and r["llm_model"] == "google/gemini-3-flash-preview":
        pid = r["prompt_id"]
        pid2entry[pid] = r

pids = [p for p, _ in PROMPT_LEVELS]
labels = [l for _, l in PROMPT_LEVELS]

engines = [
    ("tesseract", "Tesseract", C["tess"], "o"),
    ("easyocr",   "EasyOCR",   C["easy"], "s"),
    ("paddle",    "PaddleOCR", C["pad"],  "^"),
]

fig, ax = plt.subplots(figsize=(10, 4.5))
for key, name, col, mk in engines:
    wers = [pid2entry[p]["by_ocr_engine"][key]["wer"] if p in pid2entry else None for p in pids]
    wers_clean = [w if w is not None else np.nan for w in wers]
    ax.plot(range(len(pids)), wers_clean, marker=mk, color=col, linewidth=2,
            markersize=7, label=name)

ax.axhline(0.2335, color=C["baseline"], linestyle="--", linewidth=1.2, label="Raw OCR (Tesseract)")
ax.set_xticks(range(len(pids)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("WER")
ax.set_xlabel("Prompt level")
ax.set_title("WER by Prompt Level  ·  Gemini 3 Flash", fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0, 0.55)
save("01_prompt_complexity_wer.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2 – Cost-efficiency bubble chart (prompt level × WER × cost)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 2 – Cost-efficiency bubble …")

full_g = [r for r in summary if r["strategy"].startswith("Full_") and
          r["llm_model"] == "google/gemini-3-flash-preview"]
baseline_wer = 0.2335

fig, ax = plt.subplots(figsize=(8, 5))
cmap = plt.cm.RdYlGn
norm = plt.Normalize(1, 10)

for r in full_g:
    pid = r["prompt_id"]
    wer_red = baseline_wer - r["overall_average_wer"]
    cost = r["cost"] * 1000  # scale for bubble size
    col = cmap(norm(pid))
    ax.scatter(r["cost"], wer_red, s=cost * 200, color=col,
               alpha=0.8, edgecolors="k", linewidths=0.6)
    label = [l for p, l in PROMPT_LEVELS if p == pid][0].replace("\n", " ")
    ax.annotate(label, (r["cost"], wer_red), fontsize=7,
                ha="center", va="bottom", xytext=(0, 6), textcoords="offset points")

ax.set_xlabel("API cost ($)")
ax.set_ylabel("WER reduction vs. baseline")
ax.set_title("Cost vs. WER Reduction by Prompt  ·  Gemini 3 Flash", fontweight="bold")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Prompt level")
save("02_prompt_efficiency_bubble.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3 – Model leaderboard: best WER per model (Tesseract, best prompt)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 3 – Model leaderboard …")

# Load all full-text tesseract corrections and find best WER per model
import glob

model_best = {}
for fpath in glob.glob("results/corrections/tesseract/tesseract_Full_*.json"):
    fname = os.path.basename(fpath)
    # extract model from filename
    parts = fname.replace(".json", "").split("_")
    # model is last two segments joined with /
    model = "/".join(parts[-2:]).replace("__", "/")
    data = load(fpath)
    wers = [r["wer"] for r in data if "wer" in r]
    if not wers:
        continue
    avg_wer = np.mean(wers)
    if model not in model_best or avg_wer < model_best[model]:
        model_best[model] = avg_wer

# Clean up model names
MODEL_NAMES = {
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-2.5-pro-preview": "Gemini 2.5 Pro",
    "google/gemma-3-27b-it": "Gemma 3 27B",
    "openai/gpt-4o": "GPT-4o",
    "openai/gpt-4o-mini": "GPT-4o-mini",
    "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "qwen/qwen-2.5-72b-instruct": "Qwen 2.5 72B",
    "google/gemma-4-31b-it": "Gemma 4 31B",
}

models_clean = {MODEL_NAMES.get(k, k): v for k, v in model_best.items()}
# sort by WER
sorted_models = sorted(models_clean.items(), key=lambda x: x[1])

names = [m for m, _ in sorted_models]
wers_m = [w for _, w in sorted_models]
colors = [C["ours"] if "Gemini 3 Flash" == n else C["confbert"] if "GPT-4o" == n else "#aec6cf" for n in names]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(names, wers_m, color=colors, edgecolor="white", height=0.65)
ax.axvline(0.2335, color=C["baseline"], linestyle="--", linewidth=1.5, label="Raw OCR (Tesseract)")
for bar, w in zip(bars, wers_m):
    ax.text(w + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}", va="center", fontsize=8)
ax.set_xlabel("WER  (best prompt, Tesseract input)")
ax.set_title("LLM Model Comparison  ·  Best WER per Model", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 0.28)
save("03_model_leaderboard.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4 – Orthographic corrector damage (waterfall)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 4 – Orthographic damage waterfall …")

ortho = load("results/summaries/ortho_summary.json")
tess_ortho = {r["strategy"]: r for r in ortho if r["ocr_engine"] == "tesseract"}

raw_wer     = 0.2335
full_llm    = 0.09533   # tesseract best (Expert Robuste, Gemini 3 Flash)
ortho_full  = tess_ortho["Full_Orthographic"]["average_wer"]
ortho_t80   = tess_ortho["SelectiveNoContext_Orthographic_thr80"]["average_wer"]

stages = ["Raw OCR", "Full\nSpell-check", "Selective\nSpell-check", "Full LLM"]
values = [raw_wer, ortho_full, ortho_t80, full_llm]
cols   = [C["baseline"], "#e74c3c", "#e67e22", C["oracle"]]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(stages, values, color=cols, width=0.5, edgecolor="white")

for bar, v in zip(bars, values):
    delta = v - raw_wer
    sign = "+" if delta >= 0 else ""
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
            f"{v:.3f}\n({sign}{delta:.3f})", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.axhline(raw_wer, color=C["baseline"], linestyle="--", linewidth=1, alpha=0.5)
ax.set_ylabel("WER  (Tesseract)")
ax.set_title("Correction Strategy Comparison", fontweight="bold")
ax.set_ylim(0, 0.36)
save("04_orthographic_damage.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5 – Routing frontier (all strategies) – clean reproduction
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 5 – Routing frontier …")

sweep = load("results/routing/routing_threshold_sweep.json")

label_map = {
    "True WER":              ("Oracle",           C["oracle"],   "-",  2.5),
    "Ridge WER Prediction":  ("Regression (Ours)", C["ours"],     "-",  2.2),
    "NN WER Prediction":     ("Neural Net",        C["confbert"], "--", 1.5),
}

fig, ax = plt.subplots(figsize=(9, 5))

for entry in sweep:
    lbl = entry["label"]
    if lbl not in label_map:
        continue
    name, col, ls, lw = label_map[lbl]
    rows = entry["rows"]
    xs = [r["frac_routed"] for r in rows]
    ys = [r["avg_wer_after"] for r in rows]
    ax.plot(xs, ys, color=col, linestyle=ls, linewidth=lw, label=name)

ax.axhline(0.2335, color=C["baseline"], linestyle="--", linewidth=1.2, alpha=0.7, label="Raw OCR")
ax.set_xlabel("Fraction of documents routed")
ax.set_ylabel("WER")
ax.set_title("Routing Frontier  ·  Tesseract + Gemini 3 Flash", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0.04, 0.26)
save("05_routing_frontier.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 6 – WER vs. tokens Pareto scatter
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 6 – WER vs tokens Pareto …")

cond = load("results/summaries/summary_conditional.json")
all_runs = []

# full strategies (gemini flash, tesseract only)
for r in summary:
    if r["llm_model"] == "google/gemini-3-flash-preview":
        wer = r["by_ocr_engine"].get("tesseract", {}).get("wer", None)
        if wer is None:
            continue
        cost = r["cost"]
        strat = "Full" if r["strategy"].startswith("Full_") else "Selective"
        all_runs.append({"wer": wer, "cost": cost, "strat": strat, "label": r["strategy"]})

# conditional
for r in cond:
    if r["ocr_engine"] == "tesseract":
        strat = "ConditionalFull"
        all_runs.append({"wer": r["average_wer"], "cost": r["cost"], "strat": strat, "label": r["strategy"]})

strat_colors = {"Full": C["ours"], "Selective": C["confbert"], "ConditionalFull": C["oracle"]}
strat_labels = {"Full": "Full correction", "Selective": "Selective correction", "ConditionalFull": "Conditional Full"}

fig, ax = plt.subplots(figsize=(9, 5))
for strat, col in strat_colors.items():
    pts = [r for r in all_runs if r["strat"] == strat]
    ax.scatter([p["cost"] for p in pts], [p["wer"] for p in pts],
               color=col, alpha=0.65, s=50, label=strat_labels[strat], edgecolors="none")

# draw Pareto frontier
pareto_pts = sorted(all_runs, key=lambda r: r["cost"])
pareto = []
min_wer = float("inf")
for p in pareto_pts:
    if p["wer"] < min_wer:
        min_wer = p["wer"]
        pareto.append(p)
if pareto:
    ax.plot([p["cost"] for p in pareto], [p["wer"] for p in pareto],
            color="k", linewidth=1.5, linestyle="--", label="Pareto frontier", zorder=5)

ax.axhline(0.2335, color=C["baseline"], linestyle=":", linewidth=1.2, label="Raw OCR")
ax.set_xlabel("API cost ($)")
ax.set_ylabel("WER  (Tesseract)")
ax.set_title("Cost vs. WER  ·  All Strategies & Models", fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0, 0.6)
save("06_pareto_wer_vs_cost.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 7 – Per-newspaper baseline heatmap (3 engines × 9 newspapers)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 7 – Per-newspaper baseline heatmap …")

engines_files = [
    ("Tesseract",  "results/baselines/baseline_tesseract.json"),
    ("EasyOCR",    "results/baselines/baseline_easyocr.json"),
    ("PaddleOCR",  "results/baselines/baseline_paddle.json"),
]

# Chronological order using actual keys from groundtruth
newspapers_order = [
    "ME", "Feuille d'Avis de Lausanne", "Nouvelliste Vaudois",
    "ACI", "esta", "RL", "TL", "LP", "RLP",
]

heat = {}
for eng, fpath in engines_files:
    data = load(fpath)
    by_news = defaultdict(list)
    for row in data:
        news = fn2news.get(row["filename"])
        if news:
            by_news[news].append(row["wer"])
    heat[eng] = {n: np.mean(v) for n, v in by_news.items()}

matrix = np.array([[heat[eng].get(n, np.nan) for n in newspapers_order]
                   for eng, _, in engines_files])

short_labels = [NEWS_SHORT.get(n, n) for n in newspapers_order]
fig, ax = plt.subplots(figsize=(12, 3.5))
im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.55)
ax.set_xticks(range(len(newspapers_order)))
ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(3))
ax.set_yticklabels([e for e, _ in engines_files])
plt.colorbar(im, ax=ax, label="WER")
ax.set_title("Baseline WER by Newspaper & OCR Engine", fontweight="bold")

for i in range(3):
    for j in range(len(newspapers_order)):
        v = matrix[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if v > 0.35 else "black")
save("07_baseline_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 8 – Prompt token count proxy vs WER (dual axis)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 8 – Token count proxy vs WER …")

# cost is proportional to tokens; use cost as proxy, normalise by num_items
full_g2 = [r for r in summary if r["strategy"].startswith("Full_")
           and r["llm_model"] == "google/gemini-3-flash-preview"
           and r["prompt_id"] in dict(PROMPT_LEVELS)]

full_g2.sort(key=lambda r: r["prompt_id"])
pids2 = [r["prompt_id"] for r in full_g2]
tess_wers = [r["by_ocr_engine"]["tesseract"]["wer"] for r in full_g2]
token_proxy = [r["cost"] / r["by_ocr_engine"]["tesseract"]["num_items"] * 1e6 for r in full_g2]
xlabels2 = [[l for p, l in PROMPT_LEVELS if p == pid][0] for pid in pids2]

fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax2 = ax1.twinx()

x = np.arange(len(pids2))
bars = ax1.bar(x, token_proxy, color="#aec6cf", alpha=0.75, label="Cost per segment (µ$)", zorder=2)
ax2.plot(x, tess_wers, color=C["ours"], marker="o", linewidth=2.2, markersize=7,
         label="WER", zorder=3)

ax1.set_xticks(x)
ax1.set_xticklabels(xlabels2, fontsize=8)
ax1.set_ylabel("Cost per segment (µ$)", color="#555")
ax2.set_ylabel("WER", color=C["ours"])
ax2.tick_params(axis="y", labelcolor=C["ours"])
ax1.set_title("Prompt Cost vs. WER  ·  Gemini 3 Flash / Tesseract", fontweight="bold")

lines1 = [mpatches.Patch(color="#aec6cf", alpha=0.75, label="Cost per segment (µ$)")]
line2_handle, = ax2.plot([], [], color=C["ours"], marker="o", linewidth=2, label="WER")
ax1.legend(handles=lines1 + [line2_handle], fontsize=9, loc="upper left")
save("08_token_cost_vs_wer.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 9 – Routing decision agreement: Ours vs Oracle (confusion-style bar)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 9 – Routing agreement …")

rt = load("results/routing/routing_table.json")
tess_rt = [r for r in rt if r["engine"] == "tesseract" and r.get("has_correction")]

thresholds = [0.20, 0.40, 0.60]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for ax, frac in zip(axes, thresholds):
    pred_wers = [r["pred_wer_ridge"] for r in tess_rt]
    actual_delta = [r["actual_wer"] - r["corrected_wer"] for r in tess_rt]

    # rank by predicted WER descending → top-frac are "routed by model"
    k = int(len(tess_rt) * frac)
    ranked_by_pred = sorted(range(len(tess_rt)), key=lambda i: pred_wers[i], reverse=True)
    ranked_by_oracle = sorted(range(len(tess_rt)), key=lambda i: actual_delta[i], reverse=True)

    routed_pred = set(ranked_by_pred[:k])
    routed_oracle = set(ranked_by_oracle[:k])

    tp = len(routed_pred & routed_oracle)
    fp = len(routed_pred - routed_oracle)
    fn = len(routed_oracle - routed_pred)
    tn = len(set(range(len(tess_rt))) - routed_pred - routed_oracle)

    labels_conf = ["TP", "FP", "FN", "TN"]
    vals = [tp, fp, fn, tn]
    cols_conf = [C["oracle"], "#e74c3c", C["confbert"], "#aec6cf"]
    bars = ax.bar(labels_conf, vals, color=cols_conf, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v),
                ha="center", va="bottom", fontsize=9)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    ax.set_title(f"Top {int(frac*100)}% routed\nP={prec:.2f}  R={rec:.2f}", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.25)

axes[0].set_ylabel("Documents")
fig.suptitle("Routing Agreement  ·  Model vs. Oracle  (Tesseract)", fontweight="bold")
plt.tight_layout()
save("09_routing_agreement.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 10 – Per-newspaper WER before/after best model (lollipop)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 10 – Per-newspaper before/after …")

raw_tess = load("results/baselines/baseline_tesseract.json")
best_corr = load("results/corrections/tesseract/tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json")

fn2raw = {r["filename"]: r["wer"] for r in raw_tess}
fn2corr = {r["filename"]: r["wer"] for r in best_corr}

by_news_raw  = defaultdict(list)
by_news_corr = defaultdict(list)
for fn, raw in fn2raw.items():
    news = fn2news.get(fn)
    if news and fn in fn2corr:
        by_news_raw[news].append(raw)
        by_news_corr[news].append(fn2corr[fn])

news_avg_raw  = {n: np.mean(v) for n, v in by_news_raw.items()}
news_avg_corr = {n: np.mean(v) for n, v in by_news_corr.items()}

ordered = sorted(news_avg_raw.keys(), key=lambda n: news_avg_raw[n], reverse=True)
y = np.arange(len(ordered))

fig, ax = plt.subplots(figsize=(10, 5.5))
for i, news in enumerate(ordered):
    r = news_avg_raw[news]
    c = news_avg_corr.get(news, np.nan)
    color = C["oracle"] if c < r else "#e74c3c"
    ax.plot([c, r], [i, i], color="#ccc", linewidth=2, zorder=1)
    ax.scatter(r, i, color=C["baseline"], s=80, zorder=2, label="Raw OCR" if i == 0 else "")
    ax.scatter(c, i, color=color, s=80, zorder=3, marker="D",
               label="After LLM" if i == 0 else "")
    gain = (r - c) / r * 100
    ax.text(max(r, c) + 0.005, i, f"{gain:+.1f}%", va="center", fontsize=8)

ax.set_yticks(y)
ax.set_yticklabels([NEWS_SHORT.get(n, n) for n in ordered])
ax.set_xlabel("WER")
ax.set_title("WER Before vs. After LLM Correction  ·  by Newspaper", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 0.55)
save("10_per_newspaper_before_after.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 11 – WER by 50-year publication era (before / after LLM correction)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 11 – WER by 50-year era …")

fn2date = {r["filename"]: r["date"][:4] for r in gt}   # extract year string

def era_label(year):
    y = int(year)
    lo = (y // 50) * 50
    return f"{lo}–{lo+49}"

# Build era → [raw_wer], era → [corr_wer]
era_raw  = defaultdict(list)
era_corr = defaultdict(list)

for fn, raw_w in fn2raw.items():
    yr = fn2date.get(fn)
    if yr and fn in fn2corr:
        era = era_label(yr)
        era_raw[era].append(raw_w)
        era_corr[era].append(fn2corr[fn])

# Sort eras chronologically
all_eras = sorted(era_raw.keys())
era_raw_mean  = [np.mean(era_raw[e])  for e in all_eras]
era_corr_mean = [np.mean(era_corr[e]) for e in all_eras]
era_raw_std   = [np.std(era_raw[e])   for e in all_eras]
era_corr_std  = [np.std(era_corr[e])  for e in all_eras]
era_n         = [len(era_raw[e])      for e in all_eras]

x = np.arange(len(all_eras))
w = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars_raw  = ax.bar(x - w/2, era_raw_mean,  w, color=C["baseline"], label="Raw OCR",
                   yerr=era_raw_std,  capsize=4, error_kw=dict(linewidth=1, color="#555"))
bars_corr = ax.bar(x + w/2, era_corr_mean, w, color=C["oracle"],   label="After LLM",
                   yerr=era_corr_std, capsize=4, error_kw=dict(linewidth=1, color="#1a7a4a"))

# annotate segment count below x-axis labels
ax.set_xticks(x)
ax.set_xticklabels([f"{e}\n(n={era_n[i]})" for i, e in enumerate(all_eras)], fontsize=9)
ax.set_ylabel("WER")
ax.set_title("WER by Publication Era  ·  Before vs. After LLM Correction", fontweight="bold")
ax.legend(fontsize=9)
ylim_top = max(era_raw_mean) * 1.65
ax.set_ylim(0, ylim_top)

# reduction label: pin to just above the bar top (ignore std, keep inside axes)
for i, (r, c) in enumerate(zip(era_raw_mean, era_corr_mean)):
    red = (r - c) / r * 100
    y_pos = min(r + 0.015, ylim_top * 0.93)   # never overflow
    ax.text(x[i], y_pos, f"−{red:.0f}%", ha="center", fontsize=8,
            color="#1a7a4a", fontweight="bold")

save("11_wer_by_era.png")

print("\nAll 11 plots written to", OUT)

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 12 – GBT Feature Importance (top 15, WER classifier)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 12 – GBT feature importance …")

gbt = load("results/ml_models/gbt_classifier_results.json")
feat_names_gbt = gbt["feature_names"]

# Pretty-print feature names
FEAT_LABELS = {
    "text_length": "Text length",
    "word_count": "Word count",
    "avg_word_length": "Avg word length",
    "unique_char_ratio": "Unique char ratio",
    "digit_ratio": "Digit ratio",
    "punct_ratio": "Punctuation ratio",
    "upper_ratio": "Uppercase ratio",
    "newline_density": "Newline density",
    "space_ratio": "Space ratio",
    "ortho_integrity_char": "Ortho. integrity (char)",
    "ortho_integrity_word": "Ortho. integrity (word)",
    "spell_length_ratio": "Spell length ratio",
    "dict_hit_rate": "Dictionary hit rate",
    "max_run_length": "Max run length",
    "avg_run_length": "Avg run length",
    "num_lines": "Num. lines",
    "avg_chars_per_line": "Avg chars / line",
    "publication_year": "Publication year",
    "avg_confidence": "OCR confidence",
    "engine_flag": "Engine (Tesseract/Easy)",
}
def pretty(n):
    if n.startswith("freq_"):
        return f"Letter freq '{n[5:]}'"
    if n.startswith("newspaper_"):
        return f"Newspaper: {n[10:].replace('_', ' ')}"
    return FEAT_LABELS.get(n, n)

# Collect WER importances from delta_gt_0 threshold
wer_pt = next(p for p in gbt["per_threshold"] if p["metric"] == "wer" and p["threshold_name"] == "delta_gt_0")
fi_list = sorted(wer_pt["feature_importances_full_model"], key=lambda x: -x["importance"])[:15]
fi_names = [pretty(x["feature"]) for x in fi_list]
fi_vals  = [x["importance"] for x in fi_list]

# Group by feature type for colour
def feat_color(raw):
    if raw.startswith("freq_"): return "#9b59b6"
    if raw.startswith("newspaper_"): return C["pad"]
    if raw in ("ortho_integrity_char", "ortho_integrity_word", "spell_length_ratio", "dict_hit_rate"): return C["ours"]
    if raw in ("publication_year", "avg_confidence", "num_lines", "avg_chars_per_line"): return C["oracle"]
    return C["confbert"]

raw_names = [x["feature"] for x in fi_list]
colors_fi = [feat_color(n) for n in raw_names]

fig, ax = plt.subplots(figsize=(9, 5.5))
y_pos = np.arange(len(fi_names))
bars = ax.barh(y_pos, fi_vals, color=colors_fi, edgecolor="white", height=0.7)
for bar, v in zip(bars, fi_vals):
    ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
            va="center", fontsize=8)
ax.set_yticks(y_pos)
ax.set_yticklabels(fi_names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("GBT Feature Importance")
ax.set_title("GBT Classifier  ·  Top 15 Features for WER Routing", fontweight="bold")

# Legend for colours
legend_patches = [
    mpatches.Patch(color=C["confbert"], label="Text surface"),
    mpatches.Patch(color=C["ours"],     label="Orthographic integrity"),
    mpatches.Patch(color=C["oracle"],   label="Metadata / layout"),
    mpatches.Patch(color="#9b59b6",     label="Letter frequency"),
    mpatches.Patch(color=C["pad"],      label="Newspaper"),
]
ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
save("12_gbt_feature_importance.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 13 – Ridge Regression Coefficients (top 15, WER, grouped)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 13 – Ridge coefficients …")

lr = load("results/ml_models/linear_regression_pooled_noemb_results.json")
top_wer_feats = lr["top_features_wer"]  # list of [name, coef]

# Take top 15 by |coef|, already sorted
names_lr = [pretty(f) for f, _ in top_wer_feats[:15]]
coefs_lr  = [c for _, c in top_wer_feats[:15]]
raw_lr    = [f for f, _ in top_wer_feats[:15]]
cols_lr   = [feat_color(n) for n in raw_lr]

fig, ax = plt.subplots(figsize=(10, 5.5))
y_pos = np.arange(len(names_lr))
bars = ax.barh(y_pos, coefs_lr, color=cols_lr, edgecolor="white", height=0.7)
ax.axvline(0, color="#555", linewidth=0.8)
for bar, v in zip(bars, coefs_lr):
    xoff = 0.03 if v >= 0 else -0.03
    ha = "left" if v >= 0 else "right"
    ax.text(v + xoff, bar.get_y() + bar.get_height() / 2, f"{v:+.3f}",
            va="center", ha=ha, fontsize=8)
ax.set_yticks(y_pos)
ax.set_yticklabels(names_lr, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Ridge Coefficient  (positive = predicts higher WER)")
ax.set_title("Ridge Regression  ·  Top 15 Coefficients for WER Prediction", fontweight="bold")

legend_patches2 = [
    mpatches.Patch(color=C["confbert"], label="Text surface"),
    mpatches.Patch(color=C["ours"],     label="Orthographic integrity"),
    mpatches.Patch(color=C["oracle"],   label="Metadata / layout"),
    mpatches.Patch(color="#9b59b6",     label="Letter frequency"),
]
ax.legend(handles=legend_patches2, fontsize=8, loc="lower right")
save("13_ridge_coefficients.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 14 – OCR Confidence vs. Actual WER (scatter + regression line)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 14 – Confidence vs WER scatter …")

conf_data = load("results/confidence_data/low_confidence_words_80_tesseract.json")
baseline_tess = load("results/baselines/baseline_tesseract.json")
fn2wer_base = {r["filename"]: r["wer"] for r in baseline_tess}

conf_vals, wer_vals = [], []
for fn, entry in conf_data.items():
    if isinstance(entry, dict) and fn in fn2wer_base:
        conf_vals.append(entry.get("avg_confidence", 1.0))
        wer_vals.append(fn2wer_base[fn])

conf_arr = np.array(conf_vals)
wer_arr  = np.array(wer_vals)

# Clip WER > 1.0 (pathological OCR failures, not meaningful for routing)
mask = wer_arr <= 1.0
conf_arr, wer_arr = conf_arr[mask], wer_arr[mask]

# Binned means for trend line
bins = np.linspace(conf_arr.min(), conf_arr.max(), 20)
bin_idx = np.digitize(conf_arr, bins)
bin_means_x, bin_means_y = [], []
for b in range(1, len(bins)):
    mask = bin_idx == b
    if mask.sum() >= 3:
        bin_means_x.append(bins[b - 1])
        bin_means_y.append(wer_arr[mask].mean())

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(conf_arr, wer_arr, alpha=0.25, s=20, color=C["confbert"], edgecolors="none",
           label="Segment")
ax.plot(bin_means_x, bin_means_y, color=C["ours"], linewidth=2.5,
        label="Binned mean", zorder=3)

# Pearson r
from scipy.stats import pearsonr as _pearsonr
r_val, p_val = _pearsonr(conf_arr, wer_arr)
ax.text(0.97, 0.95, f"r = {r_val:.3f}", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, color=C["ours"],
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax.set_xlabel("Avg OCR confidence score")
ax.set_ylabel("WER")
ax.set_title("OCR Confidence vs. WER  ·  Tesseract segments", fontweight="bold")
ax.legend(fontsize=9)
save("14_confidence_vs_wer.png")

print("\nAll 14 plots written to", OUT)

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 15 – Ridge LOO: Predicted vs. Actual WER (calibration scatter)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 15 – Predicted vs Actual WER …")

import glob as _glob

preds = load("results/ml_models/predictions_ridge_noemb.json")
act = np.array([r["actual_wer"] for r in preds])
pred = np.array([r["pred_wer"]  for r in preds])

# Clip to [0,1] for display
act  = np.clip(act,  0, 1)
pred = np.clip(pred, 0, 1)

from scipy.stats import pearsonr as _pearsonr2
r2_val = 1 - np.sum((act - pred)**2) / np.sum((act - np.mean(act))**2)
r_val2, _ = _pearsonr2(act, pred)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(act, pred, alpha=0.35, s=18, color=C["confbert"], edgecolors="none")
ax.plot([0, 1], [0, 1], color="#e74c3c", linewidth=1.5, linestyle="--", label="Perfect prediction")
ax.set_xlabel("Actual WER")
ax.set_ylabel("Predicted WER")
ax.set_title("Ridge Regression  ·  LOO Predicted vs. Actual WER", fontweight="bold")
ax.text(0.03, 0.93, f"R² = {r2_val:.3f}\nr = {r_val2:.3f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save("15_ridge_predicted_vs_actual.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 16 – WER Distribution Shift (histogram before / after LLM correction)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 16 – WER distribution shift …")

raw_wers_all  = [r["wer"] for r in load("results/baselines/baseline_tesseract.json")
                 if r["wer"] <= 1.0]
corr_wers_all = [r["wer"] for r in load(
    "results/corrections/tesseract/tesseract_Full_Expert_Robuste_8_google__gemini-3-flash-preview.json")
                 if r["wer"] <= 1.0]

bins = np.linspace(0, 1, 31)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(raw_wers_all,  bins=bins, color=C["baseline"], alpha=0.65, label=f"Raw OCR  (μ={np.mean(raw_wers_all):.3f})")
ax.hist(corr_wers_all, bins=bins, color=C["oracle"],   alpha=0.65, label=f"After LLM (μ={np.mean(corr_wers_all):.3f})")
ax.set_xlabel("WER")
ax.set_ylabel("Segments")
ax.set_title("WER Distribution  ·  Before vs. After LLM Correction  (Tesseract)", fontweight="bold")
ax.legend(fontsize=9)
save("16_wer_distribution_shift.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 17 – ROC Curve (GBT classifier, WER delta > 0)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 17 – ROC curve …")

from sklearn.metrics import roc_curve, auc as sk_auc

gbt = load("results/ml_models/gbt_classifier_results.json")
wer_records = [r for r in gbt["per_record"]
               if r["metric"] == "wer" and r["threshold_name"] == "delta_gt_0"]

y_true_roc  = np.array([r["true_label"]  for r in wer_records])
y_score_roc = np.array([r["pred_proba"]  for r in wer_records])

fpr, tpr, _ = roc_curve(y_true_roc, y_score_roc)
roc_auc = sk_auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot(fpr, tpr, color=C["ours"], linewidth=2.2, label=f"GBT  (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="#aaa", linestyle="--", linewidth=1.2, label="Random")
ax.fill_between(fpr, tpr, alpha=0.12, color=C["ours"])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve  ·  GBT Routing Classifier (WER)", fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
save("17_roc_curve.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 18 – Model × Prompt WER Heatmap (Tesseract, Full strategy)
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 18 – Model × Prompt heatmap …")

summary = load("results/summaries/summary.json")
MODEL_SHORT = {
    "google/gemini-3-flash-preview":            "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview":     "Gemini 3.1 Lite",
    "google/gemini-2.5-pro-preview":            "Gemini 2.5 Pro",
    "google/gemma-3-27b-it":                    "Gemma 3 27B",
    "openai/gpt-4o":                            "GPT-4o",
    "openai/gpt-4o-mini":                       "GPT-4o-mini",
    "meta-llama/llama-3.3-70b-instruct":        "Llama 3.3 70B",
    "google/gemma-4-31b-it":                    "Gemma 4",
    "qwen/qwen-2.5-72b-instruct":               "Qwen 2.5 72B",
}
prompt_ids = [p for p, _ in PROMPT_LEVELS]
prompt_lbls = [l.replace("\n", " ") for _, l in PROMPT_LEVELS]
models_ordered = list(MODEL_SHORT.keys())

# Build matrix: rows = models, cols = prompt levels
hm_matrix = np.full((len(models_ordered), len(prompt_ids)), np.nan)
for r in summary:
    if not r["strategy"].startswith("Full_"):
        continue
    tess = r["by_ocr_engine"].get("tesseract", {})
    wer = tess.get("wer")
    pid = r.get("prompt_id")
    mdl = r.get("llm_model")
    if wer is None or pid not in prompt_ids or mdl not in models_ordered:
        continue
    ri = models_ordered.index(mdl)
    ci = prompt_ids.index(pid)
    hm_matrix[ri, ci] = wer

model_lbls = [MODEL_SHORT[m] for m in models_ordered]
fig, ax = plt.subplots(figsize=(13, 5))
im = ax.imshow(hm_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0.06, vmax=0.35)
ax.set_xticks(range(len(prompt_ids)))
ax.set_xticklabels(prompt_lbls, fontsize=8, rotation=20, ha="right")
ax.set_yticks(range(len(models_ordered)))
ax.set_yticklabels(model_lbls, fontsize=9)
plt.colorbar(im, ax=ax, label="WER")

for i in range(len(models_ordered)):
    for j in range(len(prompt_ids)):
        v = hm_matrix[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.25 else "black")

ax.set_title("WER by Model × Prompt Level  ·  Tesseract / Full Correction", fontweight="bold")
save("18_model_prompt_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 19 – Per-segment Correction Gain Distribution
# ═══════════════════════════════════════════════════════════════════════════
print("Plot 19 – Segment gain distribution …")

rt_data = load("results/routing/routing_table.json")
tess_rt2 = [r for r in rt_data if r["engine"] == "tesseract" and r.get("has_correction")]

deltas = np.array([r["actual_wer"] - r["corrected_wer"] for r in tess_rt2])
# Positive delta = correction helped; negative = hurt
helped  = (deltas > 0.01).sum()
neutral = (np.abs(deltas) <= 0.01).sum()
hurt    = (deltas < -0.01).sum()
total   = len(deltas)

bins2 = np.linspace(-0.5, 1.0, 46)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(deltas[deltas > 0.01],          bins=bins2, color=C["oracle"],   alpha=0.8,
        label=f"Helped  ({helped/total*100:.0f}%)")
ax.hist(deltas[np.abs(deltas) <= 0.01], bins=bins2, color=C["baseline"], alpha=0.8,
        label=f"Neutral ({neutral/total*100:.0f}%)")
ax.hist(deltas[deltas < -0.01],         bins=bins2, color=C["ours"],     alpha=0.8,
        label=f"Hurt    ({hurt/total*100:.0f}%)")
ax.axvline(0, color="#555", linewidth=1, linestyle="--")
ax.set_xlabel("WER reduction per segment  (raw − corrected)")
ax.set_ylabel("Segments")
ax.set_title("Per-Segment Correction Gain  ·  Tesseract + Expert Robuste", fontweight="bold")
ax.legend(fontsize=9)
save("19_segment_gain_distribution.png")

print("\nAll 19 plots written to", OUT)



