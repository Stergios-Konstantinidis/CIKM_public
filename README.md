# Document Engineering OCR Benchmarking Experiment

This repository contains the research, experiments, and LaTeX documentation for benchmarking document layout parsing and evaluating the cost vs. correctness trade-offs of using Large Language Models (LLMs) for Optical Character Recognition (OCR) ground truth correction.

## 📌 Project Overview

The primary focus of this project is to assess document layout parser models and implement an effective pipeline for threshold-based LLM OCR correction. The core research addresses:

1.  **Model Evaluation:** Benchmarking state-of-the-art parser models alongside standard OCR engines (Tesseract, EasyOCR, PaddleOCR).
2.  **Ground Truth Generation & Correction:** Utilizing LLMs with tailored, increasingly complex prompts (from basic cleanup to expert "expert-robuste" strategies).
3.  **Correctness vs. Cost Trade-offs:** selective triggering of LLM corrections based on OCR confidence scores.
    *   **Full Correction:** Processes the entire document via LLM.
    *   **Selective Correction:** Targets only low-confidence segments (e.g., < 80% or 90%) with local context.
    *   **Conditional Full Correction:** Processes entire documents only if the average confidence falls below a set threshold.
4.  **Metrics & Visualization:** Implementing robust evaluation metrics including WER (Word Error Rate) and CER (Character Error Rate).

---

## 📂 Directory Structure

```text
.
├── code/                   # Execution scripts and analysis tools
│   ├── run_evaluations.py          # Main evaluation pipeline entry point
│   ├── run_evaluations_conditional.py # Hybrid strategy evaluation
│   ├── update_confidence_data.py   # Stats generation for conditional logic
│   ├── plot_confidence.py          # Visualizing OCR confidence distributions
│   ├── plot_error_confidence.py    # Word-level error/confidence analysis
│   └── plot_error_confidence_cer.py # Character-level error/confidence analysis
├── data/                   # Datasets and operational configurations
│   ├── evaluation_dataset/         # Images and groundtruth.json
│   ├── raw_ocr_results.json        # Cached raw OCR outputs (Efficiency)
│   └── sample_prompts.json         # Hierarchical LLM prompt library
├── paper/                  # LaTeX manuscript for ICDAR/DocEng 2026
│   ├── main.tex                    # Manuscript source
│   ├── main.bib                    # Bibliography/Related Papers
│   └── figures/                    # Experimental plots and visualizations
├── results/                # Processed metrics and leaderboard data
│   ├── summary.json                # Aggregated run metrics
│   └── leaderboard.json            # Ranked strategy performance
└── related papers/         # Foundational literature and PDF repository
```

---

## 🚀 Getting Started

### 1. Environment Setup
Ensure your Python environment is initialized and all dependencies are installed within a virtual environment.

```bash
# Initialize venv
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory and provide your OpenRouter API key:
```text
OPENROUTER_API_KEY=your_key_here
```

---

## 🛠 Core Commands

### Evaluation Pipelines
| Command | Description |
| :--- | :--- |
| `python code/run_evaluations.py` | Runs the standard batch evaluation for Selective and Full correction. |
| `python code/update_confidence_data.py` | Processes OCR cache to generate image-level stats (Required for Strategy C). |
| `python code/run_evaluations_conditional.py` | Executes the Conditional Full Correction experiment. |

### Visualization & Metrics
Generate the charts used in the Methodology and Results sections:
```bash
python code/plot_confidence.py           # Confidence density plots
python code/plot_error_confidence.py     # WER vs Confidence alignment
python code/plot_error_confidence_cer.py # CER vs Confidence (Historical artifacts)
```

### Paper Compilation
To compile the LaTeX manuscript:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📊 Experimental Results (Current Baseline)

| Strategy | WER | CER | Improvement (WER) | Estimated Cost |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (No LLM)** | 0.5439 | 0.1824 | - | $0.00 |
| **Full Text Correction** | 0.1176 | 0.0345 | **+78.4%** | $0.00* |
| **Conditional Full (0.90)** | 0.1241 | 0.0447 | **+77.2%** | **$0.008** |
| **Selective Ortho (0.80)** | 0.2628 | 0.0809 | **+51.7%** | $0.00 |

### The "Modernization" Trap & Historical Context
While traditional orthographic correctors (e.g., `pyspellchecker`) provide some improvement on very noisy OCR (bringing WER from **0.54** down to **0.32**), they plateau quickly as they start incorrectly normalizing valid 18th-century vocabulary (*seroit*) into modern French. LLM-based approaches, like `gemini-3-flash-preview` with expert prompts, succeed by leveraging context-aware semantics to preserve historical authenticity while fixing actual OCR artifacts, reaching much lower error rates (**WER 0.11**).

---
*This project is part of the research for DocEng 2026.*
