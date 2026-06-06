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

## 🔬 Additional Experimental Results (Not in Paper)

Due to length constraints in the 4-page CIKM Short Paper manuscript, several detailed analyses and ablations were omitted. We present them in full below:

### 1. Prompt Design & Ablation Study
We evaluated 10 hierarchical prompt templates ranging from basic cleanup to expert-engineered and Chain-of-Thought (CoT) instructions. Below is the performance of the full correction strategy under each template using **Gemini 3 Flash** as the downstream corrector across all three OCR engines (**Tesseract**, **EasyOCR**, and **PaddleOCR**).

| Prompt Level | Strategy Key | Tesseract (WER/CER) | EasyOCR (WER/CER) | PaddleOCR (WER/CER) |
|---|---|---|---|---|
| **Baseline (No LLM)** | - | 0.2335/0.0632 | 0.5992/0.1511 | 0.1673/0.0420 |
| **Prompt A** | `Full_Basic` | 0.1185/0.0324 | 0.4725/0.1051 | 0.1150/0.0306 |
| **Prompt B** | `Full_Basic_plus` | 0.1409/0.0374 | 0.4470/0.0946 | 0.1398/0.0376 |
| **Prompt C** | `Full_Intermediate` | 0.1192/0.0328 | 0.1458/0.0397 | 0.0902/0.0277 |
| **Prompt D** | `Full_Intermediate_plus` | 0.1131/0.0327 | 0.1622/0.0436 | 0.0971/0.0316 |
| **Prompt E** | `Full_Advanced` | 0.1135/0.0364 | 0.2915/0.1806 | 0.0888/0.0334 |
| **Prompt F** | `Full_Advanced_plus` | 0.1309/0.0418 | 0.1999/0.0565 | 0.0997/0.0323 |
| **Prompt G** | `Full_Expert_Few_Shot` | 0.1381/0.0397 | 0.1890/0.0514 | 0.1306/0.0391 |
| **Prompt H** | `Full_Expert_Robuste` | **0.0953**/0.0292 | **0.1180**/0.0388 | **0.0774**/0.0268 |
| **Prompt I** | `Full_Master_Chain_of_Thought` | 0.1065/**0.0278** | 0.1642/0.0466 | 0.0858/**0.0265** |
| **Prompt J** | `Full_Ultimate_Master` | 0.1238/0.0356 | 0.1618/0.0489 | 0.0987/0.0316 |

#### Key Insights from Prompt Ablation:
- **Prompt H** achieves the absolute lowest Word Error Rate (WER) across all three engines. Its brute-force instruction template avoids conversational padding/formatting and forces the LLM to output pure corrected text directly.
- **Prompt I** achieves the lowest Character Error Rate (CER) on Tesseract and PaddleOCR. The structured introspection step (`<plan_et_analyse>`) lets the model capture fine-grained character details before producing the final text.
- A "Modernization Trap" is visible at **Prompts B, F, and G** where prompts over-optimize for modern styling, resulting in a higher WER on historically spelling-rich documents.

#### Prompt Level Groupings & Definitions (Demystified)
To understand the complexity scaling across the 10 hierarchical prompt templates (A through J), here is what each level represents:

* **Group 1: Prompts A-C (Basic Formatting & Historical Rules)**
  * **Prompt A (Correction Minimale)**: Direct instructions setting the global task context of historical French OCR post-correction without listing specific rules.
  * **Prompt B (Spacing and Layout)**: Adds instructions on whitespace reduction, preserving newline characters, and leaving historical vocabulary untouched.
  * **Prompt C (History & Punctuation)**: Adds explicit rules for handling spacing around punctuation and transcribing the historical long "S" (`ſ` $\rightarrow$ `s`) while preserving archaic spellings (e.g., *seroit*).
* **Group 2: Prompts D-G (OCR Sensitivity & Intermediate Regularization)**
  * **Prompt D (OCR Error Sensitivity)**: Injects hints on typical OCR-specific confusions (e.g., mapping `u` $\leftrightarrow$ `n`, `9` $\leftrightarrow$ `g`) and unifying case consistency (e.g., `LAUsANNE` $\rightarrow$ `LAUSANNE`).
  * **Prompt E (Ligatures & Ratures)**: Introduces cleaning rules for historical ligatures and accents (e.g., `E'` $\rightarrow$ `É`, `&z` $\rightarrow$ `&`), subscript/superscript alignment, and the absolute removal of crossed-out/strike-through text.
  * **Prompt F (Bulleted Role Structure)**: Reformulates all previous instructions into clean, bulleted categories (e.g., Margins, Typo, Casse) to optimize instruction following for LLMs.
  * **Prompt G (Contextual Examples)**: Integrates in-context few-shot learning by providing concrete examples of raw OCR string inputs alongside their expected target corrections.
* **Group 3: Prompts H-J (Advanced & Structural Expert Guidelines)**
  * **Prompt H (Brute-Force Extraction)**: Employs a zero-preamble, direct extraction command that forces the LLM to output the corrected text directly, stripping all chat padding, meta-information, or formatting blocks.
  * **Prompt I (Chain-of-Thought)**: Instructs the model to think step-by-step and write a detailed analysis inside a `<plan_et_analyse>` block before outputting the corrected text.
  * **Prompt J (Exhaustive Manual)**: A comprehensive instruction set merging all previous guidelines into a highly formal, categorized annotator manual format.

### 1.1. Contextual Routing Performance (Selective vs. No-Context)

To correct low-confidence text segments, we compare two routing strategies that differ in how much text is sent to the LLM:
- **Selective (with Context)**: When a low-confidence text line is routed for correction, it is sent to the LLM along with the **preceding and succeeding lines** from the original document layout. The LLM prompt explicitly instructs the model to *only* correct the target line, using the surrounding lines strictly as semantic context to help resolve word boundaries, OCR hyphenations, and spelling ambiguities.
- **Selective No-Context (No-Ctx)**: The routed low-confidence line is sent to the LLM completely in **isolation**. The model has no access to surrounding lines and must perform the correction using only the character cues on the target line itself.

Below is the performance comparison across all 10 prompt levels for Tesseract OCR at confidence thresholds of $\text{thr} = 80$ and $\text{thr} = 90$:

| Prompt Level | Full (WER/CER) | Selective thr80 (Ctx) | Selective thr80 (No-Ctx) | Selective thr90 (Ctx) | Selective thr90 (No-Ctx) | Ours (Δ >= 0.03) | Ours Routed % |
|---|---|---|---|---|---|---|---|
| **Prompt A** | 0.1185/0.0324 | 0.1974/0.0644 | 0.2485/0.1375 | 0.2396/0.1021 | 0.2216/0.1124 | 0.1346/0.0362 | 40.4% |
| **Prompt B** | 0.1409/0.0374 | 0.1898/0.0657 | 0.2045/0.0670 | 0.2233/0.0962 | 0.2143/0.0943 | 0.1533/0.0407 | 32.8% |
| **Prompt C** | 0.1192/0.0328 | 0.1822/0.0594 | 0.1879/0.0641 | 0.1945/0.0895 | 0.1959/0.0902 | 0.1480/0.0389 | 35.0% |
| **Prompt D** | 0.1131/0.0327 | 0.1959/0.0668 | 0.1969/0.0658 | 0.2024/0.0911 | 0.1955/0.0904 | 0.1320/0.0365 | 39.0% |
| **Prompt E** | 0.1135/0.0364 | 0.1841/0.0647 | 0.2514/0.1381 | 0.1916/0.0863 | 0.1968/0.0870 | 0.1367/0.0391 | 32.8% |
| **Prompt F** | 0.1309/0.0418 | 0.1870/0.0645 | 0.2488/0.1355 | 0.1945/0.0884 | 0.1932/0.0866 | 0.1454/0.0397 | 29.0% |
| **Prompt G** | 0.1381/0.0397 | 0.1944/0.0651 | 0.2162/0.0662 | 0.2234/0.0806 | 0.2252/0.0818 | 0.1471/0.0398 | 32.5% |
| **Prompt H** | 0.0953/0.0292 | 0.2108/0.0652 | 0.2086/0.0637 | 0.2161/0.0785 | 0.2149/0.0777 | 0.1198/0.0326 | 40.2% |
| **Prompt I** | 0.1065/0.0278 | 0.2123/0.0663 | 0.2086/0.0636 | 0.2325/0.0805 | 0.2303/0.0891 | 0.1234/0.0319 | 41.4% |
| **Prompt J** | 0.1238/0.0356 | 0.2127/0.0675 | 0.2152/0.0657 | 0.2183/0.0780 | 0.2142/0.0770 | 0.1366/0.0357 | 38.2% |

#### Key Observations:
- **Importance of Context**: For **Prompts A, E, and F**, removing context causes a massive spike in character error rates (CER increases from ~0.06 to ~0.13). Without surrounding context, the LLM struggles to resolve layout segmentation or line breaks, leading to counter-productive formatting edits.
- **Robustness of Advanced Prompts**: For highly structured expert prompts (**Prompts H, I, and J**), the difference between having context and not having context is much smaller, as the prompt's strong internal constraints prevent format deviations even when local context is absent.
- **Efficiency of Predicted Delta Routing (Ours)**: By routing only documents predicted to yield an improvement of $\Delta_{\text{CER}} \geq 0.03$, our strategy routes on average **only ~30–40%** of the corpus (e.g., 40.2% for Prompt H). Despite correcting less than half of the documents, it captures the majority of the Full correction's quality gains (e.g., Prompt H WER of 0.1198 vs. Full correction's 0.0953 and raw OCR baseline's 0.2335), presenting a highly optimized cost-accuracy trade-off.

### 2. Oracle Curves for Prompting Strategies
Below is the graph comparing the Oracle curves for all prompting strategies, indicating how error rate declines as we correct an increasing percentage of documents.

![Oracle Prompting Strategies](results/figures/validationprompts/oracle_prompting_strategies.png)

### 3. Downstream Correction LLM Model Comparison
We evaluated six open- and closed-source LLMs using the `Full_Expert_Robuste_8` prompt template on Tesseract OCR:

| Model | Average WER | Average CER | Key Takeaway |
|---|---|---|---|
| **Google Gemini 3 Flash** | **0.0969** | **0.0298** | Absolute best correction quality with high speed and low cost. |
| **Google Gemini 3.1 Flash Lite** | 0.1093 | 0.0403 | Exceptional speed and cost efficiency, perfect for large scale runs. |
| **OpenAI GPT-4o** | 0.1282 | 0.0486 | Solid correction, but significantly more expensive without quality gains. |
| **OpenAI GPT-4o Mini** | 0.1646 | 0.0631 | Performs worse than baseline on character-level metrics (0.0631 vs 0.0632). |
| **Qwen 2.5 72B Instruct** | 0.1742 | 0.0861 | Struggles with historical French typography. |
| **Meta Llama 3.3 70B Instruct** | 0.1924 | 0.0981 | Tends to over-modernize or hallucinate text structure. |
| **Mistral Small 3.1 24B Instruct** | 5.8122 | 5.7844 | Fails completely due to output format formatting errors/hallucinations. |
| *Baseline Tesseract (No LLM)* | *0.2335* | *0.0632* | Reference baseline. |

### 4. Model Understanding & Feature Salience
To understand how different routing models prioritize features, we extract the top 10 feature importances (for tree-based models) or standardized coefficients (for linear models).

#### 4.1. Gradient Boosted Trees (GBT) Classifier
*Predicts whether correction improves WER by >3%*

| Rank | Feature | Importance | Type | Description |
|---|---|---|---|---|
| 1 | `avg_confidence` | 0.2141 | Metadata | Average OCR confidence |
| 2 | `ortho_integrity_char` | 0.0642 | Surface | Char similarity vs spell-correct |
| 3 | `ortho_integrity_word` | 0.0482 | Surface | Word-level dictionary hit rate |
| 4 | `avg_chars_per_line` | 0.0353 | Layout | Layout character density |
| 5 | `newline_density` | 0.0342 | Surface | Frequency of newlines |
| 6 | `word_count` | 0.0326 | Surface | Total words |
| 7 | `publication_year` | 0.0324 | Metadata | Document printing year |
| 8 | `space_ratio` | 0.0323 | Surface | Space character ratio |
| 9 | `upper_ratio` | 0.0283 | Surface | Uppercase letter ratio |
| 10 | `punct_ratio` | 0.0235 | Surface | Punctuation ratio |

#### 4.2. Linear Support Vector Machine (SVM) Classifier
*Predicts whether correction improves WER by >3%*

| Rank | Feature | Coefficient | Type | Description |
|---|---|---|---|---|
| 1 | `avg_confidence` | -1.0607 | Metadata | Low confidence indicates routing |
| 2 | `ortho_integrity_word` | -0.8691 | Surface | High dictionary errors indicate routing |
| 3 | `publication_year` | -0.8489 | Metadata | Older docs are routed more |
| 4 | `unique_char_ratio` | -0.8050 | Surface | Diverse spelling/corruption density |
| 5 | `freq_d` | -0.6886 | Surface | Frequency of letter 'd' |
| 6 | `freq_s` | -0.5989 | Surface | Frequency of letter 's' |
| 7 | `freq_p` | -0.5215 | Surface | Frequency of letter 'p' |
| 8 | `freq_v` | -0.4887 | Surface | Frequency of letter 'v' |
| 9 | `freq_o` | -0.4833 | Surface | Frequency of letter 'o' |
| 10 | `freq_j` | -0.4806 | Surface | Frequency of letter 'j' |

#### 4.3. Ridge Regression
*Predicts raw Word Error Rate (WER)*

| Rank | Feature | Coefficient | Type | Description |
|---|---|---|---|---|
| 1 | `avg_confidence` | -0.0483 | Metadata | Lower confidence strongly correlates with higher error |
| 2 | `newline_density` | +0.0341 | Surface | Higher line density suggests layout noise |
| 3 | `freq_y` | +0.0317 | Surface | Frequency of letter 'y' |
| 4 | `ortho_integrity_char` | -0.0304 | Surface | Char similarity vs spell-correct |
| 5 | `avg_word_length` | -0.0279 | Surface | Short words indicate higher fragmentation |
| 6 | `ortho_integrity_word` | -0.0277 | Surface | Word-level dictionary hit rate |
| 7 | `dict_hit_rate` | -0.0241 | Surface | Vocabulary validation hit rate |
| 8 | `freq_l` | -0.0185 | Surface | Frequency of letter 'l' |
| 9 | `freq_g` | +0.0174 | Surface | Frequency of letter 'g' |
| 10 | `space_ratio` | +0.0172 | Surface | Higher spacing ratio |

#### 4.4. Lasso Regression
*Predicts raw Word Error Rate (WER)*

| Rank | Feature | Coefficient | Type | Description |
|---|---|---|---|---|
| 1 | `newline_density` | +0.1638 | Surface | Higher line density suggests layout noise |
| 2 | `space_ratio` | +0.1447 | Surface | Higher space ratio indicates layout formatting issues |
| 3 | `avg_confidence` | -0.1289 | Metadata | Lower confidence strongly correlates with higher error |
| 4 | `ortho_integrity_word` | -0.0750 | Surface | Word-level dictionary errors |
| 5 | `freq_y` | +0.0689 | Surface | Frequency of letter 'y' |
| 6 | `freq_j` | -0.0332 | Surface | Frequency of letter 'j' |
| 7 | `freq_g` | +0.0329 | Surface | Frequency of letter 'g' |
| 8 | `newspaper_TL` | +0.0285 | Metadata | Target newspaper indicator |
| 9 | `freq_c` | -0.0274 | Surface | Frequency of letter 'c' |
| 10 | `freq_l` | -0.0273 | Surface | Frequency of letter 'l' |

#### 4.5. Lasso Delta Regression (Our Routing Signal)
*Predicts Delta (Improvement in WER/CER) directly*

| Rank | Feature | Coefficient | Type | Description |
|---|---|---|---|---|
| 1 | `avg_confidence` | -0.0867 | Metadata | Lower confidence indicates higher potential correction gain |
| 2 | `freq_y` | +0.0346 | Surface | Frequency of letter 'y' |
| 3 | `newline_density` | +0.0041 | Surface | Higher line density suggests potential layout correction gain |
| 4 | `freq_e` | 0.0000 | Surface | Zeroed (Shrunk by Lasso L1 regularisation) |
| 5 | `freq_o` | 0.0000 | Surface | Zeroed (Shrunk by Lasso L1 regularisation) |
| 6 | `freq_n` | 0.0000 | Surface | Zeroed (Shrunk by Lasso L1 regularisation) |
| 7 | `freq_m` | 0.0000 | Surface | Zeroed (Shrunk by Lasso L1 regularisation) |
| 8 | `freq_l` | 0.0000 | Surface | Zeroed (Shrunk by Lasso L1 regularisation) |
| 9 | `freq_k` | 0.0000 | Surface | Zeroed (Shrunk by Lasso L1 regularisation) |
| 10 | `freq_j` | 0.0000 | Surface | Zeroed (Shrunk by Lasso L1 regularisation) |



### 5. Routing Strategy Performance Comparison Chart
Below is the visualization comparing our three main routing strategies against the baseline and full correction.

![Correction Strategies Comparison](results/figures/correction_strategies_comparison.png)

---
*This project is part of the research for CIKM 2026.*
