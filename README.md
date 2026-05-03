# SVG Scaling Laws: Decoder-Only Transformers on Scalable Vector Graphics

**CS-GY 6923 — Machine Learning · NYU Tandon School of Engineering · Spring 2026**

This project trains a family of decoder-only GPT-style language models on SVG (Scalable Vector Graphics) source code, fits neural scaling laws to the resulting loss curves, compares Standard Parameterization (SP) against Maximal Update Parameterization (μP) for learning-rate transfer, and evaluates the quality of generated SVG samples.

---

## Key Results

| Part | Result |
|------|--------|
| **Data** | 402K cleaned SVGs · 101.7M train / 1.03M val / 1.03M test tokens · vocab = 1024 |
| **SP best LR** | 3×10⁻³ (val loss 2.321 on Tiny at 20% epoch) |
| **Scaling law** | L(N) = 392.15 · N^(−0.478) + 1.085 · R² = 0.956 |
| **μP best LR** | 1×10⁻² nominal (val loss 2.259 on Tiny at 20% epoch) |
| **Generation** | 86.7% XML valid · 86.7% renderable · 73.3% structurally valid |
| **Report** | [`reports/final_report.pdf`](reports/final_report.pdf) |

---

## Repository Structure

```
.
├── configs/                     # Model size configurations (JSON)
│   ├── tiny.json                #   ~1.05M params
│   ├── small.json               #   ~3.05M params
│   ├── medium.json              #  ~11.4M params
│   ├── large.json               #  ~32.5M params
│   └── xl.json                  #  ~86.5M params
│
├── scripts/                     # All runnable Python scripts
│   ├── 01_download.py           # Part 1 – download SVG corpus
│   ├── 02_clean.py              # Part 1 – XML clean + normalize
│   ├── 03_train_tokenizer.py    # Part 1 – train SentencePiece BPE tokenizer
│   ├── 04_split_and_encode.py   # Part 1 – encode to .npy train/val/test splits
│   ├── 05_statistics.py         # Part 1 – corpus statistics & histogram
│   ├── model.py                 # Part 2 – decoder-only GPT implementation
│   ├── train.py                 # Part 2 – single-model training script
│   ├── lr_sweep.py              # Part 2 – SP learning-rate sweep (Tiny)
│   ├── fit_scaling_law.py       # Part 2 – power-law fit L(N) = a·N^(−α)+c
│   ├── mup_model.py             # Part 3 – μP GPT (MuAdamW + MuReadout)
│   ├── mup_train.py             # Part 3 – single μP model training
│   ├── mup_lr_sweep.py          # Part 3 – μP LR sweep (Tiny)
│   ├── mup_train_all.py         # Part 3 – train all 5 sizes under μP
│   ├── compare_scaling.py       # Part 3 – SP vs μP comparison plots
│   ├── generate.py              # Part 4 – unconditional + prefix-conditioned generation
│   └── evaluate.py              # Part 4 – XML validity, render rate, perplexity
│
├── notebooks/                   # Jupyter notebooks (see Notebooks section below)
│   ├── task2_1_model_architecture.ipynb
│   ├── task2_2_lr_sweep.ipynb
│   ├── task2_3_train_all_models.ipynb
│   ├── task2_4_scaling_law.ipynb
│   ├── colab_large_model.ipynb
│   ├── colab_xl_model.ipynb
│   ├── colab_mup_training.ipynb
│   ├── task3_1_3_mup_lr_sweep.ipynb
│   ├── task3_5_6_comparison.ipynb
│   ├── colab_part4_generation_v2.ipynb
│   └── archive/                 # Superseded drafts (do not grade)
│
├── tokenizer/                   # Trained SentencePiece tokenizer artifacts
│   ├── svg_bpe.model            # Binary model (used at inference)
│   ├── svg_bpe.vocab            # Human-readable vocabulary
│   ├── tokenizer.json           # HuggingFace-format wrapper
│   └── tokenizer_config.json
│
├── outputs/                     # All generated artifacts (gitignored except plots)
│   ├── results/                 # JSON result files for all experiments
│   ├── plots/                   # PNG figures used in the report
│   ├── generated/               # 15 generated SVG files (10 unconditional + 5 prefix)
│   └── logs/                    # Per-run training curve JSON logs
│
├── reports/
│   ├── final_report.pdf         # SUBMITTED REPORT
│   └── final_report.tex         # LaTeX source
│
├── requirements.txt
└── README.md
```

> **data/** and **outputs/checkpoints/** are gitignored (too large). All numeric results
> are reproduced in `outputs/results/*.json` and summarized in the report.

---

## Setup

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

The `mup` package (for Part 3) and `sentencepiece` (for the tokenizer) are included in `requirements.txt`. CairoSVG is required for SVG rendering evaluation.

---

## Reproducing the Results

### Part 1 — Data Pipeline

```bash
# 1. Download SVG corpus (~414K files, uses HuggingFace datasets)
python scripts/01_download.py

# 2. Clean: XML validation, coordinate rounding, whitespace normalization
python scripts/02_clean.py

# 3. Train SentencePiece BPE tokenizer (vocab_size=1024)
python scripts/03_train_tokenizer.py

# 4. Encode corpus → train/val/test .npy splits
python scripts/04_split_and_encode.py

# 5. Print corpus statistics and generate sequence-length histogram
python scripts/05_statistics.py
```

**Output:** `data/tokenized/train.npy` (101.7M tokens), `val.npy` (1.03M), `test.npy` (1.03M).

---

### Part 2 — Scaling Study (Standard Parameterization)

```bash
# LR sweep on Tiny model (20% of one epoch) to find optimal LR
python scripts/lr_sweep.py --config configs/tiny.json

# Train all 5 model sizes at best LR=3e-3 (large/XL ran on Google Colab)
python scripts/train.py --config configs/tiny.json   --lr 3e-3
python scripts/train.py --config configs/small.json  --lr 3e-3
python scripts/train.py --config configs/medium.json --lr 3e-3
python scripts/train.py --config configs/large.json  --lr 3e-3   # Colab
python scripts/train.py --config configs/xl.json     --lr 3e-3   # Colab

# Fit power-law scaling curve and generate plots
python scripts/fit_scaling_law.py
```

**Outputs:** `outputs/results/scaling_results.json`, `scaling_law_fit.json`, `outputs/plots/scaling_law.png`, `extrapolation.png`.

---

### Part 3 — Maximal Update Parameterization (μP)

```bash
# μP LR sweep on Tiny (nominal LR transfers across all widths)
python scripts/mup_lr_sweep.py --config configs/tiny.json

# Train all 5 sizes under μP at best nominal LR=1e-2 (large/XL on Colab)
python scripts/mup_train_all.py

# Generate SP vs μP comparison plots
python scripts/compare_scaling.py
```

**Outputs:** `outputs/results/mup_scaling_results.json`, `mup_lr_sweep_results.json`, `comparison_results.json`, `outputs/plots/sp_vs_mup_scaling.png`.

---

### Part 4 — Generation and Evaluation

Generation was performed using the Large model trained for 3 epochs (Colab). The self-contained generation loop is in `notebooks/colab_part4_generation_v2.ipynb`.

```bash
# Unconditional + prefix-conditioned generation
python scripts/generate.py \
    --checkpoint outputs/checkpoints/large_extended/checkpoint_final.pt \
    --tokenizer tokenizer/svg_bpe.model \
    --output_dir outputs/generated

# Evaluate XML validity, render rate, structural validity, perplexity
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/large_extended/checkpoint_final.pt \
    --generated_dir outputs/generated \
    --tokenizer tokenizer/svg_bpe.model
```

**Outputs:** `outputs/generated/*.svg` (15 files), `outputs/plots/generated_grid.png`, `outputs/results/evaluation_results.json`.

---

## Notebooks Guide

| Notebook | Where run | Purpose |
|----------|-----------|---------|
| `task2_1_model_architecture.ipynb` | Local | Architecture walkthrough, parameter counts |
| `task2_2_lr_sweep.ipynb` | Local | SP LR sweep analysis and plots |
| `task2_3_train_all_models.ipynb` | Local | Tiny/Small/Medium training |
| `task2_4_scaling_law.ipynb` | Local | Power-law fitting and scaling plots |
| `colab_large_model.ipynb` | **Google Colab** | Train Large model (32.5M params, 1 epoch) |
| `colab_xl_model.ipynb` | **Google Colab** | Train XL model (86.5M params, 1 epoch) |
| `colab_mup_training.ipynb` | **Google Colab** | μP training all 5 sizes |
| `task3_1_3_mup_lr_sweep.ipynb` | Local | μP architecture + LR sweep analysis |
| `task3_5_6_comparison.ipynb` | Local | SP vs μP comparison plots |
| `colab_part4_generation_v2.ipynb` | **Google Colab** | Generation + evaluation (final, Part 4) |

> Colab notebooks require a GPU runtime and a Google Drive mount for checkpoint access.
> `notebooks/archive/` contains a superseded draft; do not grade it.

---

## Model Sizes

| Name | d\_model | n\_layers | n\_heads | d\_ff | Parameters |
|------|---------|-----------|----------|-------|------------|
| Tiny | 128 | 4 | 4 | 512 | 1.05M |
| Small | 256 | 4 | 4 | 1024 | 3.05M |
| Medium | 512 | 6 | 8 | 2048 | 11.4M |
| Large | 768 | 12 | 12 | 3072 | 32.5M |
| XL | 1024 | 16 | 16 | 4096 | 86.5M |

All models: Pre-LN, GELU activation, weight tying (embedding ↔ lm\_head), causal self-attention, no bias terms.

---

## Scaling Law Results

Fitted power law (SP, 5 model sizes):

**L(N) = 392.15 · N^(−0.478) + 1.085** &nbsp;&nbsp; (R² = 0.956)

The scaling exponent α = 0.478 is substantially larger than the Kaplan et al. language-model exponent (α\_NL = 0.076), indicating that SVG is a more compressible domain where performance improves rapidly with scale.

Extrapolated loss at 865M parameters: **1.106 ± 0.200**.

---

## Report

The full 6-section academic report is at [`reports/final_report.pdf`](reports/final_report.pdf).  
LaTeX source: [`reports/final_report.tex`](reports/final_report.tex).
