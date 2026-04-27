# Personal Embedding Transformation (PET)

Implementation of **"Personal Embedding Transformation for Privacy-Preserving Split Learning on Textual Data"**
(Kim et al., IEEE Access 2026, DOI: 10.1109/ACCESS.2026.3668872)

---

## Overview

Split Learning (SL) reduces client-side compute by sending only intermediate *smashed data* to the server, but an honest-but-curious server can reconstruct the original input via a Data Reconstruction Attack (DRA). This repository implements:

| Component | Description |
|-----------|-------------|
| **PET** | Personal Embedding Transformation — applies random amplify/suppress masks to 128+128 of 512 embedding dimensions, making smashed data client-specific and hard to invert |
| **CRS** | Contrastive Representation Separation — InfoNCE loss that preserves class-level discriminability while suppressing instance-level details |
| **DRA** | FORA-style reconstruction attack: mimic client + MK-MMD distribution alignment + seq2seq inversion model |
| **Baselines** | Vanilla SL, Differential Privacy (DP), Selective Noise (SN) |

---

## Repository Structure

```
PET/
├── run.py          # Entry point — training + attack + evaluation
├── models.py       # T5ClientModel (PET), T5ServerModel, InversionModel
├── data.py         # AGNewsDataset loader
├── trainer.py      # Training loop, CRS loss (InfoNCE, Eq. 9–10)
├── attack.py       # SmashBuffer, MK-MMD alignment, inversion training
└── evaluate.py     # CS, BLEU, ROUGE-1/2/L metrics
```

---

## Requirements

```bash
pip install torch transformers datasets scikit-learn rouge-score nltk
```

Tested with:
- Python 3.9
- PyTorch 2.0.1
- Transformers 4.x
- CUDA 11.x (GPU recommended; CPU works but is slow)

---

## Dataset

### AGNews (used in this implementation)

The dataset is downloaded automatically on first run via the Hugging Face `datasets` library — **no manual download required**.

```python
# Handled inside data.py
from datasets import load_dataset
raw = load_dataset("ag_news", split="train")
```

**Class split (following paper Section IV-A-1):**

| Label | Category | Role |
|-------|----------|------|
| 0 | World | Client downstream task |
| 1 | Sports | Client downstream task |
| 2 | Business | Server auxiliary (attack only) |
| 3 | Sci/Tech | Server auxiliary (attack only) |

> The client trains on labels 0 and 1. The server's adversarial auxiliary dataset uses labels 2 and 3, simulating a different domain — matching the threat model assumption in the paper.

### WikiText-103 (original paper, not included here)

The paper also evaluates on WikiText-103 for the Next Sentence Prediction (NSP) task. If you want to extend this codebase:

```python
from datasets import load_dataset
wiki = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
```

Build positive pairs (consecutive sentences, label=1) and negative pairs (random sentences, label=0) as described in Section IV-A-1.

---

## Quick Start

### 1. Run proposed method (PET + CRS)

```bash
python run.py --method pet_crs
```

### 2. Run Vanilla SL baseline

```bash
python run.py --method vanilla
```

### 3. Run PET-only ablation

```bash
python run.py --method pet
```

### 4. Fast smoke test (small data, no attack)

```bash
python run.py --method pet_crs --n_per_class 500 --epochs 3 --skip_attack
```

### 5. Full experiment (closer to paper scale)

```bash
python run.py --method pet_crs \
    --n_per_class 5000 \
    --epochs 10 \
    --batch_size 32 \
    --mimic_epochs 5 \
    --inversion_epochs 10
```

---

## All Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | `pet_crs` | `vanilla` \| `pet` \| `pet_crs` |
| `--epochs` | `5` | Main task training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--alpha` | `0.1` | CRS loss weight α (Eq. 10) |
| `--temperature` | `1.0` | InfoNCE temperature τ (Eq. 9) |
| `--n_per_class` | `3000` | AGNews samples per class |
| `--mimic_epochs` | `2` | MK-MMD alignment epochs |
| `--inversion_epochs` | `3` | Inversion model training epochs |
| `--skip_attack` | `False` | Skip reconstruction attack stage |
| `--seed` | `42` | Random seed |
| `--gpu` | `0` | CUDA device index |

---

## Architecture

```
Client (device)                         Server
┌─────────────────────────────┐         ┌──────────────────────────┐
│  Input tokens                │         │  T5 Encoder blocks 0–5   │
│       │                      │         │          │               │
│  Embedding layer (T5)        │         │  Layer Norm + Dropout    │
│       │                      │         │          │               │
│  PET: apply plus/minus masks │  ──────▶│  Linear classifier       │
│       │                      │smashed  │          │               │
│  CRS: InfoNCE loss           │  data   │     Prediction           │
└─────────────────────────────┘         └──────────────────────────┘
         ▲
    Gradients flow back
    through smashed data
```

**PET masks (Section III-C-1):**
- 128 dimensions → **amplify** (`+` mask, L1 regularization to maximize)
- 128 dimensions → **suppress** (`−` mask, L1 regularization to minimize)
- 256 dimensions → free (no constraint)
- Masks are randomized per client using a fixed seed → each client has a unique embedding space

---

## Notes

- The T5-small **embedding layer** is randomly re-initialized on the client side (not shared with the server), as in the original implementation. The server uses its own T5 encoder weights.
- For ablation: `--method pet` runs PET without CRS; `--method vanilla` is plain split learning with no defense.
- The reconstruction metrics reflect **attack success** — lower values mean better privacy protection.
