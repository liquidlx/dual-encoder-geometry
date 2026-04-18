# Dual Encoder for Code Generation

Empirical experiment testing the hypothesis that **instruction and context occupy geometrically distinct regions in embedding space** — and that enforcing this separation architecturally improves code generation and robustness to prompt injection.

> *"If instruction and data live in the same vector space, injection is hard to prevent. What if they were mathematically incompatible by construction?"*

---

## Results

| Metric | Baseline | Dual Encoder | Delta |
|--------|----------|--------------|-------|
| Cross-similarity (I↔C) | 0.7151 | -0.0732 | -0.79 |
| Separation score | 0.1424 | 0.5366 | **+0.39** |
| Cohesion score | 0.5528 | 0.8582 | **+0.31** |
| Val loss | 3.6083 | 3.4958 | **-0.11** |

The dual encoder achieved near-orthogonal instruction/context spaces (cross-similarity = -0.07) without any adversarial training — purely by architectural construction.

---

## Architecture

### Baseline — standard encoder-decoder
```
[instruction + <sep> + context] → Encoder → Decoder → code
```
Both inputs share the same vector space. Standard LLM architecture.

### Dual Encoder — the hypothesis
```
[instruction]  → Encoder I → vector_I ─┐
                                         ├→ CrossAttention(Q=I, K=C, V=C) → Decoder → code
[context]      → Encoder C → vector_C ─┘
```

Separate encoders with independent weights. The only communication is via a controlled, auditable cross-attention gate. The decoder never sees raw instruction or context — only the result of the gate.

**Key design decision:** instruction queries context (`Q=I, K=C, V=C`). The instruction drives what to look for; the context responds.

**Gate values** are exposed during training — you can audit how much the context influenced each generation. This is impossible in the baseline.

---

## Setup

### 1. Install PyTorch with CUDA

```bash
# RTX 20xx / GTX 10xx (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# RTX 30xx / 40xx (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify GPU
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Prepare data and train BPE tokenizer

```bash
python3 data/prepare.py
```

Downloads HumanEval (164 examples) + MBPP (~768 examples) from public GitHub repos.
Trains a BPE tokenizer on the training corpus — no external tokenizer files needed.

### 3. Train both models

```bash
# Standard transformer — shared encoder space
python3 train.py --model baseline --epochs 30 --batch_size 8

# Dual encoder — separated spaces with cross-attention gate
python3 train.py --model dual --epochs 30 --batch_size 8
```

Training output (dual encoder):
```
Epoch   1/30 | train=9.09 | val=11.51 | lr=2.08e-03 | 6.5s | gates=['0.494', '0.495']
Epoch  16/30 | train=1.28 | val=3.49  | lr=1.61e-03 | 6.1s | gates=['0.352', '0.355']
Epoch  30/30 | train=0.50 | val=3.92  | lr=1.18e-03 | 6.2s | gates=['0.293', '0.303']
```

Gate values start at ~0.5 (neutral) and converge to ~0.3 — the model learned that context should influence ~30% of instruction-driven generation.

### 4. Compare

```bash
python3 eval.py --compare --geometry
```

---

## Hardware

Tested on **RTX 2060 6GB** (Windows 11 + WSL2). ~25 min per model, ~1h total.

| Component | Baseline | Dual Encoder |
|-----------|----------|--------------|
| Parameters | ~12M | ~17M |
| VRAM (batch=8) | ~2.5GB | ~3GB |
| Time/epoch | ~10s | ~6s |

---

## Project structure

```
stage2/
├── data/
│   └── prepare.py          # Download datasets + train BPE tokenizer
├── models/
│   ├── tokenizer.py        # BPE tokenizer + CodeDataset
│   ├── baseline.py         # Standard encoder-decoder (~12M params)
│   └── dual_encoder.py     # Dual encoder with gate (~17M params)
├── checkpoints/            # Saved model weights (git-ignored)
├── results/                # Evaluation outputs (git-ignored)
├── train.py                # Training loop (identical for both models)
├── eval.py                 # pass@k evaluation + geometry analysis
└── requirements.txt
```

---

## Experimental controls

To isolate architecture as the cause of any difference:

| Parameter | Baseline | Dual Encoder |
|-----------|----------|--------------|
| Dataset | identical | identical |
| Training loop | identical | identical |
| Optimizer | AdamW | AdamW |
| LR schedule | identical | identical |
| Seed | 42 | 42 |
| **Architecture** | shared encoder | **dual encoder** |

---

## What the gate values mean

The dual encoder exposes a learned scalar gate per cross-attention layer. This gate controls how much the context influences instruction-driven generation:

- Gate → 0: model relies on instruction only
- Gate → 1: model fully incorporates context
- Gate ~0.3 (observed): context influences ~30% of generation

This auditability is impossible in the baseline — instruction and context are fused from the first layer onward.

---

## Next steps

1. **BPE from GPT-2** — replace corpus-trained BPE with a pretrained tokenizer for better pass@1
2. **Separation curve** — train dual encoder variants with `n_cross_attn_layers = 1, 2, 4, 8` and plot separation score vs val loss to find the optimal sweet spot
3. **Injection robustness test** — embed injection payloads in context and measure whether the dual encoder resists by construction
4. **Formal proof** — prove why independent encoders converge to orthogonal spaces and what invariance guarantee this provides

---

## Background

This experiment is Stage 2 of a broader research hypothesis about LLM architecture:

- **Stage 1:** Geometric analysis of existing embedding models — do instruction and data tokens already occupy distinct regions? ([see Stage 1 code](../embedding-geometry))
- **Stage 2 (this):** Controlled experiment — does architectural separation improve geometry and generation quality?
- **Stage 3 (planned):** Formal proof of separation invariance + injection resistance guarantee

The core question: can separation between instruction space and data space be a *construction guarantee* rather than an emergent training property?

---

## Citation

If you use this code or build on these ideas:

```
@misc{dual-encoder-code-2026,
  title   = {Dual Encoder Architecture for Instruction-Context Separation in Code Generation},
  year    = {2026},
  url     = {https://github.com/YOUR_USERNAME/dual-encoder-geometry}
}
```
