# Stage 3 -- Dual Encoder on CodeT5+ 220M

Applies the dual encoder architecture to a pretrained code model.

**Question answered:** Does the geometric separation found in Stage 2 (random init) survive when the base model is pretrained on billions of tokens of Python code? And does pass@1 become nonzero?

---

## Key differences from Stage 2

| | Stage 2 | Stage 3 |
|---|---|---|
| Base model | Random initialization | CodeT5+ 220M (pretrained) |
| Tokenizer | Corpus BPE / tiktoken | CodeT5+ tokenizer |
| Expected pass@1 | 0% (overfitting on 746 examples) | >0% (pretrained Python knowledge) |
| Geometry separation | Demonstrated | Expected to replicate |
| Params | ~44M | ~440M (two encoder copies + decoder) |

## Why this doesn't invalidate Stage 2

Both baseline and dual encoder use the **same pretrained weights** as starting point. Any geometric separation that emerges after fine-tuning is attributable to the dual encoder architecture, not to pretraining differences.

Stage 2 proved the architectural effect from scratch (clean, no confounds).
Stage 3 proves the effect survives at scale with a real code model (practical).

---

## Setup

```bash
cd stage3

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers sentencepiece

# Download datasets (same as stage2)
python3 data/prepare.py

# Train baseline (CodeT5+ single encoder, control)
python3 train.py --model baseline --epochs 20 --batch_size 4

# Train dual encoder (CodeT5+ dual encoder, experimental)
python3 train.py --model dual --epochs 20 --batch_size 4

# Compare
python3 eval.py --compare --geometry
```

---

## Architecture

### Baseline
```
[ instruction + <sep> + context ] -> CodeT5+ encoder -> CodeT5+ decoder -> code
```
Standard CodeT5+ with no changes. Single shared encoder space.

### Dual Encoder
```
[ instruction ] -> CodeT5+ encoder copy I -> vector_I -+
                                                         +-> Gate(Q=I, K=C, V=C) -> CodeT5+ decoder -> code
[ context ]     -> CodeT5+ encoder copy C -> vector_C -+
```

Both encoder copies start from identical CodeT5+ pretrained weights.
Fine-tuning allows independent weight evolution -> geometric separation.
The decoder is shared and pretrained.

---

## VRAM estimate (RTX 2060 6GB)

Two encoder copies (~60M params each) + decoder (~100M) + gate (~5M) = ~225M trainable params.

With batch_size=4 and gradient_accumulation=8 (effective batch=32):
- Estimated VRAM: ~5.2GB
- If OOM: reduce batch_size to 2

---

## Expected results

Based on Stage 2 findings, expected after fine-tuning:

| Metric | Baseline | Dual Encoder |
|---|---|---|
| pass@1 | ~5-15% | ~5-15% (similar -- same decoder) |
| Cross-similarity | ~0.7 (shared space) | ~0.0 (near-orthogonal) |
| Separation score | ~0.15 | ~0.50 |
| Cohesion | ~0.45 | ~0.85 |

The pass@1 numbers should be similar between baseline and dual encoder
(same pretrained decoder, same data). The geometry numbers should
replicate Stage 2 -- that is the hypothesis being tested.

---

## Geometry analysis before fine-tuning

Run this before training to get the pretrained baseline geometry:

```bash
python3 measure_pretrained_geometry.py
```

This gives a third data point:
- Stage 1: Voyage AI (existing embedding model) -- ratio 1.13
- Stage 2: Random init baseline -- separation score 0.12
- Stage 3 pre-finetune: CodeT5+ baseline -- TBD
- Stage 3 post-finetune dual: CodeT5+ dual encoder -- expected ~0.50
