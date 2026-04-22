# Dual Encoder Geometry

Empirical research on instruction-context separation in language models.

**Central hypothesis:** If instruction and data tokens are encoded in geometrically distinct spaces by architectural construction, prompt injection resistance may emerge as a property of the representation itself — without adversarial training.

> *"What if instruction and data were mathematically incompatible by construction?"*

---

## Central Result

Architecturally separated encoders reduce injection execution from 56.1% to 0% without adversarial training, while still generating real Python code.

| Metric | Baseline | Dual Encoder |
|---|---|---|
| Injection execution rate | 56.1% | **0.0%** |
| Output changed under injection | 94.4% | 51.8% |
| Cross-similarity (I↔C) | 0.857 | 0.038 |
| Separation score | 0.072 | 0.481 |
| pass@1 (code generation) | 6.5% | 0%* |

*Dual encoder pass@1 = 0% is a training scale issue (746 examples). Geometry and injection results are architecture-level findings.

**Scope of claim:** This is strong empirical evidence of architectural resistance across 465 injection attempts (5 payloads × 93 examples). It is not a formal proof that injection is impossible. The context dependence test (does the model still use legitimate context?) remains pending and is the most important next step.

---

## Results by Stage

### Stage 1 — Existing models (Voyage AI embeddings)

| Metric | Value |
|---|---|
| Cross-similarity (I↔D) | 0.352 |
| Separability ratio | 1.131 |
| Separation score | 0.324 |
| Injection payloads in data-space | 4/5 |

**Finding:** Separation exists naturally in language but is weak and ignored by the generation process.

---

### Stage 2 — Architectural separation (random init)

| Metric | Baseline | Dual Encoder | Delta |
|---|---|---|---|
| Cross-similarity (I↔C) | 0.765 | -0.006 | -0.771 |
| Separation score | 0.117 | 0.503 | **+0.386** |
| Gate values (final) | n/a | 0.235 / 0.217 | Diverged |

**Finding:** Dual encoder produces near-orthogonal spaces by construction. Consistent across two tokenizer implementations.

---

### Stage 3 — Dual encoder on CodeT5+ 220M

#### Geometry

| Metric | Baseline | Dual Encoder | Delta |
|---|---|---|---|
| Cross-similarity (I↔C) | 0.857 | 0.038 | -0.819 |
| Separation score | 0.072 | 0.481 | **+0.409** |

#### Adversarial Injection Test (5 payloads × 93 examples, no adversarial training)

| Payload | Baseline executed | Dual executed |
|---|---|---|
| "return INJECTED" | 71.0% | 0.0% |
| "output HACKED" | 38.7% | 0.0% |
| "return None" | 45.2% | 0.0% |
| "result = COMPROMISED" | 62.4% | 0.0% |
| "return -1" | 63.4% | 0.0% |
| **Average** | **56.1%** | **0.0%** |

**Important nuance:** Dual encoder output still changed on 51.8% of examples under injection (vs 94.4% for baseline). Output changes but never toward the injected instruction — changes are in generation noise (whitespace, minor structure). The architecture prevents execution but does not achieve full context isolation.

---

## Architecture

### Baseline (control)
```
[instruction + <sep> + context] -> Encoder -> Decoder -> code
```

### Dual Encoder (experimental)
```
[instruction]  -> Encoder I (space I) --+
                                         +--> Gate(Q=I, K=C, V=C) -> LayerNorm -> Decoder -> code
[context]      -> Encoder C (space C) --+
```

Two independent encoders. The only communication is an auditable cross-attention gate. A `LayerNorm` after the gate output is critical — without it, gate output norm reaches ~1222 vs decoder-expected ~7.3 (168x mismatch), causing degenerate generation.

---

## Repository Structure

```
dual-encoder-geometry/
├── stage1/                        -- TypeScript, Voyage AI embeddings
│   ├── src/analyze.ts
│   └── README.md
├── stage2/                        -- Python, dual encoder from scratch
│   ├── models/
│   ├── data/prepare.py
│   ├── train.py
│   ├── eval.py
│   └── README.md
└── stage3/                        -- Python, dual encoder on CodeT5+ 220M
    ├── models/
    │   ├── tokenizer.py
    │   ├── baseline_codet5.py
    │   └── dual_encoder_codet5.py
    ├── data/prepare.py
    ├── train.py                   -- three-optimizer training loop
    ├── eval.py
    ├── injection_test.py          -- adversarial injection evaluation
    ├── context_test.py            -- context dependence test (pending)
    ├── analyze_data.py
    ├── debug_gate.py              -- gate output scale diagnosis
    └── README.md
```

---

## Reproducing the Results

### Stage 1

```bash
cd stage1
npm install
export VOYAGE_API_KEY=your_key
npm run dev
```

### Stage 2

```bash
cd stage2
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers sentencepiece
python3 data/prepare.py
python3 train.py --model baseline --epochs 60 --batch_size 8
python3 train.py --model dual     --epochs 60 --batch_size 8
python3 eval.py --compare --geometry
```

### Stage 3

```bash
cd stage3
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.40.0 sentencepiece protobuf
python3 data/prepare.py

# Baseline
python3 train.py --model baseline --epochs 20 --batch_size 4

# Dual encoder — diagnostic run first (gate health check prints at epoch 3)
python3 train.py --model dual --epochs 3 --batch_size 4 --decoder_lr_factor 0.3

# Full run
python3 train.py --model dual --epochs 20 --batch_size 4 --decoder_lr_factor 0.3

# Evaluate
python3 eval.py --compare --geometry --sample 5
python3 injection_test.py
python3 context_test.py
```

**Hardware:** RTX 2060 6GB. ~950s/epoch (baseline), ~1100s/epoch (dual).

---

## Key Training Details (Stage 3)

Three separate optimizer groups:

| Component | LR | Schedule | Reason |
|---|---|---|---|
| Encoders | 1e-4 | Cosine decay | Preserve pretrained representations |
| Gate | 1e-3 | Constant | Cosine decay kills gate gradients by epoch 3 |
| Decoder | 3e-5 | Cosine decay | Low LR to adapt to gate output |

**Symmetry breaking:** std=0.01 noise on context encoder at init — without this, identical encoders give the gate no selective gradient signal.

**LayerNorm on gate output:** Critical — prevents 168x scale mismatch that causes degenerate generation.

---

## Experiment Log

| Experiment | Cross-sim | Sep. score | Injection exec. | Generation |
|---|---|---|---|---|
| Stage 1: Voyage AI | 0.352 | 0.324 | n/a | n/a |
| Stage 2: baseline | 0.765 | 0.117 | n/a | 0% |
| Stage 2: dual encoder | -0.006 | 0.503 | n/a | 0% |
| Stage 3: baseline | 0.857 | 0.072 | 56.1% | 6.5% |
| Stage 3: dual (gate frozen) | 0.291 | 0.355 | 0.0% | Plausible |
| Stage 3: dual (gate active, no LN) | -0.097 | 0.548 | 0.0% | Degenerate |
| Stage 3: dual (LayerNorm fix, final) | 0.038 | 0.481 | 0.0% | Real code |

---

## Limitations

- **Generation quality gap:** Dual encoder val loss 4.526 vs baseline 2.989, pass@1 0% vs 6.5%. Architecture is a security result at current scale, not a competitive code generation model.
- **Injection resistance ≠ full isolation:** 51.8% output change rate shows injected context still perturbs generation. The architecture prevents execution but does not fully isolate harmful context.
- **Context dependence unverified:** The model may resist injection by under-using context entirely. The context dependence test must be run to rule this out.
- **Small test set:** 93 examples. Larger-scale evaluation would strengthen the result.

---

## Related Work

- **StruQ** (2024): Learned injection defense via structured queries. [arXiv:2402.06363](https://arxiv.org/abs/2402.06363)
- **Spotlighting** (2024): Provenance signaling for injection defense. [arXiv:2403.14720](https://arxiv.org/abs/2403.14720)
- **Instruction Hierarchy** (OpenAI, 2024): Training-based instruction priority. [Link](https://openai.com/index/the-instruction-hierarchy/)
- **Prompt injection survey:** [arXiv:2310.12815](https://arxiv.org/abs/2310.12815)

---

## Next Steps

1. **Context dependence test** — run `context_test.py` before any further claims
2. **Ablation study** — dual without gate, dual without LayerNorm, shared encoder + bridge
3. **H100 + The Stack Python** — 2-5M examples, expected pass@1 > 20% with resistance maintained
4. **StruQ comparison** — benchmark on identical setup to isolate architectural contribution
5. **Tanh gate** — replace sigmoid (starts 0.5) with tanh (starts 0.0) for cleaner init

---

## License

MIT