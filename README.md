# Dual Encoder Geometry

Empirical research on instruction-context separation in language models.

**Central hypothesis:** Instruction and data tokens occupy geometrically distinct regions in embedding space -- and enforcing this separation architecturally improves code generation quality and robustness to prompt injection.

> *"If instruction and data live in the same vector space, injection is hard to prevent. What if they were mathematically incompatible by construction?"*

---

## Results

### Stage 1 -- Existing models (Voyage AI embeddings)

Measures whether separation already exists implicitly in current embedding models.

| Metric | Value |
|--------|-------|
| Cross-similarity (I↔D) | ~0.71 (high overlap) |
| Separability ratio | ~0.77 |
| Injection payloads in instruction-space | majority |

**Finding:** Instruction and data are similar in existing models (cross-similarity 0.71). Separation exists but is weak -- vulnerable to injection.

### Stage 2 -- Architectural separation (trained from scratch)

Trains two model variants on the same data and measures whether architectural separation produces stronger geometric separation.

| Metric | Baseline | Dual Encoder | Delta |
|--------|----------|--------------|-------|
| Cross-similarity (I↔C) | 0.7151 | -0.0732 | -0.79 |
| Separation score | 0.1424 | 0.5366 | **+0.39** |
| Cohesion score | 0.5528 | 0.8582 | **+0.31** |
| Val loss | 3.6083 | 3.4958 | **-0.11** |

**Finding:** Dual encoder achieves near-orthogonal spaces (cross-similarity = -0.07) without any adversarial training -- purely by architectural construction.

---

## Structure

```
dual-encoder-geometry/
├── stage1/                  -- TypeScript, Voyage AI embeddings
│   ├── src/analyze.ts       -- geometry analysis + injection vulnerability
│   ├── package.json
│   └── README.md
└── stage2/                  -- Python, dual encoder training
    ├── models/
    │   ├── tokenizer.py     -- BPE tokenizer trained on corpus
    │   ├── baseline.py      -- standard encoder-decoder (~12M params)
    │   └── dual_encoder.py  -- dual encoder with gate (~17M params)
    ├── data/prepare.py      -- download HumanEval + MBPP, train BPE
    ├── train.py             -- shared training loop
    ├── eval.py              -- pass@k + geometry analysis
    └── README.md
```

---

## Reproducing the results

### Stage 1

```bash
cd stage1
npm install
export VOYAGE_API_KEY=your_key   # free tier at https://dash.voyageai.com
npm run dev
```

### Stage 2

```bash
cd stage2

# Install PyTorch with CUDA (RTX 20xx)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Download datasets and train BPE tokenizer
python3 data/prepare.py

# Train both variants (~25 min each on RTX 2060)
python3 train.py --model baseline --epochs 30 --batch_size 8
python3 train.py --model dual     --epochs 30 --batch_size 8

# Compare
python3 eval.py --compare --geometry
```

---

## The architecture

### Baseline (control)
```
[instruction + <sep> + context] -> Encoder -> Decoder -> code
```
Both inputs share a single vector space from the first layer onward.

### Dual Encoder (experimental)
```
[instruction]  -> Encoder I -> vector_I --+
                                           +--> CrossAttention(Q=I, K=C, V=C) -> Decoder -> code
[context]      -> Encoder C -> vector_C --+
```

Independent encoders with no shared weights. The only communication is an auditable cross-attention gate. Instruction drives (`Q=I`), context responds (`K=C, V=C`).

**Gate values** are logged during training -- they show how much context influences each generation step. Observed convergence: ~0.3 (context influences ~30% of generation).

---

## Why code generation

Code is verifiable. A function either passes its tests or it doesn't -- no subjective judgment needed. This gives a clean binary metric (`pass@k`) and separates the geometry question from opinion.

It also makes the instruction/context distinction natural and unambiguous:
- **Instruction** = docstring (what should be done)
- **Context** = function signature (what already exists)
- **Target** = function body (what the model generates)

---

## Next steps

1. **BPE from pretrained tokenizer** -- replace corpus BPE with tiktoken for better pass@k
2. **Separation curve** -- train dual encoder variants with `n_cross_attn_layers = 1, 2, 4, 8` and plot separation score vs performance
3. **Injection robustness benchmark** -- embed payloads in context, measure attack success rate in both models
4. **Formal proof** -- prove why independent encoders converge to orthogonal spaces and what invariance guarantee this provides

---

## License

MIT
