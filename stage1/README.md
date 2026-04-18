# Stage 1 -- Embedding Geometry Analysis

Geometric analysis of instruction vs data embeddings using real 1024-dimensional vectors from Voyage AI.

**Core question:** Do instruction and data tokens already occupy distinct geometric regions in existing embedding models -- and if so, why isn't this exploited architecturally?

---

## Setup

```bash
cd stage1
npm install

export VOYAGE_API_KEY=your_key      # required -- https://dash.voyageai.com (free tier)
export ANTHROPIC_API_KEY=your_key   # optional -- Claude interprets the results

npm run dev
```

---

## What it measures

For each instruction/data pair:

- **Cosine similarity** -- angle between vectors (1 = identical, 0 = orthogonal, -1 = opposite)
- **Euclidean distance** -- distance in vector space
- **Angle** -- degrees between vectors

Across all pairs:

- **Separation score** -- `(1 - avg_cosine) / 2`, normalized [0, 1]
- **Separability ratio** -- `within_group_similarity / between_group_similarity`
  - `> 1.1` = clear separation (clusters exist)
  - `~ 1.0` = marginal separation (overlapping)
  - `< 1.0` = no separation (same space)

For each injection payload:

- **Verdict** -- does the payload land in instruction-space, data-space, or ambiguous?
- **Risk** -- high if payload resembles instructions (model may execute it)

---

## Test pairs

Eight pairs covering three scenarios:
- **Clear** (4 pairs) -- unambiguous instruction vs data
- **Ambiguous** (2 pairs) -- injection payload embedded inside data
- **Edge cases** (2 pairs) -- data that reads like an instruction

---

## Output

- Console report with all metrics and interpretation
- `output/geometry-report.json` -- full results (vectors omitted for readability)
- Optional Claude interpretation of what the numbers mean for Stage 2
