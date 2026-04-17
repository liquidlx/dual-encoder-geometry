# Contributing

This is an active research experiment. Contributions welcome in these areas:

## Good first issues

- Replace corpus BPE with a pretrained tokenizer (tiktoken / sentencepiece)
- Add beam search to `eval.py` for better pass@k estimates
- Add `--n_cross_attn_layers` flag to `train.py` for easier ablation runs
- Port to a larger dataset (CodeContests, The Stack Python subset)

## Research contributions

- Injection robustness benchmark — embed payloads in context, measure attack success rate
- Separation vs performance curve — train across `n_cross_attn_layers = 1, 2, 4, 8`
- Formal analysis — why do independent encoders converge to orthogonal spaces?

## Running tests

```bash
python3 -c "
import torch, sys
sys.path.insert(0, '.')
from models.tokenizer import BPETokenizer
from models.baseline import BaselineModel, BaselineConfig
from models.dual_encoder import DualEncoderModel, DualEncoderConfig

# Quick smoke test
vocab = {'<pad>':0,'<unk>':1,'<bos>':2,'<eos>':3,'<sep>':4}
for i, c in enumerate('abcdefghijklmnopqrstuvwxyz_():. =,'):
    vocab[c] = i + 5

b = BaselineModel(BaselineConfig(vocab_size=len(vocab), d_model=64, n_encoder_layers=2, n_decoder_layers=2, d_ff=128))
d = DualEncoderModel(DualEncoderConfig(vocab_size=len(vocab), d_model=64, n_instruction_layers=2, n_context_layers=2, n_cross_attn_layers=1, n_decoder_layers=2, d_ff=128))

B, L = 2, 16
batch = {k: torch.randint(0, len(vocab), (B, L)) if 'ids' in k else torch.ones(B, L, dtype=torch.long) for k in ['instruction_ids','context_ids','target_ids','instruction_mask','context_mask','target_mask']}

b_out = b(batch)
d_out, gates = d(batch)
print('baseline:', b_out.shape)
print('dual:', d_out.shape, 'gates:', [round(g,3) for g in gates])
print('OK')
"
```

## Code style

- Python 3.10+
- No external dependencies beyond PyTorch
- Each model must implement `get_encoder_embeddings()` for geometry analysis
