"""
tokenizer.py -- Tiktoken-based tokenizer + CodeDataset.

Replaces the corpus-trained BPE with OpenAI's tiktoken (cl100k_base encoding).
This is the same tokenizer used by GPT-4 and CodeX -- it handles Python syntax
correctly out of the box, with no training required.

Why tiktoken over corpus BPE:
  Corpus BPE: trained on ~750 examples, limited vocabulary coverage
  Tiktoken:   trained on hundreds of billions of tokens including code,
              'return', 'def', 'self.', '__init__' are all single tokens

Special tokens are appended to the tiktoken vocabulary:
  <pad> = base_vocab_size + 0
  <bos> = base_vocab_size + 1
  <eos> = base_vocab_size + 2
  <sep> = base_vocab_size + 3  -- baseline model only
"""

import json
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import Dataset

# cl100k_base: used by GPT-4, CodeX, text-embedding-3 -- strong on code
_BASE_ENCODING = "cl100k_base"
_BASE_VOCAB_SIZE = 100256  # cl100k_base vocab size

PAD_ID  = _BASE_VOCAB_SIZE + 0
BOS_ID  = _BASE_VOCAB_SIZE + 1
EOS_ID  = _BASE_VOCAB_SIZE + 2
SEP_ID  = _BASE_VOCAB_SIZE + 3
VOCAB_SIZE = _BASE_VOCAB_SIZE + 4


class TiktokenWrapper:
    """
    Wraps tiktoken with the same interface as the previous BPETokenizer,
    so no changes are needed in train.py, eval.py, baseline.py, or dual_encoder.py.

    save() / load() / from_file() are no-ops -- tiktoken needs no vocab file.
    The 'vocab.json' written by prepare.py contains only metadata.
    """

    def __init__(self):
        self._enc = tiktoken.get_encoding(_BASE_ENCODING)
        self.vocab_size = VOCAB_SIZE

    def encode(
        self,
        text: str,
        max_len: int,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text to token ids, truncate and pad to max_len."""
        ids = self._enc.encode(text, disallowed_special=())

        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]

        ids = ids[:max_len]
        ids += [PAD_ID] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token ids back to string, ignoring special tokens."""
        special = {PAD_ID, BOS_ID, EOS_ID, SEP_ID}
        clean = [i for i in ids if not (skip_special and i in special)]
        # tiktoken can only decode base vocab ids
        base = [i for i in clean if i < _BASE_VOCAB_SIZE]
        return self._enc.decode(base)

    def save(self, path: str | Path):
        """Write metadata file -- tiktoken itself needs no serialization."""
        meta = {
            "tokenizer": "tiktoken",
            "encoding": _BASE_ENCODING,
            "vocab_size": VOCAB_SIZE,
            "pad_id": PAD_ID,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
            "sep_id": SEP_ID,
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TiktokenWrapper":
        """Load from metadata file -- just returns a fresh instance."""
        return cls()

    @classmethod
    def from_file(cls, path: str | Path) -> "TiktokenWrapper":
        """Alias for load() -- keeps backward compatibility with train.py / eval.py."""
        return cls.load(path)


# Public alias -- rest of codebase imports this name
CodeTokenizer = TiktokenWrapper


class CodeDataset(Dataset):
    """
    Loads a .jsonl split and returns tensors ready for both model variants.

    Each item contains:
      instruction_ids  : (max_instruction_len,)  -- tokenized docstring
      context_ids      : (max_context_len,)       -- tokenized signature
      target_ids       : (max_target_len,)        -- tokenized function body
      instruction_mask : (max_instruction_len,)   -- 1 where not padding
      context_mask     : (max_context_len,)       -- 1 where not padding
      target_mask      : (max_target_len,)        -- 1 where not padding

    The baseline model concatenates instruction + context internally.
    The dual encoder uses the fields separately.
    """

    def __init__(
        self,
        path,
        tokenizer: TiktokenWrapper,
        max_instruction_len: int = 128,
        max_context_len: int = 128,
        max_target_len: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_instruction_len = max_instruction_len
        self.max_context_len = max_context_len
        self.max_target_len = max_target_len
        self.examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tok = self.tokenizer

        i_ids = tok.encode(ex["instruction"], self.max_instruction_len, add_bos=True)
        c_ids = tok.encode(ex["context"], self.max_context_len, add_bos=True)
        t_ids = tok.encode(ex["target"], self.max_target_len, add_bos=True, add_eos=True)

        def mask(ids):
            return [0 if i == PAD_ID else 1 for i in ids]

        return {
            "instruction_ids": torch.tensor(i_ids, dtype=torch.long),
            "context_ids": torch.tensor(c_ids, dtype=torch.long),
            "target_ids": torch.tensor(t_ids, dtype=torch.long),
            "instruction_mask": torch.tensor(mask(i_ids), dtype=torch.long),
            "context_mask": torch.tensor(mask(c_ids), dtype=torch.long),
            "target_mask": torch.tensor(mask(t_ids), dtype=torch.long),
            "entry_point": ex.get("entry_point", ""),
            "tests": ex.get("tests", []),
            "id": ex.get("id", str(idx)),
        }
