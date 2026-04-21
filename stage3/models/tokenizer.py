"""
tokenizer.py -- CodeT5+ tokenizer wrapper + CodeDataset.

Wraps the CodeT5+ tokenizer with the same interface as Stage 2,
so train.py and eval.py work without changes.

CodeT5+ uses a SentencePiece-based tokenizer trained on code.
It handles Python keywords, operators, and indentation correctly.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

CHECKPOINT = "Salesforce/codet5p-220m-py"


class CodeT5Tokenizer:
    """
    Wraps CodeT5+ tokenizer with the same interface as Stage 2 tokenizers.
    save() / load() / from_file() write metadata only -- no vocab serialization needed.

    CodeT5+ uses a RoBERTa-based tokenizer.
    use_fast=False avoids a tokenizers library bug with added_tokens in transformers 5.x.
    """

    def __init__(self):
        self._tok = RobertaTokenizer.from_pretrained(CHECKPOINT, use_fast=False)
        self.vocab_size = self._tok.vocab_size
        self.pad_token_id = self._tok.pad_token_id or 0
        self.bos_token_id = self._tok.bos_token_id or self._tok.pad_token_id
        self.eos_token_id = self._tok.eos_token_id
        self.sep_token_id = self._tok.sep_token_id or self._tok.eos_token_id

    def encode(
        self,
        text: str,
        max_len: int,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text to token ids, truncate and pad to max_len."""
        ids = self._tok.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len - int(add_bos) - int(add_eos),
        )

        if add_bos and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]

        ids = ids[:max_len]
        ids += [self.pad_token_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token ids back to string."""
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def save(self, path: str | Path):
        """Write metadata file -- tokenizer itself needs no serialization."""
        meta = {
            "tokenizer": "codet5plus",
            "checkpoint": CHECKPOINT,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CodeT5Tokenizer":
        """Load from metadata file -- just returns a fresh instance."""
        return cls()

    @classmethod
    def from_file(cls, path: str | Path) -> "CodeT5Tokenizer":
        """Alias for load() -- backward compatibility."""
        return cls.load(path)


class CodeDataset(Dataset):
    """
    Loads a .jsonl split and returns tensors ready for both model variants.

    Each item contains:
      instruction_ids  : (max_instruction_len,)
      context_ids      : (max_context_len,)
      target_ids       : (max_target_len,)
      instruction_mask : (max_instruction_len,)  -- 1 where not padding
      context_mask     : (max_context_len,)
      target_mask      : (max_target_len,)
    """

    def __init__(
        self,
        path,
        tokenizer: CodeT5Tokenizer,
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

        pad = tok.pad_token_id

        def mask(ids):
            return [0 if i == pad else 1 for i in ids]

        return {
            "instruction_ids": torch.tensor(i_ids, dtype=torch.long),
            "context_ids":     torch.tensor(c_ids, dtype=torch.long),
            "target_ids":      torch.tensor(t_ids, dtype=torch.long),
            "instruction_mask": torch.tensor(mask(i_ids), dtype=torch.long),
            "context_mask":    torch.tensor(mask(c_ids), dtype=torch.long),
            "target_mask":     torch.tensor(mask(t_ids), dtype=torch.long),
            "entry_point": ex.get("entry_point", ""),
            "tests":       ex.get("tests", []),
            "id":          ex.get("id", str(idx)),
        }