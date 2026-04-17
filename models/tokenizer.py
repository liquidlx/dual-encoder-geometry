"""
tokenizer.py -- BPE tokenizer trained on the project corpus + CodeDataset.

Why BPE instead of the previous simple tokenizer:
  Simple: 'return' -> separate character tokens, model can't reconstruct keywords
  BPE:    'return' -> single learned token, model generates valid Python syntax

BPE (Byte Pair Encoding) learns frequent subword units from the corpus:
  'def '      -> single token
  'return '   -> single token
  'self.'     -> single token
  '__init__'  -> single token

This allows the model to generate syntactically correct Python code.

Special tokens:
  <pad> = 0  -- padding
  <unk> = 1  -- unknown token
  <bos> = 2  -- beginning of sequence
  <eos> = 3  -- end of sequence
  <sep> = 4  -- separator between instruction and context (baseline only)
"""

import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID, SEP_ID = 0, 1, 2, 3, 4


class BPETokenizer:
    """
    BPE tokenizer trained from scratch on the project corpus.

    Training process:
      1. Start with character-level vocabulary
      2. Count adjacent symbol pair frequencies across corpus
      3. Merge the most frequent pair into a new symbol
      4. Repeat until target vocab_size is reached
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}
        self.merges: list[tuple] = []
        self.vocab_size = 0

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 4000, min_freq: int = 2) -> "BPETokenizer":
        """Train BPE on a list of text strings."""
        tok = cls()
        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")

        # Step 1: build character vocabulary
        char_counter: Counter = Counter()
        for text in texts:
            char_counter.update(text)

        vocab = list(SPECIAL_TOKENS)
        for char, _ in char_counter.most_common():
            if char not in vocab:
                vocab.append(char)

        tok.vocab = {s: i for i, s in enumerate(vocab)}
        tok.inv_vocab = {i: s for s, i in tok.vocab.items()}

        # Represent corpus as word frequency map
        word_freqs: Counter = Counter()
        for text in texts:
            for word in re.findall(r"\S+|\s+", text):
                word_freqs[word] += 1

        # Initialize splits as character sequences
        splits = {word: list(word) for word in word_freqs}
        merges = []
        n_merges = vocab_size - len(vocab)

        for step in range(n_merges):
            # Count adjacent pairs weighted by word frequency
            pair_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                syms = splits[word]
                for i in range(len(syms) - 1):
                    pair_freqs[(syms[i], syms[i + 1])] += freq

            if not pair_freqs:
                break

            best_pair, best_freq = pair_freqs.most_common(1)[0]
            if best_freq < min_freq:
                break

            # Merge best pair into new symbol
            new_sym = best_pair[0] + best_pair[1]
            merges.append(best_pair)

            if new_sym not in tok.vocab:
                idx = len(tok.vocab)
                tok.vocab[new_sym] = idx
                tok.inv_vocab[idx] = new_sym

            # Apply merge to all splits
            for word in splits:
                syms = splits[word]
                new_syms = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best_pair:
                        new_syms.append(new_sym)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                splits[word] = new_syms

            if (step + 1) % 500 == 0:
                print(f"  Merge {step+1}/{n_merges} | vocab={len(tok.vocab)}")

        tok.merges = merges
        tok.vocab_size = len(tok.vocab)
        print(f"BPE ready. Vocab size: {tok.vocab_size}")
        return tok

    def _apply_merges(self, chars: list[str]) -> list[str]:
        """Apply learned merges to a character sequence (greedy, priority by merge order)."""
        syms = list(chars)
        merge_set = {pair: i for i, pair in enumerate(self.merges)}
        while True:
            if len(syms) < 2:
                break
            # Find the highest-priority applicable merge
            best_idx = None
            best_pos = None
            for i in range(len(syms) - 1):
                pair = (syms[i], syms[i + 1])
                if pair in merge_set:
                    idx = merge_set[pair]
                    if best_idx is None or idx < best_idx:
                        best_idx = idx
                        best_pos = i
            if best_pos is None:
                break
            new_sym = syms[best_pos] + syms[best_pos + 1]
            syms = syms[:best_pos] + [new_sym] + syms[best_pos + 2:]
        return syms

    def encode(
        self,
        text: str,
        max_len: int,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text to token ids, truncate and pad to max_len."""
        words = re.findall(r"\S+|\s+", text)
        ids = []
        for word in words:
            syms = self._apply_merges(list(word))
            for sym in syms:
                ids.append(self.vocab.get(sym, UNK_ID))
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        ids = ids[:max_len]
        ids += [PAD_ID] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token ids back to string."""
        specials = {PAD_ID, BOS_ID, EOS_ID, SEP_ID}
        parts = []
        for i in ids:
            if skip_special and i in specials:
                continue
            parts.append(self.inv_vocab.get(i, "<unk>"))
        return "".join(parts)

    def save(self, path: str | Path):
        data = {"vocab": self.vocab, "merges": self.merges, "vocab_size": self.vocab_size}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = data["vocab"]
        tok.inv_vocab = {int(i): s for s, i in tok.vocab.items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.vocab_size = data["vocab_size"]
        return tok

    @classmethod
    def from_file(cls, path: str | Path) -> "BPETokenizer":
        """Alias for load() -- kept for backward compatibility."""
        return cls.load(path)


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
        tokenizer: BPETokenizer,
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
