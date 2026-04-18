"""
prepare.py -- Download HumanEval + MBPP and write tokenizer metadata.

With tiktoken there is no training step -- the tokenizer is pretrained.
This script only downloads the datasets and writes a vocab.json metadata file
so the rest of the codebase knows which tokenizer is in use.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).parent
sys.path.insert(0, str(OUT_DIR.parent))

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"


def parse_humaneval(raw: dict) -> dict | None:
    prompt = raw["prompt"]
    doc_match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if not doc_match:
        doc_match = re.search(r"'''(.*?)'''", prompt, re.DOTALL)
    if not doc_match:
        return None
    docstring = doc_match.group(1).strip()
    sig_match = re.match(r"(def\s+\w+\([^)]*\)\s*(?:->\s*\S+)?\s*:)", prompt)
    context = sig_match.group(1) if sig_match else prompt.split("\n")[0]
    target = raw["canonical_solution"].strip()
    return {
        "id": raw["task_id"],
        "source": "humaneval",
        "instruction": docstring,
        "context": context,
        "target": target,
        "entry_point": raw["entry_point"],
        "tests": [raw["test"]],
    }


def parse_mbpp(raw: dict) -> dict | None:
    code = raw.get("code", "")
    text = raw.get("text", "").strip()
    if not code or not text:
        return None
    sig_match = re.match(r"(def\s+\w+\([^)]*\)\s*(?:->\s*\S+)?\s*:)", code)
    if not sig_match:
        return None
    context = sig_match.group(1)
    body_start = code.find("\n", code.find(context)) + 1
    target = code[body_start:].strip()
    if not target:
        return None
    ep_match = re.search(r"def\s+(\w+)", context)
    return {
        "id": f"mbpp_{raw['task_id']}",
        "source": "mbpp",
        "instruction": text,
        "context": context,
        "target": target,
        "entry_point": ep_match.group(1) if ep_match else "",
        "tests": raw.get("test_list", []),
    }


def download_humaneval() -> list[dict]:
    import gzip, io
    print("Downloading HumanEval...")
    with urllib.request.urlopen(HUMANEVAL_URL) as r:
        compressed = r.read()
    with gzip.open(io.BytesIO(compressed)) as f:
        lines = f.read().decode().strip().split("\n")
    raw = [json.loads(l) for l in lines if l.strip()]
    return [p for p in (parse_humaneval(r) for r in raw) if p]


def download_mbpp() -> list[dict]:
    print("Downloading MBPP...")
    with urllib.request.urlopen(MBPP_URL) as r:
        lines = r.read().decode().strip().split("\n")
    raw = [json.loads(l) for l in lines if l.strip()]
    return [p for p in (parse_mbpp(r) for r in raw) if p]


def split(examples, val_ratio=0.1, test_ratio=0.1, seed=42):
    import random
    rng = random.Random(seed)
    data = examples[:]
    rng.shuffle(data)
    n = len(data)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    return {
        "train": data[:n - n_val - n_test],
        "val":   data[n - n_val - n_test:n - n_test],
        "test":  data[n - n_test:],
    }


def main():
    examples = []

    try:
        examples += download_humaneval()
        print(f"  HumanEval: {len([e for e in examples if e['source']=='humaneval'])} examples")
    except Exception as e:
        print(f"  HumanEval failed: {e}")

    try:
        mbpp = download_mbpp()
        examples += mbpp
        print(f"  MBPP: {len(mbpp)} examples")
    except Exception as e:
        print(f"  MBPP failed: {e}")

    if not examples:
        print("No data downloaded.")
        sys.exit(1)

    print(f"\nTotal: {len(examples)} examples")

    splits = split(examples)
    for name, data in splits.items():
        path = OUT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} -> {path}")

    # Write tokenizer metadata -- tiktoken needs no training
    from models.tokenizer import TiktokenWrapper, VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID, SEP_ID
    tok = TiktokenWrapper()

    # Sanity check
    sample = "def add(a, b):\n    return a + b"
    ids = tok.encode(sample, 64)
    decoded = tok.decode(ids)
    print(f"\nTiktoken sanity check:")
    print(f"  Original: {repr(sample)}")
    print(f"  Token ids (first 10): {[i for i in ids if i != PAD_ID][:10]}")
    print(f"  Decoded:  {repr(decoded)}")
    print(f"  Lossless: {sample.strip() == decoded.strip()}")

    vocab_path = OUT_DIR / "vocab.json"
    tok.save(vocab_path)
    print(f"  Tokenizer: tiktoken ({VOCAB_SIZE} tokens) -> {vocab_path}")

    stats_path = OUT_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "total": len(examples),
            "splits": {k: len(v) for k, v in splits.items()},
            "vocab_size": VOCAB_SIZE,
            "tokenizer": "tiktoken_cl100k_base",
            "sources": {
                "humaneval": len([e for e in examples if e["source"] == "humaneval"]),
                "mbpp":      len([e for e in examples if e["source"] == "mbpp"]),
            }
        }, f, indent=2)

    print("\nDone. Next:")
    print("  python3 train.py --model baseline --epochs 60 --batch_size 8")
    print("  python3 train.py --model dual     --epochs 60 --batch_size 8")


if __name__ == "__main__":
    main()
