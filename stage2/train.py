"""
train.py -- Shared training loop for both model variants.

Usage:
  python3 train.py --model baseline  --epochs 30 --batch_size 8
  python3 train.py --model dual      --epochs 30 --batch_size 8

The loop is identical for both models. The only difference is the model
instantiated. Any performance difference is attributable to architecture alone.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.tokenizer import TiktokenWrapper as BPETokenizer, CodeDataset
from models.baseline import BaselineModel, BaselineConfig
from models.dual_encoder import DualEncoderModel, DualEncoderConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "dual"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--checkpoint_dir", type=str, default=str(ROOT / "checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", help="Mixed precision (requires CUDA)")
    return p.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """Original Transformer schedule: linear warmup + inverse square root decay."""
    if step == 0:
        return 1e-7
    if step < warmup_steps:
        return (d_model ** -0.5) * step * (warmup_steps ** -1.5)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """Cross-entropy loss ignoring padding positions."""
    labels = targets[:, 1:]
    mask = target_mask[:, 1:].bool()

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    )
    loss = loss.reshape(labels.shape)
    loss = (loss * mask).sum() / mask.sum().clamp(min=1)
    return loss


def collate_fn(batch):
    """Custom collate to keep non-tensor fields as lists."""
    tensor_keys = ["instruction_ids", "context_ids", "target_ids",
                   "instruction_mask", "context_mask", "target_mask"]
    out = {}
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch])
    for k in ["entry_point", "tests", "id"]:
        out[k] = [b[k] for b in batch]
    return out


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    data_dir = Path(args.data_dir)
    vocab_path = data_dir / "vocab.json"

    if not vocab_path.exists():
        print(f"\nVocab not found at {vocab_path}")
        print("Run first: python3 data/prepare.py")
        sys.exit(1)

    tokenizer = BPETokenizer.from_file(vocab_path)
    print(f"Vocabulary: {tokenizer.vocab_size} tokens")

    train_ds = CodeDataset(data_dir / "train.jsonl", tokenizer)
    val_ds = CodeDataset(data_dir / "val.jsonl", tokenizer)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    # Model
    if args.model == "baseline":
        config = BaselineConfig(vocab_size=tokenizer.vocab_size, d_model=args.d_model)
        model = BaselineModel(config)
    else:
        config = DualEncoderConfig(vocab_size=tokenizer.vocab_size, d_model=args.d_model)
        model = DualEncoderModel(config)

    model = model.to(device)
    print(f"\nModel: {args.model} | Params: {model.param_count():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01
    )

    scaler = torch.cuda.amp.GradScaler() if (args.fp16 and device.type == "cuda") else None

    ckpt_dir = Path(args.checkpoint_dir) / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")
    history = []

    total_steps = args.epochs * math.ceil(len(train_loader) / args.grad_accum)
    print(f"Total steps: {total_steps} | Warmup: {args.warmup_steps}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            global_step += 1
            lr = get_lr(global_step, args.d_model, args.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                if args.model == "baseline":
                    logits = model(batch)
                    gate_vals = None
                else:
                    logits, gate_vals = model(batch)

                loss = compute_loss(logits, batch["target_ids"], batch["target_mask"])
                loss = loss / args.grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        gate_log = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                if args.model == "baseline":
                    logits = model(batch)
                else:
                    logits, gate_vals = model(batch)
                    gate_log.append(gate_vals)

                loss = compute_loss(logits, batch["target_ids"], batch["target_mask"])
                val_loss += loss.item()
                n_val += 1

        avg_train = epoch_loss / max(n_batches, 1)
        avg_val = val_loss / max(n_val, 1)
        elapsed = time.time() - epoch_start

        gate_str = ""
        if gate_log:
            avg_gates = [
                sum(b[i] for b in gate_log) / len(gate_log)
                for i in range(len(gate_log[0]))
            ]
            gate_str = f" | gates={[f'{g:.3f}' for g in avg_gates]}"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={avg_train:.4f} | val={avg_val:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s{gate_str}"
        )

        record = {
            "epoch": epoch,
            "train_loss": avg_train,
            "val_loss": avg_val,
            "lr": lr,
            "global_step": global_step,
        }
        if gate_log:
            record["gate_values"] = avg_gates
        history.append(record)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": avg_val,
                "config": config.__dict__,
                "model_type": args.model,
            }, ckpt_path)
            print(f"  Checkpoint saved (val={avg_val:.4f})")

    hist_path = ckpt_dir / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints at: {ckpt_dir}")
    print(f"\nNext: python3 eval.py --model {args.model}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
