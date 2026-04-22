"""
train.py -- Training loop for Stage 3: CodeT5+ dual encoder experiment.

Usage:
  python3 train.py --model baseline  --epochs 20 --batch_size 4
  python3 train.py --model dual      --epochs 20 --batch_size 4

Key differences from Stage 2:
  - Models are loaded from CodeT5+ pretrained weights
  - Tokenizer is CodeT5+ (not corpus BPE or tiktoken)
  - batch_size=4 default (larger models, less VRAM headroom)
  - epochs=20 default (pretrained model converges faster)
  - Learning rate is lower (1e-4) to avoid destroying pretrained weights
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.tokenizer import CodeT5Tokenizer, CodeDataset
from models.baseline_codet5 import BaselineCodeT5
from models.dual_encoder_codet5 import DualEncoderCodeT5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "dual"], required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)      # lower than stage2 -- pretrained weights
    p.add_argument("--grad_accum", type=int, default=8)   # higher -- smaller batch
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--checkpoint_dir", type=str, default=str(ROOT / "checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_decoder", action="store_true",
                   help="Freeze decoder weights -- train only encoder and gate")
    p.add_argument("--decoder_lr_factor", type=float, default=0.3,
                   help="Decoder LR as fraction of base LR (default: 0.3). "
                        "0.1=too slow, 1.0=destabilizes, 0.3=recommended sweet spot")
    p.add_argument("--gate_lr", type=float, default=1e-3,
                   help="Constant gate LR, no decay (default: 1e-3)")
    return p.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = max(0.0, min(1.0, progress))
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    labels = targets[:, 1:]
    mask   = target_mask[:, 1:].bool()
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    )
    loss = loss.reshape(labels.shape)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def collate_fn(batch):
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

    # Tokenizer
    print("\nLoading CodeT5+ tokenizer...")
    tokenizer = CodeT5Tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Data
    data_dir = Path(args.data_dir)
    if not (data_dir / "train.jsonl").exists():
        print(f"\nData not found at {data_dir}")
        print("Run first: python3 data/prepare.py")
        sys.exit(1)

    train_ds = CodeDataset(data_dir / "train.jsonl", tokenizer)
    val_ds   = CodeDataset(data_dir / "val.jsonl",   tokenizer)
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
    print(f"\nLoading CodeT5+ {args.model} model...")
    if args.model == "baseline":
        model = BaselineCodeT5(tokenizer)
    else:
        model = DualEncoderCodeT5(tokenizer)

    if args.freeze_decoder:
        for p in model.decoder.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False
        print("Decoder frozen -- training encoder(s) and gate only")

    model = model.to(device)
    print(f"Model: {args.model} | Params: {model.param_count():,}")

    # Separate optimizer groups for dual encoder.
    # The gate starts from random init and needs a CONSTANT LR --
    # the cosine decay schedule reaches near-zero by epoch 3, which is
    # too small for the gate to receive meaningful gradient updates.
    # Pretrained encoders/decoder use the decaying schedule as normal.
    if args.model == "dual" and hasattr(model, "gate"):
        # Three-group optimizer for dual encoder:
        #
        # 1. Encoders: base LR with cosine decay
        #    -- preserve pretrained representations while allowing divergence
        #
        # 2. Gate: constant LR (no decay)
        #    -- gate starts from scratch and needs sustained signal
        #    -- cosine decay kills gate gradients by epoch 3
        #
        # 3. Decoder + lm_head: 0.1x base LR with cosine decay
        #    -- decoder was pretrained to decode single-encoder output
        #    -- now receives gate output (distribution shift)
        #    -- low LR lets it adapt gradually without losing pretrained knowledge
        #    -- this prevents empty generation (decoder outputting only padding)

        GATE_LR    = args.gate_lr                        # constant -- no decay
        DECODER_LR = args.lr * args.decoder_lr_factor   # fraction of base LR

        gate_ids   = {id(p) for p in model.gate.parameters()}
        shared_ids = {id(p) for p in model.shared.parameters()}

        def excl(module, *excl_id_sets):
            ids = set().union(*excl_id_sets)
            return [p for p in module.parameters() if id(p) not in ids]

        # Encoder optimizer (cosine decay)
        encoder_optimizer = torch.optim.AdamW([
            {"params": excl(model.instruction_encoder, gate_ids, shared_ids)},
            {"params": excl(model.context_encoder,     gate_ids, shared_ids)},
            {"params": list(model.shared.parameters())},
        ], lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        # Gate optimizer (constant LR)
        gate_optimizer = torch.optim.AdamW(
            model.gate.parameters(),
            lr=GATE_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )

        # Decoder optimizer (low LR with cosine decay)
        decoder_optimizer = torch.optim.AdamW([
            {"params": excl(model.decoder,  gate_ids, shared_ids)},
            {"params": excl(model.lm_head,  gate_ids, shared_ids)},
        ], lr=DECODER_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        optimizer = (encoder_optimizer, gate_optimizer, decoder_optimizer)
        print(f"Dual encoder optimizers:")
        print(f"  encoder_lr     = {args.lr:.0e} (cosine decay)")
        print(f"  gate_lr        = {GATE_LR:.0e} (constant, no decay)")
        print(f"  decoder_lr     = {DECODER_LR:.0e} (cosine decay, {args.decoder_lr_factor}x base)")
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )

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
        optimizers = optimizer if isinstance(optimizer, tuple) else (optimizer,)
        for opt in optimizers:
            opt.zero_grad()
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            global_step += 1
            lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr)
            decoder_lr = get_lr(
                global_step,
                args.warmup_steps,
                total_steps,
                args.lr * args.decoder_lr_factor,
            )

            # Update encoder optimizer (index 0) with cosine-decayed LR
            for pg in optimizers[0].param_groups:
                pg["lr"] = lr
            # Gate optimizer (index 1) keeps constant LR -- no update
            # Decoder optimizer (index 2) uses 0.1x cosine-decayed LR
            if len(optimizers) > 2:
                for pg in optimizers[2].param_groups:
                    pg["lr"] = decoder_lr

            if args.model == "baseline":
                logits = model(batch)
                gate_vals = None
            else:
                logits, gate_vals = model(batch)

            loss = compute_loss(
                logits, batch["target_ids"], batch["target_mask"],
                tokenizer.pad_token_id
            )
            loss = loss / args.grad_accum
            loss.backward()

            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                for opt in optimizers:
                    opt.step()
                for opt in optimizers:
                    opt.zero_grad()

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

                loss = compute_loss(
                    logits, batch["target_ids"], batch["target_mask"],
                    tokenizer.pad_token_id
                )
                val_loss += loss.item()
                n_val += 1

        avg_train = epoch_loss / max(n_batches, 1)
        avg_val   = val_loss   / max(n_val, 1)
        elapsed   = time.time() - epoch_start

        gate_str = ""
        if gate_log:
            avg_gates = [
                sum(b[i] for b in gate_log) / len(gate_log)
                for i in range(len(gate_log[0]))
            ]
            gate_str = f" | gates={[f'{g:.3f}' for g in avg_gates]}"

        lr_str = f"lr={lr:.2e}"
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={avg_train:.4f} | val={avg_val:.4f} | "
            f"{lr_str} | {elapsed:.1f}s{gate_str}"
        )

        record = {"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val, "lr": lr}
        if gate_log:
            record["gate_values"] = avg_gates
        history.append(record)

        # Gate health diagnostic after epoch 3
        # Prints a clear verdict so you can stop early if the gate is frozen
        if epoch == 3 and args.model == "dual" and gate_log:
            gate_movement = max(abs(g - 0.0) for g in avg_gates)
            print(f"  Gate values: {[f'{g:.4f}' for g in avg_gates]}")
            print(f"  Max deviation from 0.0: {gate_movement:.4f}")
            if gate_movement > 0.01:
                print(f"  VERDICT: HEALTHY -- gate is moving, continue training")
            else:
                print(f"  VERDICT: FROZEN -- gate not moving, stop and investigate")
                print(f"  Suggestion: increase GATE_LR or SYMMETRY_BREAK_STD")
            print()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": avg_val,
                "model_type": args.model,
            }, ckpt_dir / "best.pt")
            print(f"  Checkpoint saved (val={avg_val:.4f})")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints at: {ckpt_dir}")
    print(f"\nNext: python3 eval.py --model {args.model}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
