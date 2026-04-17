"""
eval.py -- Evaluation: pass@k and geometry analysis for both model variants.

Usage:
  python3 eval.py --model baseline
  python3 eval.py --model dual
  python3 eval.py --compare --geometry   # full comparative report
"""

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.tokenizer import BPETokenizer, CodeDataset
from models.baseline import BaselineModel, BaselineConfig
from models.dual_encoder import DualEncoderModel, DualEncoderConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "dual"], default=None)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--checkpoint_dir", type=str, default=str(ROOT / "checkpoints"))
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--geometry", action="store_true", help="Also run geometry analysis")
    return p.parse_args()


def collate_fn(batch):
    tensor_keys = ["instruction_ids", "context_ids", "target_ids",
                   "instruction_mask", "context_mask", "target_mask"]
    out = {}
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch])
    for k in ["entry_point", "tests", "id"]:
        out[k] = [b[k] for b in batch]
    return out


def load_model(model_type: str, checkpoint_dir: Path, device: torch.device):
    ckpt_path = checkpoint_dir / model_type / "best.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print(f"Run first: python3 train.py --model {model_type}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["config"]

    if model_type == "baseline":
        config = BaselineConfig(**cfg_dict)
        model = BaselineModel(config)
    else:
        config = DualEncoderConfig(**cfg_dict)
        model = DualEncoderModel(config)

    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_type} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return model


def execute_tests(generated_code: str, entry_point: str, tests: list[str]) -> bool:
    """Run test assertions in an isolated subprocess. Returns True if all pass."""
    if not tests:
        return False

    test_code = textwrap.dedent(f"""
import sys
try:
{textwrap.indent(generated_code, '    ')}
except Exception as e:
    print(f"COMPILE_ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)

errors = []
{chr(10).join(f'''
try:
    {t}
except Exception as e:
    errors.append(f"FAIL: {t} -> {{e}}")
''' for t in tests)}

if errors:
    for e in errors:
        print(e, file=sys.stderr)
    sys.exit(1)
sys.exit(0)
""")

    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def evaluate_pass_at_k(
    model,
    model_type: str,
    dataset: CodeDataset,
    tokenizer: BPETokenizer,
    device: torch.device,
    k: int = 1,
    max_new_tokens: int = 128,
    batch_size: int = 4,
) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    results = []
    total = 0
    passed = 0

    print(f"\nEvaluating {model_type} (pass@{k})...")

    for batch in loader:
        batch_tensors = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            generated = model.generate(
                instruction_ids=batch_tensors["instruction_ids"],
                context_ids=batch_tensors["context_ids"],
                instruction_mask=batch_tensors["instruction_mask"],
                context_mask=batch_tensors["context_mask"],
                max_new_tokens=max_new_tokens,
            )

        for i in range(generated.size(0)):
            gen_ids = generated[i].tolist()
            gen_text = tokenizer.decode(gen_ids)

            # Reconstruct function signature from context_ids
            ctx_ids = batch_tensors["context_ids"][i].tolist()
            ctx = tokenizer.decode(ctx_ids)
            full_code = f"{ctx}:\n    {gen_text}"

            entry_point = batch["entry_point"][i] if isinstance(batch["entry_point"], list) else ""
            tests = batch["tests"][i] if isinstance(batch["tests"], list) else []

            ok = execute_tests(full_code, entry_point, tests)
            total += 1
            if ok:
                passed += 1

            results.append({
                "id": batch["id"][i] if isinstance(batch["id"], list) else str(i),
                "generated": gen_text,
                "passed": ok,
            })

    pass_rate = passed / max(total, 1)
    print(f"  pass@{k}: {passed}/{total} = {pass_rate:.3f} ({pass_rate*100:.1f}%)")

    return {
        "model": model_type,
        "pass_at_k": pass_rate,
        "passed": passed,
        "total": total,
        "k": k,
        "results": results,
    }


def analyze_geometry(model, model_type: str, dataset: CodeDataset, device: torch.device) -> dict:
    """
    Run Stage 1 geometry analysis on the trained model's embeddings.

    Measures whether instruction/context separation increased after training.
    Key metric: separation_score = (1 - cross_similarity) / 2
      0.0 = fully overlapping spaces
      1.0 = perfectly separated spaces
    """
    def collate_fn_geo(batch):
        tensor_keys = ["instruction_ids", "context_ids", "target_ids",
                       "instruction_mask", "context_mask", "target_mask"]
        out = {}
        for k in tensor_keys:
            out[k] = torch.stack([b[k] for b in batch])
        for k in ["entry_point", "tests", "id"]:
            out[k] = [b[k] for b in batch]
        return out

    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_geo)

    i_vecs, c_vecs = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            embs = model.get_encoder_embeddings(
                batch["instruction_ids"],
                batch["context_ids"],
                batch["instruction_mask"],
                batch["context_mask"],
            )
            i_vecs.append(embs["instruction"].cpu())
            c_vecs.append(embs["context"].cpu())

    i_all = torch.cat(i_vecs)
    c_all = torch.cat(c_vecs)

    def cos_sim(a, b):
        a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (a * b).sum(dim=-1)

    # Cross-group similarity (instruction vs context, pairwise)
    cross = cos_sim(i_all, c_all).mean().item()

    # Within-group similarity (subsample to avoid O(n^2))
    n = min(len(i_all), 64)
    i_sub = i_all[:n]
    c_sub = c_all[:n]

    within_i, within_c = [], []
    for j in range(n):
        for l in range(j + 1, n):
            within_i.append(cos_sim(i_sub[j:j+1], i_sub[l:l+1]).item())
            within_c.append(cos_sim(c_sub[j:j+1], c_sub[l:l+1]).item())

    avg_wi = sum(within_i) / max(len(within_i), 1)
    avg_wc = sum(within_c) / max(len(within_c), 1)
    avg_within = (avg_wi + avg_wc) / 2

    # Normalized separation score: 0 = overlapping, 1 = orthogonal
    separation_score = (1.0 - cross) / 2.0
    cohesion_score = avg_within

    print(f"\n  Geometry [{model_type}]:")
    print(f"    Cross-similarity (I<->C): {cross:.4f}")
    print(f"    Within instruction:       {avg_wi:.4f}")
    print(f"    Within context:           {avg_wc:.4f}")
    print(f"    Separation score:         {separation_score:.4f}  (0=overlapping, 1=separated)")
    print(f"    Cohesion score:           {cohesion_score:.4f}  (internal cluster cohesion)")

    return {
        "model": model_type,
        "cross_similarity": cross,
        "within_instruction": avg_wi,
        "within_context": avg_wc,
        "separation_score": separation_score,
        "cohesion_score": cohesion_score,
    }


def run_eval(model_type: str, args, device: torch.device, tokenizer: BPETokenizer, test_ds: CodeDataset) -> dict:
    model = load_model(model_type, Path(args.checkpoint_dir), device)
    if model is None:
        return {}

    result = evaluate_pass_at_k(
        model, model_type, test_ds, tokenizer, device,
        k=args.k, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
    )

    if args.geometry:
        result["geometry"] = analyze_geometry(model, model_type, test_ds, device)

    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    data_dir = Path(args.data_dir)
    tokenizer = BPETokenizer.from_file(data_dir / "vocab.json")
    test_ds = CodeDataset(data_dir / "test.jsonl", tokenizer)
    print(f"Test set: {len(test_ds)} examples")

    results = {}

    if args.compare:
        for mtype in ["baseline", "dual"]:
            r = run_eval(mtype, args, device, tokenizer, test_ds)
            if r:
                results[mtype] = r

        if len(results) == 2:
            print("\n" + "-" * 60)
            print("COMPARATIVE RESULTS")
            print("-" * 60)

            b = results["baseline"]
            d = results["dual"]

            print(f"\n  pass@{args.k}:")
            print(f"    baseline:     {b['pass_at_k']:.3f} ({b['passed']}/{b['total']})")
            print(f"    dual_encoder: {d['pass_at_k']:.3f} ({d['passed']}/{d['total']})")
            delta = d['pass_at_k'] - b['pass_at_k']
            sign = "+" if delta >= 0 else ""
            print(f"    delta:        {sign}{delta:.3f} ({sign}{delta*100:.1f}%)")

            if args.geometry and "geometry" in b and "geometry" in d:
                bg = b["geometry"]
                dg = d["geometry"]
                print(f"\n  Geometric separation (0=overlapping, 1=separated):")
                print(f"    baseline:     {bg['separation_score']:.4f}  (cross={bg['cross_similarity']:.4f})")
                print(f"    dual_encoder: {dg['separation_score']:.4f}  (cross={dg['cross_similarity']:.4f})")
                delta_sep = dg["separation_score"] - bg["separation_score"]
                sign = "+" if delta_sep >= 0 else ""
                print(f"    delta:        {sign}{delta_sep:.4f}")
                print(f"\n  Internal cohesion:")
                print(f"    baseline:     {bg['cohesion_score']:.4f}")
                print(f"    dual_encoder: {dg['cohesion_score']:.4f}")
                if delta_sep > 0.1:
                    print("\n  Geometric separation higher in dual encoder -- hypothesis supported")
                else:
                    print("\n  Similar separation -- investigate gate architecture")

    elif args.model:
        r = run_eval(args.model, args, device, tokenizer, test_ds)
        results[args.model] = r
    else:
        print("Use --model baseline, --model dual, or --compare")
        sys.exit(1)

    out_path = ROOT / "results" / "eval_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        slim = {k: {kk: vv for kk, vv in v.items() if kk != "results"} for k, v in results.items()}
        json.dump(slim, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
