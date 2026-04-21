"""
eval.py -- Evaluation for Stage 3: pass@k and geometry analysis.

Usage:
  python3 eval.py --model baseline
  python3 eval.py --model dual
  python3 eval.py --compare --geometry
  python3 eval.py --model dual --sample 5
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

from models.tokenizer import CodeT5Tokenizer, CodeDataset
from models.baseline_codet5 import BaselineCodeT5
from models.dual_encoder_codet5 import DualEncoderCodeT5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "dual"], default=None)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--checkpoint_dir", type=str, default=str(ROOT / "checkpoints"))
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--geometry", action="store_true")
    p.add_argument("--sample", type=int, default=0)
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


def load_model(model_type: str, checkpoint_dir: Path, tokenizer, device):
    ckpt_path = checkpoint_dir / model_type / "best.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print(f"Run first: python3 train.py --model {model_type}")
        return None

    if model_type == "baseline":
        model = BaselineCodeT5(tokenizer)
    else:
        model = DualEncoderCodeT5(tokenizer)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_type} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return model


def execute_tests(generated_code: str, entry_point: str, tests: list[str]) -> bool:
    if not tests:
        return False
    test_code = textwrap.dedent(f"""
import sys
try:
{textwrap.indent(generated_code, '    ')}
except Exception as e:
    sys.exit(1)
errors = []
{chr(10).join(f'''
try:
    {t}
except Exception as e:
    errors.append(str(e))
''' for t in tests)}
sys.exit(1 if errors else 0)
""")
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def evaluate_pass_at_k(model, model_type, dataset, tokenizer, device,
                        k=1, max_new_tokens=128, batch_size=2, n_samples=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    results = []
    total = passed = 0

    bos_id = tokenizer.bos_token_id or tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    print(f"\nEvaluating {model_type} (pass@{k})...")

    for batch in loader:
        batch_t = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            generated = model.generate(
                instruction_ids=batch_t["instruction_ids"],
                context_ids=batch_t["context_ids"],
                instruction_mask=batch_t["instruction_mask"],
                context_mask=batch_t["context_mask"],
                max_new_tokens=max_new_tokens,
                bos_id=bos_id,
                eos_id=eos_id,
            )

        for i in range(generated.size(0)):
            gen_text = tokenizer.decode(generated[i].tolist())
            ctx_ids  = batch_t["context_ids"][i].tolist()
            ctx      = tokenizer.decode(ctx_ids)
            ctx_clean = ctx.rstrip(":")

            # Normalize indentation -- ensure every line of generated body
            # has exactly 4 spaces. The model generates correct logic but
            # indentation may vary due to tokenization artifacts.
            gen_lines = gen_text.split("\n")
            normalized = []
            for line in gen_lines:
                stripped = line.lstrip()
                if stripped:
                    # Preserve relative indentation -- detect base indent
                    orig_indent = len(line) - len(stripped)
                    # Map to multiples of 4 spaces, minimum 4
                    level = max(1, round(orig_indent / 4)) if orig_indent > 0 else 1
                    normalized.append("    " * level + stripped)
                else:
                    normalized.append("")
            gen_normalized = "\n".join(normalized)
            full_code = f"{ctx_clean}:\n{gen_normalized}"

            ep    = batch["entry_point"][i] if isinstance(batch["entry_point"], list) else ""
            tests = batch["tests"][i] if isinstance(batch["tests"], list) else []
            ok    = execute_tests(full_code, ep, tests)

            total += 1
            if ok:
                passed += 1

            results.append({
                "id":          batch["id"][i] if isinstance(batch["id"], list) else str(i),
                "instruction": tokenizer.decode(batch_t["instruction_ids"][i].tolist()),
                "generated":   gen_text,
                "full_code":   full_code,
                "passed":      ok,
            })

    pass_rate = passed / max(total, 1)
    print(f"  pass@{k}: {passed}/{total} = {pass_rate:.3f} ({pass_rate*100:.1f}%)")

    if n_samples > 0:
        print(f"\n  --- {n_samples} sample generations ---")
        for r in results[:n_samples]:
            print(f"\n  [{r['id']}]")
            print(f"  Instruction: {r['instruction'][:80].strip()!r}")
            print(f"  Generated:")
            for line in r["generated"][:200].split("\n")[:8]:
                print(f"    {line}")
            print(f"  Passed: {r['passed']}")
        print(f"  --- end samples ---\n")

    return {"model": model_type, "pass_at_k": pass_rate, "passed": passed,
            "total": total, "k": k, "results": results}


def analyze_geometry(model, model_type, dataset, device):
    """Stage 1 geometry analysis applied to Stage 3 trained models."""
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    i_vecs, c_vecs = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            embs = model.get_encoder_embeddings(
                batch["instruction_ids"], batch["context_ids"],
                batch["instruction_mask"], batch["context_mask"],
            )
            i_vecs.append(embs["instruction"].cpu())
            c_vecs.append(embs["context"].cpu())

    i_all = torch.cat(i_vecs)
    c_all = torch.cat(c_vecs)

    def cos_sim(a, b):
        a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (a * b).sum(dim=-1)

    cross = cos_sim(i_all, c_all).mean().item()

    n = min(len(i_all), 64)
    within_i, within_c = [], []
    for j in range(n):
        for l in range(j + 1, n):
            within_i.append(cos_sim(i_all[j:j+1], i_all[l:l+1]).item())
            within_c.append(cos_sim(c_all[j:j+1], c_all[l:l+1]).item())

    avg_wi = sum(within_i) / max(len(within_i), 1)
    avg_wc = sum(within_c) / max(len(within_c), 1)
    separation_score = (1.0 - cross) / 2.0
    cohesion_score   = (avg_wi + avg_wc) / 2

    print(f"\n  Geometry [{model_type}]:")
    print(f"    Cross-similarity (I<->C): {cross:.4f}")
    print(f"    Within instruction:       {avg_wi:.4f}")
    print(f"    Within context:           {avg_wc:.4f}")
    print(f"    Separation score:         {separation_score:.4f}  (0=overlapping, 1=separated)")
    print(f"    Cohesion score:           {cohesion_score:.4f}")

    return {"model": model_type, "cross_similarity": cross,
            "within_instruction": avg_wi, "within_context": avg_wc,
            "separation_score": separation_score, "cohesion_score": cohesion_score}


def run_eval(model_type, args, device, tokenizer, test_ds):
    model = load_model(model_type, Path(args.checkpoint_dir), tokenizer, device)
    if model is None:
        return {}

    result = evaluate_pass_at_k(
        model, model_type, test_ds, tokenizer, device,
        k=args.k, max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size, n_samples=args.sample,
    )

    if args.geometry:
        result["geometry"] = analyze_geometry(model, model_type, test_ds, device)

    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading CodeT5+ tokenizer...")
    tokenizer = CodeT5Tokenizer()
    test_ds   = CodeDataset(Path(args.data_dir) / "test.jsonl", tokenizer)
    print(f"Test set: {len(test_ds)} examples")

    results = {}

    if args.compare:
        for mtype in ["baseline", "dual"]:
            r = run_eval(mtype, args, device, tokenizer, test_ds)
            if r:
                results[mtype] = r

        if len(results) == 2:
            b, d = results["baseline"], results["dual"]
            print("\n" + "-" * 60)
            print("COMPARATIVE RESULTS -- STAGE 3")
            print("-" * 60)
            print(f"\n  pass@{args.k}:")
            print(f"    baseline:     {b['pass_at_k']:.3f} ({b['passed']}/{b['total']})")
            print(f"    dual_encoder: {d['pass_at_k']:.3f} ({d['passed']}/{d['total']})")
            delta = d['pass_at_k'] - b['pass_at_k']
            sign  = "+" if delta >= 0 else ""
            print(f"    delta:        {sign}{delta:.3f} ({sign}{delta*100:.1f}%)")

            if args.geometry and "geometry" in b and "geometry" in d:
                bg, dg = b["geometry"], d["geometry"]
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
        results[args.model] = run_eval(args.model, args, device, tokenizer, test_ds)
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