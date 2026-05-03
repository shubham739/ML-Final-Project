import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.model import GPT, ModelConfig


BOS_ID = 1
EOS_ID = 2
SVG_PREFIX = "<svg"


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> tuple[GPT, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    config = ModelConfig.from_dict(cfg_dict)
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg_dict


def ids_to_svg(ids: list[int], tokenizer: Tokenizer) -> str:
    # Strip BOS/EOS/PAD before decoding
    clean = [i for i in ids if i not in (BOS_ID, EOS_ID, 0)]
    return tokenizer.decode(clean)


def generate_sample(
    model: GPT,
    tokenizer: Tokenizer,
    device: torch.device,
    prefix: str = None,
    max_new_tokens: int = 900,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
) -> str:
    """Generate one SVG string. Returns raw SVG text."""
    if prefix:
        enc = tokenizer.encode(prefix)
        prompt_ids = [BOS_ID] + enc.ids
    else:
        # Unconditional: start with bos + "<svg"
        svg_enc = tokenizer.encode(SVG_PREFIX)
        prompt_ids = [BOS_ID] + svg_enc.ids

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=EOS_ID,
        )

    generated_ids = out[0].tolist()
    # Remove the prompt prefix from output for prefix-conditioned case
    new_ids = generated_ids[len(prompt_ids):]
    if prefix:
        return prefix + ids_to_svg(new_ids, tokenizer)
    else:
        return ids_to_svg(generated_ids, tokenizer)


def run_generation(
    checkpoint_path: str,
    tokenizer_path: str = "tokenizer/tokenizer.json",
    output_dir: str = "outputs",
    n_samples: int = 10,
    prefixes: list[str] = None,
    max_new_tokens: int = 900,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = None,
    device_str: str = None,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    device = get_device(device_str)

    print(f"Loading model from: {checkpoint_path}")
    model, cfg = load_model(checkpoint_path, device)
    n_params = model.count_parameters()
    print(f"Model: {cfg.get('name','?')}  ({n_params/1e6:.1f}M params)  device={device}")
    print(f"Sampling: temperature={temperature}  top_k={top_k}  top_p={top_p}")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    out_dir = Path(output_dir) / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # --- Unconditional samples ---
    print(f"\nGenerating {n_samples} unconditional samples...")
    for i in range(n_samples):
        t0 = time.time()
        svg = generate_sample(
            model, tokenizer, device,
            prefix=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        elapsed = time.time() - t0
        fname = f"unconditional_{i+1:02d}.svg"
        (out_dir / fname).write_text(svg)
        print(f"  [{i+1}/{n_samples}] {len(svg)} chars  ({elapsed:.1f}s)  → {fname}")
        results.append({
            "type": "unconditional",
            "index": i + 1,
            "filename": fname,
            "length_chars": len(svg),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })

    # --- Prefix-conditioned samples ---
    if prefixes:
        print(f"\nGenerating {len(prefixes)} prefix-conditioned samples...")
        for i, prefix in enumerate(prefixes):
            t0 = time.time()
            svg = generate_sample(
                model, tokenizer, device,
                prefix=prefix,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            elapsed = time.time() - t0
            fname = f"prefix_{i+1:02d}.svg"
            (out_dir / fname).write_text(svg)
            print(f"  [prefix {i+1}] {len(prefix)} → {len(svg)} chars  ({elapsed:.1f}s)  → {fname}")
            results.append({
                "type": "prefix_conditioned",
                "index": i + 1,
                "filename": fname,
                "prefix": prefix,
                "length_chars": len(svg),
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            })

    # Save metadata
    meta = {
        "checkpoint": checkpoint_path,
        "model_name": cfg.get("name", "?"),
        "param_count": n_params,
        "n_unconditional": n_samples,
        "n_prefix_conditioned": len(prefixes) if prefixes else 0,
        "settings": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
        },
        "samples": results,
    }
    meta_path = Path(output_dir) / "results" / "generation_results.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved: {meta_path}")
    print(f"SVG files in:   {out_dir}/")
    return meta


# Default prefixes for Task 4.3
DEFAULT_PREFIXES = [
    # Partial circle (face) — does model add features?
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="none" stroke="black" stroke-width="2"/>',
    # Open path — does model close it?
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M4 4 L20 4 L20 20"',
    # Group with one rect — does model add related shapes?
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><g><rect x="2" y="2" width="8" height="8" fill="steelblue"/>',
    # Arrow shaft — does model complete the arrowhead?
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><line x1="2" y1="12" x2="18" y2="12" stroke="black" stroke-width="2"/>',
    # Half of a symmetric icon — does model mirror it?
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 2 L12 22 M12 2 L4 8"',
]


def main():
    p = argparse.ArgumentParser(description="Generate SVG samples from a trained model.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint_final.pt")
    p.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--n_samples", type=int, default=10,
                   help="Number of unconditional samples")
    p.add_argument("--n_prefix", type=int, default=5,
                   help="Number of prefix-conditioned samples (uses built-in prefixes)")
    p.add_argument("--prefix", default=None,
                   help="Custom single prefix string (overrides built-in prefixes)")
    p.add_argument("--max_new_tokens", type=int, default=900)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.prefix:
        prefixes = [args.prefix]
    else:
        prefixes = DEFAULT_PREFIXES[:args.n_prefix]

    run_generation(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        prefixes=prefixes,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
