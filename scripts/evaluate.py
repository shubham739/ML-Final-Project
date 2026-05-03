import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.model import GPT, ModelConfig
from scripts.train import load_data, compute_val_loss, get_device

def compute_perplexity(
    model: GPT,
    test_path: str,
    device: torch.device,
    block_size: int,
) -> float:
    test_data = load_data(test_path)
    avg_loss = compute_val_loss(model, test_data, block_size, device)
    return math.exp(avg_loss) if not math.isnan(avg_loss) else float("nan")

def check_xml_valid(svg_text: str) -> bool:
    try:
        from lxml import etree
        etree.fromstring(svg_text.encode())
        return True
    except Exception:
        return False

def check_renderable(svg_text: str, out_path: str = None) -> bool:
    try:
        import cairosvg
        png = cairosvg.svg2png(bytestring=svg_text.encode())
        if out_path:
            Path(out_path).write_bytes(png)
        return True
    except Exception:
        return False

def check_structural(svg_text: str) -> dict:
    text = svg_text.strip()
    has_svg_root    = text.startswith("<svg") and "</svg>" in text
    has_viewbox     = "viewBox" in text or "viewbox" in text
    has_closed_tags = text.endswith("</svg>")
    has_path_or_shape = any(f"<{t}" in text for t in
                            ["path", "circle", "rect", "line", "polygon",
                             "polyline", "ellipse", "g"])
    return {
        "has_svg_root":    has_svg_root,
        "has_viewbox":     has_viewbox,
        "has_closed_tags": has_closed_tags,
        "has_shape":       has_path_or_shape,
        "fully_structural": all([has_svg_root, has_closed_tags, has_path_or_shape]),
    }

def evaluate_samples(generated_dir: str, output_dir: str, render_pngs: bool = True) -> dict:
    gen_path = Path(generated_dir)
    svg_files = sorted(gen_path.glob("*.svg"))

    if not svg_files:
        print(f"No .svg files found in {generated_dir}")
        return {}

    print(f"Evaluating {len(svg_files)} SVG files in {generated_dir}...")

    png_dir = Path(output_dir) / "generated_png"
    if render_pngs:
        png_dir.mkdir(parents=True, exist_ok=True)

    xml_valid   = []
    renderable  = []
    structural  = []
    lengths     = []

    for svg_file in svg_files:
        text = svg_file.read_text(errors="replace")
        lengths.append(len(text))

        xv = check_xml_valid(text)
        xml_valid.append(xv)

        png_out = str(png_dir / (svg_file.stem + ".png")) if render_pngs else None
        rv = check_renderable(text, out_path=png_out if xv else None)
        renderable.append(rv)

        sv = check_structural(text)
        structural.append(sv)

        status = ("XML✓" if xv else "XML✗") + " " + ("PNG✓" if rv else "PNG✗")
        print(f"  {svg_file.name:<35} {len(text):>5} chars  {status}")

    n = len(svg_files)
    xml_rate    = sum(xml_valid) / n
    render_rate = sum(renderable) / n
    struct_rate = sum(s["fully_structural"] for s in structural) / n

    print(f"\nResults over {n} samples:")
    print(f"  XML validity rate:    {xml_rate:.1%}  ({sum(xml_valid)}/{n})")
    print(f"  SVG render rate:      {render_rate:.1%}  ({sum(renderable)}/{n})")
    print(f"  Structural validity:  {struct_rate:.1%}  ({sum(s['fully_structural'] for s in structural)}/{n})")
    print(f"  Avg length:           {np.mean(lengths):.0f} chars  "
          f"(min={min(lengths)}, max={max(lengths)})")

    return {
        "n_samples": n,
        "xml_validity_rate": xml_rate,
        "render_rate": render_rate,
        "structural_validity_rate": struct_rate,
        "avg_length_chars": float(np.mean(lengths)),
        "min_length_chars": min(lengths),
        "max_length_chars": max(lengths),
        "per_sample": [
            {
                "filename": f.name,
                "xml_valid": xv,
                "renderable": rv,
                **sv,
            }
            for f, xv, rv, sv in zip(svg_files, xml_valid, renderable, structural)
        ],
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate generated SVG samples.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint_final.pt")
    p.add_argument("--generated_dir", default="outputs/generated",
                   help="Directory containing generated .svg files")
    p.add_argument("--data_dir", default="data/tokenized",
                   help="Directory with train/val/test.npy")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--no_render", action="store_true",
                   help="Skip CairoSVG rendering (faster)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: {config.d_model}d  {config.n_layers}L  "
          f"{model.count_parameters()/1e6:.1f}M params")

    # Perplexity
    test_path = os.path.join(args.data_dir, "test.npy")
    if os.path.exists(test_path):
        print("\nComputing test set perplexity...")
        ppl = compute_perplexity(model, test_path, device, config.block_size)
        print(f"  Test perplexity: {ppl:.4f}")
    else:
        print(f"[WARN] test.npy not found at {test_path} — skipping perplexity")
        ppl = None

    # Sample quality metrics
    sample_metrics = evaluate_samples(
        generated_dir=args.generated_dir,
        output_dir=args.output_dir,
        render_pngs=not args.no_render,
    )

    # Save
    results = {
        "checkpoint": args.checkpoint,
        "model_name": ckpt["config"].get("name", "?"),
        "test_perplexity": ppl,
        "sample_metrics": sample_metrics,
    }
    out_path = Path(args.output_dir) / "results" / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation saved: {out_path}")
    return results


if __name__ == "__main__":
    main()
