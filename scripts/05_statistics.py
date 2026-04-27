"""
Compute and visualize dataset statistics for the report.

Produces:
  outputs/plots/seq_len_histogram.png   — token count distribution
  outputs/plots/svg_examples.png        — rendered grid of SVGs at varied complexity
  data/cleaned/dataset_stats.json       — all numbers for the report

Requires cairosvg for rendering (pip install cairosvg).
Falls back gracefully if cairosvg is not available.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parent.parent
CLEAN_FILE = ROOT / "data" / "cleaned" / "cleaned.jsonl"
TOK_FILE   = ROOT / "tokenizer" / "svg_bpe.model"
META_FILE  = ROOT / "data" / "tokenized" / "split_metadata.json"
PLOT_DIR   = ROOT / "outputs" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import cairosvg
    HAS_CAIRO = True
except (ImportError, OSError):
    cairosvg = None
    HAS_CAIRO = False
    print("cairosvg/libcairo not available — SVG rendering will be skipped.")

try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def load_svgs(path: Path) -> list[str]:
    svgs = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading SVGs for stats"):
            try:
                svgs.append(json.loads(line)["svg"])
            except (json.JSONDecodeError, KeyError):
                continue
    return svgs


def compute_token_lengths(svgs: list[str], tokenizer: spm.SentencePieceProcessor) -> list[int]:
    lengths = []
    for svg in tqdm(svgs, desc="Tokenizing for lengths"):
        try:
            lengths.append(len(tokenizer.encode(svg, out_type=int)))
        except Exception:
            continue
    return lengths


def plot_histogram(lengths: list[int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Sequence length (tokens)", fontsize=13)
    ax.set_ylabel("Number of SVGs", fontsize=13)
    ax.set_title("Token-length distribution of cleaned SVG corpus", fontsize=14)
    ax.axvline(np.median(lengths), color="tomato", linestyle="--",
               label=f"Median: {int(np.median(lengths))}")
    ax.axvline(np.mean(lengths), color="orange", linestyle=":",
               label=f"Mean: {np.mean(lengths):.0f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Histogram saved → {out_path}")


def render_svg_to_pil(svg_str: str, size: int = 128):
    """Render SVG string to a PIL Image. Returns None on failure."""
    if not (HAS_CAIRO and HAS_PIL):
        return None
    try:
        png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"),
                                     output_width=size, output_height=size)
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception:
        return None


def plot_svg_grid(svgs: list[str], lengths: list[int], out_path: Path) -> None:
    """Render a 3×3 grid of SVGs spanning complexity levels."""
    if not (HAS_CAIRO and HAS_PIL):
        print("  Skipping SVG grid (cairosvg / Pillow not available)")
        return

    sorted_idx = np.argsort(lengths)
    # Pick 9 indices spread across short, medium, long
    n = len(sorted_idx)
    picks = [int(n * frac) for frac in (0.02, 0.10, 0.20, 0.35, 0.50, 0.65, 0.75, 0.88, 0.97)]
    picks = [min(p, n - 1) for p in picks]

    cell = 128
    cols, rows = 3, 3
    grid = Image.new("RGBA", (cols * cell, rows * cell), (240, 240, 240, 255))
    placed = 0
    for pos, idx in enumerate(picks):
        if placed >= rows * cols:
            break
        svg = svgs[sorted_idx[idx]]
        img = render_svg_to_pil(svg, cell)
        if img is None:
            continue
        row, col = divmod(placed, cols)
        grid.paste(img, (col * cell, row * cell))
        placed += 1

    grid.save(out_path)
    print(f"  SVG grid saved → {out_path}")


def print_stats(lengths: list[int], svgs: list[str]) -> dict:
    lengths_arr = np.array(lengths)
    stats = {
        "total_svgs": len(svgs),
        "token_lengths": {
            "min":    int(lengths_arr.min()),
            "max":    int(lengths_arr.max()),
            "mean":   float(lengths_arr.mean()),
            "median": float(np.median(lengths_arr)),
            "p25":    float(np.percentile(lengths_arr, 25)),
            "p75":    float(np.percentile(lengths_arr, 75)),
            "p95":    float(np.percentile(lengths_arr, 95)),
        },
        "char_lengths": {
            "mean": float(np.mean([len(s) for s in svgs])),
        },
    }
    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Total SVGs         : {stats['total_svgs']:>10,}")
    print(f"  Token length")
    for k, v in stats["token_lengths"].items():
        print(f"    {k:<8}           : {v:>10.1f}")
    return stats


def main():
    for path in (CLEAN_FILE, TOK_FILE):
        if not path.exists():
            print(f"Missing: {path}. Run previous scripts first.")
            sys.exit(1)

    print("Loading tokenizer...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(TOK_FILE))
    vocab_size = tokenizer.get_piece_size()
    print(f"  Vocab size: {vocab_size}")

    print("\nLoading SVGs...")
    svgs = load_svgs(CLEAN_FILE)

    print("\nComputing token lengths (this takes a few minutes)...")
    lengths = compute_token_lengths(svgs, tokenizer)

    stats = print_stats(lengths, svgs)
    stats["vocab_size"] = vocab_size

    # Add split token counts if available
    if META_FILE.exists():
        with open(META_FILE) as f:
            meta = json.load(f)
        stats["splits"] = meta["splits"]
        print(f"\n  Train tokens : {meta['splits']['train']['total_tokens']:>12,}")
        print(f"  Val tokens   : {meta['splits']['val']['total_tokens']:>12,}")
        print(f"  Test tokens  : {meta['splits']['test']['total_tokens']:>12,}")

    # Save numeric stats
    out_stats = ROOT / "data" / "cleaned" / "dataset_stats.json"
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats written → {out_stats}")

    # Plots
    print("\nGenerating plots...")
    plot_histogram(lengths, PLOT_DIR / "seq_len_histogram.png")
    plot_svg_grid(svgs, lengths, PLOT_DIR / "svg_examples.png")

    # Sample token vocabulary
    print(f"\nSample vocabulary entries (first 30 by id):")
    for idx in range(min(30, vocab_size)):
        piece = tokenizer.id_to_piece(idx)
        print(f"  [{idx:4d}]  {repr(piece)}")

    print(f"\nDone. All outputs in {PLOT_DIR} and {ROOT / 'data' / 'cleaned'}")


if __name__ == "__main__":
    main()
