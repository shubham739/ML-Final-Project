"""
Download SVG datasets from HuggingFace and save as JSONL.

Priority order:
  1. starvector/svg-icons-simple  (~89K icons, primary)
  2. starvector/svg-emoji-simple  (~3.5K emoji, supplement)

Each output line: {"svg": "...", "source": "dataset-name", "id": N}
"""

import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Datasets in priority order; (hf_name, output_filename, max_samples or None)
# svg-fonts-simple is 2.38 GB; cap at 200K to stay manageable while reaching 100M tokens.
DATASETS = [
    ("starvector/svg-icons-simple", "icons.jsonl",  None),
    ("starvector/svg-emoji-simple", "emoji.jsonl",  None),
    ("starvector/svg-fonts-simple", "fonts.jsonl",  320_000),
]

# Minimum tokens we want in the training set.
# ~89K icons × ~700 avg chars/icon ÷ ~4 chars/token ≈ ~15M tokens from icons alone.
# We'll need emoji too. The stats script will confirm final counts.
TARGET_TRAIN_TOKENS = 100_000_000


def detect_svg_field(example: dict) -> str:
    """Return the key that holds the SVG string."""
    for candidate in ("image", "svg", "svg_code", "code", "text"):
        if candidate in example:
            val = example[candidate]
            if isinstance(val, str) and val.strip().startswith("<"):
                return candidate
    # Fallback: pick the first string field that looks like XML
    for k, v in example.items():
        if isinstance(v, str) and "<svg" in v:
            return k
    raise ValueError(f"Cannot find SVG field in keys: {list(example.keys())}")


def download_dataset(hf_name: str, out_path: Path, max_samples: int | None = None) -> int:
    """Download one HuggingFace dataset, write JSONL, return count saved."""
    print(f"\n{'='*60}")
    print(f"Downloading: {hf_name}")
    print(f"Output:      {out_path}")

    # Load all available splits and concatenate — we do our own splits later.
    from datasets import get_dataset_split_names, concatenate_datasets
    try:
        split_names = get_dataset_split_names(hf_name)
    except Exception:
        split_names = ["train"]

    ds_parts = []
    for sname in split_names:
        try:
            part = load_dataset(hf_name, split=sname)
            ds_parts.append(part)
            print(f"  Loaded split '{sname}': {len(part):,} examples")
        except Exception as e:
            print(f"  Skipped split '{sname}': {e}")

    ds = concatenate_datasets(ds_parts) if len(ds_parts) > 1 else ds_parts[0]
    print(f"  Total combined: {len(ds):,} examples")
    print(f"  Fields: {ds.column_names}")

    # Detect SVG field on first example
    svg_field = detect_svg_field(ds[0])
    print(f"  SVG field: '{svg_field}'")

    if max_samples is not None and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
        print(f"  Capped at {max_samples:,} samples")

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(tqdm(ds, desc="  Writing")):
            svg = example[svg_field]
            if not isinstance(svg, str) or not svg.strip():
                continue
            record = {"svg": svg, "source": hf_name, "id": i}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count:,} SVGs → {out_path}")
    return count


def main():
    total = 0
    for hf_name, fname, max_samples in DATASETS:
        out_path = RAW_DIR / fname
        if out_path.exists():
            with open(out_path) as f:
                existing = sum(1 for _ in f)
            print(f"Already exists: {out_path} ({existing:,} lines) — skipping download.")
            total += existing
        else:
            total += download_dataset(hf_name, out_path, max_samples)

    print(f"\n{'='*60}")
    print(f"Total raw SVGs across all datasets: {total:,}")
    print(f"Raw data saved to: {RAW_DIR}")
    print("\nNext step: python scripts/02_clean.py")


if __name__ == "__main__":
    main()
