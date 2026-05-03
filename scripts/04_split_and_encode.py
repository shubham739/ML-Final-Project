import json
import random
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parent.parent
CLEAN_FILE = ROOT / "data" / "cleaned" / "cleaned.jsonl"
TOK_FILE   = ROOT / "tokenizer" / "svg_bpe.model"
OUT_DIR    = ROOT / "data" / "tokenized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
VAL_FRAC    = 0.01
TEST_FRAC   = 0.01
MAX_SEQ_LEN = 1024   # drop SVGs longer than this many tokens


def load_all_svgs(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading cleaned SVGs"):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def encode_split(
    records: list[dict],
    tokenizer: spm.SentencePieceProcessor,
    eos_id: int,
    bos_id: int,
    split_name: str,
    max_len: int,
) -> tuple[np.ndarray, dict]:
    """
    Encode all SVGs in records, filter by length, concatenate with <eos>.
    Returns (token_array, stats_dict).
    """
    all_ids: list[int] = []
    kept = 0
    dropped = 0

    for rec in tqdm(records, desc=f"  Encoding {split_name}"):
        svg = rec["svg"]
        ids = tokenizer.encode(svg, out_type=int)

        if len(ids) > max_len:
            dropped += 1
            continue

        # BOS + token ids + EOS
        all_ids.append(bos_id)
        all_ids.extend(ids)
        all_ids.append(eos_id)
        kept += 1

    arr = np.array(all_ids, dtype=np.uint16)
    stats = {
        "svgs_kept": kept,
        "svgs_dropped_too_long": dropped,
        "total_tokens": len(arr),
    }
    return arr, stats


def main():
    # Check inputs
    for path in (CLEAN_FILE, TOK_FILE):
        if not path.exists():
            print(f"Missing: {path}. Run previous scripts first.")
            sys.exit(1)

    out_files = [OUT_DIR / f"{s}.npy" for s in ("train", "val", "test")]
    if all(f.exists() for f in out_files):
        print("All split files already exist. Delete them to re-run.")
        sys.exit(0)

    print("Loading tokenizer...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(TOK_FILE))
    eos_id = tokenizer.piece_to_id("<eos>")
    bos_id = tokenizer.piece_to_id("<bos>")
    print(f"  Vocab size : {tokenizer.get_piece_size()}")
    print(f"  BOS id     : {bos_id}")
    print(f"  EOS id     : {eos_id}")

    print("\nLoading cleaned SVGs...")
    records = load_all_svgs(CLEAN_FILE)
    total = len(records)
    print(f"  {total:,} records loaded")

    # Reproducible shuffle
    rng = random.Random(SEED)
    rng.shuffle(records)

    # Compute split sizes
    n_val  = max(1, int(total * VAL_FRAC))
    n_test = max(1, int(total * TEST_FRAC))
    n_train = total - n_val - n_test

    train_recs = records[:n_train]
    val_recs   = records[n_train:n_train + n_val]
    test_recs  = records[n_train + n_val:]

    print(f"\nSplit sizes (before token-length filtering):")
    print(f"  Train : {len(train_recs):>8,}")
    print(f"  Val   : {len(val_recs):>8,}")
    print(f"  Test  : {len(test_recs):>8,}")

    metadata = {
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "max_seq_len": MAX_SEQ_LEN,
        "splits": {},
    }

    for split_name, split_records in [
        ("train", train_recs),
        ("val",   val_recs),
        ("test",  test_recs),
    ]:
        print(f"\nEncoding {split_name}...")
        arr, stats = encode_split(
            split_records, tokenizer, eos_id, bos_id, split_name, MAX_SEQ_LEN
        )
        out_path = OUT_DIR / f"{split_name}.npy"
        np.save(out_path, arr)
        metadata["splits"][split_name] = stats
        print(f"  Saved {len(arr):,} tokens → {out_path}")
        print(f"  SVGs kept: {stats['svgs_kept']:,}  |  dropped (too long): {stats['svgs_dropped_too_long']:,}")

    with open(OUT_DIR / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Final summary
    train_tokens = metadata["splits"]["train"]["total_tokens"]
    print(f"\n{'='*60}")
    print(f"Training tokens  : {train_tokens:>12,}")
    print(f"Val tokens       : {metadata['splits']['val']['total_tokens']:>12,}")
    print(f"Test tokens      : {metadata['splits']['test']['total_tokens']:>12,}")

    if train_tokens < 100_000_000:
        print(f"\n⚠  Training tokens ({train_tokens:,}) < 100M target.")
        print("   Add svg-emoji-simple or svg-fonts-simple to 01_download.py,")
        print("   re-run 01_download.py → 02_clean.py → 03_train_tokenizer.py → this script.")
    else:
        print(f"\n✓  Training tokens exceed 100M target.")

    print(f"\nNext step: python scripts/05_statistics.py")


if __name__ == "__main__":
    main()
