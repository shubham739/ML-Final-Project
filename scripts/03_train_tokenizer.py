"""
Train a BPE tokenizer on the cleaned SVG corpus using sentencepiece.

sentencepiece is preferred over HuggingFace ByteLevel BPE for SVG because:
  - SVG text doesn't split cleanly on whitespace (coordinates, path data)
  - sentencepiece learns merges across all character boundaries
  - Works well for structured, non-linguistic text

Vocabulary size: 4096
  - Captures SVG-specific tokens: <path, fill=, viewBox, d="M, etc.
  - Small enough for embedding tables in 1M–88M parameter models
  - Justified: SVG has ~60–70 unique ASCII chars; 4096 merges gives
    ~5–7 chars/token, efficient compression of path data

Special token IDs (fixed by sentencepiece convention):
  <pad>  = 0
  <bos>  = 1
  <eos>  = 2
  <unk>  = 3

Output:
  tokenizer/svg_bpe.model   — sentencepiece model (binary)
  tokenizer/svg_bpe.vocab   — human-readable vocab
  tokenizer/tokenizer_config.json
"""

import json
import sys
from pathlib import Path

import sentencepiece as spm
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parent.parent
CLEAN_FILE = ROOT / "data" / "cleaned" / "cleaned.jsonl"
TOK_DIR    = ROOT / "tokenizer"
TOK_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE  = 1024  # SVG's limited char set after normalization can't fill 4096
MODEL_PREFIX = str(TOK_DIR / "svg_bpe")
CORPUS_TXT   = TOK_DIR / "corpus_tmp.txt"


def write_corpus(clean_file: Path, out_txt: Path) -> int:
    """Write one SVG per line to a plain text file for sentencepiece."""
    count = 0
    with open(clean_file, encoding="utf-8") as fin, \
         open(out_txt, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Writing corpus"):
            try:
                svg = json.loads(line)["svg"]
                # Keep SVG on a single line for sentencepiece
                fout.write(svg.replace("\n", " ") + "\n")
                count += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return count


def main():
    if not CLEAN_FILE.exists():
        print(f"Missing: {CLEAN_FILE}. Run 02_clean.py first.")
        sys.exit(1)

    model_file = Path(MODEL_PREFIX + ".model")
    if model_file.exists():
        print(f"Tokenizer model already exists at {model_file}. Delete to retrain.")
        sys.exit(0)

    # Step 1: write corpus text file
    if not CORPUS_TXT.exists():
        print("Writing corpus to text file...")
        n = write_corpus(CLEAN_FILE, CORPUS_TXT)
        print(f"  Wrote {n:,} lines → {CORPUS_TXT}")
    else:
        print(f"Corpus text file already exists: {CORPUS_TXT}")

    # Step 2: train sentencepiece BPE
    print(f"\nTraining sentencepiece BPE (vocab_size={VOCAB_SIZE})...")
    spm.SentencePieceTrainer.train(
        input=str(CORPUS_TXT),
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,      # cover all chars in training data
        pad_id=0,  pad_piece="<pad>",
        bos_id=1,  bos_piece="<bos>",
        eos_id=2,  eos_piece="<eos>",
        unk_id=3,  unk_piece="<unk>",
        # Treat these structural chars as individual tokens (not split by BPE)
        user_defined_symbols=[],
        # Don't split on digits — keeps coordinate numbers together
        split_digits=False,
        byte_fallback=True,          # never produce <unk> for unseen chars
        train_extremely_large_corpus=False,
    )
    print(f"  Model saved → {model_file}")

    # Step 3: load and verify
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))
    actual_vocab = sp.get_piece_size()
    print(f"  Actual vocabulary size: {actual_vocab}")

    # Verify special token IDs
    for piece, expected_id in [("<pad>", 0), ("<bos>", 1), ("<eos>", 2), ("<unk>", 3)]:
        actual = sp.piece_to_id(piece)
        status = "OK" if actual == expected_id else f"MISMATCH (got {actual})"
        print(f"  {piece}: id={expected_id}  [{status}]")

    # Show sample SVG-specific tokens
    print("\nSample tokens learned (by id):")
    svg_keywords = ["<svg", "path", "fill", "view", "rect", "circ", "d=\"", "M", "L", "Z"]
    vocab_list = [sp.id_to_piece(i) for i in range(actual_vocab)]
    matches = [(i, p) for i, p in enumerate(vocab_list)
               if any(k in p for k in svg_keywords)][:20]
    for idx, piece in matches:
        print(f"  [{idx:4d}]  {repr(piece)}")

    # Sanity check on first SVG
    with open(CLEAN_FILE) as f:
        sample = json.loads(f.readline())["svg"]
    ids = sp.encode(sample, out_type=int)
    print(f"\nSanity check on first SVG:")
    print(f"  char length : {len(sample)}")
    print(f"  token count : {len(ids)}")
    print(f"  chars/token : {len(sample)/len(ids):.2f}")

    # Save config
    cfg = {
        "vocab_size": actual_vocab,
        "model_type": "BPE",
        "library": "sentencepiece",
        "model_file": str(model_file),
        "special_tokens": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3},
    }
    with open(TOK_DIR / "tokenizer_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nConfig saved → {TOK_DIR / 'tokenizer_config.json'}")
    print("\nNext step: python scripts/04_split_and_encode.py")


if __name__ == "__main__":
    main()
