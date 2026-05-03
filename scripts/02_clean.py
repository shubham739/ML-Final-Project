import json
import re
import sys
from pathlib import Path

from lxml import etree
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR  = ROOT / "data" / "raw"
OUT_DIR  = ROOT / "data" / "cleaned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE   = OUT_DIR / "cleaned.jsonl"
STATS_FILE = OUT_DIR / "clean_stats.json"

MIN_CHARS = 50
MAX_CHARS = 1024 * 4
_FLOAT_RE = re.compile(r"-?\d+\.\d+")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_META_RE = re.compile(r"<metadata[^>]*>.*?</metadata>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _round_float(match: re.Match) -> str:
    return f"{float(match.group()):.1f}"


def normalize_svg(svg: str) -> str:
    """Apply all normalization steps and return cleaned SVG string."""
    # 1. Strip comments
    svg = _COMMENT_RE.sub("", svg)
    # 2. Strip metadata blocks
    svg = _META_RE.sub("", svg)
    # 3. Collapse whitespace
    svg = _WS_RE.sub(" ", svg).strip()
    # 4. Round all floats to 1 decimal place
    svg = _FLOAT_RE.sub(_round_float, svg)
    return svg


def is_valid_xml(svg: str) -> bool:
    """Return True if svg parses as valid XML."""
    try:
        etree.fromstring(svg.encode("utf-8"))
        return True
    except etree.XMLSyntaxError:
        return False


def process_file(raw_path: Path, out_f, stats: dict) -> None:
    """Read one JSONL file, clean every SVG, write passing ones to out_f."""
    with open(raw_path, encoding="utf-8") as f:
        lines = f.readlines()

    stats["input"][raw_path.name] = len(lines)

    for line in tqdm(lines, desc=f"  {raw_path.name}", unit="svg"):
        stats["total_raw"] += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats["dropped_json_error"] += 1
            continue

        svg = record.get("svg", "")
        if not isinstance(svg, str):
            stats["dropped_json_error"] += 1
            continue

        # --- Normalize ---
        svg = normalize_svg(svg)

        # --- Length filter ---
        if len(svg) < MIN_CHARS:
            stats["dropped_too_short"] += 1
            continue
        if len(svg) > MAX_CHARS:
            stats["dropped_too_long"] += 1
            continue

        # --- XML validation ---
        if not is_valid_xml(svg):
            stats["dropped_invalid_xml"] += 1
            continue

        record["svg"] = svg
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats["kept"] += 1


def main():
    raw_files = sorted(RAW_DIR.glob("*.jsonl"))
    if not raw_files:
        print(f"No JSONL files found in {RAW_DIR}. Run 01_download.py first.")
        sys.exit(1)

    if OUT_FILE.exists():
        print(f"Cleaned file already exists: {OUT_FILE}")
        print("Delete it to re-run cleaning. Exiting.")
        sys.exit(0)

    stats: dict = {
        "total_raw": 0,
        "kept": 0,
        "dropped_too_short": 0,
        "dropped_too_long": 0,
        "dropped_invalid_xml": 0,
        "dropped_json_error": 0,
        "input": {},
        "min_chars": MIN_CHARS,
        "max_chars": MAX_CHARS,
    }

    print(f"Cleaning {len(raw_files)} raw file(s)...")
    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        for raw_path in raw_files:
            process_file(raw_path, out_f, stats)

    # Summary
    kept = stats["kept"]
    total = stats["total_raw"]
    print(f"\n{'='*60}")
    print(f"Raw SVGs total      : {total:>10,}")
    print(f"Kept (passed filter): {kept:>10,}  ({100*kept/max(total,1):.1f}%)")
    print(f"  Dropped too short : {stats['dropped_too_short']:>10,}")
    print(f"  Dropped too long  : {stats['dropped_too_long']:>10,}")
    print(f"  Dropped bad XML   : {stats['dropped_invalid_xml']:>10,}")
    print(f"  Dropped JSON err  : {stats['dropped_json_error']:>10,}")
    print(f"\nOutput: {OUT_FILE}")

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats : {STATS_FILE}")
    print("\nNext step: python scripts/03_train_tokenizer.py")


if __name__ == "__main__":
    main()
