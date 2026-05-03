"""
μP Learning Rate Sweep on the Tiny model.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.mup_train import train_mup_model


LEARNING_RATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


def run_mup_sweep(
    config_path: str = "configs/tiny.json",
    data_dir: str = "data/tokenized",
    output_dir: str = "outputs",
    sweep_fraction: float = 1.0,
    device: str = None,
    seed: int = 42,
) -> dict:
    # Load any previously saved results so we can skip them
    res_path = Path(output_dir) / "results" / "mup_lr_sweep_results.json"
    existing = {}
    if res_path.exists():
        raw = json.load(open(res_path))
        existing = {float(k): v for k, v in raw.items()}
        print(f"Found {len(existing)} existing result(s) — will skip those LRs.")

    print("\n" + "=" * 60)
    print("μP LR SWEEP  —  Task 3.3 (extended)")
    print(f"Config : {config_path}")
    print(f"LRs    : {LEARNING_RATES}")
    print(f"Fraction of epoch: {sweep_fraction:.0%}")
    print("KEY: Under μP, this best LR transfers directly to wider models.")
    print("=" * 60)

    max_steps_for_fraction = None
    if sweep_fraction < 1.0:
        full_steps = 1551
        max_steps_for_fraction = max(50, int(full_steps * sweep_fraction))
        print(f"Using max_steps={max_steps_for_fraction} per LR trial "
              f"(out of ~{full_steps} full-epoch steps)")

    results = dict(existing)
    for lr in LEARNING_RATES:
        if lr in existing:
            print(f"\n--- μP LR = {lr:.0e}  [SKIP — already have val_loss={existing[lr]:.4f}] ---")
            continue
        print(f"\n--- μP LR = {lr:.0e} ---")
        try:
            r = train_mup_model(
                config_path=config_path,
                lr=lr,
                data_dir=data_dir,
                output_dir=output_dir,
                seed=seed,
                eval_interval=50,
                device_str=device,
                max_steps=max_steps_for_fraction,
                save_checkpoint=False,
            )
            results[lr] = r["val_loss"]
            print(f"  → val_loss = {r['val_loss']:.4f}")
        except Exception as e:
            print(f"  → FAILED: {e}")
            results[lr] = float("nan")

    return results


def plot_mup_lr_sweep(results: dict, output_dir: str) -> str:
    lrs   = sorted(results.keys())
    vals  = [results[lr] for lr in lrs]
    valid = [(lr, v) for lr, v in zip(lrs, vals) if not np.isnan(v)]

    best_lr, best_val = min(valid, key=lambda x: x[1]) if valid else (None, None)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(lrs, vals, "s-", color="darkorchid", linewidth=2, markersize=7,
                label="μP val loss")

    if best_lr is not None:
        ax.axvline(best_lr, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"best LR = {best_lr:.0e}  (val={best_val:.4f})")

    ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("μP LR Sweep — Tiny Model (Task 3.3)\n"
                 "Best LR transfers to all wider models without retuning",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    out_path = Path(output_dir) / "plots" / "mup_lr_sweep.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")
    return str(out_path)


def main():
    p = argparse.ArgumentParser(description="μP LR sweep on the Tiny model (Task 3.3).")
    p.add_argument("--config",         default="configs/tiny.json")
    p.add_argument("--data_dir",       default="data/tokenized")
    p.add_argument("--output_dir",     default="outputs")
    p.add_argument("--sweep_fraction", type=float, default=1.0,
                   help="Fraction of full epoch per LR (0.2 = fast sweep)")
    p.add_argument("--device",         default=None)
    p.add_argument("--seed",           type=int, default=42)
    args = p.parse_args()

    results = run_mup_sweep(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sweep_fraction=args.sweep_fraction,
        device=args.device,
        seed=args.seed,
    )

    # Save raw results
    res_path = Path(args.output_dir) / "results" / "mup_lr_sweep_results.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {f"{k:.2e}": v for k, v in results.items()}
    with open(res_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved: {res_path}")

    plot_mup_lr_sweep(results, args.output_dir)

    valid = [(lr, v) for lr, v in results.items() if not np.isnan(v)]
    if valid:
        best_lr, best_val = min(valid, key=lambda x: x[1])
        print("\n" + "=" * 60)
        print(f"μP BEST LR: {best_lr:.2e}  (val_loss = {best_val:.4f})")
        print("This LR transfers to ALL model sizes without retuning:")
        for cfg in ["tiny", "small", "medium", "large", "xl"]:
            print(f"  python scripts/mup_train.py "
                  f"--config configs/{cfg}.json --lr {best_lr:.2e}")
        print("=" * 60)
        print("\nOr use mup_train_all.py for a one-shot run:")
        print(f"  python scripts/mup_train_all.py --best_lr {best_lr:.2e}")

    return results


if __name__ == "__main__":
    main()
