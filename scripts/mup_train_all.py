"""
Train all 5 μP model sizes with the transferred best LR.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.mup_train import train_mup_model


CONFIGS = [
    "configs/tiny.json",
    "configs/small.json",
    "configs/medium.json",
    "configs/large.json",
    "configs/xl.json",
]


def get_best_lr_from_sweep(output_dir: str) -> float:
    sweep_path = Path(output_dir) / "results" / "mup_lr_sweep_results.json"
    if not sweep_path.exists():
        raise FileNotFoundError(
            f"μP sweep results not found at {sweep_path}. "
            "Run mup_lr_sweep.py first."
        )
    with open(sweep_path) as f:
        data = json.load(f)
    valid = {float(k): v for k, v in data.items()
             if not isinstance(v, float) or not (v != v)}  # exclude NaN
    best_lr = min(valid, key=lambda k: valid[k])
    print(f"Auto-selected best μP LR: {best_lr:.2e} "
          f"(val_loss={valid[best_lr]:.4f} from sweep)")
    return best_lr


def already_trained(name: str, output_dir: str) -> bool:
    ckpt = Path(output_dir) / "checkpoints" / f"{name}_mup" / "checkpoint_final.pt"
    return ckpt.exists()


def main():
    p = argparse.ArgumentParser(
        description="Train all 5 μP model sizes with LR transfer (Task 3.4)."
    )
    lr_group = p.add_mutually_exclusive_group(required=True)
    lr_group.add_argument("--best_lr", type=float,
                          help="Best LR from mup_lr_sweep.py")
    lr_group.add_argument("--auto_lr", action="store_true",
                          help="Auto-read best LR from mup_lr_sweep_results.json")
    p.add_argument("--data_dir",    default="data/tokenized")
    p.add_argument("--output_dir",  default="outputs")
    p.add_argument("--effective_batch_tokens", type=int, default=65536)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--grad_clip",   type=float,  default=1.0)
    p.add_argument("--seed",        type=int,    default=42)
    p.add_argument("--eval_interval", type=int,  default=100)
    p.add_argument("--device",      default=None)
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip configs whose checkpoint already exists")
    args = p.parse_args()

    if args.auto_lr:
        best_lr = get_best_lr_from_sweep(args.output_dir)
    else:
        best_lr = args.best_lr

    print("\n" + "=" * 60)
    print("μP TRAIN ALL SIZES  —  Task 3.4")
    print(f"Transferred LR: {best_lr:.2e}  (same for all sizes — μP guarantee)")
    print(f"Configs: {[Path(c).stem for c in CONFIGS]}")
    print("=" * 60)

    all_results = []
    for cfg_path in CONFIGS:
        name = Path(cfg_path).stem
        if args.skip_existing and already_trained(name, args.output_dir):
            print(f"\n[SKIP] {name}_mup — checkpoint already exists")
            continue

        print(f"\n{'─'*60}")
        print(f"Training μP {name}  |  LR = {best_lr:.2e}")
        print(f"{'─'*60}")
        try:
            result = train_mup_model(
                config_path=cfg_path,
                lr=best_lr,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                effective_batch_tokens=args.effective_batch_tokens,
                batch_size=args.batch_size,
                weight_decay=args.weight_decay,
                warmup_fraction=args.warmup_fraction,
                grad_clip=args.grad_clip,
                seed=args.seed,
                eval_interval=args.eval_interval,
                device_str=args.device,
                save_checkpoint=True,
            )
            all_results.append(result)
            print(f"  ✓ {name}_mup  val_loss={result['val_loss']:.4f}")
        except Exception as e:
            print(f"  ✗ {name}_mup FAILED: {e}")
            raise

    # Summary table
    if all_results:
        print("\n" + "=" * 80)
        print("μP SCALING RESULTS SUMMARY  (Task 3.4)")
        print(f"{'Name':<14} {'Params':>10} {'d_model':>8} "
              f"{'n_lay':>6} {'Val Loss':>10} {'Time(h)':>8}")
        print("-" * 60)
        for r in sorted(all_results, key=lambda x: x["param_count"]):
            t = r.get("total_time_seconds", 0)
            print(f"{r['name']:<14} {r['param_count']:>10,} {r['d_model']:>8} "
                  f"{r['n_layers']:>6} {r['val_loss']:>10.4f} {t/3600:>8.2f}")
        print("=" * 80)
        print(f"\nAll results saved to: outputs/results/mup_scaling_results.json")
        print("Next: python scripts/compare_scaling.py")


if __name__ == "__main__":
    main()
