"""
Train a GPT model on the tokenized SVG dataset.

Usage (single run):
    python scripts/train.py --config configs/tiny.json --lr 3e-4

The script saves:
  - outputs/checkpoints/<name>/checkpoint_final.pt
  - outputs/logs/<name>_lr<lr>.json   (training curve + final val loss)
  - outputs/results/scaling_results.json  (appended; used by fit_scaling_law.py)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.model import GPT, ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(requested: str = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lr(step: int, warmup_steps: int, total_steps: int,
           max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def load_data(path: str) -> np.ndarray:
    """Load tokenized split into RAM for fast random access."""
    return np.array(np.load(path, mmap_mode="r"))  # np.load handles .npy header; raw memmap skips it


def get_batch(data: np.ndarray, block_size: int, batch_size: int,
              device: torch.device):
    ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix
    ])
    return x.to(device), y.to(device)


def get_memory_mb(device: torch.device) -> float:
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024 ** 2
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


@torch.no_grad()
def compute_val_loss(model: GPT, val_data: np.ndarray, block_size: int,
                     device: torch.device, max_batches: int = None) -> float:
    """Compute validation loss over non-overlapping windows of val_data."""
    model.eval()
    batch_size = 8
    losses = []
    step = 0
    for i in range(0, len(val_data) - block_size, block_size * batch_size):
        starts = list(range(i, min(i + block_size * batch_size,
                                   len(val_data) - block_size), block_size))
        if not starts:
            break
        x = torch.stack([
            torch.from_numpy(val_data[s:s + block_size].astype(np.int64))
            for s in starts
        ]).to(device)
        y = torch.stack([
            torch.from_numpy(val_data[s + 1:s + block_size + 1].astype(np.int64))
            for s in starts
        ]).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
        step += 1
        if max_batches and step >= max_batches:
            break
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def upsert_scaling_results(results_path: str, entry: dict) -> None:
    """Add or update an entry in scaling_results.json keyed by model name."""
    results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    # Replace existing entry for the same model name if present
    results = [r for r in results if r.get("name") != entry["name"]]
    results.append(entry)
    results.sort(key=lambda r: r.get("param_count", 0))
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(
    config_path: str,
    lr: float,
    data_dir: str = "data/tokenized",
    output_dir: str = "outputs",
    effective_batch_tokens: int = 65536,
    batch_size: int = 4,
    weight_decay: float = 0.1,
    warmup_fraction: float = 0.05,
    grad_clip: float = 1.0,
    seed: int = 42,
    eval_interval: int = 100,
    device_str: str = None,
    max_steps: int = None,
    save_checkpoint: bool = True,
) -> dict:
    """
    Train one model and return metrics dict.

    Returns
    -------
    dict with keys: name, param_count, val_loss, train_losses,
                    tokens_per_second, peak_memory_mb, total_time_seconds,
                    config, lr
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device(device_str)
    print(f"\n{'='*60}")
    print(f"Training: {config_path}  |  lr={lr:.2e}  |  device={device}")
    print(f"{'='*60}")

    # Load model config
    with open(config_path) as f:
        cfg_dict = json.load(f)
    model_name = cfg_dict.get("name", Path(config_path).stem)
    config = ModelConfig.from_dict(cfg_dict)

    model = GPT(config).to(device)
    param_count = model.count_parameters()
    print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")

    # Data
    train_data = load_data(os.path.join(data_dir, "train.npy"))
    val_data   = load_data(os.path.join(data_dir, "val.npy"))
    print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

    # Training schedule
    block_size = config.block_size
    grad_accum_steps = max(1, effective_batch_tokens // (batch_size * block_size))
    tokens_per_step = batch_size * block_size * grad_accum_steps
    total_steps_full = math.ceil(len(train_data) / tokens_per_step)
    total_steps = max_steps if max_steps else total_steps_full
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    min_lr = lr * 0.1

    print(f"Effective batch: {tokens_per_step:,} tokens  "
          f"(phys_bs={batch_size}, accum={grad_accum_steps})")
    print(f"Steps: {total_steps:,}  |  Warmup: {warmup_steps}")

    # Optimizer — betas from GPT-3 paper
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    # Output paths
    ckpt_dir = Path(output_dir) / "checkpoints" / model_name
    log_dir  = Path(output_dir) / "logs"
    res_path = Path(output_dir) / "results" / "scaling_results.json"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    res_path.parent.mkdir(parents=True, exist_ok=True)

    lr_tag = f"{lr:.0e}".replace("-0", "-")
    log_path = log_dir / f"{model_name}_lr{lr_tag}.json"

    # Training loop
    train_losses = []
    peak_mem_mb = 0.0
    tokens_seen = 0
    t_start = time.time()
    t_last_log = t_start

    model.train()
    for step in range(total_steps):
        # Update LR
        current_lr = get_lr(step, warmup_steps, total_steps, lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = get_batch(train_data, block_size, batch_size, device)
            _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss.backward()
            step_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        tokens_seen += tokens_per_step
        mem = get_memory_mb(device)
        if mem > peak_mem_mb:
            peak_mem_mb = mem

        # Logging
        if step % eval_interval == 0 or step == total_steps - 1:
            t_now = time.time()
            elapsed = t_now - t_start
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            # Quick val estimate (50 batches) for training curve
            val_loss_est = compute_val_loss(model, val_data, block_size, device,
                                            max_batches=50)
            train_losses.append({
                "step": step,
                "train_loss": round(step_loss, 5),
                "val_loss": round(val_loss_est, 5),
                "lr": round(current_lr, 8),
                "tokens_per_sec": round(tps, 1),
            })
            dt = t_now - t_last_log
            t_last_log = t_now
            pct = 100 * step / total_steps
            print(f"  step {step:5d}/{total_steps} ({pct:4.1f}%)  "
                  f"train={step_loss:.4f}  val={val_loss_est:.4f}  "
                  f"lr={current_lr:.2e}  {tps:,.0f} tok/s  "
                  f"mem={peak_mem_mb:.0f}MB")

    # Final full validation loss
    print("Computing final validation loss over full val set...")
    final_val_loss = compute_val_loss(model, val_data, block_size, device)
    total_time = time.time() - t_start
    avg_tps = tokens_seen / total_time if total_time > 0 else 0

    print(f"\nFinal val loss: {final_val_loss:.4f}")
    print(f"Total time: {total_time/3600:.2f} hrs  |  "
          f"Avg throughput: {avg_tps:,.0f} tok/s")
    print(f"Peak memory: {peak_mem_mb:.1f} MB")

    # Save checkpoint
    if save_checkpoint:
        ckpt = {
            "step": total_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg_dict,
            "val_loss": final_val_loss,
            "lr": lr,
        }
        torch.save(ckpt, ckpt_dir / "checkpoint_final.pt")
        print(f"Checkpoint saved: {ckpt_dir / 'checkpoint_final.pt'}")

    result = {
        "name": model_name,
        "param_count": param_count,
        "val_loss": final_val_loss,
        "lr": lr,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "d_ff": config.d_ff,
        "train_losses": train_losses,
        "total_time_seconds": round(total_time, 1),
        "tokens_per_second": round(avg_tps, 1),
        "peak_memory_mb": round(peak_mem_mb, 1),
        "total_steps": total_steps,
        "effective_batch_tokens": tokens_per_step,
    }

    # Save per-run log
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Log saved: {log_path}")

    # Append to global scaling results (only full-epoch runs)
    if not max_steps:
        upsert_scaling_results(str(res_path), {
            k: v for k, v in result.items() if k != "train_losses"
        })
        print(f"Results appended: {res_path}")

    # Free MPS cache
    if device.type == "mps":
        torch.mps.empty_cache()

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train one GPT model size.")
    p.add_argument("--config", required=True, help="Path to model config JSON")
    p.add_argument("--lr", type=float, required=True, help="Learning rate")
    p.add_argument("--data_dir", default="data/tokenized")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--effective_batch_tokens", type=int, default=65536)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Physical sequences per micro-step")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Cap steps (for LR sweep; omit for full epoch)")
    p.add_argument("--no_checkpoint", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        config_path=args.config,
        lr=args.lr,
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
        max_steps=args.max_steps,
        save_checkpoint=not args.no_checkpoint,
    )
