"""
Task 2.4 — Fit a power-law scaling curve and produce the log-log plot.

Reads outputs/results/scaling_results.json (written by train.py) and fits:

    L(N) = a * N^(-alpha) + c

Outputs:
    outputs/plots/scaling_law.png
    outputs/results/scaling_law_fit.json
    Console: fitted a, alpha, c + comparison with Kaplan et al.

Usage:
    python scripts/fit_scaling_law.py
    python scripts/fit_scaling_law.py --results path/to/scaling_results.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def power_law(N, a, alpha, c):
    return a * N ** (-alpha) + c


def fit_and_plot(results_path: str, output_dir: str) -> dict:
    with open(results_path) as f:
        data = json.load(f)

    if len(data) < 3:
        raise ValueError(
            f"Need at least 3 data points to fit; found {len(data)}. "
            "Train more model sizes first."
        )

    data.sort(key=lambda r: r["param_count"])
    names  = [r["name"]        for r in data]
    params = np.array([r["param_count"] for r in data], dtype=float)
    losses = np.array([r["val_loss"]    for r in data], dtype=float)
    times  = [r.get("total_time_seconds", None) for r in data]

    print("\n" + "=" * 60)
    print("SCALING LAW FIT — Task 2.4")
    print("=" * 60)
    print(f"{'Model':<10} {'Params':>12} {'Val Loss':>10}")
    print("-" * 35)
    for n, p, l in zip(names, params, losses):
        print(f"{n:<10} {p:>12,.0f} {l:>10.4f}")

    # Fit: L = a * N^(-alpha) + c
    # Initial guess: a=1, alpha=0.1 (Kaplan-like), c = min_loss
    p0 = [1.0, 0.1, max(0.01, losses.min() * 0.5)]
    try:
        popt, pcov = curve_fit(
            power_law, params, losses,
            p0=p0,
            bounds=([0, 0, 0], [np.inf, 2.0, np.inf]),
            maxfev=20000,
        )
    except RuntimeError as e:
        print(f"\nWarning: curve_fit did not converge ({e}). "
              "Trying without bounds...")
        popt, pcov = curve_fit(power_law, params, losses, p0=p0, maxfev=50000)

    a, alpha, c = popt
    perr = np.sqrt(np.diag(pcov))

    print(f"\nFitted: L(N) = {a:.4f} * N^(-{alpha:.4f}) + {c:.4f}")
    print(f"  a     = {a:.4f} ± {perr[0]:.4f}")
    print(f"  alpha = {alpha:.4f} ± {perr[1]:.4f}  "
          f"(Kaplan et al. NL: ~0.076)")
    print(f"  c     = {c:.4f} ± {perr[2]:.4f}  (irreducible loss floor)")

    # Interpret alpha
    if alpha > 0.076:
        comparison = "steeper than"
    elif alpha < 0.076:
        comparison = "shallower than"
    else:
        comparison = "equal to"
    print(f"\n  SVG scaling exponent ({alpha:.3f}) is {comparison} "
          f"natural language (~0.076).")

    # Residuals / R²
    preds = power_law(params, *popt)
    ss_res = np.sum((losses - preds) ** 2)
    ss_tot = np.sum((losses - losses.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"  R² = {r2:.4f}")

    # Extrapolation to 10× XL (Task 3.6 preview)
    xl_params = max(params)
    target_params = xl_params * 10
    pred_loss = power_law(target_params, *popt)
    # Propagate uncertainty using delta method
    grad_a     = target_params ** (-alpha)
    grad_alpha = -a * np.log(target_params) * target_params ** (-alpha)
    grad_c     = 1.0
    pred_var   = (grad_a * perr[0]) ** 2 + (grad_alpha * perr[1]) ** 2 + (grad_c * perr[2]) ** 2
    pred_std   = np.sqrt(pred_var)
    print(f"\n  Extrapolation: N = {target_params:.2e} (10× XL)")
    print(f"    Predicted val_loss = {pred_loss:.4f} ± {pred_std:.4f} (1σ)")

    # Plot
    fig, (ax_main, ax_log) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Log-log plot (main result) ---
    N_fit = np.logspace(np.log10(params.min() * 0.7),
                        np.log10(params.max() * 1.5), 300)
    L_fit = power_law(N_fit, *popt)

    ax_main.scatter(params, losses, s=80, zorder=5, label="Trained models")
    for n, p, l in zip(names, params, losses):
        ax_main.annotate(n, (p, l), textcoords="offset points",
                         xytext=(6, 4), fontsize=9)
    ax_main.plot(N_fit, L_fit, "--", color="tomato", linewidth=1.8,
                 label=rf"$L = {a:.3f} \cdot N^{{-{alpha:.3f}}} + {c:.3f}$")

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_xlabel("Parameters (N)", fontsize=12)
    ax_main.set_ylabel("Validation Loss", fontsize=12)
    ax_main.set_title("Scaling Law: Val Loss vs. Parameters (log-log)", fontsize=12)
    ax_main.legend(fontsize=9)
    ax_main.grid(True, which="both", alpha=0.3)

    # --- Training curves (loss vs step) ---
    # Load per-model logs for training curves if available
    log_dir = Path(output_dir) / "logs"
    found_any = False
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(names)))
    for i, (r, color) in enumerate(zip(data, colors)):
        # Find the log file for this model
        candidates = list(log_dir.glob(f"{r['name']}_lr*.json"))
        if not candidates:
            continue
        # Use the one with the lowest final val loss (= the full-epoch run)
        best_log = min(candidates, key=lambda p: json.load(open(p)).get("val_loss", 1e9))
        with open(best_log) as f:
            log = json.load(f)
        train_losses = log.get("train_losses", [])
        if train_losses:
            steps = [e["step"] for e in train_losses]
            tlosses = [e["train_loss"] for e in train_losses]
            ax_log.plot(steps, tlosses, label=r["name"], color=color, linewidth=1.5)
            found_any = True

    if found_any:
        ax_log.set_xlabel("Training Step", fontsize=12)
        ax_log.set_ylabel("Training Loss", fontsize=12)
        ax_log.set_title("Training Curves — All Models (Task 2.3)", fontsize=12)
        ax_log.legend(fontsize=9)
        ax_log.grid(True, alpha=0.3)
    else:
        ax_log.set_visible(False)
        # Single wider main plot
        ax_main.set_position([0.1, 0.12, 0.85, 0.78])

    plot_path = Path(output_dir) / "plots" / "scaling_law.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {plot_path}")

    # Compile statistics table
    print("\n" + "=" * 80)
    print(f"{'Name':<10} {'Params':>10} {'d_model':>8} {'n_lay':>6} "
          f"{'n_hd':>5} {'d_ff':>6} {'ValLoss':>8} {'Time(h)':>8} {'tok/s':>8}")
    print("-" * 80)
    for r in data:
        t = r.get("total_time_seconds", 0)
        tps = r.get("tokens_per_second", 0)
        print(f"{r['name']:<10} {r['param_count']:>10,} {r['d_model']:>8} "
              f"{r['n_layers']:>6} {r['n_heads']:>5} {r['d_ff']:>6} "
              f"{r['val_loss']:>8.4f} {t/3600:>8.2f} {tps:>8,.0f}")
    print("=" * 80)

    # Save fit results
    fit_result = {
        "a": a, "alpha": alpha, "c": c,
        "a_std": perr[0], "alpha_std": perr[1], "c_std": perr[2],
        "r_squared": r2,
        "kaplan_alpha_nl": 0.076,
        "extrapolation": {
            "target_params": target_params,
            "predicted_val_loss": pred_loss,
            "predicted_val_loss_std": pred_std,
        },
        "data_points": [
            {"name": r["name"], "param_count": r["param_count"],
             "val_loss": r["val_loss"]}
            for r in data
        ],
    }
    fit_path = Path(output_dir) / "results" / "scaling_law_fit.json"
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fit_path, "w") as f:
        json.dump(fit_result, f, indent=2)
    print(f"Fit results saved: {fit_path}")

    return fit_result


def main():
    p = argparse.ArgumentParser(description="Fit scaling law and plot (Task 2.4).")
    p.add_argument("--results", default="outputs/results/scaling_results.json",
                   help="Path to scaling_results.json")
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: {args.results} not found.")
        print("Train at least 3 model sizes first using train.py, then re-run.")
        return

    fit_and_plot(args.results, args.output_dir)


if __name__ == "__main__":
    main()
