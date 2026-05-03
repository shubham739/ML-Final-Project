import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def power_law(N, a, alpha, c):
    return a * N ** (-alpha) + c


def fit_power_law(params: np.ndarray, losses: np.ndarray):
    """Fit L = a*N^(-alpha) + c. Returns (popt, pcov)."""
    p0 = [1.0, 0.1, max(0.01, losses.min() * 0.5)]
    try:
        popt, pcov = curve_fit(
            power_law, params, losses,
            p0=p0,
            bounds=([0, 0, 0], [np.inf, 2.0, np.inf]),
            maxfev=30000,
        )
    except RuntimeError:
        popt, pcov = curve_fit(power_law, params, losses, p0=p0, maxfev=80000)
    return popt, pcov

def extrapolate(target_params: float, popt, pcov):
    """Predict loss + 1σ confidence interval at target_params."""
    a, alpha, c = popt
    perr = np.sqrt(np.diag(pcov))
    pred = power_law(target_params, *popt)
    # Delta method for uncertainty propagation
    grad_a     = target_params ** (-alpha)
    grad_alpha = -a * np.log(target_params) * target_params ** (-alpha)
    grad_c     = 1.0
    pred_std   = np.sqrt(
        (grad_a * perr[0])**2 + (grad_alpha * perr[1])**2 + (grad_c * perr[2])**2
    )
    return pred, pred_std

def plot_scaling_comparison(sp_data, mup_data, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 6))

    sp_params  = np.array([r["param_count"] for r in sp_data],  dtype=float)
    sp_losses  = np.array([r["val_loss"]    for r in sp_data],  dtype=float)
    mup_params = np.array([r["param_count"] for r in mup_data], dtype=float)
    mup_losses = np.array([r["val_loss"]    for r in mup_data], dtype=float)

    sp_names  = [r["name"]      for r in sp_data]
    mup_names = [r.get("base_name", r["name"].replace("_mup","")) for r in mup_data]

    # --- fit both ---
    sp_popt,  sp_pcov  = fit_power_law(sp_params,  sp_losses)
    mup_popt, mup_pcov = fit_power_law(mup_params, mup_losses)

    N_fit = np.logspace(
        np.log10(min(sp_params.min(), mup_params.min()) * 0.7),
        np.log10(max(sp_params.max(), mup_params.max()) * 1.5),
        300,
    )

    # SP curve
    ax.scatter(sp_params,  sp_losses,  s=80, color="steelblue",  zorder=5,
               label="SP (Standard)", marker="o")
    for n, p, l in zip(sp_names, sp_params, sp_losses):
        ax.annotate(n, (p, l), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color="steelblue")
    a, alpha, c = sp_popt
    ax.plot(N_fit, power_law(N_fit, *sp_popt), "--", color="steelblue", linewidth=1.8,
            label=rf"SP fit: $L={a:.2f}\cdot N^{{-{alpha:.3f}}}+{c:.3f}$")

    # μP curve
    ax.scatter(mup_params, mup_losses, s=80, color="darkorchid", zorder=5,
               label="μP (MuP)", marker="s")
    for n, p, l in zip(mup_names, mup_params, mup_losses):
        ax.annotate(n, (p, l), textcoords="offset points",
                    xytext=(6, -12), fontsize=8, color="darkorchid")
    a_m, alpha_m, c_m = mup_popt
    ax.plot(N_fit, power_law(N_fit, *mup_popt), "--", color="darkorchid", linewidth=1.8,
            label=rf"μP fit: $L={a_m:.2f}\cdot N^{{-{alpha_m:.3f}}}+{c_m:.3f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters (N)", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Scaling Laws: Standard (SP) vs μP — Log-Log Plot\n"
                 "Tasks 3.5 & 3.6", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    out = Path(output_dir) / "plots" / "sp_vs_mup_scaling.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Scaling comparison plot saved: {out}")
    return str(out)

def plot_lr_comparison(sp_sweep: dict, mup_sweep: dict, output_dir: str) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, label, color, marker in [
        (ax1, sp_sweep,  "SP (Standard)",       "steelblue",  "o"),
        (ax2, mup_sweep, "μP (transferred LR)", "darkorchid", "s"),
    ]:
        lrs   = sorted(float(k) for k in data.keys())
        vals  = [data[f"{lr:.2e}"] if f"{lr:.2e}" in data else data.get(lr, float("nan"))
                 for lr in lrs]
        # handle both string and float keys
        vals  = []
        for lr in lrs:
            v = data.get(f"{lr:.2e}", data.get(f"{lr:.1e}", data.get(lr, float("nan"))))
            vals.append(v)

        valid = [(lr, v) for lr, v in zip(lrs, vals)
                 if isinstance(v, (int, float)) and not np.isnan(v)]
        best_lr, best_val = min(valid, key=lambda x: x[1]) if valid else (None, None)

        ax.semilogx(lrs, vals, f"{marker}-", color=color, linewidth=2, markersize=7,
                    label=label)
        if best_lr is not None:
            ax.axvline(best_lr, color="tomato", linestyle="--", linewidth=1.5,
                       label=f"best={best_lr:.0e}  (val={best_val:.4f})")
        ax.set_xlabel("Learning Rate (log scale)", fontsize=11)
        ax.set_ylabel("Validation Loss", fontsize=11)
        ax.set_title(f"{label}\nLR Sweep — Tiny Model", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("LR Sweep Comparison: SP vs μP (Task 3.5)", fontsize=13)
    out = Path(output_dir) / "plots" / "lr_sweep_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"LR comparison plot saved: {out}")
    return str(out)

def analyze(sp_path, mup_path, sp_sweep_path, mup_sweep_path, output_dir):
    # Load results
    with open(sp_path)  as f: sp_data  = json.load(f)
    with open(mup_path) as f: mup_data = json.load(f)

    sp_data.sort( key=lambda r: r["param_count"])
    mup_data.sort(key=lambda r: r["param_count"])

    sp_params  = np.array([r["param_count"] for r in sp_data],  dtype=float)
    sp_losses  = np.array([r["val_loss"]    for r in sp_data],  dtype=float)
    mup_params = np.array([r["param_count"] for r in mup_data], dtype=float)
    mup_losses = np.array([r["val_loss"]    for r in mup_data], dtype=float)

    # Fit both curves
    sp_popt,  sp_pcov  = fit_power_law(sp_params,  sp_losses)
    mup_popt, mup_pcov = fit_power_law(mup_params, mup_losses)

    sp_perr  = np.sqrt(np.diag(sp_pcov))
    mup_perr = np.sqrt(np.diag(mup_pcov))

    sp_r2  = 1 - np.sum((sp_losses  - power_law(sp_params,  *sp_popt))**2)  / np.sum((sp_losses  - sp_losses.mean())**2)
    mup_r2 = 1 - np.sum((mup_losses - power_law(mup_params, *mup_popt))**2) / np.sum((mup_losses - mup_losses.mean())**2)

    # Extrapolation: 10× XL params
    xl_params     = max(sp_params.max(), mup_params.max())
    target_params = xl_params * 10
    sp_pred,  sp_pred_std  = extrapolate(target_params, sp_popt,  sp_pcov)
    mup_pred, mup_pred_std = extrapolate(target_params, mup_popt, mup_pcov)

    # Choose best fit for final extrapolation report
    better = "μP" if mup_r2 > sp_r2 else "SP"
    best_pred = mup_pred if better == "μP" else sp_pred
    best_pred_std = mup_pred_std if better == "μP" else sp_pred_std

    # Print report
    print("\n" + "=" * 70)
    print("SCALING LAW COMPARISON — Tasks 3.5 & 3.6")
    print("=" * 70)

    print(f"\n{'Model':<14} {'Params':>12} {'SP Loss':>10} {'μP Loss':>10} {'Δ Loss':>10}")
    print("-" * 58)
    for sp_r, mup_r in zip(sp_data, mup_data):
        delta = mup_r["val_loss"] - sp_r["val_loss"]
        sign  = "+" if delta > 0 else ""
        print(f"{sp_r['name']:<14} {sp_r['param_count']:>12,} "
              f"{sp_r['val_loss']:>10.4f} {mup_r['val_loss']:>10.4f} "
              f"{sign}{delta:>9.4f}")

    print(f"\n{'':=<70}")
    print("POWER LAW FITS")
    print(f"{'':=<70}")
    a, alpha, c = sp_popt
    print(f"  SP  : L = {a:.4f} · N^(-{alpha:.4f}) + {c:.4f}")
    print(f"         a={a:.4f}±{sp_perr[0]:.4f}  "
          f"alpha={alpha:.4f}±{sp_perr[1]:.4f}  "
          f"c={c:.4f}±{sp_perr[2]:.4f}  R²={sp_r2:.4f}")

    a_m, alpha_m, c_m = mup_popt
    print(f"  μP  : L = {a_m:.4f} · N^(-{alpha_m:.4f}) + {c_m:.4f}")
    print(f"         a={a_m:.4f}±{mup_perr[0]:.4f}  "
          f"alpha={alpha_m:.4f}±{mup_perr[1]:.4f}  "
          f"c={c_m:.4f}±{mup_perr[2]:.4f}  R²={mup_r2:.4f}")

    print(f"\n  Kaplan et al. (natural language) α ≈ 0.076")
    print(f"  SP  scaling exponent: {alpha:.3f}  "
          f"({'steeper' if alpha > 0.076 else 'shallower'} than NL)")
    print(f"  μP  scaling exponent: {alpha_m:.3f}  "
          f"({'steeper' if alpha_m > 0.076 else 'shallower'} than NL)")
    print(f"  Better power-law fit (higher R²): {better}")

    print(f"\n{'':=<70}")
    print(f"EXTRAPOLATION TO ~880M PARAMS  (10× XL = {target_params:.2e})")
    print(f"{'':=<70}")
    print(f"  SP  predicted val_loss = {sp_pred:.4f} ± {sp_pred_std:.4f} (1σ)")
    print(f"  μP  predicted val_loss = {mup_pred:.4f} ± {mup_pred_std:.4f} (1σ)")
    print(f"  Using {better} fit (R²={max(sp_r2, mup_r2):.4f}): "
          f"{best_pred:.4f} ± {best_pred_std:.4f}")

    # Analysis questions (Tasks 3.5)
    print(f"\n{'':=<70}")
    print("ANALYSIS  (Task 3.5 discussion questions)")
    print(f"{'':=<70}")
    improvements = [(sp_r["param_count"], sp_r["val_loss"] - mup_r["val_loss"])
                    for sp_r, mup_r in zip(sp_data, mup_data)]
    print(f"  μP vs SP val_loss differences (negative = μP better):")
    for pc, diff in improvements:
        sign = "μP better" if diff > 0 else "SP better"
        print(f"    {pc:>12,.0f} params: SP-μP = {diff:+.4f}  ({sign})")

    best_improvement = max(improvements, key=lambda x: x[1])
    print(f"  Largest improvement at {best_improvement[0]:,} params: {best_improvement[1]:+.4f}")
    print(f"  μP steeper scaling: {'YES' if alpha_m > alpha else 'NO'} "
          f"(μP α={alpha_m:.3f} vs SP α={alpha:.3f})")

    # Save results
    comparison = {
        "sp_fit":  {"a": sp_popt[0],  "alpha": sp_popt[1],  "c": sp_popt[2],
                    "r_squared": sp_r2,
                    "a_std": sp_perr[0], "alpha_std": sp_perr[1], "c_std": sp_perr[2]},
        "mup_fit": {"a": mup_popt[0], "alpha": mup_popt[1], "c": mup_popt[2],
                    "r_squared": mup_r2,
                    "a_std": mup_perr[0], "alpha_std": mup_perr[1], "c_std": mup_perr[2]},
        "kaplan_nl_alpha": 0.076,
        "better_fit": better,
        "extrapolation": {
            "target_params": target_params,
            "sp_predicted_loss":      sp_pred,
            "sp_predicted_loss_std":  sp_pred_std,
            "mup_predicted_loss":     mup_pred,
            "mup_predicted_loss_std": mup_pred_std,
        },
        "data_points": {
            "sp":  [{"name": r["name"], "param_count": r["param_count"],
                     "val_loss": r["val_loss"]} for r in sp_data],
            "mup": [{"name": r["name"], "param_count": r["param_count"],
                     "val_loss": r["val_loss"]} for r in mup_data],
        },
    }
    out_path = Path(output_dir) / "results" / "comparison_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison results saved: {out_path}")

    # Plots
    plot_scaling_comparison(sp_data, mup_data, output_dir)

    if sp_sweep_path and mup_sweep_path:
        try:
            with open(sp_sweep_path)  as f: sp_sweep  = json.load(f)
            with open(mup_sweep_path) as f: mup_sweep = json.load(f)
            plot_lr_comparison(sp_sweep, mup_sweep, output_dir)
        except FileNotFoundError as e:
            print(f"[WARN] LR sweep plot skipped: {e}")

    return comparison


def main():
    p = argparse.ArgumentParser(
        description="Compare SP vs μP scaling and extrapolate (Tasks 3.5 & 3.6)."
    )
    p.add_argument("--sp_results",
                   default="outputs/results/scaling_results.json")
    p.add_argument("--mup_results",
                   default="outputs/results/mup_scaling_results.json")
    p.add_argument("--sp_sweep",
                   default="outputs/results/lr_sweep_results.json")
    p.add_argument("--mup_sweep",
                   default="outputs/results/mup_lr_sweep_results.json")
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    for path, label in [(args.sp_results, "SP results"),
                        (args.mup_results, "μP results")]:
        if not Path(path).exists():
            print(f"Error: {label} not found at {path}")
            print("Train all μP models first: python scripts/mup_train_all.py --best_lr <lr>")
            return

    sp_sweep_path  = args.sp_sweep  if Path(args.sp_sweep).exists()  else None
    mup_sweep_path = args.mup_sweep if Path(args.mup_sweep).exists() else None

    analyze(args.sp_results, args.mup_results,
            sp_sweep_path, mup_sweep_path,
            args.output_dir)


if __name__ == "__main__":
    main()
