"""Mean learning curves for the single-point withdrawal decomposition, one panel per body.

  python plot_withdrawal.py            # -> figs/withdrawal_curves.png (myoElbow + myoFinger)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_withdrawal import load_dir  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = {"none": ("#999999", "--"), "constant": ("#2ca02c", "-"), "abrupt": ("#d62728", "-"),
         "selfanchor": ("#1f77b4", "-"), "randanchor": ("#ff7f0e", ":")}
BODIES = [("myoelbow", "mrad", 200), ("myofinger", "mm", 250)]


def main():
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (body, unit, ylim) in zip(axes, BODIES):
        d = os.path.join(HERE, f"results_withdrawal_{body}")
        if not os.path.isdir(d):
            continue
        curves = load_dir(d)
        for arm, (color, ls) in STYLE.items():
            if arm not in curves:
                continue
            steps = sorted(next(iter(curves[arm].values())))
            mean = [sum(c[s] for c in curves[arm].values()) / len(curves[arm]) for s in steps]
            ax.plot(steps, mean, color=color, ls=ls, label=f"{arm} (n={len(curves[arm])})")
        ax.axvline(8000, color="k", lw=0.8, ls=":")
        ax.set_ylim(0, ylim)
        ax.set_title(f"{body}: withdraw at 8000")
        ax.set_xlabel("env steps")
        ax.set_ylabel(f"held-out error [{unit}]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "figs", "withdrawal_curves.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
