"""Entropy-temperature (alpha) and eval-return traces per arm, both bodies.
Shows the myoFinger abrupt drift is not a temperature effect.  python plot_alpha.py"""
import glob, json, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = {"none": ("#999999", "--"), "constant": ("#2ca02c", "-"), "abrupt": ("#d62728", "-"),
         "selfanchor": ("#1f77b4", "-"), "randanchor": ("#ff7f0e", ":")}
fig, axes = plt.subplots(2, 2, figsize=(9, 5.6))
for col, body in enumerate(["myoelbow", "myofinger"]):
    for arm, (c, ls) in STYLE.items():
        acc, ret = defaultdict(list), defaultdict(list)
        for f in glob.glob(os.path.join(HERE, f"results_withdrawal_{body}", f"{arm}_seed*.json")):
            for r in json.load(open(f)):
                acc[r["step"]].append(r["alpha"]); ret[r["step"]].append(r["eval_return"])
        steps = sorted(acc)
        axes[0, col].plot(steps, [sum(acc[s]) / len(acc[s]) for s in steps], color=c, ls=ls, label=arm)
        axes[1, col].plot(steps, [sum(ret[s]) / len(ret[s]) for s in steps], color=c, ls=ls, label=arm)
    for row, yl in enumerate(["entropy temperature alpha", "eval return"]):
        ax = axes[row, col]; ax.axvline(8000, color="k", lw=0.8, ls=":"); ax.grid(alpha=0.3)
        ax.set_ylabel(yl); ax.set_xlabel("env steps")
    axes[0, col].set_title(body); axes[0, col].set_yscale("log")
axes[0, 0].legend(fontsize=8)
fig.tight_layout(); out = os.path.join(HERE, "figs", "alpha_traces.png"); fig.savefig(out, dpi=150); print("wrote", out)
