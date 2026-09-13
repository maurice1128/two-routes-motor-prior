"""Figures for the attachment-duration sweep.

  figs/attachment_components.png   the three withdrawal-cost components vs t_w
                                   (dip, error 4000 steps after withdrawal, endpoint),
                                   paired-t 95% CIs over seeds
  figs/attachment_curves.png       mean learning curves, one panel per t_w:
                                   abrupt / selfanchor / constant / none

Run with a Python that has matplotlib (the system python or .venv_mm), from this dir:
  python plot_attachment.py --dir results_attachment_myoelbow --shared results_withdrawal_myoelbow
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_attachment import load_sweep, load_shared, paired_ci  # noqa: E402

COMPS = [("abrupt", "constant", "total  (abrupt − constant)", "#444444"),
         ("selfanchor", "constant", "guidance only  (selfanchor − constant)", "#1f77b4"),
         ("abrupt", "selfanchor", "loss-term only  (abrupt − selfanchor)", "#d62728")]


def metrics(ser, t_w, K, const):
    pre, post = ser[t_w], ser[t_w + 1000]
    return {"dip": post - pre, "errk": ser[t_w + K], "end": ser[max(ser)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_attachment_myoelbow")
    ap.add_argument("--shared", default="results_withdrawal_myoelbow")
    ap.add_argument("--k", type=int, default=4000)
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(here, args.dir)
    sd = os.path.join(here, args.shared)
    os.makedirs(os.path.join(here, args.out), exist_ok=True)
    unit = "mrad" if "elbow" in args.dir else "mm"
    K = args.k

    sweep = load_sweep(d)
    constant = load_shared(sd, "constant")
    none = load_shared(sd, "none")
    tws = sorted({t for a in sweep for t in sweep[a]})

    # per (arm, t_w, seed) metrics; constant is "withdrawn at t_w" with nothing happening
    M = {}
    for arm in ("abrupt", "selfanchor"):
        for t_w in tws:
            for s, ser in sweep[arm][t_w].items():
                if t_w + 1000 in ser and t_w + K in ser:
                    M[(arm, t_w, s)] = metrics(ser, t_w, K, constant)
    for t_w in tws:
        for s, ser in constant.items():
            M[("constant", t_w, s)] = metrics(ser, t_w, K, constant)

    # ---- figure 1: components vs t_w ---------------------------------------
    titles = {"dip": f"immediate dip\nerr(t_w+1000) − err(t_w)  [{unit}]",
              "errk": f"error {K} steps after withdrawal\n(equal footing across t_w)  [{unit}]",
              "end": f"endpoint error at 12k\n(late t_w has fewer post steps)  [{unit}]"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, metric in zip(axes, ("dip", "errk", "end")):
        for i, (a, b, label, color) in enumerate(COMPS):
            xs, ms, los, his = [], [], [], []
            for t_w in tws:
                seeds = [s for s in range(12) if (a, t_w, s) in M and (b, t_w, s) in M]
                diffs = [M[(a, t_w, s)][metric] - M[(b, t_w, s)][metric] for s in seeds]
                if len(diffs) < 2:
                    continue
                m, lo, hi, _ = paired_ci(diffs)
                xs.append(t_w + (i - 1) * 120); ms.append(m); los.append(m - lo); his.append(hi - m)
            ax.errorbar(xs, ms, yerr=[los, his], fmt="o-", color=color, label=label,
                        capsize=3, lw=1.6, ms=5)
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xticks(tws)
        ax.set_xlabel("withdrawal step t_w  (= attachment duration)")
        ax.set_title(titles[metric], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Withdrawal cost decomposed along the attachment axis  (12 seeds, paired-t 95% CI)",
                 fontsize=11)
    fig.tight_layout()
    p1 = os.path.join(here, args.out, f"attachment_components_{unit}.png")
    fig.savefig(p1, dpi=150)

    # ---- figure 2: mean curves per t_w ---------------------------------------
    fig, axes = plt.subplots(1, len(tws), figsize=(3.6 * len(tws), 3.8), sharey=True)
    steps = sorted(next(iter(constant.values())))

    def mean_curve(curves):
        return [sum(c[s] for c in curves.values() if s in c) /
                max(1, sum(1 for c in curves.values() if s in c)) for s in steps]

    for ax, t_w in zip(axes, tws):
        ax.plot(steps, mean_curve(none), color="#999999", ls="--", label="none")
        ax.plot(steps, mean_curve(constant), color="#2ca02c", label="constant")
        ax.plot(steps, mean_curve(sweep["abrupt"][t_w]), color="#d62728", label="abrupt")
        ax.plot(steps, mean_curve(sweep["selfanchor"][t_w]), color="#1f77b4", label="selfanchor")
        ax.axvline(t_w, color="k", lw=0.8, ls=":")
        ax.set_title(f"withdraw at {t_w}")
        ax.set_xlabel("env steps")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"held-out error [{unit}]")
    axes[0].set_ylim(0, 200 if unit == "mrad" else 250)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(here, args.out, f"attachment_curves_{unit}.png")
    fig.savefig(p2, dpi=150)
    print("wrote", p1, "\n      ", p2)


if __name__ == "__main__":
    main()
