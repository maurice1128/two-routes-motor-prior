# -*- coding: utf-8 -*-
"""v4 figures, drawn from the same per-seed JSON as the tables (common.py). Endpoint panels use the
late mean (mean of the evaluations at 90k, 92k, ..., 100k). Curves: mean across seeds, band +-1 SE;
component plots: paired-t 95% intervals."""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
plt.rcParams.update({"font.size": 8.5, "legend.fontsize": 7.5, "axes.titlesize": 8.5})

OUT = os.path.join(W, "tcds_v4", "figs"); os.makedirs(OUT, exist_ok=True)
CK = list(range(2000, 100001, 2000))


def arr(run, seeds, steps=CK):
    return np.array([[run[s][k] for k in steps] for s in seeds])


def band(ax, y, label, color, ls="-", steps=CK):
    m = y.mean(0); se = y.std(0, ddof=1) / math.sqrt(y.shape[0])
    ax.plot(steps, m, ls, color=color, lw=1.2, label=label)
    ax.fill_between(steps, m - se, m + se, color=color, alpha=0.15, lw=0)


def ci95(v):
    v = np.asarray(v, float); n = len(v); return v.mean(), T2[n - 1] * v.std(ddof=1) / math.sqrt(n)


def late_window(ax):
    ax.axvspan(90000, 100000, color="0.85", lw=0, zorder=0)


E, F = core()

# ---- Fig 1: the core conditions over training, both MyoSuite bodies (legend below the axes) ----
ARMS = [("blank", "blank (model-free)", "0.25", "-"), ("blank64", "blank64 (64 real per update)", "0.55", "--"),
        ("prior", "prior (body model)", "tab:blue", "-"), ("randprior", "randprior (random model)", "tab:cyan", "--"),
        ("coach", "coach (teacher T1)", "tab:red", "-"), ("randcoach", "randcoach (random teacher)", "tab:orange", "--"),
        ("priorcoach", "priorcoach (T1)", "tab:purple", "-"),
        ("coachT2", "coach (teacher T2, finger)", "tab:green", "-"), ("priorcoachT2", "priorcoach (T2, finger)", "tab:olive", "-")]
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5))
for ax, B, tag, unit in ((axes[0], E, "elbow", "mrad"), (axes[1], F, "finger", "mm")):
    late_window(ax)
    for k, name, col, ls in ARMS:
        if k in B:
            band(ax, arr(B[k], S12), name, col, ls)
    ax.set_ylim(*((0, 150) if tag == "elbow" else (20, 175)))
    ax.set_xlabel("environment steps"); ax.set_ylabel("held-out error (%s)" % unit)
    ax.set_title("myoElbow" if tag == "elbow" else "myoFinger")
    ax.set_xlim(0, 100000); ax.grid(alpha=0.3)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.savefig(os.path.join(OUT, "converged_curves.png"), dpi=300); plt.close(fig)

# ---- Fig 2: joint-count series (late mean) ----
AR = arms()
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
ax = axes[0]; A4, s4 = AR["arm4"]; late_window(ax)
for k, name, col, ls in (("blank", "blank", "0.25", "-"), ("blank64", "blank64", "0.55", "--"),
                         ("prior", "prior (body model)", "tab:blue", "-"), ("randprior", "randprior", "tab:cyan", "--")):
    band(ax, arr(A4[k], s4), name, col, ls)
ax.set_title("4-joint arm ($n{=}%d$)" % len(s4)); ax.set_xlabel("environment steps"); ax.set_ylabel("held-out error (mm)")
ax.set_xlim(0, 100000); ax.set_ylim(0, 600); ax.grid(alpha=0.3); ax.legend(frameon=False)
ax = axes[1]
for k, (name, x, y, color) in enumerate((("prior $-$ blank", "prior", "blank", "tab:blue"), ("prior $-$ blank64", "prior", "blank64", "tab:purple"),
                                         ("randprior $-$ blank", "randprior", "blank", "tab:cyan"))):
    ms, es = [], []
    for b in ("arm1", "arm2", "arm3", "arm4"):
        A, seeds = AR[b]; bl = np.mean([end(A["blank"], s) for s in seeds])
        m, e = ci95([(end(A[x], s) - end(A[y], s)) / bl for s in seeds]); ms.append(m); es.append(e)
    ax.errorbar(np.arange(1, 5) + (k - 1) * 0.08, ms, yerr=es, fmt="o-", ms=3, lw=1, capsize=2, color=color, label=name)
ax.axhline(0, color="k", lw=0.6); ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("joints")
ax.set_ylabel("paired difference\n(fraction of blank mean)")
ax.set_title("late mean, by joint count"); ax.grid(alpha=0.3); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "joint_series.png"), dpi=300); plt.close(fig)

# ---- Fig 3: teacher withdrawal over training (additive): constant, none, abrupt at 2k and 8k ----
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
for ax, body, unit, ylim in ((axes[0], "elbow", "mrad", (0, 150)), (axes[1], "finger", "mm", (20, 175))):
    sw = sweep(body); late_window(ax)
    band(ax, arr(sw["none"], S12), "never guided (none)", "0.25")
    band(ax, arr(sw["con"], S12), "teacher kept (constant)", "tab:red")
    band(ax, arr(sw["arms"][2000][0], S12), "withdrawn at 2k", "tab:green", "--")
    band(ax, arr(sw["arms"][8000][0], S12), "withdrawn at 8k", "tab:blue", "--")
    ax.set_xlim(0, 100000); ax.set_ylim(*ylim); ax.grid(alpha=0.3)
    ax.set_xlabel("environment steps"); ax.set_ylabel("held-out error (%s)" % unit)
    ax.set_title("myoElbow (teacher T1)" if body == "elbow" else "myoFinger (teacher T2)")
axes[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "withdrawal_curves_v4.png"), dpi=300); plt.close(fig)

# ---- Fig 4: attachment sweep, components vs t_w: immediate dip and late mean ----
COMP = [("total (abrupt $-$ constant)", "ab", "con", "k"), ("teacher (selfanchor $-$ constant)", "sa", "con", "tab:blue"),
        ("loss term (abrupt $-$ selfanchor)", "ab", "sa", "tab:red"), ("abrupt $-$ none", "ab", "none", "tab:green")]
fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8))
for row, (body, unit) in enumerate((("elbow", "mrad"), ("finger", "mm"))):
    sw = sweep(body)
    for col, metric in enumerate(("dip", "late")):
        ax = axes[row, col]
        for k, (name, x, y, color) in enumerate(COMP):
            ms, es = [], []
            for tw in TWS:
                if metric == "dip":
                    a, sa = sw["dip"][tw]; src = {"ab": a, "sa": sa, "con": sw["dcon"], "none": sw["dnone"]}
                    f = lambda r, s: r[s][tw + 1000] - r[s][tw]
                else:
                    a, sa = sw["arms"][tw]; src = {"ab": a, "sa": sa, "con": sw["con"], "none": sw["none"]}
                    f = end
                m, e = ci95([f(src[x], s) - f(src[y], s) for s in S12]); ms.append(m); es.append(e)
            ax.errorbar(np.array(TWS) / 1000.0 + (k - 1.5) * 0.08, ms, yerr=es, fmt="o-", ms=3, lw=1, capsize=2, color=color, label=name)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title("%s: %s" % ("myoElbow" if body == "elbow" else "myoFinger (teacher T2)", "immediate dip" if metric == "dip" else "late mean (90–100k)"))
        ax.set_xlabel("attachment duration $t_w$ (k steps)"); ax.set_ylabel("paired difference (%s)" % unit); ax.grid(alpha=0.3)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(OUT, "sweep_components_v4.png"), dpi=300); plt.close(fig)
print("figs written")
