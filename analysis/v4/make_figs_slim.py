# -*- coding: utf-8 -*-
"""Figures of the v4 paper (same data access as common.py). IEEE graphics rules: text >= 8 pt at final
size (figures are included at full text width, so 1:1), 600 dpi, and every series distinguishable by
line style or marker as well as color.
Fig. 1 (teaser): (a) myoFinger teacher withdrawal curves; (b) random-minus-trained body-model gap by
                 joint count, both endpoints.
Fig. 3: the drop at withdrawal, teacher vs imitation-term component, against t_w.
Fig. 4: learning curves of blank / blank64 / prior / randprior on myoElbow, myoFinger, 4-joint arm."""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from common import *
plt.rcParams.update({"font.size": 9, "legend.fontsize": 8, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
OUT = os.path.join(W, "tcds_v4", "figs")
DPI = 600
CK = list(range(2000, 100001, 2000))
CK1 = list(range(1000, 100001, 1000))   # withdrawal runs are evaluated every 1,000 steps
KFMT = FuncFormatter(lambda x, _: "0" if x == 0 else "%dk" % (x / 1000))


def arr(run, seeds, steps=CK): return np.array([[run[s][k] for k in steps] for s in seeds])


def band(ax, y, label, color, ls="-", steps=CK):
    m = y.mean(0); se = y.std(0, ddof=1) / math.sqrt(y.shape[0])
    ax.plot(steps, m, ls, color=color, lw=1.2, label=label); ax.fill_between(steps, m - se, m + se, color=color, alpha=0.15, lw=0)


def ci95(v):
    v = np.asarray(v, float); n = len(v); return v.mean(), T2[n - 1] * v.std(ddof=1) / math.sqrt(n)


E, F = core(); AR = arms()

# ---- Fig 1 teaser
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[1]
for k, (e, f, col, lab, hatch) in enumerate((("late", end, "tab:blue", "Mean over 90–100k", None),
                                             ("100k", lambda r, s: r[s][100000], "tab:orange", "Single 100k evaluation", "///"))):
    ms, es = [], []
    for b in ("arm1", "arm2", "arm3", "arm4"):
        A, s = AR[b]; m, h = ci95([f(A["randprior"], t) - f(A["prior"], t) for t in s]); ms.append(m); es.append(h)
    ax.bar(np.arange(1, 5) + (k - 0.5) * 0.36, ms, 0.34, yerr=es, capsize=2, color=col, label=lab, hatch=hatch, edgecolor="white" if hatch else None)
ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("Joints of the planar muscle arm")
ax.set_ylabel("Random minus trained model\n(held-out error, mm)")
ax.set_title("(b) A body model is worth its content"); ax.grid(alpha=0.3, axis="y"); ax.legend(frameon=False, loc="upper left")
ax = axes[0]; sw = sweep("finger"); ax.axvspan(90000, 100000, color="0.85", lw=0, zorder=0)
band(ax, arr(sw["none"], S12, CK1), "Never guided", "0.25", steps=CK1); band(ax, arr(sw["con"], S12, CK1), "Teacher kept", "tab:red", steps=CK1)
band(ax, arr(sw["arms"][2000][0], S12, CK1), "Withdrawn at 2k", "tab:green", "--", steps=CK1)
band(ax, arr(sw["arms"][8000][0], S12, CK1), "Withdrawn at 8k", "tab:blue", "-.", steps=CK1)
ax.set_xlim(0, 100000); ax.set_ylim(20, 185); ax.grid(alpha=0.3); ax.set_xlabel("Environment steps"); ax.set_ylabel("Held-out error (mm)")
ax.xaxis.set_major_formatter(KFMT)
ax.set_title("(a) Taking a competent teacher away (myoFinger)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "teaser_slim.png"), dpi=DPI); plt.close(fig)

# ---- Fig 4 learning curves
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.9))
A4, s4 = AR["arm4"]
for ax, B, seeds, title, unit, ylim in ((axes[0], E, S12, "myoElbow (1 joint)", "mrad", (0, 150)), (axes[1], F, S12, "myoFinger (4 joints)", "mm", (20, 175)),
                                       (axes[2], A4, s4, "4-joint arm ($n{=}24$)", "mm", (0, 600))):
    ax.axvspan(90000, 100000, color="0.85", lw=0, zorder=0)
    for k, name, col, ls in (("blank", "Model-free (blank)", "0.25", "-"), ("blank64", "Model-free, 64 real (blank64)", "0.55", ":"),
                             ("prior", "Trained body model", "tab:blue", "-"), ("randprior", "Random body model", "tab:cyan", "--")):
        band(ax, arr(B[k], seeds), name, col, ls)
    ax.set_xlim(0, 100000); ax.set_ylim(*ylim); ax.grid(alpha=0.3); ax.set_title(title); ax.set_xlabel("Environment steps")
    ax.set_ylabel("Held-out error (%s)" % unit); ax.set_xticks([0, 50000, 100000]); ax.xaxis.set_major_formatter(KFMT)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(os.path.join(OUT, "curves_slim.png"), dpi=DPI); plt.close(fig)

# ---- Fig 3 the drop at withdrawal
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7))
d = lambda r, s, tw: r[s][tw + 1000] - r[s][tw]
for ax, body, unit in ((axes[0], "elbow", "mrad"), (axes[1], "finger", "mm")):
    sw = sweep(body)
    for k, (name, x, y, col, fmt) in enumerate((("Total (abrupt $-$ constant)", "ab", "con", "k", "o-"),
                                               ("Teacher (selfanchor $-$ constant)", "sa", "con", "tab:blue", "s--"),
                                               ("Imitation term (abrupt $-$ selfanchor)", "ab", "sa", "tab:red", "^:"))):
        ms, es = [], []
        for tw in TWS:
            a, sa = sw["dip"][tw]; src = {"ab": a, "sa": sa, "con": sw["dcon"]}
            m, h = ci95([d(src[x], s, tw) - d(src[y], s, tw) for s in S12]); ms.append(m); es.append(h)
        ax.errorbar(np.array(TWS) / 1000.0 + (k - 1) * 0.1, ms, yerr=es, fmt=fmt, ms=4, lw=1.1, capsize=2, color=col, label=name)
    ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3)
    ax.set_title("myoElbow (teacher T1)" if body == "elbow" else "myoFinger (teacher T2)")
    ax.set_xlabel("Teacher attached for $t_w$ (k steps)"); ax.set_ylabel("Rise in error over 1k steps (%s)" % unit)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(OUT, "dip_slim.png"), dpi=DPI); plt.close(fig)
print("figs written")
