# -*- coding: utf-8 -*-
"""Figures for the merged TCDS paper, drawn from the same per-seed JSON the
tables are audited against. Mean across 12 seeds, band = +-1 SE."""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
OUT = os.path.join(W, "tcds_merged", "figs")
os.makedirs(OUT, exist_ok=True)
CK = list(range(2000, 100001, 2000))


def load(d, c):
    out = []
    for s in range(12):
        r = {int(x["step"]): x["eval_dist"] * 1000 for x in json.load(open(os.path.join(W, d, "%s_seed%d.json" % (c, s))))}
        out.append([r[k] for k in CK])
    return np.array(out)


def band(ax, y, label, color, ls="-"):
    m = y.mean(0); se = y.std(0, ddof=1) / math.sqrt(y.shape[0])
    ax.plot(CK, m, ls, color=color, lw=1.6, label=label)
    ax.fill_between(CK, m - se, m + se, color=color, alpha=0.15, lw=0)


ARMS = [("blank", "blank", "model-free", "0.25"), ("prior", "prior", "body model", "tab:blue"),
        ("coach", "coach", "coach", "tab:red"), ("randprior", "prior", "random model", "tab:cyan"),
        ("randcoach", "coach", "random teacher", "tab:orange"), ("priorcoach", "priorcoach", "both", "tab:purple")]

# ---- Fig: converged learning curves ----
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, tag, unit in ((axes[0], "elbow", "mrad"), (axes[1], "finger", "mm")):
    for lab, cond, name, col in ARMS:
        band(ax, load("results_conv_%s_%s" % (tag, lab), cond), name, col, "--" if lab.startswith("rand") else "-")
    ax.set_ylim(*((0, 150) if tag == "elbow" else (40, 175)))
    ax.axvline(12000, color="k", ls=":", lw=1)
    ax.text(12500, ax.get_ylim()[1] * 0.97, "12k", va="top", fontsize=8)
    ax.set_xlabel("environment steps"); ax.set_ylabel("held-out error (%s)" % unit)
    ax.set_title("myoElbow" if tag == "elbow" else "myoFinger", fontsize=10)
    ax.set_xlim(0, 100000); ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "converged_curves.png"), dpi=200); plt.close(fig)

# ---- Fig: both withdrawals to 100k, elbow ----
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
ax = axes[0]
for arm, name, col in (("keep", "keep", "tab:blue"), ("soft", "soft", "tab:green"), ("purge", "purge", "tab:orange"), ("matched", "matched", "tab:red")):
    band(ax, load("results_matched100k_elbow", arm), name, col)
ax.set_title("body model withdrawn at 8k", fontsize=10)
ax = axes[1]
for d, arm, name, col in (("results_coach100k_elbow", "constant", "constant", "tab:blue"), ("results_coach100k_elbow", "abrupt", "abrupt", "tab:red"),
                          ("results_coach100k_elbow_off2", "selfanchor", "selfanchor", "tab:green")):
    band(ax, load(d, arm), name, col)
ax.set_title("teacher withdrawn at 8k", fontsize=10)
for ax in axes:
    ax.axvline(8000, color="k", ls=":", lw=1); ax.axvline(12000, color="k", ls=":", lw=1)
    ax.set_xlabel("environment steps"); ax.set_ylabel("held-out error (mrad)")
    ax.set_xlim(0, 100000); ax.set_ylim(0, 140); ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "withdrawal100k_curves.png"), dpi=200); plt.close(fig)
print("figs written")
