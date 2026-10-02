# -*- coding: utf-8 -*-
"""Schematic of the teacher-withdrawal conditions and the self-anchor decomposition (no data).
Column width (3.5 in) at 1:1, all text >= 8 pt, 600 dpi."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "schematic_withdrawal.png")
plt.rcParams.update({"font.size": 8})
C_T, C_S, C_N = "#d62728", "#9ecae1", "#d9d9d9"
TW, END = 3.0, 10.0

fig, ax = plt.subplots(figsize=(3.5, 2.75))
rows = [("Never guided\n(none)", [(0, END, C_N, "Reward only", None)]),
        ("Teacher kept\n(constant)", [(0, END, C_T, "Reward + imitate teacher", None)]),
        ("Self-anchor\n(selfanchor)", [(0, TW, C_T, "", None), (TW, END, C_S, "Reward + imitate\nfrozen student copy", "...")]),
        ("Withdrawn\n(abrupt)", [(0, TW, C_T, "", None), (TW, END, C_N, "Reward only", None)])]
for i, (name, segs) in enumerate(rows):
    y = 3 - i
    ax.text(-0.25, y, name, ha="right", va="center", fontsize=8)
    for x0, x1, c, lab, hatch in segs:
        ax.add_patch(FancyBboxPatch((x0, y - 0.31), x1 - x0, 0.62, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=c, ec="none", lw=0, zorder=2))
        if lab:
            ax.text((x0 + x1) / 2, y, lab, ha="center", va="center", fontsize=8,
                    color="white" if c == C_T else "black", zorder=3)
ax.axvline(TW, color="k", lw=0.8, ls="--", zorder=1)
ax.text(TW + 0.12, 3.62, "$t_w$: teacher withdrawn", ha="left", va="bottom", fontsize=8)


def brace(y0, y1, x, text, col):
    ax.annotate("", xy=(x, y0), xytext=(x, y1), arrowprops=dict(arrowstyle="<->", lw=0.9, color=col))
    ax.text(x + 0.15, (y0 + y1) / 2, text, ha="left", va="center", fontsize=8, color=col)


brace(2, 1, END + 0.25, "Teacher", C_T)
brace(1, 0, END + 0.25, "Imitation\nterm", "#2171b5")
ax.set_xlim(-0.1, END + 3.1); ax.set_ylim(-0.55, 3.95)
ax.set_xticks([0, TW, END]); ax.set_xticklabels(["0", "$t_w$", "100k"]); ax.set_yticks([])
ax.tick_params(labelsize=8); ax.set_xlabel("Training steps", fontsize=8)
ax.spines["bottom"].set_bounds(0, END)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.27, right=0.99, top=0.97, bottom=0.16)
fig.savefig(OUT, dpi=600); plt.close(fig)
print("wrote", OUT)
