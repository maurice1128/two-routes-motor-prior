# -*- coding: utf-8 -*-
"""Schematic of the teacher-withdrawal conditions and the self-anchor decomposition (no data)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "schematic_withdrawal.png")
plt.rcParams.update({"font.size": 8})
C_T, C_S, C_N = "#d62728", "#1f77b4", "#bdbdbd"
TW, END = 3.0, 10.0

fig, ax = plt.subplots(figsize=(3.5, 2.35))
rows = [("never guided\n(none)", [(0, END, C_N, "reward only")]),
        ("teacher kept\n(constant)", [(0, END, C_T, "reward + imitate teacher")]),
        ("self-anchor\n(selfanchor)", [(0, TW, C_T, ""), (TW, END, C_S, "reward + imitate\nfrozen student copy")]),
        ("withdrawn\n(abrupt)", [(0, TW, C_T, ""), (TW, END, C_N, "reward only")])]
for i, (name, segs) in enumerate(rows):
    y = 3 - i
    ax.text(-0.25, y, name, ha="right", va="center", fontsize=7.5)
    for x0, x1, c, lab in segs:
        ax.add_patch(FancyBboxPatch((x0, y - 0.28), x1 - x0, 0.56, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=c, ec="none", alpha=0.85))
        if lab:
            ax.text((x0 + x1) / 2, y, lab, ha="center", va="center", fontsize=6.4,
                    color="white" if c != C_N else "black")
ax.axvline(TW, color="k", lw=0.8, ls="--")
ax.text(TW + 0.12, 3.62, "$t_w$: teacher withdrawn", ha="left", va="bottom", fontsize=7)
ax.annotate("student\nfrozen here", xy=(TW, 1.0), xytext=(1.45, 0.42), fontsize=6.3, ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.6))


def brace(y0, y1, x, text, col):
    ax.annotate("", xy=(x, y0), xytext=(x, y1), arrowprops=dict(arrowstyle="<->", lw=0.9, color=col))
    ax.text(x + 0.15, (y0 + y1) / 2, text, ha="left", va="center", fontsize=6.6, color=col)


brace(2, 1, END + 0.25, "teacher", C_T)
brace(1, 0, END + 0.25, "imitation\nterm", C_S)
ax.set_xlim(-0.1, END + 2.9); ax.set_ylim(-0.55, 3.95)
ax.set_xticks([0, TW, END]); ax.set_xticklabels(["0", "$t_w$", "100k"]); ax.set_yticks([])
ax.set_xlabel("training steps", fontsize=7.5)
ax.spines["bottom"].set_bounds(0, END)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.27, right=0.99, top=0.97, bottom=0.19)
fig.savefig(OUT, dpi=300); plt.close(fig)
print("wrote", OUT)
