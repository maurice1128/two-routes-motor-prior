"""
figstyle.py -- shared visual grammar for the MotorPrior figure set.

VISUAL GRAMMAR
  hue          = condition family (Okabe-Ito, colourblind-safe, fixed)
  bar hatch    = guidance schedule (kept on / faded or soft-withdrawn / strict cut)
  marker shape = guidance schedule (circle = kept on, square = annealed/withdrawn)
  line style   = guidance schedule (solid = kept on, dashed = annealed/withdrawn)
  NOTE: the ANNEALED coach is a taper that fades from step 0 and reaches 0 at 8k.
  It is NOT a withdrawal at 8k, and is never labelled as one (paper S3.2).
  MARKER FILL  = STATISTICAL FLAG  (filled = ROBUST, hollow = ROBUST-NULL,
                 grey marker + dashed CI + hatched band + explicit label =
                 METHOD-DEPENDENT)
  The flag is NEVER encoded by hue.

Units: elbow panels are milliradians, finger panels millimetres. The two bodies
never share a magnitude axis and are never compared numerically.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
})

COND_COLORS = {"blank": "#999999", "prior": "#0072B2",
               "coach": "#E69F00", "priorcoach": "#009E73"}
RAND_COLOR = "#D55E00"
MID_COLOR = "#FFFFFF"
COND_LABEL = {"blank": "blank", "prior": "prior", "coach": "coach",
              "priorcoach": "prior+coach"}
BODY_COLOR = {"elbow": "#332288", "finger": "#CC6677"}
BODY_LABEL = {"elbow": "Elbow", "finger": "Finger"}

KEY_COLOR = {"blank": COND_COLORS["blank"], "prior": COND_COLORS["prior"],
             "prior_soft": COND_COLORS["prior"], "prior_strict": COND_COLORS["prior"],
             "coach_const": COND_COLORS["coach"], "coach_ann": COND_COLORS["coach"],
             "coach_abrupt": COND_COLORS["coach"],
             "pc_const": COND_COLORS["priorcoach"], "pc_ann": COND_COLORS["priorcoach"]}
KEY_HATCH = {"blank": "", "prior": "", "prior_soft": "//", "prior_strict": "xx",
             "coach_const": "", "coach_ann": "//", "coach_abrupt": "xx",
             "pc_const": "", "pc_ann": "//"}
KEY_LABEL = {"blank": "blank", "prior": "prior\nkept",
             "prior_soft": "prior\nwithdrawn\nsoft",
             "prior_strict": "prior\nwithdrawn\nstrict",
             "coach_const": "coach\nkept\n(constant)",
             "coach_ann": "coach\nannealed\n(faded to 0)",
             "coach_abrupt": "coach\nwithdrawn\n(abrupt)",
             "pc_const": "prior+coach\n(constant)",
             "pc_ann": "prior+coach\n(annealed)"}
KEY_LABEL_1L = {"blank": "blank", "prior": "prior kept",
                "prior_soft": "prior withdrawn (soft)",
                "prior_strict": "prior withdrawn (strict)",
                "coach_const": "coach (constant)",
                "coach_ann": "coach (annealed, faded to 0 by 8k)",
                "coach_abrupt": "coach (abrupt cut)",
                "pc_const": "prior+coach (constant)", "pc_ann": "prior+coach (annealed)"}

ERR_LABEL_FINAL = {"elbow": "Final joint-angle error (mrad)",
                   "finger": "Final fingertip error (mm)"}
ERR_LABEL_HELDOUT = {"elbow": "Held-out joint-angle error (mrad)",
                     "finger": "Held-out fingertip error (mm)"}
BODY_TITLE = {"elbow": "Elbow (myoElbow) — joint angle, mrad",
              "finger": "Finger (myoFinger) — fingertip, mm"}

CI_NOTE = ("Every error bar is a PAIRED t 95% CI (df = n−1 = 11) — the paper's "
           "declared primary estimator. The BCa bootstrap is used only to assign the flag.")

# ------------------------------------------------------------- flag rendering
MD_GREY = "#8A8A8A"
MD_BAND = "#DCDCDC"
FLAG_TAG = {"ROBUST": "ROBUST",
            "ROBUST-NULL": "ROBUST-NULL (n.s.)",
            "METHOD-DEPENDENT": "METHOD-DEPENDENT — NOT A FINDING"}


def flag_kw(fl, color, marker="o"):
    if fl == "ROBUST":
        return dict(marker=marker, mfc=color, mec="black", mew=0.8, ms=8.5,
                    ecolor=color, elw=2.0, ci_ls="-", txt=color)
    if fl == "ROBUST-NULL":
        return dict(marker=marker, mfc="white", mec=color, mew=1.7, ms=8.5,
                    ecolor=color, elw=1.6, ci_ls="-", txt="#555555")
    return dict(marker=marker, mfc=MD_GREY, mec="black", mew=1.0, ms=9.0,
                ecolor=MD_GREY, elw=1.9, ci_ls=(0, (3, 2)), txt="#4A4A4A")


def flag_errorbar(ax, x, y, lo, hi, fl, color, marker="o", horizontal=True):
    """Draw one point + its paired t 95% CI, styled by flag (never by hue)."""
    kw = flag_kw(fl, color, marker)
    err = dict(fmt=kw["marker"], mfc=kw["mfc"], mec=kw["mec"], mew=kw["mew"],
               ms=kw["ms"], ecolor=kw["ecolor"], elinewidth=kw["elw"],
               capsize=5, zorder=3)
    if horizontal:
        eb = ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], **err)
    else:
        eb = ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], **err)
    for bl in eb[2]:
        bl.set_linestyle(kw["ci_ls"])
    return kw


def md_band(ax, ypos, half=0.42, horizontal=True):
    """Grey hatched band marking a METHOD-DEPENDENT cell."""
    f = ax.axhspan if horizontal else ax.axvspan
    f(ypos - half, ypos + half, facecolor=MD_BAND, edgecolor=MD_GREY,
      hatch="////", alpha=0.55, lw=0.8, zorder=0)


# ----------------------------------------------- Holm multiplicity marking
# A cell can be ROBUST (uncorrected t and BCa agree, both exclude 0) and still
# fail Holm-Bonferroni over the paper's declared 28-cell claim family (§4.9).
# That is marked THREE redundant ways -- never by hue: a glyph prefixed to the
# row label, a back-hatched band across the row, and an explicit tag beside the
# interval -- plus a legend entry.
HOLM_EDGE = "#4A4A4A"
HOLM_GLYPH = {"pass": "✓", "fail": "✗"}
HOLM_TAG = {"pass": "✓ survives Holm",
            "fail": "✗ FAILS Holm"}


def holm_band(ax, ypos, half=0.42, horizontal=True):
    """Back-hatched band marking a cell that does NOT survive Holm."""
    f = ax.axhspan if horizontal else ax.axvspan
    f(ypos - half, ypos + half, facecolor="none", edgecolor=HOLM_EDGE,
      hatch="\\\\\\\\", alpha=0.45, lw=0.9, zorder=0)


def holm_legend_handles():
    return [
        Patch(facecolor="white", edgecolor="white",
              label="✓ / ✗ = survives / fails Holm–Bonferroni over the LITERAL 72-cell family (§4.9)"),
        Patch(facecolor="none", edgecolor=HOLM_EDGE, hatch="\\\\\\\\",
              label="✗ + back-hatched band = fails it (uncorrected CI still drawn); per-cell tag gives BOTH families"),
    ]


def flag_legend_handles():
    return [
        Line2D([0], [0], marker="o", ls="-", color="#444444", mfc="#444444",
               mec="black", ms=8, lw=2.0,
               label="ROBUST — t and BCa agree, both exclude 0"),
        Line2D([0], [0], marker="o", ls="-", color="#444444", mfc="white",
               mec="#444444", mew=1.7, ms=8, lw=1.6,
               label="ROBUST-NULL — both intervals contain 0"),
        Line2D([0], [0], marker="o", ls=(0, (3, 2)), color=MD_GREY, mfc=MD_GREY,
               mec="black", ms=8, lw=1.9,
               label="METHOD-DEPENDENT — t and BCa disagree, NOT a finding"),
        Patch(facecolor=MD_BAND, edgecolor=MD_GREY, hatch="////",
              label="hatched grey band = METHOD-DEPENDENT cell"),
    ]


def schedule_legend_handles():
    return [
        Line2D([0], [0], marker="o", ls="-", color="#444444", ms=8, lw=1.8,
               label="circle / solid = guidance KEPT ON (constant)"),
        Line2D([0], [0], marker="s", ls="--", color="#444444", ms=8, lw=1.8,
               label="square / dashed = guidance ANNEALED (faded from step 0, reaches 0 at 8k)"),
    ]


def footer(fig, *lines, **kw):
    """Bottom caption block. Keep each line under ~135 characters so that it
    cannot run off the canvas at any of the figure widths used here."""
    color = kw.get("color", "#555555")
    size = kw.get("size", 7.0)
    step = kw.get("step") or _line_step(fig, size)
    y0 = kw.get("y0", 0.008)
    for i, line in enumerate(reversed(lines)):
        fig.text(0.5, y0 + i * step, line, ha="center", fontsize=size, color=color)


def _line_step(fig, size=7.0):
    """Line spacing in FIGURE FRACTION, so captions never self-overlap on short
    figures (a fixed fraction would collapse as the figure gets shorter)."""
    return (size * 1.5) / (fig.get_figheight() * 72.0)


def footer_room(fig, n_lines, size=7.0, pad=0.022):
    """Bottom fraction to reserve in tight_layout(rect=...) for `footer`."""
    return pad + n_lines * _line_step(fig, size)


def flag_legend_top(fig, y=0.885, ncol=4, fontsize=6.8, schedule=False, holm=False):
    """Flag key placed in the band under the suptitle, where it cannot collide
    with data or with in-axes annotations."""
    h = flag_legend_handles()
    if schedule:
        h = h + schedule_legend_handles()
    if holm:
        h = h + holm_legend_handles()
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol, fontsize=fontsize, handlelength=2.4, columnspacing=1.8)


def bar_ci(ax, x, stats, keys, annotate="{:.1f}", edge="black"):
    """Grouped bars with one-sample t 95% CI whiskers; hatch encodes schedule."""
    means = [s[0] for s in stats]
    errs = [s[0] - s[1] for s in stats]
    bars = ax.bar(x, means, yerr=errs, capsize=4,
                  color=[KEY_COLOR[k] for k in keys],
                  edgecolor=edge, linewidth=0.8,
                  error_kw=dict(lw=1.0, ecolor="#333333"))
    for b, k in zip(bars, keys):
        h = KEY_HATCH.get(k, "")
        if h:
            b.set_hatch(h)
    for xi, m, er in zip(x, means, errs):
        ax.text(xi, m + er + max(errs) * 0.18, annotate.format(m),
                ha="center", va="bottom", fontsize=8.5)
    return means, errs


def forest(ax, rows, unit, xlabel, title=None, title_color="#222222",
           label_fs=8.2, val_fs=8.0):
    """Horizontal forest panel.

    rows: list of dicts with keys
      label   row label
      res     result dict (mean/tlo/thi/flag)
      color   hue = condition family
      marker  schedule marker ('o' kept on, 's' withdrawn)
      holm    optional 'pass' / 'fail' -- Holm verdict over the declared
              28-cell claim family; drawn as a label glyph, a band and a tag.
    Draws the paired t 95% CI, the flag styling and an explicit flag tag.
    """
    y = list(range(len(rows)))[::-1]
    labels = []
    for i, r in enumerate(rows):
        res = r["res"]
        holm = r.get("holm")
        if res["flag"] == "METHOD-DEPENDENT":
            md_band(ax, y[i])
        if holm == "fail":
            holm_band(ax, y[i])
        kw = flag_errorbar(ax, res["mean"], y[i], res["tlo"], res["thi"],
                           res["flag"], r["color"], r.get("marker", "o"))
        tag = FLAG_TAG[res["flag"]]
        if not res.get("in_rec", True):
            tag += "  [not in RECOMPUTED.md]"
        if r.get("holm_text"):
            tag += "  ·  " + r["holm_text"]
        elif holm in HOLM_TAG:
            tag += "  ·  " + HOLM_TAG[holm]
        labels.append(("%s %s" % (HOLM_GLYPH[holm], r["label"])) if holm in HOLM_GLYPH
                      else r["label"])
        ax.annotate("%+.1f [%+.1f, %+.1f] %s   %s"
                    % (res["mean"], res["tlo"], res["thi"], unit, tag),
                    (res["mean"], y[i]), textcoords="offset points",
                    xytext=(0, 12), ha="center", va="bottom",
                    fontsize=val_fs, color=kw["txt"],
                    fontweight="bold" if (res["flag"] == "METHOD-DEPENDENT"
                                          or holm == "fail") else "normal")
    ax.axvline(0, color="black", lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_fs)
    ax.set_ylim(-0.62, len(rows) - 1 + 0.80)
    ax.set_xlabel(xlabel, fontsize=8.5)
    if title:
        ax.set_title(title, color=title_color, fontweight="bold", fontsize=9.5)
    ax.margins(x=0.26)
