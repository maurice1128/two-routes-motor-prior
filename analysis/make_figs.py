"""
make_figs.py -- Publication-quality figures for the MotorPrior paper.

EVERY PLOTTED INTERVAL IS THE PAPER'S DECLARED STATISTICAL METHOD
================================================================
  PRIMARY      paired t 95% CI, mean(d) +/- t_{0.975,n-1} * sd(d)/sqrt(n), df = 11.
               (The z interval `mean +/- 1.96*SE` drawn by every previous version
               of this script is declared INVALID by the paper: it is only 89.1%
               of the correct half-width.)
  SENSITIVITY  BCa bootstrap 95% CI (20000 resamples, RNG seed 0, resampling
               seeds) -- used ONLY to assign the flag, never drawn as the bar.
  FLAG         ROBUST / ROBUST-NULL / METHOD-DEPENDENT.

`t_ci`, `bca_ci`, `flag` are imported from `recompute_all.py` (via figstats.py);
nothing is reimplemented. Every contrast is verified against `RECOMPUTED.md`
BEFORE any PNG is written: on any disagreement the script prints the offending
rows and exits WITHOUT plotting.

SIGN CONVENTION: each contrast is plotted in the orientation used by
`RECOMPUTED.md` (e.g. `coach(const) - prior`), so the plotted number IS the
authoritative number. Lower error = better, so a NEGATIVE difference means the
first-named condition is ahead.

UNITS: myoElbow eval_dist is a joint-angle error in radians (x1000 = mrad);
myoFinger is a 3-D fingertip error in metres (x1000 = mm). The bodies never
share a magnitude axis and are never compared numerically.

Data provenance: see `figstats.py` (identical directory/condition map to
`recompute_all.py`), plus `results_plateau2_{body}/*.log` for the n=3 Fig. 8.

TWO VARIANTS, ONE COMMAND: this script writes the AUDIT set used by
`PAPER_TMLR.md` and then calls `make_figs_clean.build_clean`, which writes the
CLEAN variants used by `PAPER_CLEAN.md` -- `fig12_crossover_clean.png`,
`fig10_withdrawal_clean.png`, `fig5_randprior_clean.png`. The clean variants use
NEW filenames and never overwrite an audit figure; their values go through the
same RECOMPUTED.md check and abort the same way.

CANONICAL INTERPRETER: `..\.venv\Scripts\python.exe` (3.12) and
`..\.venv_mm\Scripts\python.exe` (3.11) produce IDENTICAL numbers, but the two
do not produce byte-identical PNGs for the most text-dense figures (10 and 12).
The archived audit PNGs were written by the 3.11 venv; regenerate them with the
same one if byte-identity matters.
"""
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import figstats as FS
import figcontrasts as FC
from figstats import (C, cond_mean, sd_ratio_ci, vals, curve, t_ci, UNIT, BODIES,
                      BLANK_HO, PRIOR_HO, BLANK_TR, PRIOR_TR, COACH_CONST,
                      COACH_ANN, COACH_ANN_TR, COACH_ABRUPT, PC_CONST, PC_ANN,
                      PC_ANN_TR, PRIOR_SOFT, PRIOR_STRICT, RANDPRIOR_TR)
from figstyle import (COND_COLORS, RAND_COLOR, MID_COLOR, COND_LABEL, BODY_COLOR,
                      BODY_LABEL, KEY_COLOR, KEY_HATCH, KEY_LABEL, KEY_LABEL_1L,
                      ERR_LABEL_FINAL, ERR_LABEL_HELDOUT, BODY_TITLE, CI_NOTE,
                      MD_GREY, MD_BAND, FLAG_TAG, flag_errorbar, md_band,
                      flag_legend_handles, schedule_legend_handles, footer,
                      footer_room, flag_legend_top, bar_ci, forest)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

HELD_SPEC = {"blank": BLANK_HO, "prior": PRIOR_HO, "prior_soft": PRIOR_SOFT,
             "prior_strict": PRIOR_STRICT, "coach_const": COACH_CONST,
             "coach_ann": COACH_ANN, "coach_abrupt": COACH_ABRUPT,
             "pc_const": PC_CONST, "pc_ann": PC_ANN}
DISP_NAME = {"blank": "blank", "prior": "prior kept",
             "prior_soft": "prior soft-withdrawn", "prior_strict": "prior strict-withdrawn",
             "coach_const": "coach constant", "coach_ann": "coach annealed",
             "coach_abrupt": "coach abrupt-withdrawn"}
TRAIN_SPEC = {"blank": BLANK_TR, "prior": PRIOR_TR,
              "coach": COACH_ANN_TR, "priorcoach": PC_ANN_TR}
PAPER_TRAIN_MEAN = {"elbow": {"blank": 65.6, "prior": 40.1, "coach": 55.4,
                              "priorcoach": 46.8, "randprior": 89.4},
                    "finger": {"blank": 123.3, "prior": 89.0, "coach": 152.2,
                               "priorcoach": 115.3, "randprior": 129.8}}
MK = {"const": "o", "ann": "s"}          # marker shape = schedule


def held_end(key, body):
    return vals(HELD_SPEC[key], body, "ENDPOINT")


def held_stat(key, body):
    """(mean, lo, hi) one-sample t 95% CI on the held-out endpoint."""
    return cond_mean("held-out endpoint %s" % KEY_LABEL_1L[key], body,
                     held_end(key, body), rec_name=DISP_NAME.get(key))


def train_stat(cond, body):
    return cond_mean("train endpoint %s" % cond, body,
                     vals(TRAIN_SPEC[cond], body, "ENDPOINT"),
                     paper_mean=PAPER_TRAIN_MEAN[body][cond])


# ============================================================ FIG 1
def fig1():
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    conds = ["blank", "prior", "coach", "priorcoach"]
    for ax, body in zip(axes[0], BODIES):
        stats = [train_stat(c, body) for c in conds]
        x = np.arange(len(conds))
        keys = ["blank", "prior", "coach_ann", "pc_ann"]
        bar_ci(ax, x, stats, keys)
        ax.set_xticks(x)
        ax.set_xticklabels(["blank", "prior", "coach\n(annealed)", "prior+coach\n(annealed)"],
                           fontsize=8.5)
        ax.set_title("%s  —  TRAIN targets" % BODY_TITLE[body],
                     color=BODY_COLOR[body], fontweight="bold", fontsize=9.5)
        ax.set_ylabel(ERR_LABEL_FINAL[body])
        ax.margins(y=0.26)
    held_keys = ["blank", "prior", "coach_ann", "coach_const", "pc_ann", "pc_const"]
    for ax, body in zip(axes[1], BODIES):
        stats = [held_stat(k, body) for k in held_keys]
        x = np.arange(len(held_keys))
        bar_ci(ax, x, stats, held_keys)
        ax.set_xticks(x)
        ax.set_xticklabels([KEY_LABEL[k] for k in held_keys], fontsize=7.2)
        ax.set_title("%s  —  HELD-OUT targets" % BODY_TITLE[body],
                     color=BODY_COLOR[body], fontweight="bold", fontsize=9.5)
        ax.set_ylabel(ERR_LABEL_FINAL[body])
        ax.margins(y=0.32)
    axes[1][0].legend(handles=[
        Patch(facecolor="white", edgecolor="black", hatch="//",
              label="guidance ANNEALED (faded from step 0, reaches 0 at 8k — not a withdrawal)"),
        Patch(facecolor="white", edgecolor="black", label="guidance kept on (constant)")],
        loc="upper right", fontsize=7.2)
    fig.suptitle("Final reach error by condition and guidance schedule (n=12; lower = better)\n"
                 "Held-out row: on the elbow the two constant-schedule arms end low (coach 39.1, prior+coach 38.5);\n"
                 "the annealed arms do not",
                 fontweight="bold", fontsize=10.5)
    footer(fig,
           "Bars are condition means with a ONE-SAMPLE t 95% CI (df = n−1 = 11) — the same estimator family as",
           "the paired t intervals used for every contrast in this set. No z interval appears anywhere.",
           "Panels use different units (mrad vs mm) and are not comparable in magnitude. All coach cells are",
           "clean-teacher runs; the constant-schedule arms exist under held-out evaluation only.")
    fig.tight_layout(rect=[0, footer_room(fig, 4), 1, 0.93])
    fp = os.path.join(FIGS, "fig1_2x2_endpoint.png")
    fig.savefig(fp); plt.close(fig); return fp


# ============================================================ FIG 2
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))
    spec = [("ENDPOINT (final)\ncoach KEPT ON", "coach(const) - prior (held-out, ENDPOINT)", "const"),
            ("ENDPOINT (final)\ncoach ANNEALED to 0 by 8k", "coach(anneal) - prior (held-out, ENDPOINT)", "ann"),
            ("EARLY (2k–6k)\ncoach KEPT ON", "coach(const) - prior (held-out, EARLY)", "const"),
            ("EARLY (2k–6k)\ncoach ANNEALED to 0 by 8k", "coach(anneal) - prior (held-out, EARLY)", "ann")]
    for ax, body in zip(axes, BODIES):
        rows = [dict(label=lab, res=C(key, body), color=COND_COLORS["coach"],
                     marker=MK[reg]) for lab, key, reg in spec]
        forest(ax, rows, UNIT[body],
               "coach − prior (%s)\n<0 = coach ahead   |   >0 = prior ahead" % UNIT[body],
               BODY_TITLE[body], BODY_COLOR[body])
        ax.axhline(1.5, color="#bbbbbb", lw=1.0)
    fig.suptitle("Prior vs coach depends on the GUIDANCE SCHEDULE (held-out, n=12, paired t 95% CIs)\n"
                 "Elbow: the kept-on coach also ends lower (−10.9, ROBUST uncorrected — this cell FAILS Holm over\n"
                 "BOTH families, §4.9); under the ANNEALED schedule that endpoint advantage is gone (ROBUST-NULL).\n"
                 "Finger: the prior ends lower under both schedules.  Holm detail in the footer.",
                 fontweight="bold", fontsize=10.0)
    flag_legend_top(fig, y=0.845, ncol=3, schedule=True)
    footer(fig,
           CI_NOTE,
           "Marker shape and line style encode the SCHEDULE; marker fill encodes the FLAG. Neither uses hue.",
           "A relative prior-vs-coach contrast, not the guidance test (Fig. 3). EARLY favours the coach under BOTH",
           "schedules; only the endpoint verdict is schedule-dependent. Elbow mrad, finger mm — not comparable.",
           "HOLM, constant schedule only (the ANNEALED cells carry no claim, §3.2): finger ENDPOINT +32.3 (p = 0.00090)",
           "survives the declared 28-cell family, survives the PARTIAL prune at K = 58 (rank 12, threshold 0.00106) and",
           "the K = 44 family §3.6's exclusion criterion ACTUALLY derives once MID is pruned alongside ENDPOINT (rank 9,",
           "threshold 0.00139; §4.9(d)), survives a by-question route-comparison split (K = 26, rank 5), and",
           "FAILS only the literal 72 — where it is the cell the step-down stops on. Elbow ENDPOINT −10.9 (p = 0.01539)",
           "fails every family computed. ROBUST is the UNCORRECTED flag and does not imply survival of any correction.")
    fig.tight_layout(rect=[0, footer_room(fig, 10), 1, 0.80])
    fp = os.path.join(FIGS, "fig2_paired_contrast.png")
    fig.savefig(fp); plt.close(fig); return fp


# ============================================================ FIG 3
def fig3():
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.6))
    panels = [("EARLY  (2k–6k window)", "EARLY"), ("ENDPOINT  (final)", "ENDPOINT")]
    rows_spec = [("train targets\ncoach annealed", "coach(anneal) - blank (train, %s)", "ann"),
                 ("held-out targets\ncoach annealed", "coach(anneal) - blank (held-out, %s)", "ann"),
                 ("held-out targets\ncoach CONSTANT", "coach(const) - blank (held-out, %s)", "const")]
    for r, body in enumerate(BODIES):
        for c, (ptitle, win) in enumerate(panels):
            ax = axes[r][c]
            rows = [dict(label=lab, res=C(key % win, body),
                         color=COND_COLORS["coach"], marker=MK[reg])
                    for lab, key, reg in rows_spec]
            forest(ax, rows, UNIT[body],
                   "coach − blank (%s)   |   <0 = coach better" % UNIT[body],
                   "%s — %s" % (BODY_LABEL[body], ptitle), BODY_COLOR[body],
                   label_fs=8.0, val_fs=7.4)
    fig.suptitle("Guidance test — coach vs blank (paired t 95% CIs, n=12)\n"
                 "EARLY: the coach is ahead in every held-out cell, under both schedules.  ENDPOINT: the gain\n"
                 "survives ONLY on the elbow with the coach never withdrawn (−31.6, ROBUST); every other endpoint\n"
                 "cell is ROBUST-NULL, including both train-target cells.  ROBUST is UNCORRECTED (see footer for Holm).",
                 fontweight="bold", fontsize=10.0)
    flag_legend_top(fig, y=0.885, ncol=4, fontsize=6.6)
    footer(fig,
           CI_NOTE,
           "Clean-teacher coach runs; the constant coach was run under held-out evaluation only.",
           "The two TRAIN-target rows are NOT among the 132 contrasts of RECOMPUTED.md; they are computed here with",
           "the same imported estimators and are labelled as such. Elbow rows mrad, finger rows mm.",
           "HOLM: all four coach − blank EARLY cells survive the declared 28-cell family; three of the four survive the",
           "literal 72 (coach(anneal) − blank EARLY elbow, p = 0.00362, falls) and only the two FINGER cells survive once",
           "the family is widened to the ~80 cells §4.9 concedes it omitted. The elbow ENDPOINT cell (−31.6, p = 0.00176)",
           "survives the declared family and FAILS the literal one — but coach(const) − blank LATE elbow (p = 0.00042) IS",
           "a literal-72 survivor and ENDPOINT ⊂ LATE (§3.6), so that is one confirmation read two ways with two verdicts",
           "(§4.2, §4.9(c)). ROBUST is the UNCORRECTED flag throughout and does not imply survival of any correction.")
    fig.tight_layout(rect=[0, footer_room(fig, 10), 1, 0.845])
    fp = os.path.join(FIGS, "fig3_guidance_coachVblank.png")
    fig.savefig(fp); plt.close(fig); return fp


# ============================================================ FIG 4
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    layout = [("train targets", "prior - blank (train, ENDPOINT)"),
              ("held-out targets", "prior - blank (held-out, ENDPOINT)")]
    for ax, body in zip(axes, BODIES):
        rows = [dict(label=lab, res=C(key, body), color=COND_COLORS["prior"],
                     marker="o") for lab, key in layout]
        forest(ax, rows, UNIT[body],
               "prior − blank (%s)   |   <0 = prior better" % UNIT[body],
               BODY_TITLE[body], BODY_COLOR[body], label_fs=9)
    fig.suptitle("Prior endpoint benefit: ROBUST (UNCORRECTED) on elbow, both target sets; METHOD-DEPENDENT on finger\n"
                 "— the finger cells are NOT findings and carry no claim; ROBUST does NOT imply Holm survival",
                 fontweight="bold", fontsize=10.5)
    flag_legend_top(fig, y=0.845, ncol=4, fontsize=6.6)
    footer(fig,
           CI_NOTE + "  Prior kept on throughout (withdrawal variants in Fig. 10).",
           "The finger TRAIN cell was significant under the invalid z interval and is METHOD-DEPENDENT under the",
           "paired t; the finger HELD-OUT cell was never z-significant and is METHOD-DEPENDENT because BCa excludes",
           "zero while t does not. Neither carries a claim. Elbow mrad, finger mm — panels are not comparable.",
           "HOLM: elbow TRAIN (p = 0.00162) survives the declared 28-cell family and FAILS the literal 72; elbow HELD-OUT",
           "(p = 0.01179) FAILS BOTH. No ENDPOINT cell here survives the literal family — but prior − blank LATE elbow",
           "(train 0.00091, held-out 0.00104) sits immediately behind its stopping cell, and ENDPOINT ⊂ LATE (§3.6),",
           "so the endpoint verdict is one reading of a confirmation whose other reading nearly passes (§4.2, §4.9(c)).")
    fig.tight_layout(rect=[0, footer_room(fig, 8), 1, 0.80])
    fp = os.path.join(FIGS, "fig4_heldout_vs_train.png")
    fig.savefig(fp); plt.close(fig); return fp


# ============================================================ FIG 5
def fig5():
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6),
                             gridspec_kw={"height_ratios": [1.25, 1.0]})
    for ax, body in zip(axes[0], BODIES):
        stats = [train_stat("blank", body),
                 cond_mean("train endpoint randprior", body,
                           vals(RANDPRIOR_TR, body, "ENDPOINT"),
                           paper_mean=PAPER_TRAIN_MEAN[body]["randprior"]),
                 train_stat("prior", body)]
        x = np.arange(3)
        means = [s[0] for s in stats]; errs = [s[0] - s[1] for s in stats]
        ax.bar(x, means, yerr=errs, capsize=4,
               color=[COND_COLORS["blank"], RAND_COLOR, COND_COLORS["prior"]],
               edgecolor="black", linewidth=0.7, error_kw=dict(lw=1.0, ecolor="#333333"))
        for xi, m, er in zip(x, means, errs):
            ax.text(xi, m + er + max(errs) * 0.20, "%.1f" % m, ha="center",
                    va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(["blank", "random\nprior", "trained\nprior"])
        ax.set_ylabel(ERR_LABEL_FINAL[body])
        ax.set_title(BODY_TITLE[body], color=BODY_COLOR[body], fontweight="bold", fontsize=9.5)
        ax.margins(y=0.26)
    for ax, body in zip(axes[1], BODIES):
        rows = [dict(label="trained prior\n− random prior",
                     res=C("trained prior - randprior (train, ENDPOINT)", body),
                     color=COND_COLORS["prior"], marker="o"),
                dict(label="random prior\n− blank",
                     res=C("randprior - blank (train, ENDPOINT)", body),
                     color=RAND_COLOR, marker="o")]
        forest(ax, rows, UNIT[body],
               "paired difference (%s)   |   <0 = first condition better" % UNIT[body],
               None, label_fs=8.2, val_fs=7.8)
    fig.suptitle("A TRAINED frozen model beats a RANDOM one on both bodies; the random one is worse than\n"
                 "nothing on elbow  (train targets, n=12, paired t 95% CIs)",
                 fontweight="bold", fontsize=10.5)
    flag_legend_top(fig, y=0.885, ncol=2, fontsize=6.6)
    footer(fig,
           CI_NOTE,
           "Bars are condition means with a one-sample t 95% CI; the lower row is the paired per-seed contrast.",
           "This figure does NOT show that only a trained prior helps: §4.4 explicitly declines that claim (a random net",
           "corrupts the analytic reward as well as the states, so this cannot isolate LEARNED dynamics), and the coach",
           "helps too (§4.2). randprior − blank is ROBUST on elbow (+23.8) but FAILS Holm over BOTH families — declared",
           "28-cell and literal 72-cell (p = 0.037); it is ROBUST-NULL on finger. Both trained-prior − randprior cells",
           "survive Holm over both families; the ELBOW one clears its literal-72 threshold by 10.7x, the widest margin",
           "of any cell in the study, and the FINGER one by 3.9x, which is 4th of the six that survive both (item 55).")
    fig.tight_layout(rect=[0, footer_room(fig, 8), 1, 0.835])
    fp = os.path.join(FIGS, "fig5_randprior.png")
    fig.savefig(fp); plt.close(fig); return fp


# ============================================================ FIG 6
def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.4))
    windows = [("EARLY\n(2k–6k)", "EARLY"), ("MID\n(6k–10k)", "MID"), ("LATE\n(10k–12k)", "LATE")]
    for ax, body in zip(axes, BODIES):
        xpos = np.arange(len(windows))
        for reg, key, off in [("const", "coach(const) - prior (held-out, %s)", -0.16),
                              ("ann", "coach(anneal) - prior (held-out, %s)", 0.16)]:
            xs, ys = [], []
            for j, (wlab, w) in enumerate(windows):
                r = C(key % w, body)
                if r["flag"] == "METHOD-DEPENDENT":
                    md_band(ax, xpos[j] + off, half=0.11, horizontal=False)
                kw = flag_errorbar(ax, xpos[j] + off, r["mean"], r["tlo"], r["thi"],
                                   r["flag"], COND_COLORS["coach"], MK[reg],
                                   horizontal=False)
                xs.append(xpos[j] + off); ys.append(r["mean"])
                lab = FLAG_TAG[r["flag"]].split(" —")[0].split(" (")[0]
                # label sits BESIDE its own marker (left for the kept-on arm,
                # right for the withdrawn arm) so it can never be read as
                # belonging to the other schedule's point
                dx, dy = (-9, -13) if reg == "const" else (9, 13)
                ax.annotate("%+.1f\n%s" % (r["mean"], lab),
                            (xpos[j] + off, r["mean"]),
                            textcoords="offset points", xytext=(dx, dy),
                            ha="right" if dx < 0 else "left",
                            va="top" if dy < 0 else "bottom",
                            fontsize=6.5, color=kw["txt"],
                            fontweight="bold" if r["flag"] == "METHOD-DEPENDENT" else "normal")
            ax.plot(xs, ys, ls="-" if reg == "const" else "--", lw=1.6,
                    color=COND_COLORS["coach"], alpha=0.85, zorder=2,
                    label="coach %s" % ("kept on (constant)" if reg == "const"
                                        else "annealed (faded to 0 by 8k)"))
        ax.axhline(0, color="black", lw=1.0, ls="--")
        ax.set_xticks(xpos); ax.set_xticklabels([w[0] for w in windows])
        ax.set_ylabel("coach − prior (%s)\n<0 = coach ahead" % UNIT[body])
        ax.set_xlabel("Training window (overlapping: MID and LATE share step 10k)")
        ax.set_title(BODY_TITLE[body], color=BODY_COLOR[body], fontweight="bold", fontsize=9.5)
        ax.margins(x=0.42, y=0.46)
        ax.legend(loc="lower right", fontsize=7.0)
    fig.suptitle("Coach-vs-prior across training windows (held-out, n=12, paired t 95% CIs)\n"
                 "Elbow: the KEPT-ON coach leads in all three windows (all ROBUST uncorrected; the LATE cell fails Holm\n"
                 "over the declared 28-cell family, and over the LITERAL 72 the elbow MID cell falls too — only elbow\n"
                 "EARLY survives there). The ANNEALED coach's late REVERSAL does not survive the correct interval —\n"
                 "MID is ROBUST-NULL, LATE METHOD-DEPENDENT.  Finger: the early lead decays and reverses late, both schedules.",
                 fontweight="bold", fontsize=9.8)
    flag_legend_top(fig, y=0.845, ncol=4, fontsize=6.6)
    footer(fig,
           CI_NOTE,
           "Marker shape and line style encode the SCHEDULE; marker fill encodes the FLAG. All windows come from",
           "the CLEAN-TEACHER runs. CHANGED UNDER t: the two elbow annealed cells (MID, LATE) were significant",
           "under the invalid z interval and are not claimable now. Elbow mrad, finger mm — not comparable.")
    fig.tight_layout(rect=[0, footer_room(fig, 4), 1, 0.795])
    fp = os.path.join(FIGS, "fig6_timecourse.png")
    fig.savefig(fp); plt.close(fig); return fp


# ============================================================ FIG 7
def fig7():
    steps = np.array([2000, 4000, 6000, 8000, 10000, 12000])
    order = ["blank", "prior", "coach_const", "coach_ann", "pc_const", "pc_ann"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))
    for ax, body in zip(axes, BODIES):
        for k in order:
            M = curve(HELD_SPEC[k], body)
            mu = np.array([t_ci(M[:, j])[0] for j in range(M.shape[1])])
            lo = np.array([t_ci(M[:, j])[1] for j in range(M.shape[1])])
            hi = np.array([t_ci(M[:, j])[2] for j in range(M.shape[1])])
            CURVE_TABLE.append((body, k, steps, mu, lo, hi))
            cond_mean("Fig7 final held-out %s" % KEY_LABEL_1L[k], body, M[:, -1],
                      rec_name=DISP_NAME.get(k))
            ann = k.endswith("_ann")
            ax.fill_between(steps, lo, hi, color=KEY_COLOR[k],
                            alpha=0.09 if ann else 0.16, linewidth=0)
            ax.plot(steps, mu, marker="s" if ann else "o", ms=4.5,
                    lw=1.5 if ann else 2.1, ls="--" if ann else "-",
                    color=KEY_COLOR[k], mec="black", mew=0.5,
                    mfc="white" if ann else KEY_COLOR[k], label=KEY_LABEL_1L[k])
        ax.axvline(8000, color="#8c3b00", lw=1.0, ls=":")
        ax.annotate("annealed coach weight reaches 0\n(a fade from step 0, not a withdrawal)", (8000, 0.005),
                    xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=7.0, color="#8c3b00")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ERR_LABEL_HELDOUT[body])
        ax.set_xlim(1500, 12500)
        ax.xaxis.set_major_locator(MultipleLocator(2000))
        ax.set_title(BODY_TITLE[body], color=BODY_COLOR[body], fontweight="bold", fontsize=9.5)
        ax.margins(y=0.20)
    axes[0].legend(loc="upper right", fontsize=7.2, ncol=2)
    axes[0].text(0.02, 0.02, "lower = better", transform=axes[0].transAxes,
                 fontsize=8, style="italic", color="#555555")
    fig.suptitle("Held-out learning curves (n=12, mean with a t 95% CI band), both guidance schedules\n"
                 "Elbow: the constant-schedule arms end low (coach 39.1, prior+coach 38.5); the ANNEALED coach gives the gain back after 8k (61.1)",
                 fontweight="bold", fontsize=10.2)
    footer(fig,
           "Bands are ONE-SAMPLE t 95% CIs (df = n−1 = 11) at each checkpoint — NOT the ±SEM bands drawn by earlier",
           "versions of this figure, and not z intervals. Contrast-level flags are in Figs. 2, 3, 6 and 12.",
           "Solid + filled = guidance kept on; dashed + open = guidance annealed, a fade from step 0 reaching 0 at 8k",
           "and NOT a withdrawal there (§3.2). Clean-teacher coach runs.",
           "Elbow mrad, finger mm — the y-axes are not comparable across panels.")
    fig.tight_layout(rect=[0, footer_room(fig, 5), 1, 0.90])
    fp = os.path.join(FIGS, "fig7_learning_curves.png")
    fig.savefig(fp); plt.close(fig); return fp


CURVE_TABLE = []

# ---- plateau logs (n=3) ----
PLATEAU_RX = re.compile(r"step\s+(\d+).*?dist\s+([-\d.]+)mm")
PLATEAU_SEEDS = [0, 1, 2]


def load_plateau(body):
    raw = {}
    for cond in ["blank", "prior"]:
        for s in PLATEAU_SEEDS:
            fp = os.path.join(HERE, "results_plateau2_%s" % body, "%s_s%d.log" % (cond, s))
            d = {}
            with open(fp, errors="ignore") as f:
                for line in f:
                    m = PLATEAU_RX.search(line)
                    if m:
                        d[int(m.group(1))] = float(m.group(2))
            raw[(cond, s)] = d
    common = sorted(set.intersection(*[set(raw[k]) for k in raw]))
    out = {c: np.array([[raw[(c, s)][st] for st in common] for s in PLATEAU_SEEDS])
           for c in ["blank", "prior"]}
    return np.array(common), out


# ============================================================ FIG 8
def fig8():
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 6.0))
    for ax, body in zip(axes, BODIES):
        steps_all, mat = load_plateau(body)
        keep = steps_all <= 30000
        steps = steps_all[keep]
        for cond in ["blank", "prior"]:
            M = mat[cond][:, keep]
            for row in M:
                ax.plot(steps, row, lw=0.9, alpha=0.45, color=COND_COLORS[cond], zorder=1)
            ax.plot(steps, M.mean(axis=0), marker="o", ms=4.5, lw=2.2,
                    color=COND_COLORS[cond], mec="black", mew=0.5,
                    label="%s (mean of 3 seeds)" % COND_LABEL[cond], zorder=2)
        gaps = (mat["blank"][:, keep].mean(axis=0) - mat["prior"][:, keep].mean(axis=0))
        post = gaps[steps >= 12000]
        gap30 = gaps[-1]
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ERR_LABEL_HELDOUT[body])
        ax.set_xlim(1000, 31000)
        ax.xaxis.set_major_locator(MultipleLocator(6000))
        ax.set_title(BODY_TITLE[body], color=BODY_COLOR[body], fontweight="bold", fontsize=9.5)
        ax.margins(y=0.14)
        extreme = ("SERIES MAXIMUM" if gap30 >= post.max() else
                   "SERIES MINIMUM" if gap30 <= post.min() else "mid-series")
        ax.text(0.98, 0.96,
                "blank − prior @30k: %+.1f %s  (%s of the\n12k–30k series, range %+.1f to %+.1f)\n"
                "(no interval: n=3)"
                % (gap30, UNIT[body], extreme, post.min(), post.max()),
                transform=ax.transAxes, ha="right", va="top", fontsize=7.6)
        ax.legend(loc="lower left", fontsize=8.4, ncol=2)
    fig.suptitle("Longer budget, n=3 seeds — SUGGESTIVE ONLY, NOT CONFIRMATORY\n"
                 "NO INTERVAL IS DRAWN: individual seeds are shown instead",
                 fontweight="bold", fontsize=10.4, color="#8c3b00")
    footer(fig,
           "These runs are n=3 and are NOT in RECOMPUTED.md: the per-seed JSONs were never written (curves were",
           "parsed from logs), so no paired t contrast exists for them and no interval of any kind is plotted.",
           "THE RUNS WERE LAUNCHED FOR 36k AND KILLED AT 32k; the data run to 32k. This plot stops at 30k because that",
           "is the last step all six logs share — prior seed 2 ends at 30k on both bodies — so the 32k points would be",
           "n=2 for prior against n=3 for blank. Both series are printed IN FULL to 32k in §4.8 and RECOMPUTED.md E2.",
           "The @30k annotation is flagged where it sits at an extremum of its own series: on FINGER it is the series",
           "MAXIMUM (+67.5), produced by a blank spike (112.6 → 131.0 → 103.5), which is correction item 53.",
           "Thin lines are the three individual seeds; thick lines are their mean. No inferential claim is made.",
           "Elbow mrad, finger mm — panels are not comparable in magnitude.")
    fig.tight_layout(rect=[0, footer_room(fig, 9), 1, 0.875])
    fp = os.path.join(FIGS, "fig8_plateau.png")
    fig.savefig(fp); plt.close(fig); return fp


# ============================================================ FIG 9
def fig9():
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8),
                             gridspec_kw={"height_ratios": [1.35, 0.85]})
    for ax, body in zip(axes[0], BODIES):
        p = held_end("prior", body); c = held_end("coach_ann", body)
        pc = held_end("pc_ann", body)
        mid = 0.5 * (p + c)
        stats = [held_stat("prior", body), held_stat("coach_ann", body),
                 cond_mean("50/50 midpoint (annealed)", body, mid),
                 held_stat("pc_ann", body)]
        names = ["prior\nonly", "coach only\n(annealed)", "50/50\nmidpoint\n(predicted)",
                 "prior+coach\n(annealed, actual)"]
        cols = [COND_COLORS["prior"], COND_COLORS["coach"], MID_COLOR, COND_COLORS["priorcoach"]]
        hatches = ["", "//", "///", "//"]
        means = [s[0] for s in stats]; errs = [s[0] - s[1] for s in stats]
        bars = ax.bar(np.arange(4), means, yerr=errs, capsize=4, color=cols,
                      edgecolor="black", linewidth=0.8, error_kw=dict(lw=1.0, ecolor="#333333"))
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)
        for xi, m, er in zip(np.arange(4), means, errs):
            ax.text(xi, m + er + max(errs) * 0.20, "%.1f" % m, ha="center", va="bottom", fontsize=9)
        ax.set_xticks(np.arange(4)); ax.set_xticklabels(names, fontsize=8.0)
        ax.set_ylabel(ERR_LABEL_HELDOUT[body])
        ax.set_title(BODY_TITLE[body], color=BODY_COLOR[body], fontweight="bold", fontsize=9.5)
        ax.margins(y=0.30)
    for ax, body in zip(axes[1], BODIES):
        rows = [dict(label="actual − 50/50 midpoint\n(annealed schedule)",
                     res=C("priorcoach(anneal) - 0.5*[prior + coach(anneal)] (ENDPOINT)", body),
                     color=COND_COLORS["priorcoach"], marker="s")]
        forest(ax, rows, UNIT[body],
               "priorcoach − midpoint (%s)" % UNIT[body], None, label_fs=8.0, val_fs=7.8)
    fig.suptitle("SUPERSEDED BY FIG. 11 — do not cite as the general result\n"
                 "Annealed-schedule special case: with the coach faded to 0 the 50/50 mixture prediction merely fails to\n"
                 "be rejected (ROBUST-NULL on both bodies). With the coach kept on it IS rejected on elbow (Fig. 11).",
                 fontweight="bold", fontsize=9.6, color="#8c3b00")
    flag_legend_top(fig, y=0.845, ncol=2, fontsize=6.6)
    footer(fig,
           CI_NOTE,
           "A failure to reject the mixture prediction is not proof of a mixture.",
           "Bars are condition means with a one-sample t 95% CI; the lower row is the paired per-seed contrast.",
           "Elbow mrad, finger mm — panels are not comparable in magnitude.")
    fig.tight_layout(rect=[0, footer_room(fig, 4), 1, 0.785])
    fp = os.path.join(FIGS, "fig9_mixture.png")
    fig.savefig(fp); plt.close(fig); return fp


# ============================================================ FIG 10
def fig10():
    keys = ["blank", "prior", "prior_soft", "prior_strict",
            "coach_const", "coach_ann", "coach_abrupt"]
    contrasts = [
        ("prior withdrawal COST\n(strict − kept)", "prior strict-withdrawn - prior kept (ENDPOINT)",
         COND_COLORS["prior"], "s"),
        ("coach withdrawal COST\n(abrupt − constant)", "coach abrupt-withdrawn - coach const (ENDPOINT)",
         COND_COLORS["coach"], "s"),
        ("coach cost − prior cost\n(difference of costs)", "coach cost - prior cost (ENDPOINT)",
         "#444444", "s"),
        ("prior after SOFT withdrawal\n− blank", "prior soft-withdrawn - blank (ENDPOINT)",
         COND_COLORS["prior"], "s"),
        ("prior after STRICT withdrawal\n− blank", "prior strict-withdrawn - blank (ENDPOINT)",
         COND_COLORS["prior"], "s"),
        ("coach after withdrawal\n− blank", "coach abrupt-withdrawn - blank (ENDPOINT)",
         COND_COLORS["coach"], "s"),
        ("coach after withdrawal\n− prior after withdrawal",
         "coach abrupt-withdrawn - prior strict-withdrawn (ENDPOINT)", "#444444", "s"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.6),
                             gridspec_kw={"height_ratios": [1.0, 1.85]})
    for ax, body in zip(axes[0], BODIES):
        stats = [held_stat(k, body) for k in keys]
        x = np.arange(len(keys))
        bar_ci(ax, x, stats, keys)
        ax.axhline(stats[0][0], color=COND_COLORS["blank"], lw=1.2, ls=":", zorder=0)
        ax.text(-0.42, stats[0][0], "blank level", va="bottom", ha="left",
                fontsize=7.4, color="#666666")
        ax.set_xticks(x); ax.set_xticklabels([KEY_LABEL[k] for k in keys], fontsize=7.0)
        ax.set_ylabel(ERR_LABEL_HELDOUT[body])
        ax.set_title(BODY_TITLE[body], color=BODY_COLOR[body], fontweight="bold", fontsize=9.5)
        ax.margins(y=0.28)
    for ax, body in zip(axes[1], BODIES):
        rows = [dict(label=lab, res=C(key, body), color=col, marker=mk)
                for lab, key, col, mk in contrasts]
        forest(ax, rows, UNIT[body],
               "paired difference (%s)   |   >0 = withdrawal hurt / worse than blank" % UNIT[body],
               None, label_fs=7.6, val_fs=7.4)
    fig.legend(handles=[
        Patch(facecolor="white", edgecolor="black", label="kept on for the whole budget"),
        Patch(facecolor="white", edgecolor="black", hatch="//",
              label="prior withdrawn @8k (soft) / coach ANNEALED — a fade from step 0, not a withdrawal"),
        Patch(facecolor="white", edgecolor="black", hatch="xx", label="withdrawn @8k (strict / abrupt)")],
        loc="lower center", ncol=3, fontsize=7.2,
        bbox_to_anchor=(0.5, footer_room(fig, 11) + 0.004))
    fig.suptitle("Matched withdrawal test (held-out endpoint, n=12, paired t 95% CIs)\n"
                 "The COACH's withdrawal cost is ROBUST on both bodies; the PRIOR's is ROBUST-NULL on both.\n"
                 "The difference between the two costs is ROBUST-NULL on elbow and METHOD-DEPENDENT on finger —\n"
                 "so this figure does NOT establish that the coach is more withdrawal-fragile than the prior.\n"
                 "ROBUST is the UNCORRECTED flag — the Holm status of every cell is in the footer.",
                 fontweight="bold", fontsize=9.6)
    flag_legend_top(fig, y=0.885, ncol=4, fontsize=6.6)
    footer(fig,
           CI_NOTE,
           "Withdrawal at 8k of a 12k budget: SOFT = model frozen out, buffer kept; STRICT = buffer purged;",
           "ABRUPT = coach cut to 0 at 8k. ANNEALED is NOT a withdrawal: its weight fades from step 0 and merely",
           "reaches 0 at 8k, so only the ABRUPT coach is matched to the prior's cut. Bars: means, one-sample t 95% CI.",
           "HOLM, per cell: coach cost ELBOW p = 0.00350 — PASSES the declared 28-cell family by 0.00007 (the narrowest",
           "passing margin in it) and FAILS the literal 72; coach cost FINGER 0.00117 — declared PASS, literal FAIL;",
           "coach-withdrawn − prior-withdrawn finger 0.00117 — declared PASS, literal FAIL; coach-withdrawn − blank finger",
           "0.01406 — FAILS every family computed. BOTH coach costs are RESCUED by the by-question removal family (K = 10,",
           "ranks 1 and 3), which is the same split that rescues the title's reversal and is applied here too (§4.5, §4.9).",
           "The FINGER cost, and coach-withdrawn − prior-withdrawn finger, ALSO survive the K = 44 family §3.6's exclusion",
           "criterion actually derives (ranks 10 and 11, §4.9(d)); the ELBOW cost, at 0.00350, fails there as it does at K = 72.",
           "The prior's own cost and prior-strict − blank are t-non-significant on both bodies and survive nothing anywhere.",
           "At n=12 the post-withdrawal nulls are underpowered in both directions (§4.8). Elbow mrad, finger mm.")
    fig.tight_layout(rect=[0, footer_room(fig, 13) + 0.040, 1, 0.885])
    fp = os.path.join(FIGS, "fig10_withdrawal.png")
    fig.savefig(fp); plt.close(fig); return fp


# ============================================================ FIG 11
def fig11():
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0),
                             gridspec_kw={"height_ratios": [1.15, 1.0]})
    for ax, body in zip(axes[0], BODIES):
        p = held_end("prior", body); c = held_end("coach_const", body)
        mid = 0.5 * (p + c)
        stats = [held_stat("prior", body), held_stat("coach_const", body),
                 cond_mean("50/50 midpoint (constant)", body, mid),
                 held_stat("pc_const", body)]
        names = ["prior\nalone", "coach alone\n(constant)", "predicted\n50/50 midpoint",
                 "prior+coach\n(constant, ACTUAL)"]
        cols = [COND_COLORS["prior"], COND_COLORS["coach"], MID_COLOR, COND_COLORS["priorcoach"]]
        hatches = ["", "", "///", ""]
        means = [s[0] for s in stats]; errs = [s[0] - s[1] for s in stats]
        bars = ax.bar(np.arange(4), means, yerr=errs, capsize=4, color=cols,
                      edgecolor="black", linewidth=0.9, error_kw=dict(lw=1.0, ecolor="#333333"))
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)
        for xi, m, er in zip(np.arange(4), means, errs):
            ax.text(xi, m + er + max(errs) * 0.18, "%.1f" % m, ha="center", va="bottom", fontsize=9)
        ax.axhline(means[1], color=COND_COLORS["coach"], lw=1.2, ls=":", zorder=0)
        ax.set_xticks(np.arange(4)); ax.set_xticklabels(names, fontsize=8.0)
        ax.set_ylabel(ERR_LABEL_HELDOUT[body])
        ax.set_title(BODY_TITLE[body], color=BODY_COLOR[body], fontweight="bold", fontsize=9.5)
        ax.margins(y=0.30)
    spec = [("actual − 50/50 midpoint", "priorcoach(const) - 0.5*[prior + coach(const)] (ENDPOINT)",
             COND_COLORS["priorcoach"]),
            ("actual − coach alone", "priorcoach(const) - coach(const) (held-out, ENDPOINT)",
             COND_COLORS["coach"]),
            ("actual − prior alone", "priorcoach(const) - prior (held-out, ENDPOINT)",
             COND_COLORS["prior"])]
    for ax, body in zip(axes[1], BODIES):
        rows = [dict(label=lab, res=C(key, body), color=col, marker="o")
                for lab, key, col in spec]
        forest(ax, rows, UNIT[body],
               "paired difference (%s)" % UNIT[body], None, label_fs=8.0, val_fs=7.6)
    fig.suptitle("Combining the two routes (held-out endpoint, constant schedule, n=12)\n"
                 "At the endpoint we CANNOT DETECT A DIFFERENCE between the combination and the coach alone on\n"
                 "either body (actual − coach alone is ROBUST-NULL on both). At n=12 that is a failure to detect,\n"
                 "NOT evidence that the combination tracks the coach. What IS measured is the move away from the\n"
                 "prior, helping on elbow and hurting on finger; the 50/50 mixture FORM is rejected on ELBOW only.",
                 fontweight="bold", fontsize=9.6)
    flag_legend_top(fig, y=0.815, ncol=4, fontsize=6.6)
    footer(fig,
           CI_NOTE,
           "ROBUST-NULL means both intervals contain zero. Per the paper's §6 rule every non-significant cell is",
           "underpowered rather than evidence of equivalence: the finger `actual − coach alone` interval",
           "[−14.7, +1.4] admits a 14 mm advantage to the combination. The midpoint is the per-seed average of the",
           "prior-alone and coach-alone endpoints; the dotted line marks the coach-alone level. Elbow mrad, finger mm.")
    fig.tight_layout(rect=[0, footer_room(fig, 4), 1, 0.765])
    fp = os.path.join(FIGS, "fig11_coach_dominance.png")
    fig.savefig(fp); plt.close(fig); return fp


# ============================================================ FIG 12 (NEW)
# Holm-Bonferroni verdicts over TWO families (§4.9):
#   DECLARED  = the 28 contrasts the paper declares (the t-significant subset)  -> 16 survive
#   LITERAL   = the 72 cells §3.6's rule enumerates without that filter          -> 12 survive
# Every one of these eight cells is ROBUST under the uncorrected primary
# estimator. Three survive the declared-family correction; ONE survives the
# literal-family one. The glyph/band marks the LITERAL verdict (the stricter of
# the two); the declared verdict is printed in the tag and the footer.
# (declared, literal, p)
FIG12_HOLM_FULL = {
    ("elbow", "EARLY"):    ("pass", "pass", 0.00034),
    ("elbow", "MID"):      ("pass", "fail", 0.00114),
    ("elbow", "LATE"):     ("fail", "fail", 0.00605),
    ("elbow", "ENDPOINT"): ("fail", "fail", 0.01539),
    ("finger", "EARLY"):   ("fail", "fail", 0.00587),
    ("finger", "MID"):     ("fail", "fail", 0.02142),
    ("finger", "LATE"):    ("fail", "fail", 0.00423),
    ("finger", "ENDPOINT"): ("pass", "fail", 0.00090)}
FIG12_HOLM = {k: (v[1], v[2]) for k, v in FIG12_HOLM_FULL.items()}


def fig12():
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.8))
    wins = [("EARLY  (2k–6k)", "EARLY"), ("MID  (6k–10k)", "MID"),
            ("LATE  (10k–12k)", "LATE"), ("ENDPOINT  (12k)", "ENDPOINT")]
    for ax, body in zip(axes, BODIES):
        rows = []
        for lab, w in wins:
            hd, hl, hp = FIG12_HOLM_FULL[(body, w)]
            rows.append(dict(
                label=lab, res=C("coach(const) - prior (held-out, %s)" % w, body),
                color=COND_COLORS["coach"], marker="o", holm=hl,
                holm_text="Holm: declared-28 %s / literal-72 %s"
                          % ("✓" if hd == "pass" else "✗",
                             "✓" if hl == "pass" else "✗")))
        forest(ax, rows, UNIT[body],
               "coach(constant) − prior (%s)\n<0 = coach ahead   |   >0 = prior ahead" % UNIT[body],
               BODY_TITLE[body], BODY_COLOR[body], label_fs=8.6, val_fs=6.6)
        ax.set_ylim(-1.15, len(wins) - 1 + 0.62)
        ax.margins(x=0.62)
        ax.axhline(0.5, color="#bbbbbb", lw=1.0)
        end = C("coach(const) - prior (held-out, ENDPOINT)", body)
        hd, hl, hp = FIG12_HOLM_FULL[(body, "ENDPOINT")]
        ax.text(0.5, 0.015,
                "ENDPOINT: %s (%+.1f %s), p = %.5f\ndeclared-28 %s   ·   literal-72 %s"
                % ("coach ahead" if end["mean"] < 0 else "prior ahead",
                   end["mean"], UNIT[body], hp,
                   "✓ survives" if hd == "pass" else "✗ FAILS",
                   "✓ survives" if hl == "pass" else "✗ FAILS"),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2,
                fontweight="bold",
                color=COND_COLORS["coach"] if end["mean"] < 0 else COND_COLORS["prior"])
    fig.suptitle("coach(constant) − prior by window (held-out, n=12, paired t 95% CIs)\n"
                 "WHAT SURVIVES THE CORRECTION — AND WHICH FAMILY: all eight cells are significant UNCORRECTED.\n"
                 "Over the paper's DECLARED 28-cell family three survive: elbow EARLY (−50.6), elbow MID (−22.5)\n"
                 "and finger ENDPOINT (+32.3). Over the LITERAL 72-cell family §3.6's rule ENUMERATES, only elbow\n"
                 "EARLY survives — the step-down STOPS ON finger ENDPOINT (p = 0.00090 vs 0.00083). ✗ = fails LITERAL,\n"
                 "which is the STRICTEST family computed and NOT the family §3.6's exclusion criterion DERIVES — that is\n"
                 "K = 44, on which finger ENDPOINT is rank 9 and SURVIVES. ✗ is not a verdict all families share.",
                 fontweight="bold", fontsize=9.5, y=0.985)
    flag_legend_top(fig, y=0.845, ncol=3, fontsize=6.5, holm=True)
    footer(fig,
           CI_NOTE,
           "ROBUST is the UNCORRECTED flag (t and BCa agree, both exclude 0). It does NOT imply survival of Holm.",
           "✗ = fails Holm over the LITERAL 72-cell family (7 of 8 cells). Five also fail the declared 28-cell family:",
           "elbow LATE (0.00605), elbow ENDPOINT (0.01539), finger EARLY (0.00587), MID (0.02142), LATE (0.00423).",
           "Elbow MID (0.00114) and finger ENDPOINT (0.00090) survive the declared family and fail the literal one.",
           "The declared 28 are the t-SIGNIFICANT SUBSET of the literal 72, so correcting within them understates it (§3.6).",
           "Windows overlap (MID and LATE share step 10k; ENDPOINT ⊂ LATE): not independent confirmations. Elbow mrad,",
           "finger mm — panels NOT comparable in magnitude; only the SIGN of the endpoint contrast differs between bodies.",
           "AND THE LITERAL 72 BREAKS THAT SAME RULE: it enters BOTH the LATE and the ENDPOINT cell of every window-spanning",
           "pair on both bodies — 14 duplicated cells — though §3.6 declares ENDPOINT ⊂ LATE 'one confirmation and not two'",
           "and excludes sub-divisions on exactly that ground. Pruning them gives K = 58 — the PARTIAL prune — on which",
           "finger ENDPOINT moves to rank 12 and SURVIVES (0.00090 vs 0.05/47 = 0.00106). THE SAME §3.6 SENTENCE SAYS MID",
           "OVERLAPS BOTH EARLY AND LATE, and MID was never pruned: pruning it too gives K = 44 — what that criterion",
           "ACTUALLY derives — on which finger ENDPOINT is rank 9 and SURVIVES (0.00090 vs 0.05/36 = 0.00139), together",
           "with both finger withdrawal cells and the finger decay interaction. It also survives a by-question split",
           "(route family, K = 26, rank 5). Pruning ENDPOINT instead of LATE gives K = 58 too and removes the finger",
           "ENDPOINT cell from the family entirely. Including correlated cells is CONSERVATIVE, not erroneous, so the",
           "literal-72 verdict stands as a verdict about that family — but PRECEDENCE GOES TO K = 44, not to the literal",
           "72, which is itself selection-filtered (D2c): preferring it BECAUSE it is strict is a second selection with",
           "the sign flipped (§4.9(c), §4.9(d); RECOMPUTED.md F1, G2).")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.745,
                        bottom=footer_room(fig, 20) + 0.095, wspace=0.26)
    fp = os.path.join(FIGS, "fig12_crossover.png")
    fig.savefig(fp); plt.close(fig); return fp


# ============================================================ FIG 13 (NEW)
SD_KEYS = ["blank", "prior", "prior_soft", "prior_strict",
           "coach_const", "coach_ann", "coach_abrupt", "pc_const", "pc_ann"]
# TRAIN-TARGET block: `randprior` was never scored held-out (it is a train-target
# control, §4.4), so it is missing from every held-out dispersion row above.  Its
# matched referent is the TRAIN-target blank, and the train-target prior is shown
# beside it so the two are read on one identical target set.
SD_TRAIN_KEYS = ["blank_tr", "prior_tr", "randprior_tr"]
TRAIN_SD_SPEC = {"blank_tr": BLANK_TR, "prior_tr": PRIOR_TR,
                 "randprior_tr": RANDPRIOR_TR}
TRAIN_SD_LABEL = {"blank_tr": "blank\n(train)", "prior_tr": "prior kept\n(train)",
                  "randprior_tr": "randprior\n(train)"}
TRAIN_SD_LABEL_1L = {"blank_tr": "blank (train)", "prior_tr": "prior kept (train)",
                     "randprior_tr": "randprior (train)"}
TRAIN_SD_COLOR = {"blank_tr": COND_COLORS["blank"], "prior_tr": COND_COLORS["prior"],
                  "randprior_tr": RAND_COLOR}
SD_ROWS = []


def train_end(key, body):
    return vals(TRAIN_SD_SPEC[key], body, "ENDPOINT")


def fig13():
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 12.6),
                             gridspec_kw={"height_ratios": [1.0, 2.05]})
    for ax, body in zip(axes[0], BODIES):
        sds = []
        for k in SD_KEYS:
            arr = held_end(k, body)
            sd = float(arr.std(ddof=1))
            ref = FS.DISP.get((DISP_NAME.get(k, k), body))
            FS.MEAN_CHECKS.append(dict(
                label="Fig13 endpoint SD %s" % KEY_LABEL_1L[k], body=body, mean=sd,
                lo=float("nan"), hi=float("nan"),
                expected=None if ref is None else ref["sd"], src="RECOMPUTED (d) SD",
                status="MATCH" if ref is not None and abs(sd - ref["sd"]) <= 0.02
                else ("MISMATCH" if ref is not None else "NOT-IN-RECOMPUTED")))
            sds.append(sd)
        for k in SD_TRAIN_KEYS:
            arr = train_end(k, body)
            sd = float(arr.std(ddof=1))
            FS.MEAN_CHECKS.append(dict(
                label="Fig13 endpoint SD %s" % TRAIN_SD_LABEL_1L[k], body=body, mean=sd,
                lo=float("nan"), hi=float("nan"), expected=None,
                src="train targets (§4.4)", status="NOT-IN-RECOMPUTED"))
            sds.append(sd)
        keys = SD_KEYS + SD_TRAIN_KEYS
        cols = [KEY_COLOR[k] for k in SD_KEYS] + [TRAIN_SD_COLOR[k] for k in SD_TRAIN_KEYS]
        x = np.arange(len(keys))
        bars = ax.bar(x, sds, color=cols, edgecolor="black", linewidth=0.8)
        for b, k in zip(bars, keys):
            h = KEY_HATCH.get(k, "..")
            if h:
                b.set_hatch(h)
        ax.axhline(sds[0], color=COND_COLORS["blank"], lw=1.3, ls=":", zorder=0)
        ax.text(len(SD_KEYS) - 0.4, sds[0], " held-out blank SD", ha="right", va="bottom",
                fontsize=7.0, color="#666666")
        ax.axhline(sds[len(SD_KEYS)], color=COND_COLORS["blank"], lw=1.3, ls="-.", zorder=0)
        ax.text(-0.4, sds[len(SD_KEYS)], "train blank SD ", ha="left", va="top",
                fontsize=7.0, color="#666666")
        ax.axvline(len(SD_KEYS) - 0.5, color="#444444", lw=1.2, ls="-", zorder=1)
        for xi, s in zip(x, sds):
            ax.text(xi, s + max(sds) * 0.03, "%.1f" % s, ha="center", va="bottom", fontsize=8.0)
        ax.set_xticks(x)
        ax.set_xticklabels([KEY_LABEL[k] for k in SD_KEYS]
                           + [TRAIN_SD_LABEL[k] for k in SD_TRAIN_KEYS], fontsize=6.0)
        ax.set_ylabel("Endpoint SD across 12 seeds (%s)" % UNIT[body])
        ax.set_title("%s\nleft of the rule: HELD-OUT targets   ·   right: TRAIN targets"
                     % BODY_TITLE[body], color=BODY_COLOR[body], fontweight="bold", fontsize=8.2)
        ax.margins(y=0.22)
    for ax, body in zip(axes[1], BODIES):
        blank = held_end("blank", body)
        blank_tr = train_end("blank_tr", body)
        rows = [(k, "held") for k in SD_KEYS[1:]] + [(k, "train") for k in SD_TRAIN_KEYS[1:]]
        y = list(range(len(rows)))[::-1]
        labels = []
        for i, (k, tgt) in enumerate(rows):
            if tgt == "held":
                r, lo, hi = sd_ratio_ci(blank, held_end(k, body))
                col = KEY_COLOR[k]
                lab = KEY_LABEL_1L[k]
                mk = "o" if k in ("prior", "coach_const") else "s"
            else:
                r, lo, hi = sd_ratio_ci(blank_tr, train_end(k, body))
                col = TRAIN_SD_COLOR[k]
                lab = TRAIN_SD_LABEL_1L[k] + "  vs train blank"
                mk = "D"
            excl = lo > 1.0 or hi < 1.0
            SD_ROWS.append((body, k if tgt == "held" else k, r, lo, hi, excl))
            labels.append(lab)
            eb = ax.errorbar(r, y[i], xerr=[[r - lo], [hi - r]], fmt=mk,
                             mfc=col if excl else "white", mec="black" if excl else col,
                             mew=0.8 if excl else 1.7, ms=8.5, ecolor=col,
                             elinewidth=2.0 if excl else 1.6, capsize=5, zorder=3)
            if not excl:
                for bl in eb[2]:
                    bl.set_linestyle((0, (4, 2)))
            ax.annotate("%.2f [%.2f, %.2f]   %s" % (r, lo, hi,
                        "excludes 1" if excl else "includes 1"),
                        (r, y[i]), textcoords="offset points", xytext=(0, 8),
                        ha="center", va="bottom", fontsize=7.0,
                        color=col if excl else "#555555")
        ax.axhline(y[len(SD_KEYS) - 2] - 0.5, color="#444444", lw=1.2)
        ax.axvline(1.0, color="black", lw=1.2, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.6)
        ax.set_ylim(-0.75, len(rows) - 1 + 0.95)
        ax.set_xlabel("SD(blank) / SD(condition), each against its OWN target set's blank\n"
                      ">1 = LESS dispersed than model-free RL", fontsize=9)
        ax.margins(x=0.22)
    fig.suptitle("Seed-to-seed dispersion: an aid SUPPRESSES dispersion while present in THREE of the FOUR kept-aid cells\n"
                 "ON THE SD RATIO, and in TWO of the four on the SCALE-FREE CV RATIO (2 aids x 2 bodies). The FOURTH SD cell —\n"
                 "finger PRIOR kept, 1.68 [0.54, 3.51] — INCLUDES 1: a failure to detect at n=12, not an absence; on CV the\n"
                 "elbow CONSTANT COACH joins it at 1.62 [0.87, 3.73]. The LARGEST ratios in the\n"
                 "study are the COMBINED arms (4.77 elbow, 4.68 finger), followed by the finger constant coach (3.53).\n"
                 "After the two MATCHED withdrawals (prior STRICT 0.87, coach ABRUPT 1.51) the ratio is NOT DISTINGUISHABLE\n"
                 "FROM 1 — an underpowered null, not a demonstration that the baseline was restored. The UNMATCHED soft prior\n"
                 "withdrawal (3.10) is TIGHTER THAN BOTH MATCHED WITHDRAWALS (vs STRICT 3.55 [2.27, 6.11] p = 0.00042; vs ABRUPT\n"
                 "COACH 2.05 [1.19, 3.68] p = 0.03204) and NOT DISTINGUISHABLE from the KEPT prior it was withdrawn from\n"
                 "(3.10 vs 2.78 = 1.114 [0.680, 1.896], p = 0.70198) — TESTED, not read off two point estimates; NO INVERSION.\n"
                 "THE ADVERSARIAL CONTROL, NEVER PREVIOUSLY SCORED HERE: on TRAIN targets the RANDOM-INIT frozen model does NOT\n"
                 "suppress dispersion on elbow (0.84) but DOES on finger (3.33 > the trained prior's 1.54) — in an arm with NO\n"
                 "TEACHER, so the FINGER dispersion result is not about what the model or the teacher knows. That 3.33-vs-1.54\n"
                 "difference is TESTED DIRECTLY here, not read off two intervals: Pitman-Morgan p = 0.02117, ratio-of-ratios\n"
                 "2.163 [1.067, 5.130]; the ELBOW twin runs the OTHER WAY, 0.250 [0.149, 0.386], p = 0.00010 (§4.6).",
                 fontweight="bold", fontsize=9.0, y=0.998, va="top")
    footer(fig,
           "Top row: per-condition endpoint SD (ddof = 1), matching RECOMPUTED.md §(d) where present. Bottom row: SD(blank)/SD(condition)",
           "with a PERCENTILE BOOTSTRAP 95% CI (20 000 resamples, RNG seed 0, resampling seeds). Filled marker = CI excludes 1.",
           "`randprior` was run on TRAIN targets only (§4.4), which is why it was absent from every earlier version of this figure; it is",
           "scored here against the matched TRAIN-target blank, with the train-target prior beside it on the identical target set (DIAMONDS).",
           "Held-out and train ratios are never mixed. An earlier version omitted the two prior+coach arms and called 3.10 the largest ratio",
           "in the study; it is not — it is the largest among WITHDRAWN arms. All twelve seeds share one target set, so none of these SDs",
           "contains any target-set variance (§3.1). POST-HOC. These ratios are NOT paired-difference contrasts, so they are NOT among",
           "RECOMPUTED.md's 132 contrasts and carry no ROBUST / ROBUST-NULL / METHOD-DEPENDENT flag. Elbow mrad, finger mm — not comparable.")
    fig.tight_layout(rect=[0, footer_room(fig, 8), 1, 0.998])
    fp = os.path.join(FIGS, "fig13_variance.png")
    fig.savefig(fp); plt.close(fig); return fp


# ============================================================ verification
def print_checks():
    print("\n" + "=" * 132)
    print("CONTRAST CHECK — computed here  vs  RECOMPUTED.md (authoritative)")
    print("all intervals are the PAIRED t 95% CI, df = n-1 = 11")
    print("=" * 132)
    hdr = ("%-62s %-7s %9s %9s %-24s %-24s %-18s %-18s %s"
           % ("contrast", "body", "MINE", "REC", "t CI (mine)", "t CI (RECOMPUTED)",
              "flag (mine)", "flag (REC)", "status"))
    print(hdr)
    print("-" * 132)
    for c in FS.CHECKS:
        ref = c["ref"]
        rm = "%9.3f" % ref["mean"] if ref else "%9s" % "--"
        rci = "[%.3f, %.3f]" % (ref["tlo"], ref["thi"]) if ref else "--"
        rfl = ref["flag"] if ref else "--"
        print("%-62s %-7s %9.3f %s %-24s %-24s %-18s %-18s %s"
              % (c["label"][:62], c["body"], c["mean"], rm,
                 "[%.3f, %.3f]" % (c["tlo"], c["thi"]), rci, c["flag"], rfl,
                 c["status"]))
    print("-" * 132)
    print("\n" + "=" * 110)
    print("POINT-ESTIMATE CHECK — plotted bar/curve means and SDs")
    print("=" * 110)
    print("%-46s %-7s %10s %10s %-22s %s" % ("quantity", "body", "COMPUTED",
                                             "EXPECTED", "source", "status"))
    print("-" * 110)
    for m in FS.MEAN_CHECKS:
        exp = "%10.3f" % m["expected"] if m["expected"] is not None else "%10s" % "--"
        print("%-46s %-7s %10.3f %s %-22s %s"
              % (m["label"][:46], m["body"], m["mean"], exp, m["src"], m["status"]))
    print("-" * 110)
    if SD_ROWS:
        print("\nSD-RATIO CHECK — SD(blank)/SD(condition), bootstrap 95% CI "
              "(NOT in RECOMPUTED.md; percentile bootstrap, 20000 resamples, seed 0)")
        print("-" * 110)
        paper = {("elbow", "prior"): (2.78, 1.48, 4.63), ("elbow", "prior_soft"): (3.10, 1.61, 5.46),
                 ("elbow", "prior_strict"): (0.87, 0.45, 1.58), ("elbow", "coach_const"): (2.93, 1.44, 7.25),
                 ("elbow", "coach_ann"): (0.74, 0.37, 1.47), ("elbow", "coach_abrupt"): (1.51, 0.78, 2.78),
                 ("elbow", "pc_const"): (4.77, 2.44, 8.96), ("elbow", "pc_ann"): (1.63, 0.85, 2.92),
                 ("finger", "prior"): (1.68, 0.54, 3.51), ("finger", "prior_soft"): (1.94, 0.63, 3.74),
                 ("finger", "prior_strict"): (1.35, 0.44, 2.68), ("finger", "coach_const"): (3.53, 1.16, 6.67),
                 ("finger", "coach_ann"): (0.93, 0.30, 2.83), ("finger", "coach_abrupt"): (1.25, 0.41, 2.69),
                 ("finger", "pc_const"): (4.68, 1.53, 8.78), ("finger", "pc_ann"): (2.41, 0.78, 4.80),
                 # train-target block (§4.6, second table) -- the randprior control
                 ("elbow", "prior_tr"): (3.34, 2.12, 5.45), ("elbow", "randprior_tr"): (0.84, 0.53, 1.27),
                 ("finger", "prior_tr"): (1.54, 0.78, 2.84), ("finger", "randprior_tr"): (3.33, 1.66, 7.59)}
        names = dict(KEY_LABEL_1L)
        names.update(TRAIN_SD_LABEL_1L)
        for body, k, r, lo, hi, excl in SD_ROWS:
            p = paper.get((body, k))
            note = ("PAPER §4.6 %.2f [%.2f, %.2f]" % p) if p else "not published in §4.6"
            if p and (abs(r - p[0]) > 0.005 or abs(lo - p[1]) > 0.005 or abs(hi - p[2]) > 0.005):
                note += "   <<< MISMATCH"
            print("  %-7s %-26s %5.2f [%5.2f, %5.2f]  %-11s  %s"
                  % (body, names[k], r, lo, hi,
                     "excludes 1" if excl else "includes 1", note))
    changed = [c for c in FS.CHECKS if c["changed"]]
    print("\n" + "=" * 110)
    print("SIGNIFICANCE CHANGES: cells whose SUPERSEDED z interval excluded zero but whose "
          "PAIRED t interval does not")
    print("=" * 110)
    if changed:
        for c in changed:
            print("  %-58s %-7s %8.3f   z [%.3f, %.3f]  ->  t [%.3f, %.3f]  %s"
                  % (c["label"][:58], c["body"], c["mean"], c["zlo"], c["zhi"],
                     c["tlo"], c["thi"], c["flag"]))
    else:
        print("  none among the plotted contrasts")
    print("(No contrast can move the other way: the t interval is uniformly wider than the z interval.)")
    n_contrast = len(FS.CHECKS)
    n_match = sum(1 for c in FS.CHECKS if c["status"] == "MATCH")
    n_notrec = sum(1 for c in FS.CHECKS if c["status"].startswith("NOT-IN"))
    n_bad = FS.n_bad()
    print("\n" + "=" * 110)
    print("SUMMARY: %d plotted contrasts | %d MATCH RECOMPUTED.md exactly | %d not in "
          "RECOMPUTED.md (computed with the same estimators) | %d DISAGREEMENTS"
          % (n_contrast, n_match, n_notrec, n_bad))
    flags = {}
    for c in FS.CHECKS:
        flags[c["flag"]] = flags.get(c["flag"], 0) + 1
    print("Flag counts over the plotted contrasts: " +
          " | ".join("%s %d" % (k, flags[k]) for k in sorted(flags)))
    print("=" * 110)
    return n_bad


if __name__ == "__main__":
    # ---- PHASE 1: compute + verify EVERYTHING before a single PNG is written
    FC.build()
    bad = [c for c in FS.CHECKS
           if c["status"] in ("MISMATCH", "MISSING-FROM-RECOMPUTED")]
    if bad:
        print("STOP — computed values disagree with RECOMPUTED.md; nothing was plotted:")
        for c in bad:
            print("  %s [%s]: mine %.3f [%.3f, %.3f] %s vs REC %s"
                  % (c["label"], c["body"], c["mean"], c["tlo"], c["thi"], c["flag"], c["ref"]))
        sys.exit(2)

    # ---- PHASE 2: plot the AUDIT set (for PAPER_TMLR.md)
    paths = [fig1(), fig2(), fig3(), fig4(), fig5(), fig6(), fig7(), fig8(),
             fig9(), fig10(), fig11(), fig12(), fig13()]

    # ---- PHASE 3: plot the CLEAN variants (for PAPER_CLEAN.md).
    # New filenames only -- `fig{12,10,5}_*_clean.png`; the audit PNGs above are
    # never touched. Every value is re-verified against RECOMPUTED.md inside
    # `build_clean`, which exits without writing on any disagreement.
    import make_figs_clean as MFC
    paths += MFC.build_clean(sys.modules[__name__])

    print("\n" + "=" * 100)
    print("FIG 7 held-out reach error, n=12 seeds, mean [paired-t-family 95% CI half-width] "
          "(elbow=mrad, finger=mm)")
    print("=" * 100)
    hdr = "  ".join("%15d" % s for s in [2000, 4000, 6000, 8000, 10000, 12000])
    cur = None
    for body, key, steps, mu, lo, hi in CURVE_TABLE:
        if body != cur:
            cur = body
            print("\n%s  [%s]" % (BODY_LABEL[body], UNIT[body]))
            print("%-26s%s" % ("cond", hdr))
        cells = "  ".join("%8.1f[%4.1f]" % (m, m - l) for m, l in zip(mu, lo))
        print("%-26s%s" % (KEY_LABEL_1L[key], cells))

    nbad = print_checks()
    print("\nFigures written:")
    for p in paths:
        print("  ", p)
    sys.exit(1 if nbad else 0)
