"""
make_figs_clean.py -- CLEAN variants of the three body figures whose audit
versions carry on-image multiplicity annotations.

WHY THIS MODULE EXISTS
======================
`PAPER_TMLR.md` is the full audit draft; `PAPER_CLEAN.md` is the readable TMLR
submission distilled from it. Three figures in the shared set were drawn for the
audit draft and carry annotations that belong to it and only to it:

  fig12_crossover.png  ✗ marks on 7 of 8 cells, the K = 44 / 58 / 72 family
                       text, and the precedence statement between families.
  fig10_withdrawal.png a per-cell Holm block for all four withdrawal cells,
                       including the K = 44 note.
  fig5_randprior.png   a footer disclosing that `randprior − blank` fails Holm
                       over both families, with the 10.7x / 3.9x margins.

This module writes CLEAN variants to NEW filenames -- `*_clean.png` -- so the
audit originals stay byte-identical for `PAPER_TMLR.md`. Nothing here overwrites
an audit figure.

WHAT IS REMOVED (audit annotation, not result)
  - the ✓ / ✗ Holm glyphs, back-hatched Holm bands and per-cell Holm tags
  - every family label (declared-28, literal-72, K = 44, K = 58, K = 26, K = 10)
  - the precedence statement between family definitions
  - the Holm footers and the per-cell p-values that only served them
  - the 10.7x / 3.9x threshold margins and the "widest margin in the study" claim

WHAT IS KEPT (result, not annotation)
  - every plotted value and its PAIRED t 95% CI, unchanged and re-verified
  - the zero line, the per-body panels, the condition colours (Okabe-Ito),
    the schedule encodings (hatch / marker shape / marker fill), 300 dpi
  - the ROBUST / ROBUST-NULL / METHOD-DEPENDENT flag grammar, which reports
    primary-vs-sensitivity estimator agreement and is a result (§3.6)
  - the honest scoping that is substantive and that `PAPER_CLEAN.md` still
    states: that the windows overlap and are therefore not independent
    confirmations; that elbow is mrad and finger is mm and the panels are not
    comparable in magnitude; that only the SIGN of the endpoint contrast differs
    between the bodies; that the cost-difference contrast is non-significant on
    both bodies; that `randprior` is an adversarial rather than an
    accuracy-matched control.

VERIFICATION
  Every value plotted here goes through the same gate `make_figs.py` applies:
  contrasts are the rows `figcontrasts.build()` already checked against
  `RECOMPUTED.md`, condition means go through `figstats.cond_mean`, and
  `build_clean` ABORTS WITHOUT WRITING A PNG if any of them disagrees. The rows
  are labelled "CLEAN ..." so they are visible in `make_figs.print_checks()`.

Run `python make_figs.py` to regenerate the audit set and these three clean
variants together; run this file directly to regenerate only the clean three.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import figstats as FS
from figstats import C, cond_mean, vals, UNIT, BODIES
from figstyle import (COND_COLORS, RAND_COLOR, BODY_COLOR, KEY_LABEL,
                      KEY_LABEL_1L, ERR_LABEL_FINAL, ERR_LABEL_HELDOUT,
                      BODY_TITLE, CI_NOTE, bar_ci, forest, footer, footer_room,
                      flag_legend_top)
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")

# contrasts each clean variant plots -- the exact RECOMPUTED.md labels
FIG12_CELLS = ["coach(const) - prior (held-out, %s)" % w
               for w in ("EARLY", "MID", "LATE", "ENDPOINT")]
FIG10_CELLS = ["prior strict-withdrawn - prior kept (ENDPOINT)",
               "coach abrupt-withdrawn - coach const (ENDPOINT)",
               "coach cost - prior cost (ENDPOINT)",
               "prior soft-withdrawn - blank (ENDPOINT)",
               "prior strict-withdrawn - blank (ENDPOINT)",
               "coach abrupt-withdrawn - blank (ENDPOINT)",
               "coach abrupt-withdrawn - prior strict-withdrawn (ENDPOINT)"]
FIG5_CELLS = ["trained prior - randprior (train, ENDPOINT)",
              "randprior - blank (train, ENDPOINT)"]

FIG10_KEYS = ["blank", "prior", "prior_soft", "prior_strict",
              "coach_const", "coach_ann", "coach_abrupt"]


# ==================================================== verification (pre-plot)
def _collect(MF):
    """Compute + verify EVERY value the three clean variants plot.

    Returns (bar_stats, bad_rows). Nothing is drawn from anything that is not
    in this dict, so a value cannot reach a canvas without passing the check.
    """
    bad = []

    # --- contrasts: already registered and checked by figcontrasts.build() ---
    for label in FIG12_CELLS + FIG10_CELLS + FIG5_CELLS:
        for body in BODIES:
            r = FS.RES.get((label, body))
            if r is None:
                bad.append("MISSING CONTRAST  %s [%s] — figcontrasts.build() "
                           "was not run" % (label, body))
            elif r["status"] in ("MISMATCH", "MISSING-FROM-RECOMPUTED"):
                bad.append("%s  %s [%s]: mine %+.3f [%+.3f, %+.3f] %s vs REC %s"
                           % (r["status"], label, body, r["mean"], r["tlo"],
                              r["thi"], r["flag"], r["ref"]))

    # --- condition means: registered here, checked against RECOMPUTED.md ------
    n0 = len(FS.MEAN_CHECKS)
    bars = {}
    for body in BODIES:
        for k in FIG10_KEYS:                       # fig10 clean, top row
            bars[("held", k, body)] = cond_mean(
                "CLEAN fig10 held-out endpoint %s" % KEY_LABEL_1L[k], body,
                MF.held_end(k, body), rec_name=MF.DISP_NAME.get(k))
        for cond in ("blank", "prior"):            # fig5 clean, top row
            bars[("train", cond, body)] = cond_mean(
                "CLEAN fig5 train endpoint %s" % cond, body,
                vals(MF.TRAIN_SPEC[cond], body, "ENDPOINT"),
                paper_mean=MF.PAPER_TRAIN_MEAN[body][cond])
        bars[("train", "randprior", body)] = cond_mean(
            "CLEAN fig5 train endpoint randprior", body,
            vals(FS.RANDPRIOR_TR, body, "ENDPOINT"),
            paper_mean=MF.PAPER_TRAIN_MEAN[body]["randprior"])

    for m in FS.MEAN_CHECKS[n0:]:
        if m["status"] == "MISMATCH":
            bad.append("MISMATCH  %s [%s]: computed %.3f vs expected %.3f (%s)"
                       % (m["label"], m["body"], m["mean"], m["expected"], m["src"]))
    return bars, bad


# ============================================================ FIG 12 (clean)
def fig12_clean(MF, bars):
    """`coach(constant) − prior` by training window, both bodies."""
    wins = [("EARLY  (2k–6k)", "EARLY"), ("MID  (6k–10k)", "MID"),
            ("LATE  (10k–12k)", "LATE"), ("ENDPOINT  (12k)", "ENDPOINT")]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.2))
    for ax, body in zip(axes, BODIES):
        rows = [dict(label=lab,
                     res=C("coach(const) - prior (held-out, %s)" % w, body),
                     color=COND_COLORS["coach"], marker="o")
                for lab, w in wins]
        forest(ax, rows, UNIT[body],
               "coach(constant) − prior (%s)\n<0 = coach ahead   |   >0 = prior ahead"
               % UNIT[body],
               BODY_TITLE[body], BODY_COLOR[body], label_fs=8.8, val_fs=7.8)
        ax.set_ylim(-1.05, len(wins) - 1 + 0.80)
        ax.margins(x=0.40)
        # ENDPOINT is the last 2k of LATE; the short rule at the left marks that
        # nesting. It stops well short of the value tags so it cannot run
        # underneath one.
        ax.axhline(0.5, xmin=0.0, xmax=0.30, color="#bbbbbb", lw=1.4)
        # the endpoint summary sits on the side of zero the winner is on, so it
        # never runs across the zero line
        end = C("coach(const) - prior (held-out, ENDPOINT)", body)
        left = end["mean"] < 0
        ax.text(0.020 if left else 0.980, 0.020,
                "ENDPOINT (12k): %s by %.1f %s"
                % ("coach ahead" if left else "prior ahead",
                   abs(end["mean"]), UNIT[body]),
                transform=ax.transAxes, ha="left" if left else "right",
                va="bottom", fontsize=8.6, fontweight="bold",
                color=COND_COLORS["coach"] if left else COND_COLORS["prior"])
    fig.suptitle("coach(constant) − prior by training window "
                 "(held-out targets, n=12, paired t 95% CIs)\n"
                 "Elbow: the coach is ahead in all four windows and the lead shrinks "
                 "(−50.6 → −10.9 mrad).\n"
                 "Finger: the early lead decays and changes sign; at the endpoint the "
                 "prior is ahead (+32.3 mm).",
                 fontweight="bold", fontsize=10.2, y=0.985)
    flag_legend_top(fig, y=0.900, ncol=4, fontsize=6.8)
    footer(fig,
           CI_NOTE,
           "All eight cells exclude zero under the primary estimator. Marker fill encodes the flag, never hue.",
           "The four windows OVERLAP — MID and LATE share step 10k, and ENDPOINT ⊂ LATE (the short grey rule marks",
           "that nesting) — so they are NOT four independent confirmations of one effect.",
           "Elbow is mrad and finger is mm: the panels are NOT comparable in magnitude, and only the SIGN of the",
           "endpoint contrast differs between the two bodies. Held-out targets; coach weight constant throughout.")
    # explicit placement (as in the audit variant) -- tight_layout leaves a wide
    # dead band under the flag key on a two-panel forest of this shape
    fig.subplots_adjust(left=0.105, right=0.985, top=0.785,
                        bottom=footer_room(fig, 6) + 0.090, wspace=0.26)
    fp = os.path.join(FIGS, "fig12_crossover_clean.png")
    fig.savefig(fp)
    plt.close(fig)
    return fp


# ============================================================ FIG 10 (clean)
def fig10_clean(MF, bars):
    """Matched withdrawal at 8k: levels, costs and post-withdrawal contrasts."""
    contrasts = [
        ("prior withdrawal COST\n(strict − kept)",
         "prior strict-withdrawn - prior kept (ENDPOINT)", COND_COLORS["prior"], "s"),
        ("coach withdrawal COST\n(abrupt − constant)",
         "coach abrupt-withdrawn - coach const (ENDPOINT)", COND_COLORS["coach"], "s"),
        ("coach cost − prior cost\n(difference of costs)",
         "coach cost - prior cost (ENDPOINT)", "#444444", "s"),
        ("prior after SOFT withdrawal\n− blank",
         "prior soft-withdrawn - blank (ENDPOINT)", COND_COLORS["prior"], "s"),
        ("prior after STRICT withdrawal\n− blank",
         "prior strict-withdrawn - blank (ENDPOINT)", COND_COLORS["prior"], "s"),
        ("coach after withdrawal\n− blank",
         "coach abrupt-withdrawn - blank (ENDPOINT)", COND_COLORS["coach"], "s"),
        ("coach after withdrawal\n− prior after withdrawal",
         "coach abrupt-withdrawn - prior strict-withdrawn (ENDPOINT)", "#444444", "s"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.8),
                             gridspec_kw={"height_ratios": [1.0, 1.85]})
    for ax, body in zip(axes[0], BODIES):
        stats = [bars[("held", k, body)] for k in FIG10_KEYS]
        x = np.arange(len(FIG10_KEYS))
        bar_ci(ax, x, stats, FIG10_KEYS)
        ax.axhline(stats[0][0], color=COND_COLORS["blank"], lw=1.2, ls=":", zorder=0)
        ax.text(-0.42, stats[0][0], "blank level", va="bottom", ha="left",
                fontsize=7.4, color="#666666")
        ax.set_xticks(x)
        ax.set_xticklabels([KEY_LABEL[k] for k in FIG10_KEYS], fontsize=7.0)
        ax.set_ylabel(ERR_LABEL_HELDOUT[body])
        ax.set_title(BODY_TITLE[body], color=BODY_COLOR[body],
                     fontweight="bold", fontsize=9.5)
        ax.margins(y=0.28)
    for ax, body in zip(axes[1], BODIES):
        rows = [dict(label=lab, res=C(key, body), color=col, marker=mk)
                for lab, key, col, mk in contrasts]
        forest(ax, rows, UNIT[body],
               "paired difference (%s)   |   >0 = withdrawal hurt / worse than blank"
               % UNIT[body], None, label_fs=7.6, val_fs=7.0)
        # the METHOD-DEPENDENT tag is the longest string in the set; widen the
        # data margins so it cannot reach either spine
        ax.margins(x=0.60)
    fig.legend(handles=[
        Patch(facecolor="white", edgecolor="black", label="kept on for the whole budget"),
        Patch(facecolor="white", edgecolor="black", hatch="//",
              label="prior withdrawn @8k (soft) / coach ANNEALED — a fade from step 0, not a withdrawal"),
        Patch(facecolor="white", edgecolor="black", hatch="xx",
              label="withdrawn @8k (strict / abrupt)")],
        loc="lower center", ncol=3, fontsize=7.2,
        bbox_to_anchor=(0.5, footer_room(fig, 7) + 0.004))
    fig.suptitle("Matched withdrawal at 8k of a 12k budget "
                 "(held-out endpoint, n=12, paired t 95% CIs)\n"
                 "Top: where every arm ends, against the model-free blank. Bottom: each route's "
                 "withdrawal cost,\nthe direct difference between the two costs, and the "
                 "post-withdrawal levels against blank.",
                 fontweight="bold", fontsize=10.0)
    flag_legend_top(fig, y=0.905, ncol=4, fontsize=6.8)
    footer(fig,
           CI_NOTE,
           "Withdrawal at 8k: SOFT = model frozen out, buffer kept; STRICT = buffer purged; ABRUPT = coach cut to 0.",
           "ANNEALED is NOT a withdrawal: that weight fades from step 0 and merely reaches 0 at 8k, so only the ABRUPT",
           "coach is matched to the prior's cut. Bars are condition means with a one-sample t 95% CI.",
           "Cutting the coach costs on both bodies; cutting the prior is not shown to cost on either. The DIRECT",
           "cost-difference contrast is non-significant on both bodies, and at n=12 these post-withdrawal nulls are",
           "underpowered in both directions (§4.8). Elbow mrad, finger mm — panels not comparable in magnitude.")
    fig.tight_layout(rect=[0, footer_room(fig, 8) + 0.030, 1, 0.888])
    fp = os.path.join(FIGS, "fig10_withdrawal_clean.png")
    fig.savefig(fp)
    plt.close(fig)
    return fp


# ============================================================= FIG 5 (clean)
def fig5_clean(MF, bars):
    """The random-initialized frozen-model control, train targets."""
    fig, axes = plt.subplots(2, 2, figsize=(10.9, 7.4),
                             gridspec_kw={"height_ratios": [1.25, 1.0]})
    for ax, body in zip(axes[0], BODIES):
        stats = [bars[("train", "blank", body)],
                 bars[("train", "randprior", body)],
                 bars[("train", "prior", body)]]
        x = np.arange(3)
        means = [s[0] for s in stats]
        errs = [s[0] - s[1] for s in stats]
        ax.bar(x, means, yerr=errs, capsize=4,
               color=[COND_COLORS["blank"], RAND_COLOR, COND_COLORS["prior"]],
               edgecolor="black", linewidth=0.7,
               error_kw=dict(lw=1.0, ecolor="#333333"))
        for xi, m, er in zip(x, means, errs):
            ax.text(xi, m + er + max(errs) * 0.20, "%.1f" % m, ha="center",
                    va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["blank", "random\nprior", "trained\nprior"])
        ax.set_ylabel(ERR_LABEL_FINAL[body])
        ax.set_title(BODY_TITLE[body], color=BODY_COLOR[body],
                     fontweight="bold", fontsize=9.5)
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
        ax.margins(x=0.36)
    fig.suptitle("Random-initialized frozen-model control, TRAIN targets "
                 "(n=12, paired t 95% CIs)\n"
                 "Top: endpoint error of blank, random prior and trained prior. "
                 "Bottom: the two paired contrasts.",
                 fontweight="bold", fontsize=10.5)
    flag_legend_top(fig, y=0.905, ncol=2, fontsize=6.6)
    footer(fig,
           CI_NOTE,
           "Bars are condition means with a one-sample t 95% CI; the lower row is the paired per-seed contrast.",
           "Scored on TRAIN targets only — the randprior arm was never evaluated on held-out targets (§4.4).",
           "The control is ADVERSARIAL rather than accuracy-matched: a random network corrupts the analytic reward",
           "evaluated at its predicted states as well as the states themselves, so it separates a trained model from an",
           "actively unhelpful one, not learned dynamics from any other ingredient (§4.4).",
           "Per §3.1 the randprior arm also trains on half the real data per update that blank does — a mundane rival",
           "account of randprior − blank. Elbow mrad, finger mm — panels are not comparable in magnitude.")
    fig.tight_layout(rect=[0, footer_room(fig, 7), 1, 0.848])
    fp = os.path.join(FIGS, "fig5_randprior_clean.png")
    fig.savefig(fp)
    plt.close(fig)
    return fp


# ================================================================== driver
def build_clean(MF):
    """Verify every value, then write the three clean variants.

    `MF` is the already-loaded `make_figs` module, passed in rather than
    imported so that this module can be imported from `make_figs.__main__`
    without a circular import.
    """
    bars, bad = _collect(MF)
    if bad:
        print("STOP — clean-variant values disagree with RECOMPUTED.md; "
              "no clean PNG was written:")
        for b in bad:
            print("  " + b)
        sys.exit(2)
    return [fig12_clean(MF, bars), fig10_clean(MF, bars), fig5_clean(MF, bars)]


if __name__ == "__main__":
    # Standalone: rebuild ONLY the three clean variants. The audit figures are
    # not touched. `make_figs.py` is imported (never executed as __main__), so
    # its own plotting phase does not run.
    import figcontrasts as FC
    import make_figs as MF

    FC.build()
    _bad = [c for c in FS.CHECKS
            if c["status"] in ("MISMATCH", "MISSING-FROM-RECOMPUTED")]
    if _bad:
        print("STOP — computed values disagree with RECOMPUTED.md; nothing was plotted:")
        for c in _bad:
            print("  %s [%s]: mine %.3f [%.3f, %.3f] %s vs REC %s"
                  % (c["label"], c["body"], c["mean"], c["tlo"], c["thi"],
                     c["flag"], c["ref"]))
        sys.exit(2)
    _paths = build_clean(MF)
    _nbad = MF.print_checks()
    print("\nClean figures written:")
    for _p in _paths:
        print("  ", _p)
    sys.exit(1 if _nbad else 0)
