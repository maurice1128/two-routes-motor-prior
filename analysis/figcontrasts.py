"""
figcontrasts.py -- registers EVERY contrast the figure set plots, computes it
with the imported `t_ci` / `bca_ci` / `flag` from `recompute_all.py`, and
verifies it cell-by-cell against `RECOMPUTED.md`.

Labels are the EXACT contrast strings used in `RECOMPUTED.md`, so the check
table is a literal row-for-row comparison. A handful of cells a figure needs are
not in `RECOMPUTED.md` at all (the train-target coach-vs-blank rows of Fig. 3);
they are registered with `in_rec=False`, computed with the same estimators, and
reported as such.
"""
import figstats as FS
from figstats import (register, d_diff, d_dd, d_mid, BODIES,
                      BLANK_HO, PRIOR_HO, BLANK_TR, PRIOR_TR,
                      COACH_CONST, COACH_ANN, COACH_ANN_TR, COACH_ABRUPT,
                      PC_CONST, PC_ANN, PC_ANN_TR,
                      PRIOR_SOFT, PRIOR_STRICT, RANDPRIOR_TR)

WINDOWS = ("EARLY", "MID", "LATE", "ENDPOINT")


def _c(label, A, B, w, in_rec=True):
    for body in BODIES:
        register(label, body, d_diff(A, B, body, w), in_rec=in_rec)


def build():
    # ---- 1. prior vs blank -------------------------------------------------
    _c("prior - blank (held-out, ENDPOINT)", PRIOR_HO, BLANK_HO, "ENDPOINT")
    _c("prior - blank (train, ENDPOINT)", PRIOR_TR, BLANK_TR, "ENDPOINT")

    # ---- 2 / 4. coach vs blank --------------------------------------------
    _c("coach(const) - blank (held-out, ENDPOINT)", COACH_CONST, BLANK_HO, "ENDPOINT")
    _c("coach(const) - blank (held-out, EARLY)", COACH_CONST, BLANK_HO, "EARLY")
    _c("coach(anneal) - blank (held-out, ENDPOINT)", COACH_ANN, BLANK_HO, "ENDPOINT")
    _c("coach(anneal) - blank (held-out, EARLY)", COACH_ANN, BLANK_HO, "EARLY")
    # train-target coach-vs-blank: NOT among the 132 recomputed contrasts
    _c("coach(anneal) - blank (train, ENDPOINT)", COACH_ANN_TR, BLANK_TR,
       "ENDPOINT", in_rec=False)
    _c("coach(anneal) - blank (train, EARLY)", COACH_ANN_TR, BLANK_TR,
       "EARLY", in_rec=False)

    # ---- 3 / L7. coach vs prior, all windows ------------------------------
    for w in WINDOWS:
        _c("coach(const) - prior (held-out, %s)" % w, COACH_CONST, PRIOR_HO, w)
        _c("coach(anneal) - prior (held-out, %s)" % w, COACH_ANN, PRIOR_HO, w)

    # ---- 5. randprior ------------------------------------------------------
    _c("trained prior - randprior (train, ENDPOINT)", PRIOR_TR, RANDPRIOR_TR, "ENDPOINT")
    _c("randprior - blank (train, ENDPOINT)", RANDPRIOR_TR, BLANK_TR, "ENDPOINT")

    # ---- 6/7/8/9. withdrawal ----------------------------------------------
    _c("prior strict-withdrawn - prior kept (ENDPOINT)", PRIOR_STRICT, PRIOR_HO, "ENDPOINT")
    _c("coach abrupt-withdrawn - coach const (ENDPOINT)", COACH_ABRUPT, COACH_CONST, "ENDPOINT")
    for body in BODIES:
        register("coach cost - prior cost (ENDPOINT)", body,
                 d_dd(COACH_ABRUPT, COACH_CONST, PRIOR_STRICT, PRIOR_HO, body, "ENDPOINT"))
    _c("prior strict-withdrawn - blank (ENDPOINT)", PRIOR_STRICT, BLANK_HO, "ENDPOINT")
    _c("coach abrupt-withdrawn - blank (ENDPOINT)", COACH_ABRUPT, BLANK_HO, "ENDPOINT")
    _c("coach abrupt-withdrawn - prior strict-withdrawn (ENDPOINT)",
       COACH_ABRUPT, PRIOR_STRICT, "ENDPOINT")
    _c("prior soft-withdrawn - blank (ENDPOINT)", PRIOR_SOFT, BLANK_HO, "ENDPOINT")

    # ---- 10 / L3. combination ---------------------------------------------
    _c("priorcoach(const) - prior (held-out, ENDPOINT)", PC_CONST, PRIOR_HO, "ENDPOINT")
    _c("priorcoach(const) - coach(const) (held-out, ENDPOINT)", PC_CONST, COACH_CONST, "ENDPOINT")
    for body in BODIES:
        register("priorcoach(const) - 0.5*[prior + coach(const)] (ENDPOINT)", body,
                 d_mid(PC_CONST, PRIOR_HO, COACH_CONST, body, "ENDPOINT"))
        register("priorcoach(anneal) - 0.5*[prior + coach(anneal)] (ENDPOINT)", body,
                 d_mid(PC_ANN, PRIOR_HO, COACH_ANN, body, "ENDPOINT"))
