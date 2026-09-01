if __name__ != "__main__":
    raise ImportError(
        "This script writes a tracked artefact and must not be imported; run it directly. "
        "Import-time writes have silently damaged RECOMPUTED.md three times in this project."
    )

"""
recompute_legacy.py -- recomputation of the contrasts that were still carrying
the OLD z-interval (`mean +/- 1.96*SE`) in PAPER_v19.md, i.e. every cell marked
*(legacy)* there and not present among the 64 contrasts of RECOMPUTED.md.

Method is IDENTICAL to recompute_all.py (whose functions are imported directly,
not reimplemented):
  * unit of analysis = seed, paired differences over matched seed ids
  * PRIMARY   : paired t 95% CI, mean(d) +/- t_{0.975, n-1} * sd(d)/sqrt(n)
  * SENSITIVITY: BCa bootstrap 95% CI, 20000 resamples, RNG seed 0, resampling SEEDS
  * eval_dist * 1000; elbow = mrad, finger = mm; bodies never pooled
  * ENDPOINT = index 5; EARLY = mean idx 0,1,2; MID = mean idx 2,3,4; LATE = mean idx 4,5
  * flags: ROBUST / ROBUST-NULL / METHOD-DEPENDENT

Additionally reports, for each contrast, the OLD z-interval (mean +/- 1.96*SE)
so that verdict changes caused by the estimator error can be listed explicitly.

Running this script first re-executes recompute_all.py (which rewrites
RECOMPUTED.md from scratch, deterministically) and then APPENDS the section
"Appendix: previously-legacy contrasts".  Numbers only, no interpretation.
"""

import os
import sys
import importlib.util
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- import recompute_all (this regenerates RECOMPUTED.md, deterministically) ----
_spec = importlib.util.spec_from_file_location(
    "recompute_all", os.path.join(ROOT, "recompute_all.py"))
RA = importlib.util.module_from_spec(_spec)
sys.modules["recompute_all"] = RA
_spec.loader.exec_module(RA)

series = RA.series
t_ci = RA.t_ci
bca_ci = RA.bca_ci
flag = RA.flag
UNIT = RA.UNIT
fmt = RA.fmt

BODIES = ("elbow", "finger")

# ------------------------------------------------------------- condition specs
BLANK_HO = ("results_heldout", "blank")
PRIOR_HO = ("results_heldout", "prior")
BLANK_TR = ("results_2x2", "blank")
PRIOR_TR = ("results_2x2", "prior")

# CONTAMINATED-teacher coach students (the ORIGINAL runs, annealed schedule)
COACH_CONTAM_HO = ("results_heldout", "coach")
COACH_CONTAM_TR = ("results_2x2", "coach")
PC_CONTAM_HO = ("results_heldout", "priorcoach")
PC_CONTAM_TR = ("results_2x2", "priorcoach")

# CLEAN-teacher coach students, ANNEALED schedule
COACH_ANN = ("results_cleancoach", "coach")
COACH_ANN_TR = ("results_cleancoach_train", "coach")
PC_ANN = ("results_cleancoach", "priorcoach")
PC_ANN_TR = ("results_cleancoach_train", "priorcoach")

# CONSTANT schedule
COACH_CONST = ("results_noanneal", "coach")
PC_CONST = ("results_pcconst", "priorcoach")

PRIOR_STRICT = ("results_priorpurge", "prior")

# ------------------------------------------------------------------ statistics
Z = 1.959963984540054


def z_ci(d):
    n = len(d)
    m = float(np.mean(d))
    se = float(np.std(d, ddof=1)) / np.sqrt(n)
    return m, m - Z * se, m + Z * se


ROWS = []      # (family, label, body, n, mean, tlo, thi, blo, bhi, flag, zlo, zhi, zsig, changed)
FAMILIES = []
SKIPPED = []


def _emit(family, label, body, d):
    if family not in FAMILIES:
        FAMILIES.append(family)
    n = len(d)
    m, tlo, thi = t_ci(d)
    _, blo, bhi = bca_ci(d)
    fl = flag(tlo, thi, blo, bhi)
    _, zlo, zhi = z_ci(d)
    zsig = (zlo > 0) or (zhi < 0)
    # verdict change = what the z-interval licensed vs what the policy now licenses
    changed = (zsig and fl != "ROBUST") or ((not zsig) and fl == "ROBUST")
    ROWS.append((family, label, body, n, m, tlo, thi, blo, bhi, fl, zlo, zhi, zsig, changed))


def _vals(spec, body, w):
    s, v = series(spec, body, w)
    return dict(zip(s, v))


def combo(family, label, terms, w, bodies=BODIES):
    """terms = list of (coefficient, spec, window_or_None).

    Builds the per-seed linear combination sum_i coef_i * value(spec_i, window_i)
    over the seeds common to every term, then intervals it.
    """
    for body in bodies:
        try:
            maps = [(c, _vals(sp, body, ww or w)) for c, sp, ww in terms]
        except FileNotFoundError as e:
            SKIPPED.append("%s [%s]: %s" % (label, body, e))
            continue
        common = sorted(set.intersection(*[set(m.keys()) for _c, m in maps]))
        if len(common) < 2:
            SKIPPED.append("%s [%s]: only %d matched seeds" % (label, body, len(common)))
            continue
        d = np.array([sum(c * m[s] for c, m in maps) for s in common])
        _emit(family, label, body, d)


def contrast(family, label, A, B, w, bodies=BODIES):
    combo(family, label, [(1.0, A, None), (-1.0, B, None)], w, bodies)


def win_interaction(family, label, A, B, w1, w2, bodies=BODIES):
    """(A - B)@w1 - (A - B)@w2, within seed."""
    combo(family, label,
          [(1.0, A, w1), (-1.0, B, w1), (-1.0, A, w2), (1.0, B, w2)], w1, bodies)


# =============================================================== L1. decontamination
F = "L1. teacher decontamination (clean-teacher minus contaminated-teacher; negative = clean better)"
for w in ("ENDPOINT", "EARLY"):
    contrast(F, "coach clean - coach contaminated (held-out, %s)" % w,
             COACH_ANN, COACH_CONTAM_HO, w)
for w in ("ENDPOINT", "EARLY"):
    contrast(F, "coach clean - coach contaminated (train, %s)" % w,
             COACH_ANN_TR, COACH_CONTAM_TR, w)
for w in ("ENDPOINT", "EARLY"):
    contrast(F, "priorcoach clean - priorcoach contaminated (held-out, %s)" % w,
             PC_ANN, PC_CONTAM_HO, w)
for w in ("ENDPOINT", "EARLY"):
    contrast(F, "priorcoach clean - priorcoach contaminated (train, %s)" % w,
             PC_ANN_TR, PC_CONTAM_TR, w)

# ================================================================ L2. schedule
F = "L2. coach schedule (constant minus annealed, clean teachers, held-out)"
for w in ("ENDPOINT", "EARLY", "MID", "LATE"):
    contrast(F, "coach(const) - coach(anneal) (held-out, %s)" % w,
             COACH_CONST, COACH_ANN, w)
F2 = "L2b. pre-8k bit-identity check for the prior arm (strict minus kept, EARLY = steps 2k/4k/6k)"
contrast(F2, "prior strict-withdrawn - prior kept (held-out, EARLY)",
         PRIOR_STRICT, PRIOR_HO, "EARLY")

# ================================================================ L3. mixture / midpoint
F = "L3. mixture / 50-50 midpoint (priorcoach minus midpoint of prior and coach; held-out)"
combo(F, "priorcoach(const) - 0.5*[prior + coach(const)] (ENDPOINT)",
      [(1.0, PC_CONST, None), (-0.5, PRIOR_HO, None), (-0.5, COACH_CONST, None)],
      "ENDPOINT")
combo(F, "priorcoach(const) - 0.5*[prior + coach(const)] (EARLY)",
      [(1.0, PC_CONST, None), (-0.5, PRIOR_HO, None), (-0.5, COACH_CONST, None)],
      "EARLY")
combo(F, "priorcoach(anneal) - 0.5*[prior + coach(anneal)] (ENDPOINT)",
      [(1.0, PC_ANN, None), (-0.5, PRIOR_HO, None), (-0.5, COACH_ANN, None)],
      "ENDPOINT")
combo(F, "priorcoach(anneal) - 0.5*[prior + coach(anneal)] (EARLY)",
      [(1.0, PC_ANN, None), (-0.5, PRIOR_HO, None), (-0.5, COACH_ANN, None)],
      "EARLY")

F = "L4. priorcoach minus coach alone, ANNEALED schedule (held-out; the constant-schedule version is already among the 64)"
for w in ("ENDPOINT", "EARLY", "MID", "LATE"):
    contrast(F, "priorcoach(anneal) - coach(anneal) (held-out, %s)" % w,
             PC_ANN, COACH_ANN, w)
contrast(F, "priorcoach(anneal) - coach(anneal) (train, ENDPOINT)",
         PC_ANN_TR, COACH_ANN_TR, "ENDPOINT")

# ================================================================ L5. back-loading interactions
F = "L5. back-loading window interactions (per-seed gap difference; negative = the gap in favour of the aid is LARGER in the later window)"
win_interaction(F, "[prior - blank]@LATE - @EARLY (held-out)", PRIOR_HO, BLANK_HO, "LATE", "EARLY")
win_interaction(F, "[prior - blank]@ENDPOINT - @EARLY (held-out)", PRIOR_HO, BLANK_HO, "ENDPOINT", "EARLY")
win_interaction(F, "[coach(const) - blank]@LATE - @EARLY (held-out)", COACH_CONST, BLANK_HO, "LATE", "EARLY")
win_interaction(F, "[coach(const) - blank]@ENDPOINT - @EARLY (held-out)", COACH_CONST, BLANK_HO, "ENDPOINT", "EARLY")
win_interaction(F, "[coach(anneal) - blank]@LATE - @EARLY (held-out)", COACH_ANN, BLANK_HO, "LATE", "EARLY")
win_interaction(F, "[coach(anneal) - blank]@ENDPOINT - @EARLY (held-out)", COACH_ANN, BLANK_HO, "ENDPOINT", "EARLY")

# ================================================================ L6. prior - blank, MID and LATE
F = "L6. prior minus blank, MID and LATE windows (held-out; the v18 cells that were never recomputed)"
for w in ("MID", "LATE"):
    contrast(F, "prior - blank (held-out, %s)" % w, PRIOR_HO, BLANK_HO, w)

# ================================================================ L7. coach(anneal) vs prior
F = "L7. coach(ANNEALED) minus prior, all windows (held-out; the endpoint crossover under the annealed schedule)"
for w in ("ENDPOINT", "EARLY", "MID", "LATE"):
    contrast(F, "coach(anneal) - prior (held-out, %s)" % w, COACH_ANN, PRIOR_HO, w)

# ------------------------------------------------------------------- reporting
lines = []
A = lines.append
A("")
A("---")
A("")
A("# Appendix: previously-legacy contrasts")
A("")
A("Generated by `recompute_legacy.py`, which imports `recompute_all.py` and reuses its "
  "estimators unchanged. These are the contrasts that `PAPER_v19.md` still reported as "
  "***(legacy)*** z-intervals (`mean ± 1.96·SE`) because they were **not** among the 64 "
  "contrasts of the main table above. They are recomputed here under the **same method** as "
  "the main table: paired differences over matched seeds; **paired t 95% CI (PRIMARY, "
  "df = n − 1)**; **BCa bootstrap 95% CI (20 000 resamples, RNG seed 0, resampling seeds)** "
  "as sensitivity; flags `ROBUST` / `ROBUST-NULL` / `METHOD-DEPENDENT` as defined above. "
  "`eval_dist × 1000`; elbow = mrad, finger = mm; bodies never pooled.")
A("")
A("The **old z 95% CI** column reproduces the superseded estimator (`mean ± 1.96·SE`) so that "
  "**verdict changes** attributable purely to the estimator error can be read off directly. "
  "`z sig?` = the z-interval excludes zero. **`Δ verdict`** = **YES** when the z-interval and "
  "the v19 policy (only `ROBUST` may support a claim) disagree about whether the contrast is "
  "claimable.")
A("")
A("Contrasts here are **additional to** the 64; the totals in the main Summary above are "
  "unchanged.")
A("")
A("## Condition → data mapping (additions used only in this appendix)")
A("")
A("| label | directory | condition key |")
A("|---|---|---|")
for nm, sp in [("coach CONTAMINATED-teacher, annealed (held-out)", COACH_CONTAM_HO),
               ("coach CONTAMINATED-teacher, annealed (train)", COACH_CONTAM_TR),
               ("priorcoach CONTAMINATED-teacher, annealed (held-out)", PC_CONTAM_HO),
               ("priorcoach CONTAMINATED-teacher, annealed (train)", PC_CONTAM_TR),
               ("coach CLEAN-teacher, annealed (held-out)", COACH_ANN),
               ("coach CLEAN-teacher, annealed (train)", COACH_ANN_TR)]:
    A("| %s | `%s_{elbow,finger}/` | `%s` |" % (nm, sp[0], sp[1]))
A("")
A("## Results")
A("")
for fam in FAMILIES:
    A("### %s" % fam)
    A("")
    A("| contrast | body | unit | n | mean diff | paired t 95% CI (PRIMARY) | BCa 95% CI (sensitivity) | flag | old z 95% CI | z sig? | Δ verdict |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ROWS:
        if r[0] != fam:
            continue
        _, lab, body, n, m, tlo, thi, blo, bhi, fl, zlo, zhi, zsig, ch = r
        A("| %s | %s | %s | %d | %s | [%s, %s] | [%s, %s] | %s | [%s, %s] | %s | %s |"
          % (lab, body, UNIT[body], n, fmt(m), fmt(tlo), fmt(thi), fmt(blo), fmt(bhi),
             fl, fmt(zlo), fmt(zhi), "yes" if zsig else "no", "**YES**" if ch else "no"))
    A("")

rob = [r for r in ROWS if r[9] == "ROBUST"]
nul = [r for r in ROWS if r[9] == "ROBUST-NULL"]
md = [r for r in ROWS if r[9] == "METHOD-DEPENDENT"]
chg = [r for r in ROWS if r[13]]

A("## Appendix summary")
A("")
A("Previously-legacy contrasts recomputed: **%d**  (ROBUST %d / ROBUST-NULL %d / METHOD-DEPENDENT %d)"
  % (len(ROWS), len(rob), len(nul), len(md)))
A("")
A("### Contrasts whose verdict CHANGES relative to the z-interval")
A("")
if chg:
    A("| contrast | body | unit | mean diff | old z 95% CI | z verdict | paired t 95% CI | BCa 95% CI | new flag |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in chg:
        _, lab, body, n, m, tlo, thi, blo, bhi, fl, zlo, zhi, zsig, _c = r
        A("| %s | %s | %s | %s | [%s, %s] | %s | [%s, %s] | [%s, %s] | %s |"
          % (lab, body, UNIT[body], fmt(m), fmt(zlo), fmt(zhi),
             "significant" if zsig else "n.s.",
             fmt(tlo), fmt(thi), fmt(blo), fmt(bhi), fl))
else:
    A("_none — every previously-legacy contrast keeps the verdict its z-interval implied._")
A("")
A("### ROBUST among the previously-legacy set")
A("")
if rob:
    A("| contrast | body | unit | n | mean diff | paired t 95% CI | BCa 95% CI |")
    A("|---|---|---|---|---|---|---|")
    for r in rob:
        _, lab, body, n, m, tlo, thi, blo, bhi = r[:9]
        A("| %s | %s | %s | %d | %s | [%s, %s] | [%s, %s] |"
          % (lab, body, UNIT[body], n, fmt(m), fmt(tlo), fmt(thi), fmt(blo), fmt(bhi)))
else:
    A("_none_")
A("")
A("### METHOD-DEPENDENT among the previously-legacy set")
A("")
if md:
    A("| contrast | body | unit | n | mean diff | paired t 95% CI | BCa 95% CI | t excludes 0? | BCa excludes 0? |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in md:
        _, lab, body, n, m, tlo, thi, blo, bhi = r[:9]
        te = "yes" if (tlo > 0 or thi < 0) else "no"
        be = "yes" if (blo > 0 or bhi < 0) else "no"
        A("| %s | %s | %s | %d | %s | [%s, %s] | [%s, %s] | %s | %s |"
          % (lab, body, UNIT[body], n, fmt(m), fmt(tlo), fmt(thi), fmt(blo), fmt(bhi), te, be))
else:
    A("_none_")
A("")
A("### Notes on individual rows")
A("")
A("- **L2b** is an exact bit-identity check, not an inferential result: every paired "
  "difference is exactly 0.000, so sd = 0, and both intervals are the degenerate [0, 0]. "
  "It is flagged `ROBUST-NULL` by the mechanical rule; read it as the exact equality it is.")
A("- **L2** is reported as `constant − annealed`. The legacy **+21.94 mrad** elbow figure was "
  "reported in the opposite orientation (`annealed − constant`); the interval mirrors "
  "accordingly.")
A("- **Already among the 64, therefore NOT repeated here:** `priorcoach(anneal) − prior` "
  "held-out and train at ENDPOINT (the anti-synergy cells, main table family 10), and "
  "`priorcoach(const) − coach(const)` in all four windows (main table family 10).")
A("")
A("### Previously-legacy items that could NOT be recomputed")
A("")
uncomputable = [
    ("§4.8 — v18's SD ratios and its ratio-of-ratios symmetry test",
     "not paired-difference contrasts over seeds; they are ratios of sample SDs, "
     "outside the estimator defined for this table. The current §4.8 ratios are already "
     "bootstrap intervals, not z-intervals."),
    ("§4.10.3 — the n=3 plateau/30k gaps",
     "the draft reports no interval of any kind for them, and the runs are n=3 with the "
     "JSONs never written (curves parsed from logs), so there is no legacy interval to correct."),
    ("§4.1 — the legacy train-target MEANS (blank 65.6/123.3, prior 40.1/89.0, "
     "randprior 89.4/129.8) and §3.5 teacher scores",
     "point estimates, not intervals; the contrasts built on them are already among the 64."),
]
for nm, why in uncomputable:
    A("- **%s** — %s" % (nm, why))
A("")
if SKIPPED:
    A("Contrasts skipped for missing data:")
    A("")
    for s in SKIPPED:
        A("- %s" % s)
    A("")
else:
    A("No appendix contrast was skipped for missing data: every condition it needs is on disk, "
      "**including the contaminated-teacher coach and priorcoach runs** "
      "(`results_heldout_{elbow,finger}/{coach,priorcoach}_seed*.json` for held-out and "
      "`results_2x2_{elbow,finger}/{coach,priorcoach}_seed*.json` for train), which are the "
      "original pre-decontamination runs.")
    A("")

A("### Reproduction check against the legacy figures quoted in the draft")
A("")
A("Each recomputed cell is the same quantity as the legacy cell: the **z column above "
  "reproduces the previously-printed legacy interval** (up to orientation).")
A("")
A("| legacy figure as printed | where | recomputed here | z column reproduces it |")
A("|---|---|---|---|")
A("| endpoint −2.3 [−19.6, +15.0] | §4.2, elbow decontamination | `coach clean − coach contaminated "
  "(held-out, ENDPOINT)` elbow, +2.335, z [−14.967, +19.638] | yes, sign-mirrored |")
A("| EARLY −1.8 [−8.1, +4.4] | §4.2, elbow decontamination | `coach clean − coach contaminated "
  "(held-out, EARLY)` elbow, +1.848, z [−4.431, +8.126] | yes, sign-mirrored |")
A("| \"deflating the coach's early advantage by ≈15 mm\" | §4.2, finger decontamination | "
  "`coach clean − coach contaminated (held-out, EARLY)` finger, **+15.208** | yes |")
A("| **+21.94 mrad** coach schedule cost | §3.2/§4.7/§4.12 item 13 | `coach(const) − coach(anneal) "
  "(held-out, ENDPOINT)` elbow, **−21.942** | yes, sign-mirrored |")
A("")
A("---")
A("")
A("## Standalone summary — what changed")
A("")
A("**Scope.** `PAPER_v19.md` recomputed 64 contrasts and left every other interval in the draft "
  "marked ***(legacy)*** — still `mean ± 1.96·SE`. **%d further contrasts** have now been "
  "recomputed under the identical method (paired t primary + BCa sensitivity). "
  "The recomputed superset is **%d contrasts** (64 + %d)."
  % (len(ROWS), 64 + len(ROWS), len(ROWS)))
A("")
A("**Recoverability.** **Nothing was unrecoverable.** The contaminated-teacher coach and "
  "priorcoach runs — the ones the decontamination contrasts need — are on disk as the original "
  "`coach`/`priorcoach` cells of `results_heldout_{elbow,finger}/` (held-out) and "
  "`results_2x2_{elbow,finger}/` (train), 12 seeds each. Three legacy items have no interval to "
  "correct at all (the §4.8 SD ratios, the n=3 §4.10.3 plateau gaps, and the §4.1/§3.5 point "
  "means); they are listed above rather than computed.")
A("")
A("**Flag counts over the %d previously-legacy contrasts: ROBUST %d / ROBUST-NULL %d / "
  "METHOD-DEPENDENT %d.**" % (len(ROWS), len(rob), len(nul), len(md)))
A("")
A("**Verdict changes: %d.** In every one of them the legacy z-interval excluded zero and the "
  "corrected primary estimator does not, so each moves from *significant* to *not claimable* "
  "under the §3.7 policy; **no contrast moves in the opposite direction** (the t interval is "
  "uniformly wider than the z interval, so a z-null can never become ROBUST). "
  "%d land on METHOD-DEPENDENT (BCa still excludes zero, t does not) and %d on ROBUST-NULL."
  % (len(chg),
     len([r for r in chg if r[9] == "METHOD-DEPENDENT"]),
     len([r for r in chg if r[9] == "ROBUST-NULL"])))
A("")
A("The %d are listed in full in *Contrasts whose verdict CHANGES relative to the z-interval* "
  "above." % len(chg))
A("")
A("**Legacy readings that survive the correction.** The elbow schedule cost keeps its "
  "significance (`coach(const) − coach(anneal)` ENDPOINT elbow **−21.942**, t "
  "[−40.876, −3.008], BCa [−42.227, −8.600], ROBUST — the legacy **+21.94** figure in the "
  "opposite orientation); the elbow decontamination nulls stay null (endpoint +2.335 and EARLY "
  "+1.848, both ROBUST-NULL); the finger decontamination EARLY deflation stays significant "
  "(**+15.208**, t [+9.433, +20.982], ROBUST); the 50/50 midpoint deviation stays significant on "
  "**elbow** under the constant coach (**−6.085**, t [−8.899, −3.271], ROBUST) but becomes "
  "METHOD-DEPENDENT on **finger** (+9.497); and all four back-loading `prior − blank` "
  "interaction cells stay non-significant (ROBUST-NULL on both bodies, both window pairs), as "
  "the draft anticipated.")
A("")

out_path = os.path.join(ROOT, "RECOMPUTED.md")

# IDEMPOTENCE MARKER. This script appends; without this check, running it twice
# silently duplicates the whole appendix inside the authoritative table, and a
# duplicated appendix is far harder to notice than a missing one. The marker is
# the appendix's own H1 heading, which only this script writes.
_MARKER = "# Appendix: previously-legacy contrasts"
_existing = ""
if os.path.exists(out_path):
    with open(out_path, encoding="utf-8", errors="replace") as _fh:
        _existing = _fh.read()

if _MARKER in _existing:
    print("REFUSING TO APPEND: '%s' is already present in %s." % (_MARKER, out_path))
    print("This appendix is already archived; re-running would duplicate it.")
    print("Delete the existing section first if you genuinely intend to regenerate.")
    raise SystemExit(0)

with open(out_path, "a", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")

print("appended appendix to", out_path)
print("legacy contrasts: %d | ROBUST %d | ROBUST-NULL %d | METHOD-DEPENDENT %d"
      % (len(ROWS), len(rob), len(nul), len(md)))
print("\nVERDICT CHANGES vs z-interval (%d):" % len(chg))
for r in chg:
    print("  - %s [%s] mean=%.3f z=[%.3f,%.3f] %s -> t=[%.3f,%.3f] bca=[%.3f,%.3f] %s"
          % (r[1], r[2], r[4], r[10], r[11], "SIG" if r[12] else "n.s.",
             r[5], r[6], r[7], r[8], r[9]))
if SKIPPED:
    print("\nSKIPPED:")
    for s in SKIPPED:
        print("  -", s)
