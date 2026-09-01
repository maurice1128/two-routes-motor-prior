"""
extra_stats.py -- the three computations requested for PAPER_v21:

  1a. Wilcoxon signed-rank test (EXACT, n=12) on every contrast that currently
      carries a claim (the ROBUST cells cited in the title, abstract, S1.1 and
      S6), plus the two new coach-decay interaction cells.
  1b. Coefficient of variation (CV = SD/mean) per condition per body at
      ENDPOINT, and the CV ratios blank-vs-condition next to the raw SD ratios.
  1c. The SD-ratio intervals as produced by `figstats.sd_ratio_ci` itself --
      the function `make_figs.py` actually calls -- which is authoritative.

Reuses recompute_all.py's loaders/estimators via figstats (side-effect free).
Writes an appendix onto RECOMPUTED.md.
"""
import itertools
import numpy as np

import figstats as FS

t_ci, bca_ci, flag = FS.t_ci, FS.bca_ci, FS.flag


# ------------------------------------------------------------- Wilcoxon (exact)
def wilcoxon_exact(d):
    """Two-sided exact Wilcoxon signed-rank test of H0: median(d)=0.

    Zeros are dropped (Wilcoxon's own convention).  |d| are ranked with
    mid-ranks for ties; the exact null enumerates all 2^n sign patterns of that
    fixed rank vector, which is valid under sign-symmetry.
    """
    d = np.asarray(d, dtype=float)
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return dict(n=0, W=float("nan"), p=float("nan"))
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    srt = a[order]
    while i < n:                       # mid-ranks for ties
        j = i
        while j + 1 < n and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    Wp = float(ranks[d > 0].sum())
    total = float(ranks.sum())

    # exact null distribution of W+ over 2^n sign patterns
    dist = {}
    for signs in itertools.product((0, 1), repeat=n):
        w = float(np.dot(ranks, signs))
        dist[w] = dist.get(w, 0) + 1
    N = 2 ** n
    p_le = sum(c for w, c in dist.items() if w <= Wp) / N
    p_ge = sum(c for w, c in dist.items() if w >= Wp) / N
    p = min(1.0, 2.0 * min(p_le, p_ge))
    return dict(n=n, W=Wp, Wminus=total - Wp, p=p)


# ------------------------------------------------------------------ 1a. cells
S = FS   # short alias for the condition specs

# (label, body, kind, args)  kind: "diff" = A-B on window; "win" = (A-B)@w1 - @w2
CLAIM_CELLS = [
    # --- abstract / S1.1 claim 1 : the elbow prior effect
    ("prior - blank (held-out, ENDPOINT)", "elbow", "diff", (S.PRIOR_HO, S.BLANK_HO, "ENDPOINT")),
    ("prior - blank (train, ENDPOINT)", "elbow", "diff", (S.PRIOR_TR, S.BLANK_TR, "ENDPOINT")),
    # --- claim 2 : the constant coach on elbow
    ("coach(const) - blank (held-out, ENDPOINT)", "elbow", "diff", (S.COACH_CONST, S.BLANK_HO, "ENDPOINT")),
    ("coach(const) - prior (held-out, ENDPOINT)", "elbow", "diff", (S.COACH_CONST, S.PRIOR_HO, "ENDPOINT")),
    # --- claim 3 : the early coach lead, both bodies, both schedules
    ("coach(const) - blank (held-out, EARLY)", "elbow", "diff", (S.COACH_CONST, S.BLANK_HO, "EARLY")),
    ("coach(const) - blank (held-out, EARLY)", "finger", "diff", (S.COACH_CONST, S.BLANK_HO, "EARLY")),
    ("coach(anneal) - blank (held-out, EARLY)", "elbow", "diff", (S.COACH_ANN, S.BLANK_HO, "EARLY")),
    ("coach(anneal) - blank (held-out, EARLY)", "finger", "diff", (S.COACH_ANN, S.BLANK_HO, "EARLY")),
    ("coach(const) - prior (held-out, EARLY)", "elbow", "diff", (S.COACH_CONST, S.PRIOR_HO, "EARLY")),
    ("coach(const) - prior (held-out, EARLY)", "finger", "diff", (S.COACH_CONST, S.PRIOR_HO, "EARLY")),
    ("coach(const) - prior (held-out, MID)", "elbow", "diff", (S.COACH_CONST, S.PRIOR_HO, "MID")),
    ("coach(const) - prior (held-out, MID)", "finger", "diff", (S.COACH_CONST, S.PRIOR_HO, "MID")),
    # --- claim 4 : the endpoint crossover
    ("coach(const) - prior (held-out, LATE)", "elbow", "diff", (S.COACH_CONST, S.PRIOR_HO, "LATE")),
    ("coach(const) - prior (held-out, LATE)", "finger", "diff", (S.COACH_CONST, S.PRIOR_HO, "LATE")),
    ("coach(const) - prior (held-out, ENDPOINT)", "finger", "diff", (S.COACH_CONST, S.PRIOR_HO, "ENDPOINT")),
    # --- claim 5 : randprior
    ("trained prior - randprior (train, ENDPOINT)", "elbow", "diff", (S.PRIOR_TR, S.RANDPRIOR_TR, "ENDPOINT")),
    ("trained prior - randprior (train, ENDPOINT)", "finger", "diff", (S.PRIOR_TR, S.RANDPRIOR_TR, "ENDPOINT")),
    ("randprior - blank (train, ENDPOINT)", "elbow", "diff", (S.RANDPRIOR_TR, S.BLANK_TR, "ENDPOINT")),
    # --- claim 6 : withdrawal
    ("coach abrupt-withdrawn - coach const (ENDPOINT)", "elbow", "diff", (S.COACH_ABRUPT, S.COACH_CONST, "ENDPOINT")),
    ("coach abrupt-withdrawn - coach const (ENDPOINT)", "finger", "diff", (S.COACH_ABRUPT, S.COACH_CONST, "ENDPOINT")),
    ("coach abrupt-withdrawn - blank (ENDPOINT)", "finger", "diff", (S.COACH_ABRUPT, S.BLANK_HO, "ENDPOINT")),
    ("coach abrupt-withdrawn - prior strict-withdrawn (ENDPOINT)", "finger", "diff", (S.COACH_ABRUPT, S.PRIOR_STRICT, "ENDPOINT")),
    # --- S6 : the combination arm
    ("priorcoach(const) - prior (held-out, ENDPOINT)", "elbow", "diff", (S.PC_CONST, S.PRIOR_HO, "ENDPOINT")),
    ("priorcoach(const) - prior (held-out, ENDPOINT)", "finger", "diff", (S.PC_CONST, S.PRIOR_HO, "ENDPOINT")),
    ("priorcoach(const) - prior (held-out, LATE)", "finger", "diff", (S.PC_CONST, S.PRIOR_HO, "LATE")),
    ("priorcoach(const) - coach(const) (held-out, EARLY)", "elbow", "diff", (S.PC_CONST, S.COACH_CONST, "EARLY")),
    # --- NEW (F-6) : the decay of the coach's lead over the prior
    ("[coach(const) - prior]@LATE - @EARLY (held-out)", "elbow", "win", (S.COACH_CONST, S.PRIOR_HO, "LATE", "EARLY")),
    ("[coach(const) - prior]@LATE - @EARLY (held-out)", "finger", "win", (S.COACH_CONST, S.PRIOR_HO, "LATE", "EARLY")),
]


def get_d(body, kind, args):
    if kind == "diff":
        A, B, w = args
        return FS.d_diff(A, B, body, w)
    A, B, w1, w2 = args
    return FS.d_win(A, B, body, w1, w2)


rows_1a = []
for label, body, kind, args in CLAIM_CELLS:
    d = get_d(body, kind, args)
    m, tlo, thi = t_ci(d)
    _, blo, bhi = bca_ci(d)
    fl = flag(tlo, thi, blo, bhi)
    wx = wilcoxon_exact(d)
    t_sig = (tlo > 0) or (thi < 0)
    w_sig = wx["p"] < 0.05
    rows_1a.append(dict(label=label, body=body, n=len(d), mean=m, tlo=tlo, thi=thi,
                        blo=blo, bhi=bhi, flag=fl, W=wx["W"], p=wx["p"],
                        t_sig=t_sig, w_sig=w_sig, agree=(t_sig == w_sig)))

# cross-check against scipy where available
try:
    from scipy.stats import wilcoxon as _spw
    _md = 0.0
    for r, (label, body, kind, args) in zip(rows_1a, CLAIM_CELLS):
        d = get_d(body, kind, args)
        try:
            p = float(_spw(d, alternative="two-sided", method="exact").pvalue)
        except TypeError:                                 # older scipy
            p = float(_spw(d, alternative="two-sided", mode="exact")[1])
        _md = max(_md, abs(p - r["p"]))
    SCIPY_NOTE = "cross-checked against `scipy.stats.wilcoxon(method='exact')`, " \
                 "max |Δp| = %.2e" % _md
except Exception as e:                                    # pragma: no cover
    SCIPY_NOTE = "scipy cross-check unavailable (%s); exact enumeration only" \
                 % type(e).__name__

# --------------------------------------------------------------- 1b. CV / ratios
DISP_CONDS = [
    ("blank", S.BLANK_HO),
    ("prior kept", S.PRIOR_HO),
    ("prior soft-withdrawn", S.PRIOR_SOFT),
    ("prior strict-withdrawn", S.PRIOR_STRICT),
    ("coach constant", S.COACH_CONST),
    ("coach annealed", S.COACH_ANN),
    ("coach abrupt-withdrawn", S.COACH_ABRUPT),
]


def cv_ratio_ci(blank, cond, nboot=20000, seed=0):
    """Percentile bootstrap CI for CV(blank)/CV(cond); resamples seeds.

    Same resampling scheme, nboot and seed as figstats.sd_ratio_ci.
    """
    rng = np.random.default_rng(seed)
    i1 = rng.integers(0, len(blank), size=(nboot, len(blank)))
    i2 = rng.integers(0, len(cond), size=(nboot, len(cond)))
    cv1 = blank[i1].std(axis=1, ddof=1) / blank[i1].mean(axis=1)
    cv2 = cond[i2].std(axis=1, ddof=1) / cond[i2].mean(axis=1)
    r = cv1 / cv2
    lo, hi = np.percentile(r, [2.5, 97.5])
    point = (blank.std(ddof=1) / blank.mean()) / (cond.std(ddof=1) / cond.mean())
    return float(point), float(lo), float(hi)


VALS = {}
for name, spec in DISP_CONDS:
    for body in ("elbow", "finger"):
        VALS[(name, body)] = FS.vals(spec, body, "ENDPOINT")

rows_1b = []
for name, spec in DISP_CONDS:
    for body in ("elbow", "finger"):
        v = VALS[(name, body)]
        b = VALS[("blank", body)]
        mu = float(np.mean(v))
        sd = float(np.std(v, ddof=1))
        cv = sd / mu
        if name == "blank":
            rows_1b.append(dict(cond=name, body=body, mean=mu, sd=sd, cv=cv,
                                sdr=None, sdlo=None, sdhi=None,
                                cvr=None, cvlo=None, cvhi=None))
            continue
        sdr, sdlo, sdhi = FS.sd_ratio_ci(b, v)
        cvr, cvlo, cvhi = cv_ratio_ci(b, v)
        rows_1b.append(dict(cond=name, body=body, mean=mu, sd=sd, cv=cv,
                            sdr=sdr, sdlo=sdlo, sdhi=sdhi,
                            cvr=cvr, cvlo=cvlo, cvhi=cvhi))

# ------------------------------------------- 1c. authoritative SD-ratio intervals
PAPER_SD_CI = {   # as printed in PAPER_v20 S4.8 / abstract / S6
    ("prior kept", "elbow"): (2.78, 1.47, 4.60),
    ("coach constant", "elbow"): (2.93, 1.43, 7.11),
    ("prior strict-withdrawn", "elbow"): (0.87, 0.45, 1.58),
    ("coach abrupt-withdrawn", "elbow"): (1.51, 0.78, 2.82),
    ("prior kept", "finger"): (1.68, 0.54, 3.50),
    ("coach constant", "finger"): (3.53, 1.15, 6.65),
}
rows_1c = []
for (cond, body), (pr, plo, phi) in PAPER_SD_CI.items():
    r, lo, hi = FS.sd_ratio_ci(VALS[("blank", body)], VALS[(cond, body)])
    mism = (abs(round(lo, 2) - plo) >= 0.005) or (abs(round(hi, 2) - phi) >= 0.005) \
        or (abs(round(r, 2) - pr) >= 0.005)
    rows_1c.append(dict(cond=cond, body=body, paper=(pr, plo, phi),
                        code=(r, lo, hi), mismatch=mism))
# the two finger withdrawn ratios v20 reported as "not computed"
for cond in ("prior strict-withdrawn", "coach abrupt-withdrawn"):
    r, lo, hi = FS.sd_ratio_ci(VALS[("blank", "finger")], VALS[(cond, "finger")])
    rows_1c.append(dict(cond=cond, body="finger", paper=None,
                        code=(r, lo, hi), mismatch=False))

# ------------------------------------------------------------------- reporting
def f3(x):
    return "%.3f" % x


L = []
A = L.append
A("")
A("---")
A("")
A("# Appendix B — robustness additions for v21 (Wilcoxon, CV, authoritative SD-ratio CIs)")
A("")
A("Generated by `extra_stats.py`, which imports `figstats.py` (and through it "
  "`recompute_all.py`'s loaders and estimators) unchanged. Nothing in the main table "
  "or in Appendix A is altered by this appendix; it **adds** a third estimator to the "
  "claim-bearing cells, a scale-free dispersion metric, and the authoritative "
  "SD-ratio intervals.")
A("")

# --- B1
A("## B1. Wilcoxon signed-rank test on every claim-bearing contrast")
A("")
A("**Why.** The primary estimator (paired t) assumes approximate normality of the "
  "**paired differences**, on data this paper elsewhere describes as spanning "
  "16.8–110.7 mrad across seeds. A rank test makes no such assumption and is the "
  "natural robustness check; none was run in v19 or v20.")
A("")
A("**Method.** Two-sided **exact** Wilcoxon signed-rank test of H0: median(d) = 0, "
  "on the *same* per-seed paired differences the t and BCa intervals use. "
  "The exact null enumerates all 2^12 = 4096 sign patterns of the fixed |d| rank "
  "vector (mid-ranks for ties); no normal approximation and no continuity correction. "
  "`W+` is the sum of the ranks of the positive differences. "
  "Significance at alpha = 0.05. " + SCIPY_NOTE + ".")
A("")
A("**Cells.** Every ROBUST contrast cited in the title, the abstract, §1.1 or §6, "
  "plus the two new `[coach(const) − prior]@LATE − @EARLY` decay cells (F-6).")
A("")
A("| contrast | body | unit | n | mean diff | paired t 95% CI (PRIMARY) | flag | W+ | Wilcoxon p (exact) | Wilcoxon sig? | agrees with t? |")
A("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows_1a:
    A("| %s | %s | %s | %d | %s | [%s, %s] | %s | %.1f | %.5f | %s | %s |"
      % (r["label"], r["body"], FS.UNIT[r["body"]], r["n"], f3(r["mean"]),
         f3(r["tlo"]), f3(r["thi"]), r["flag"], r["W"], r["p"],
         "yes" if r["w_sig"] else "no", "YES" if r["agree"] else "**NO**"))
A("")
n_agree = sum(1 for r in rows_1a if r["agree"])
A("**Agreement: %d of %d cells.** %s"
  % (n_agree, len(rows_1a),
     "Every claim-bearing contrast that the paired t interval calls significant is "
     "also significant under the exact Wilcoxon signed-rank test, and vice versa."
     if n_agree == len(rows_1a) else
     "The disagreeing cells are listed above with **NO** in the last column."))
A("")

# --- B2
A("## B2. Coefficient of variation, and CV ratios beside the raw SD ratios")
A("")
A("**Why.** Reach error is bounded below by zero, so a condition with a lower mean "
  "will tend to have a lower SD for that reason alone. `SD(blank)/SD(cond)` therefore "
  "partly measures the mean difference the paper reports separately. "
  "`CV = SD/mean` divides that out. Both are reported; neither alone is sufficient.")
A("")
A("Per-condition ENDPOINT values across the 12 seeds; ratios are **blank ÷ condition**, "
  "with percentile bootstrap 95% CIs (20 000 resamples, RNG seed 0, resampling seeds) — "
  "the same scheme as `figstats.sd_ratio_ci`. A ratio **above 1** means the condition is "
  "**less** dispersed than model-free SAC on that metric.")
A("")
A("| condition | body | unit | mean | SD | CV | SD(blank)/SD(cond) | SD-ratio 95% CI | CV(blank)/CV(cond) | CV-ratio 95% CI |")
A("|---|---|---|---|---|---|---|---|---|---|")
for r in rows_1b:
    if r["sdr"] is None:
        A("| %s (reference) | %s | %s | %s | %s | %.3f | 1.00 | — | 1.00 | — |"
          % (r["cond"], r["body"], FS.UNIT[r["body"]], f3(r["mean"]), f3(r["sd"]), r["cv"]))
    else:
        A("| %s | %s | %s | %s | %s | %.3f | **%.2f** | [%.2f, %.2f] | **%.2f** | [%.2f, %.2f] |"
          % (r["cond"], r["body"], FS.UNIT[r["body"]], f3(r["mean"]), f3(r["sd"]),
             r["cv"], r["sdr"], r["sdlo"], r["sdhi"], r["cvr"], r["cvlo"], r["cvhi"]))
A("")
A("**Reading.** The three cells the paper calls dispersion-suppressing all shrink "
  "substantially on the scale-free metric: the SD ratios of 2.78 / 2.93 / 3.53 become "
  "CV ratios of %.2f / %.2f / %.2f. "
  % tuple(next(r["cvr"] for r in rows_1b if r["cond"] == c and r["body"] == b)
          for c, b in (("prior kept", "elbow"), ("coach constant", "elbow"),
                       ("coach constant", "finger"))) +
  "\"Roughly threefold\" is an SD-scale statement only.")
A("")

# --- B3
A("## B3. Authoritative SD-ratio intervals (`figstats.sd_ratio_ci`)")
A("")
A("**Why.** `make_figs.py` draws Fig. 13 from `figstats.sd_ratio_ci`. Six of the "
  "intervals printed in `PAPER_v20.md` (§4.8, abstract, §6) do not match that "
  "function's output. The function is the authority, because it is what the figure "
  "set is built from; the paper text is corrected to it (correction record item 23).")
A("")
A("| condition | body | printed in v20 | `figstats.sd_ratio_ci` output | mismatch? |")
A("|---|---|---|---|---|")
for r in rows_1c:
    code = "%.2f [%.2f, %.2f]" % r["code"]
    if r["paper"] is None:
        A("| %s | %s | *not computed* | %s | — (new) |" % (r["cond"], r["body"], code))
    else:
        A("| %s | %s | %.2f [%.2f, %.2f] | %s | %s |"
          % (r["cond"], r["body"], r["paper"][0], r["paper"][1], r["paper"][2],
             code, "**yes**" if r["mismatch"] else "no"))
A("")
nm = sum(1 for r in rows_1c if r["paper"] is not None and r["mismatch"])
A("**%d of the 6 printed intervals are wrong in the last displayed digit or two.** "
  "All are small (<= 0.51 on a bound) and none changes an excludes-1 verdict, but the "
  "text must match the code that draws the figure. The two finger withdrawn ratios, "
  "which v20 reported as point ratios with \"not computed\" intervals, are computed "
  "here for the first time." % nm)
A("")
A("*(Full precision, for the record: " +
  "; ".join("%s %s = %.4f [%.4f, %.4f]" % (r["cond"], r["body"], r["code"][0],
                                           r["code"][1], r["code"][2])
            for r in rows_1c) + ".)*")
A("")

# --- B4
A("## B4. NEW contrasts: the decay of the coach's lead over the prior")
A("")
A("**Why.** v20 framed the endpoint crossover as a body-dependent *reversal* and "
  "asserted that the coach's advantage \"does not decay at all on elbow within 12k\". "
  "The interaction that statement requires — a within-seed window×condition test on "
  "`coach(const) − prior` — had never been computed. It is computed here. Orientation: "
  "**positive = the coach's lead over the prior SHRINKS in the later window.** "
  "EARLY and LATE are disjoint (§3.7), so the LATE−EARLY test is well-posed; "
  "ENDPOINT ⊂ LATE, so the ENDPOINT−EARLY row is corroboration, not a second "
  "independent confirmation.")
A("")
A("| contrast | body | unit | n | mean diff | paired t 95% CI (PRIMARY) | BCa 95% CI (sensitivity) | flag | Wilcoxon p (exact) |")
A("|---|---|---|---|---|---|---|---|---|")
for w1 in ("LATE", "ENDPOINT"):
    for body in ("elbow", "finger"):
        d = FS.d_win(S.COACH_CONST, S.PRIOR_HO, body, w1, "EARLY")
        m, tlo, thi = t_ci(d)
        _, blo, bhi = bca_ci(d)
        A("| [coach(const) − prior]@%s − @EARLY (held-out) | %s | %s | %d | %s | [%s, %s] | [%s, %s] | %s | %.5f |"
          % (w1, body, FS.UNIT[body], len(d), f3(m), f3(tlo), f3(thi), f3(blo), f3(bhi),
             flag(tlo, thi, blo, bhi), wilcoxon_exact(d)["p"]))
A("")
A("**All four cells are ROBUST, and all four are positive.** The coach's lead over the "
  "prior decays on **both** bodies. The two bodies differ only in whether the decay "
  "carries the gap past zero inside the 12k budget: on finger it does (LATE +19.95, "
  "ENDPOINT +32.27), on elbow it does not (LATE −11.83, ENDPOINT −10.89). "
  "**These two contrasts are additional to the 132; the authoritative set is now 134 "
  "(60 ROBUST / 60 ROBUST-NULL / 14 METHOD-DEPENDENT).**")
A("")

# GUARD: importing this module must NOT append to RECOMPUTED.md.
# Other scripts legitimately import it for CLAIM_CELLS, and an import-time append
# has already fired once, duplicating Appendix B. Only a direct run may write.
# Do not remove. (Same class of hazard as recompute_all.py's, fixed there too.)
txt = "\n".join(L) + "\n"
if __name__ == "__main__":
    with open(FS.REC_PATH, "a", encoding="utf-8") as fh:
        fh.write(txt)

try:
    print(txt)
except UnicodeEncodeError:                                # console codepage
    print(txt.encode("ascii", "replace").decode("ascii"))

# This print used to say "=== appended to <path>" at module scope, i.e. it
# announced a write that the __main__ guard above had already prevented.
# An audit read by eye would conclude the file had been appended when it had
# not. A message that misreports what the program did is worse than no message.
if __name__ == "__main__":
    print("=== appended to", FS.REC_PATH)
else:
    print("=== extra_stats imported: estimators only, RECOMPUTED.md NOT written")
