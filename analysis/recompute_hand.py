r"""recompute_hand.py -- Appendix G of RECOMPUTED.md.

Five analyses the r12 draft either never ran, ran at the wrong K, or asserted
without a test:

  G1  **myoHand -- the third MyoSuite body.**  `results_hand/` holds twelve
      complete runs ({blank,prior} x seed0..5), thirty checkpoints each, at
      60000 steps, produced by `launch_hand.ps1` (`WM_BODY=myohand`,
      `prior_hand.pt`, `--utd 2 --ntargets 8`, the same `run_reach.py`).  The
      body was disclosed in `PAPER_v21.md:747` and the sentence was DELETED,
      not corrected, in the TMLR rewrite.  Appendix F4 -- the corrected
      completeness register written specifically to close the
      run-but-never-scored species -- does not list it either.  Scored here.

  G2  **K = 44: Sec. 3.6's exclusion criterion applied in full.**  F1 pruned
      the 14 duplicated LATE cells and stopped, calling the resulting K = 58
      "the family Sec. 3.6's own exclusion rule defines".  It is not.  The
      same sentence of Sec. 3.6 that declares ENDPOINT (subset) LATE also
      declares that MID overlaps both EARLY and LATE, and MID was never
      pruned.  Pruning it as well gives **K = 44**, on which the title's
      reversal is rank 9 and survives.

  G3  **The Pitman-Morgan test clause (v) needed and never ran.**  Sec. 4.6
      concludes from two separate dispersion verdicts (finger `randprior`
      3.33 excludes 1, finger train `prior` 1.54 does not) that the finger
      contraction is not about what the model knows -- which is exactly the
      inference-from-two-verdicts Sec. 1.2 forbids.  The direct test is a
      paired variance-ratio test (Pitman-Morgan) plus a paired bootstrap of
      the ratio of the two SD ratios, over the twelve shared seeds.

  G4  **Clause (v)'s count is scale-dependent.**  Sec. 4.6 justifies printing
      the CV ratio beside every SD ratio -- "because reach error is bounded
      below by zero, SD tracks the mean" -- and item 58 applies that same
      reasoning to the decay interaction.  It was never applied to Sec. 4.6's
      own headline count, which is three-of-four on SD and two-of-four on the
      scale-free statistic.

  G5  **F4b: the completeness register, corrected a second time.**  F4 was
      itself the correction to E5's negative register, and F4 is also wrong:
      it does not list `results_hand/`.  F4 is NOT edited; F4b is appended.

Append-only, exactly like `movement_index_holm72.py`, `sweep_selective.py` and
`recompute_addendum.py`: this script never opens RECOMPUTED.md in write mode
and refuses to append if its marker heading is already present.

NOTE: this module deliberately does NOT import `extra_stats`, which has no
`__main__` guard and appends Appendix B to RECOMPUTED.md at import time.  The
exact Wilcoxon routine is reimplemented below.

Usage:  python recompute_hand.py            # print only
        python recompute_hand.py --append   # print and append
"""

import itertools
import json
import math
import os
import sys

import numpy as np

import movement_index_holm72 as MI

ROOT = os.path.dirname(os.path.abspath(__file__))
REC_PATH = os.path.join(ROOT, "RECOMPUTED.md")
HAND_DIR = os.path.join(ROOT, "results_hand")
MARKER = ("# Appendix G — the third body (myoHand), Sec. 3.6's criterion applied in full "
          "(K = 44), and the dispersion test clause (v) needed")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ------------------------------------------------------------------ estimators
def t_crit(df, alpha=0.05):
    """Two-sided critical t, by bisection on MI.t_sf (which returns two-sided p)."""
    lo, hi = 0.0, 200.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if MI.t_sf(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired(d):
    """(n, mean, lo, hi, p) with the CORRECT df = n-1 multiplier.

    MI.paired hardcodes t(.975, 11) = 2.20099 because every contrast in the
    main study is n = 12.  The hand study is n = 6, so that multiplier is
    wrong here by a factor of 2.570582/2.200985 = 1.1679.
    """
    d = np.asarray(d, float)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1)) / math.sqrt(n)
    half = t_crit(n - 1) * se
    return n, m, m - half, m + half, MI.t_sf(m / se, n - 1)


def wilcoxon_exact(d):
    """Two-sided exact Wilcoxon signed-rank p (same routine as extra_stats)."""
    d = np.asarray(d, float)
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return float("nan")
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, float)
    srt = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    Wp = float(ranks[d > 0].sum())
    dist = {}
    for signs in itertools.product((0, 1), repeat=n):
        w = float(np.dot(ranks, signs))
        dist[w] = dist.get(w, 0) + 1
    N = float(2 ** n)
    p_le = sum(c for w, c in dist.items() if w <= Wp) / N
    p_ge = sum(c for w, c in dist.items() if w >= Wp) / N
    return min(1.0, 2.0 * min(p_le, p_ge))


def holm(cells):
    """Step-down Holm at alpha = 0.05 over `cells` (each a dict with 'p')."""
    rs = sorted(cells, key=lambda r: r["p"])
    K = len(rs)
    surv, stop = [], None
    for i, r in enumerate(rs, 1):
        t = 0.05 / (K - i + 1)
        r["_rank"], r["_thr"] = i, t
        if stop is None and r["p"] <= t:
            r["_surv"] = True
            surv.append(r)
        else:
            r["_surv"] = False
            if stop is None:
                stop = (i, r, t)
    return rs, surv, stop


# ================================================================== G1
HAND_SEEDS = list(range(6))


def hand_load(cond, seed):
    with open(os.path.join(HAND_DIR, "%s_seed%d.json" % (cond, seed))) as f:
        j = json.load(f)
    return {r["step"]: r["eval_dist"] * 1000.0 for r in j}   # metres -> mm


def g1():
    B = {s: hand_load("blank", s) for s in HAND_SEEDS}
    P = {s: hand_load("prior", s) for s in HAND_SEEDS}
    steps = sorted(B[0])
    ok = all(sorted(D[s]) == steps for D in (B, P) for s in HAND_SEEDS)

    def cell(ws):
        b = np.array([np.mean([B[s][w] for w in ws]) for s in HAND_SEEDS])
        p = np.array([np.mean([P[s][w] for w in ws]) for s in HAND_SEEDS])
        d = p - b
        n, m, lo, hi, pv = paired(d)
        return dict(blank=float(b.mean()), prior=float(p.mean()), mean=m, lo=lo, hi=hi,
                    p=pv, w=wilcoxon_exact(d), pos=int((d > 0).sum()), n=n)

    windows = [("EARLY {2,4,6k}", [2000, 4000, 6000]),
               ("MID {6,8,10k}", [6000, 8000, 10000]),
               ("LATE {10,12k}", [10000, 12000]),
               ("ENDPOINT-of-main {12k}", [12000]),
               ("ENDPOINT-of-run {60k}", [60000])]
    rows = [(nm, cell(ws)) for nm, ws in windows]
    per_ckpt = [(w, cell([w])) for w in steps]
    sig = [w for w, c in per_ckpt if c["p"] < 0.05]
    ck = [dict(label="checkpoint %dk" % (w // 1000), p=c["p"]) for w, c in per_ckpt]
    _, ck_surv, ck_stop = holm(ck)
    return dict(steps=steps, aligned=ok, rows=rows, per_ckpt=per_ckpt, sig=sig,
                ck_surv=ck_surv, ck_stop=ck_stop,
                lvl2k=cell([2000]), lvl12k=cell([12000]), lvl60k=cell([60000]))


# ================================================================== G2
def g2():
    cells = MI.build_cells()
    base = holm([dict(c) for c in cells])
    late_only = holm([dict(c) for c in cells if c["w"] != "LATE"])
    full = holm([dict(c) for c in cells if c["w"] not in ("LATE", "MID")])
    title = ("coach(const) − prior", "ENDPOINT", "finger")

    def find(rs):
        return [r for r in rs if (r["label"], r["w"], r["body"]) == title][0]

    # which of the literal-72 ranks 1..12 leave under the full prune
    leaving = [(r["_rank"], r) for r in base[0][:12] if r["w"] in ("LATE", "MID")]
    return dict(cells=cells, base=base, late_only=late_only, full=full,
                t_base=find(base[0]), t_late=find(late_only[0]), t_full=find(full[0]),
                leaving=leaving)


# ================================================================== G3
def g3():
    out = {}
    for body in ("elbow", "finger"):
        Bl = MI.ser(MI.BLANK_TR, body, "ENDPOINT")
        Pr = MI.ser(MI.PRIOR_TR, body, "ENDPOINT")
        Rp = MI.ser(MI.RANDPRIOR_TR, body, "ENDPOINT")
        seeds = sorted(set(Bl) & set(Pr) & set(Rp))
        b = np.array([Bl[s] for s in seeds])
        p = np.array([Pr[s] for s in seeds])
        r = np.array([Rp[s] for s in seeds])
        n = len(seeds)
        sdb, sdp, sdr = b.std(ddof=1), p.std(ddof=1), r.std(ddof=1)
        # Pitman-Morgan paired variance-ratio test on (prior, randprior)
        u, v = p + r, p - r
        rho = float(np.corrcoef(u, v)[0, 1])
        t = rho * math.sqrt(n - 2) / math.sqrt(1.0 - rho * rho)
        pm_p = MI.t_sf(t, n - 2)
        # paired bootstrap of the ratio-of-ratios, resampling seeds jointly
        rng = np.random.default_rng(0)          # fresh generator per body
        idx = rng.integers(0, n, size=(20000, n))
        bs = p[idx].std(axis=1, ddof=1) / r[idx].std(axis=1, ddof=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[body] = dict(n=n, sd_blank=float(sdb), sd_prior=float(sdp), sd_rand=float(sdr),
                         ratio_prior=float(sdb / sdp), ratio_rand=float(sdb / sdr),
                         rr=float(sdp / sdr), rho=rho, t=t, pm_p=pm_p,
                         lo=float(lo), hi=float(hi))
    return out


# ================================================================== G4
KEPT = [("prior kept", "elbow", MI.PRIOR_HO, MI.BLANK_HO),
        ("coach constant", "elbow", MI.COACH_CONST, MI.BLANK_HO),
        ("prior kept", "finger", MI.PRIOR_HO, MI.BLANK_HO),
        ("coach constant", "finger", MI.COACH_CONST, MI.BLANK_HO)]


def _ratio_ci(blank, cond, nboot=20000, seed=0):
    """figstats.sd_ratio_ci's scheme, reimplemented (independent resampling)."""
    rng = np.random.default_rng(seed)
    i1 = rng.integers(0, len(blank), size=(nboot, len(blank)))
    i2 = rng.integers(0, len(cond), size=(nboot, len(cond)))
    sd = blank[i1].std(axis=1, ddof=1) / cond[i2].std(axis=1, ddof=1)
    cv = ((blank[i1].std(axis=1, ddof=1) / blank[i1].mean(axis=1))
          / (cond[i2].std(axis=1, ddof=1) / cond[i2].mean(axis=1)))
    return (np.percentile(sd, [2.5, 97.5]), np.percentile(cv, [2.5, 97.5]))


def g4():
    rows = []
    for name, body, spec, blankspec in KEPT:
        bl = MI.ser(blankspec, body, "ENDPOINT")
        cd = MI.ser(spec, body, "ENDPOINT")
        seeds = sorted(set(bl) & set(cd))
        b = np.array([bl[s] for s in seeds])
        c = np.array([cd[s] for s in seeds])
        sdr = b.std(ddof=1) / c.std(ddof=1)
        cvr = (b.std(ddof=1) / b.mean()) / (c.std(ddof=1) / c.mean())
        (slo, shi), (clo, chi) = _ratio_ci(b, c)
        rows.append(dict(name=name, body=body, sd_ratio=float(sdr), sd_lo=slo, sd_hi=shi,
                         cv_ratio=float(cvr), cv_lo=clo, cv_hi=chi,
                         sd_excl=bool(slo > 1.0 or shi < 1.0),
                         cv_excl=bool(clo > 1.0 or chi < 1.0)))
    return rows


# ================================================================== G5
SCORED = {
    "results_2x2_elbow": "Sec. 4.4 / Sec. 4.6 train block (main study, train targets)",
    "results_2x2_finger": "Sec. 4.4 / Sec. 4.6 train block (main study, train targets)",
    "results_heldout_elbow": "Sec. 4.1–4.6 (main study, held-out targets)",
    "results_heldout_finger": "Sec. 4.1–4.6 (main study, held-out targets)",
    "results_noanneal_elbow": "Sec. 4.2–4.6 (`coach(const)`)",
    "results_noanneal_finger": "Sec. 4.2–4.6 (`coach(const)`)",
    "results_cleancoach_elbow": "Sec. 4.1–4.6 (`coach(anneal)`, `priorcoach(anneal)`)",
    "results_cleancoach_finger": "Sec. 4.1–4.6 (`coach(anneal)`, `priorcoach(anneal)`)",
    "results_cleancoach_train_elbow": "Appendix C (decontamination, train targets)",
    "results_cleancoach_train_finger": "Appendix C (decontamination, train targets)",
    "results_abruptcoach_elbow": "Sec. 4.5 (matched coach withdrawal)",
    "results_abruptcoach_finger": "Sec. 4.5 (matched coach withdrawal)",
    "results_pcconst_elbow": "Sec. 4.7 (`priorcoach(const)`)",
    "results_pcconst_finger": "Sec. 4.7 (`priorcoach(const)`)",
    "results_prioroff_elbow": "Sec. 4.5 (soft prior withdrawal)",
    "results_prioroff_finger": "Sec. 4.5 (soft prior withdrawal)",
    "results_priorpurge_elbow": "Sec. 4.5 (strict prior withdrawal)",
    "results_priorpurge_finger": "Sec. 4.5 (strict prior withdrawal)",
    "results_randprior_elbow": "Sec. 4.4 (adversarial control)",
    "results_randprior_finger": "Sec. 4.4 (adversarial control)",
    "results_plateau2_elbow": "Sec. 4.8 (long-budget continuation)",
    "results_plateau2_finger": "Sec. 4.8 (long-budget continuation)",
    "results_plateau_elbow": "E5 — duplicate of `results_plateau2_elbow`",
    "results_plateau_finger": "E5 — duplicate of `results_plateau2_finger`",
    "results_myo": "**Sec. 4.10 / F3** (`warm`, `colearn`, `imag`; found by the F4 register)",
    "results_hand": "**Sec. 4.11 / G1** (myoHand — found by *this* register)",
}


def enumerate_results():
    out = []
    for name in sorted(os.listdir(ROOT)):
        if not name.startswith("results") or not os.path.isdir(os.path.join(ROOT, name)):
            continue
        out.append((name, SCORED.get(name, "— not scored (pre-study pilot)")))
    return out


# ------------------------------------------------------------------ rendering
def render():
    L = []
    A = L.append
    A("")
    A("---")
    A("")
    A(MARKER)
    A("")
    A("Generated by `recompute_hand.py` (append-only; it refuses to run twice). It imports "
      "`movement_index_holm72.py` for the cell definitions and loaders and reimplements the "
      "exact Wilcoxon routine locally, because `extra_stats.py` has no `__main__` guard and "
      "appends Appendix B to this file at import time.")
    A("")

    # ---------------------------------------------------------------- G1
    d = g1()
    A("## G1. myoHand — the third MyoSuite body, run in full and never reported")
    A("")
    A("`results_hand/` holds **twelve complete runs** — `{blank,prior}_seed0..5`, **30 "
      "checkpoints each**, 60 000 environment steps — produced by `launch_hand.ps1`, which "
      "sets `WM_BODY=myohand`, loads `prior_hand.pt`, and calls the same `run_reach.py` with "
      "`--utd 2 --ntargets 8 --out results_hand`. No `--heldout` flag is passed, so these are "
      "**train-target** scores on the 8 training targets. Distances are metres in the archive "
      "and are reported here in **mm**, as for myoFinger.")
    A("")
    A("**This is not a newly discovered directory. It was disclosed in prose and the "
      "disclosure was deleted.** `PAPER_v21.md:747` reads: *\"On the harder MyoSuite "
      "**myoHand** (39 muscles), vanilla SAC at a ~60k-step CPU budget does not reach a clean "
      "solve, so 'a harder body amplifies a prior's advantage' is **underpowered, not "
      "refuted**.\"* **In the r12 draft of `PAPER_TMLR.md`, and in this file up to the end of "
      "Appendix F, `grep -c 'myoHand\\|myohand\\|results_hand'` returned 0.** The rewrite "
      "**deleted** the disclosure rather than correcting it, and the runs were never scored "
      "at any point. (The r13 draft carries Sec. 4.11, which is where the scoring below now "
      "lives.)")
    A("")
    A("**And the completeness register missed it twice.** E5 closed the run-but-never-scored "
      "species with a negative register; F4 was written because that register was false; F4 "
      "enumerates `results_myo/` and does not list `results_hand/` either. See G5.")
    A("")
    A("All twelve runs carry the identical 30-checkpoint step grid (2k … 60k): **%s**."
      % ("verified" if d["aligned"] else "NOT ALIGNED"))
    A("")
    A("**Paired over the 6 shared seeds, `prior − blank` on train targets (positive = the "
      "prior is worse). n = 6, so the t multiplier is t(.975, 5) = 2.570582, not the "
      "t(.975, 11) = 2.200985 the rest of this file uses.**")
    A("")
    A("| window | blank (mm) | prior (mm) | `prior − blank` | paired t 95% CI | t p | exact Wilcoxon p | seeds on the sign |")
    A("|---|---|---|---|---|---|---|---|")
    for nm, c in d["rows"]:
        A("| **%s** | %.2f | %.2f | **%+.2f** | [%+.2f, %+.2f] | %.5f | %.5f | %d/6 |"
          % (nm, c["blank"], c["prior"], c["mean"], c["lo"], c["hi"], c["p"], c["w"],
             max(c["pos"], 6 - c["pos"])))
    A("")
    A("Levels at the three checkpoints the windows above single out: **blank %.1f → %.1f → "
      "%.1f** and **prior %.1f → %.1f → %.1f** at 2k → 12k → 60k."
      % (d["lvl2k"]["blank"], d["lvl12k"]["blank"], d["lvl60k"]["blank"],
         d["lvl2k"]["prior"], d["lvl12k"]["prior"], d["lvl60k"]["prior"]))
    A("")
    A("### G1a. All 30 checkpoints")
    A("")
    A("| step | blank (mm) | prior (mm) | `prior − blank` | paired t 95% CI | p |")
    A("|---|---|---|---|---|---|")
    for w, c in d["per_ckpt"]:
        star = " **\\***" if c["p"] < 0.05 else ""
        A("| %dk | %.2f | %.2f | %+.2f | [%+.2f, %+.2f] | %.5f%s |"
          % (w // 1000, c["blank"], c["prior"], c["mean"], c["lo"], c["hi"], c["p"], star))
    A("")
    A("**%d of the 30 checkpoints reach p < 0.05** (%s), against a chance expectation of "
      "30 × 0.05 = **1.5**."
      % (len(d["sig"]), ", ".join("%dk" % (w // 1000) for w in d["sig"])))
    A("")
    i, row, thr = d["ck_stop"]
    A("**Holm–Bonferroni over the 30 checkpoints kills the 12k cell immediately.** Ordered by "
      "ascending p, the step-down stops at **rank %d** (%s, p = %.5f against 0.05/30 = "
      "%.5f), so **nothing survives** and the 12k cell — whose own p is %.5f and whose most "
      "generous possible threshold in this family is 0.05/29 = %.5f — never gets near one."
      % (i, row["label"], row["p"], thr, d["per_ckpt"][5][1]["p"], 0.05 / 29))
    A("")
    A("### G1b. The honest verdict")
    A("")
    A("**Neither arm learns myoHand.** `blank` runs %.1f mm at 2k and %.1f mm at 60k; `prior` "
      "runs %.1f and %.1f. Thirty checkpoints of a 39-muscle body produce a flat series "
      "around 100–120 mm in both arms. The one cell that clears p < 0.05 with all six seeds "
      "on the same sign — 12k, +12.49 [+0.99, +23.99], Wilcoxon 0.03125 — is one of two "
      "cells at the chance expectation of 1.5, dies under any correction over the grid it "
      "sits in, has no neighbour supporting it (10k is +8.34 with a CI containing zero, 14k "
      "is **−10.62**), and points the **wrong way for the forward-model route** in any case."
      % (d["lvl2k"]["blank"], d["lvl60k"]["blank"], d["lvl2k"]["prior"], d["lvl60k"]["prior"]))
    A("")
    A("**The body is uninformative about the forward-model route; it does not refute it.** "
      "What the omission is, precisely, is a **scope defect and not a result**: Sec. 6's "
      "bullet \"Two bodies, one of which (myoElbow) is 1-DoF\" is false as a description of "
      "what was run. Three MyoSuite bodies were run.")
    A("")

    # ---------------------------------------------------------------- G2
    g = g2()
    base_rs, base_surv, base_stop = g["base"]
    lo_rs, lo_surv, lo_stop = g["late_only"]
    fu_rs, fu_surv, fu_stop = g["full"]
    A("## G2. Sec. 3.6's exclusion criterion applied in full: K = 44, not 58")
    A("")
    A("F1 pruned the 14 duplicated **LATE** cells from the literal 72, obtained K = 58, and "
      "labelled the result *\"the family Sec. 3.6's own exclusion rule defines\"*. **That "
      "label is wrong.** The sentence of Sec. 3.6 that F1 invokes reads, in one breath: "
      "*\"EARLY and LATE are disjoint … **MID overlaps both**, and ENDPOINT ⊂ LATE, so a "
      "late-window result and an endpoint result are one confirmation and not two.\"* MID "
      "shares its 6k checkpoint with EARLY and its 10k checkpoint with LATE. By the very "
      "criterion F1 applies — *sub-divisions of windows already in the family, "
      "double-counting the same seeds* — MID is at least as excludable as ENDPOINT, and F1 "
      "never pruned it.")
    A("")
    A("Pruning MID as well leaves **K = %d**: 72 − 14 LATE − 14 MID." % len(fu_rs))
    A("")
    A("| family | what it is | K | t-significant | Holm survivors | rank of the title's cell | verdict |")
    A("|---|---|---|---|---|---|---|")
    A("| literal | unpruned enumeration | %d | %d | %d | %d | **fails** (p = %.8f vs %.8f) |"
      % (len(base_rs), sum(1 for r in base_rs if r["sig"]), len(base_surv),
         g["t_base"]["_rank"], g["t_base"]["p"], g["t_base"]["_thr"]))
    A("| LATE-pruned | **partial** prune (F1) | %d | %d | %d | %d | **survives** (p = %.8f vs %.8f) |"
      % (len(lo_rs), sum(1 for r in lo_rs if r["sig"]), len(lo_surv),
         g["t_late"]["_rank"], g["t_late"]["p"], g["t_late"]["_thr"]))
    A("| **MID+LATE-pruned** | **what Sec. 3.6's criterion derives** | **%d** | %d | **%d** | **%d** | **survives** (p = %.8f vs 0.05/%d = %.8f) |"
      % (len(fu_rs), sum(1 for r in fu_rs if r["sig"]), len(fu_surv),
         g["t_full"]["_rank"], g["t_full"]["p"],
         len(fu_rs) - g["t_full"]["_rank"] + 1, g["t_full"]["_thr"]))
    A("")
    A("**Four of the literal family's ranks 1–12 leave**, which is what moves the reversal "
      "from rank 12 (under the partial prune) to rank %d:" % g["t_full"]["_rank"])
    A("")
    A("| literal rank | contrast | window | body | p |")
    A("|---|---|---|---|---|")
    for rk, r in g["leaving"]:
        A("| %d | %s | **%s** | %s | %.7f |" % (rk, r["label"], r["w"], r["body"], r["p"]))
    A("")
    A("### G2a. Holm over the MID+LATE-pruned family (K = %d)" % len(fu_rs))
    A("")
    A("| # | contrast | window | body | uncorrected p | Holm threshold | survives |")
    A("|---|---|---|---|---|---|---|")
    for r in fu_rs[:16]:
        mark = "**yes**" if r["_surv"] else ("no — **procedure stops here**"
                                             if r["_rank"] == fu_stop[0] else "no")
        star = "**" if (r["label"], r["w"], r["body"]) == ("coach(const) − prior", "ENDPOINT", "finger") else ""
        A("| %s%d%s | %s%s%s | %s | %s | %s%.7f%s | %.7f | %s |"
          % (star, r["_rank"], star, star, r["label"], star, r["w"], r["body"],
             star, r["p"], star, r["_thr"], mark))
    A("")
    A("Ranks %d–%d are the remaining cells, all failing." % (17, len(fu_rs)))
    A("")
    A("**All eight cells above the reversal clear their own thresholds**, so the step-down "
      "reaches it, and **it survives**: p = %.8f against 0.05/%d = %.8f. Twelve cells "
      "survive; the step-down then stops at rank %d on `%s` %s %s (%.7f against %.7f)."
      % (g["t_full"]["p"], len(fu_rs) - g["t_full"]["_rank"] + 1, g["t_full"]["_thr"],
         fu_stop[0], fu_stop[1]["label"], fu_stop[1]["w"], fu_stop[1]["body"],
         fu_stop[1]["p"], fu_stop[2]))
    A("")
    A("**This costs the paper nothing and is not a rescue.** The reversal already survived "
      "the partial prune at K = 58; K = 44 is the same verdict with a wider margin. What is "
      "corrected is a **labelling defect**: the paper said its own criterion picks out K = 58 "
      "when that criterion picks out K = 44. The three families are now named for what they "
      "are — **K = 44 is what Sec. 3.6's criterion derives, K = 58 is the partial prune, "
      "K = 72 is the unpruned literal set** — and the literal 72 remains the strictest "
      "alternative and the only family on which the reversal fails.")
    A("")

    # ---------------------------------------------------------------- G3
    pm = g3()
    A("## G3. The dispersion difference clause (v) asserted and never tested "
      "(Pitman–Morgan + paired bootstrap)")
    A("")
    A("Sec. 4.6 and clause (v) conclude, in bold, that *\"the finger dispersion result is not "
      "attributable to what the model or the teacher knows\"* — from **two separate "
      "verdicts**: finger `randprior` 3.33 [1.66, 7.59] excludes 1 and finger train `prior` "
      "1.54 [0.78, 2.84] does not. Sec. 1.2 lists *\"a direct paired within-axis contrast "
      "whenever two routes are compared, avoiding inference from two separate significance "
      "verdicts\"* as one of the paper's three method commitments, and 3.33-versus-1.54 was "
      "never tested. It is tested here two ways, on the twelve shared train-target seeds.")
    A("")
    A("**Pitman–Morgan** is the paired analogue of an F test for equal variances: with "
      "u = x + y and v = x − y, H0: var(x) = var(y) is equivalent to H0: corr(u, v) = 0, "
      "tested with t = r√(n−2)/√(1−r²) on n − 2 df. **Ratio-of-ratios** is "
      "[SD(blank)/SD(randprior)] ÷ [SD(blank)/SD(prior)] = SD(prior)/SD(randprior); its "
      "interval is a percentile bootstrap resampling the twelve seeds **jointly** (paired), "
      "20 000 resamples, a fresh `default_rng(0)` per body.")
    A("")
    A("| body | SD blank | SD prior | SD randprior | prior SD ratio | randprior SD ratio | ratio-of-ratios | paired bootstrap 95% CI | Pitman–Morgan r | t (df 10) | **p** |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for body in ("finger", "elbow"):
        r = pm[body]
        A("| **%s** | %.2f | %.2f | %.2f | %.2f | %.2f | **%.3f** | [%.3f, %.3f] | %+.4f | %+.4f | **%.5f** |"
          % (body, r["sd_blank"], r["sd_prior"], r["sd_rand"], r["ratio_prior"],
             r["ratio_rand"], r["rr"], r["lo"], r["hi"], r["rho"], r["t"], r["pm_p"]))
    A("")
    f, e = pm["finger"], pm["elbow"]
    A("**The test passes on finger and the paper's conclusion stands — now on a test rather "
      "than on two verdicts.** The randprior arm's dispersion is significantly tighter than "
      "the trained prior's on the identical train targets: ratio-of-ratios **%.3f "
      "[%.3f, %.3f]**, excluding 1, Pitman–Morgan **p = %.5f**. So on myoFinger an "
      "*uninformative* frozen model, in an arm with no teacher at all, contracts seed-to-seed "
      "spread **significantly more** than the babble-trained model does."
      % (f["rr"], f["lo"], f["hi"], f["pm_p"]))
    A("")
    A("**The elbow twin runs the other way and is also significant**, which is the honest "
      "half: ratio-of-ratios **%.3f [%.3f, %.3f]**, Pitman–Morgan **p = %.5f**. On elbow the "
      "trained prior contracts dispersion significantly more than the random one. The two "
      "bodies therefore disagree about content-specificity under a direct test, exactly as "
      "they did under the two-verdict reading — but the disagreement is now a measured "
      "difference on both bodies rather than a significant cell beside a null one."
      % (e["rr"], e["lo"], e["hi"], e["pm_p"]))
    A("")
    A("Both cells are **post hoc** and neither is entered in any Holm family; they are "
      "variance-ratio tests, not paired-difference contrasts, and Sec. 3.6 excludes such "
      "statistics from the correction for that reason.")
    A("")

    # ---------------------------------------------------------------- G4
    rows = g4()
    A("## G4. Clause (v)'s count on the scale-free statistic")
    A("")
    A("Sec. 4.6 prints the CV ratio beside every SD ratio and says why: *\"because reach "
      "error is bounded below by zero, SD tracks the mean.\"* Item 58 applies exactly that "
      "reasoning to the decay interaction and demotes abstract clause (i) for it. **The same "
      "move was never made to Sec. 4.6's own headline count**, which ranges over the four "
      "kept-aid cells (two aids × two bodies).")
    A("")
    A("| kept-aid cell | body | SD ratio | 95% CI | excludes 1? | CV ratio | 95% CI | excludes 1? |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        A("| %s | %s | **%.2f** | [%.2f, %.2f] | %s | **%.2f** | [%.2f, %.2f] | %s |"
          % (r["name"], r["body"], r["sd_ratio"], r["sd_lo"], r["sd_hi"],
             "**yes**" if r["sd_excl"] else "no",
             r["cv_ratio"], r["cv_lo"], r["cv_hi"],
             "**yes**" if r["cv_excl"] else "**no**"))
    A("")
    nsd = sum(1 for r in rows if r["sd_excl"])
    ncv = sum(1 for r in rows if r["cv_excl"])
    A("**%d of the four kept-aid cells exclude 1 on SD and %d of the four exclude it on the "
      "scale-free statistic.** The cell that changes verdict is the **elbow constant coach**: "
      "SD 2.93 [1.44, 7.25] excludes 1, CV 1.62 [0.87, 3.73] includes it. The word \"CV\" "
      "appears in neither the abstract, nor Sec. 1.1 claim 5, nor Sec. 7 of the r12 draft; "
      "all three print the SD count alone. Both counts are now printed at all three sites."
      % (nsd, ncv))
    A("")

    # ---------------------------------------------------------------- G5
    A("## G5. F4b — the completeness register, corrected a second time")
    A("")
    A("E5 closed with *\"nothing else in the repository was found computed-but-unprinted\"*. "
      "**F4 was written because that was false.** F4 is also false: it enumerates "
      "`results_myo/` and does not list `results_hand/` — twelve complete runs of a third "
      "MyoSuite body, thirty checkpoints each, whose existence a previous draft had already "
      "disclosed in prose.")
    A("")
    A("**F4 is not edited in place.** This appendix is append-only and F4 stands as written. "
      "The register, corrected a second time:")
    A("")
    A("| repository artefact | status in E5 | status in F4 | corrected status (F4b) |")
    A("|---|---|---|---|")
    A("| `results_plateau_*` | duplicate | unchanged | unchanged — still a duplicate |")
    A("| seed-3 plateau runs | never launched | unchanged | unchanged — still absent |")
    A("| `results_myo/warm_seed0..4` | not registered | **computed, never scored** | scored in Sec. 4.10 / F3 |")
    A("| `results_myo/colearn_seed0..4` | not registered | **computed, never scored** | scored in Sec. 4.10 / F3 |")
    A("| `results_myo/imag_seed0..4` | not registered | **computed, never scored** | scored in Sec. 4.10 / F3 |")
    A("| `results_myo/{blank,prior}_seed0..4` | not registered | duplicates of `results_2x2_elbow` | unchanged |")
    A("| **`results_hand/{blank,prior}_seed0..5`** | **not registered** | **not registered** | **a third body, 12 runs × 30 checkpoints, run at 60k, disclosed in `PAPER_v21.md:747` and the disclosure deleted — scored in G1 and Sec. 4.11** |")
    A("")
    A("**A completeness register has now been wrong twice, and that fact belongs in the "
      "record more than either individual omission does.** E5 asserted completeness over a "
      "repository it had not enumerated; F4 was the correction, was written specifically to "
      "close the run-but-never-scored species, and repeated the failure on a larger artefact "
      "— a whole body rather than three arms of a body already reported. Neither register was "
      "produced by enumerating `results_*` and checking each directory against the draft. "
      "That enumeration is what F4b is, and it is the reason the species keeps recurring: "
      "each correction was written against the instance that had just been found rather than "
      "against the repository.")
    A("")
    A("**The enumeration itself**, because that is the step both earlier registers skipped. "
      "Every `results*` directory on disk, partitioned into those a section of this paper "
      "scores and those it does not:")
    A("")
    A("| directory | scored by |")
    A("|---|---|")
    for name, where in enumerate_results():
        A("| `%s` | %s |" % (name, where))
    A("")
    A("The unscored rows are pre-study pilots on the toy planar Hill-muscle arms "
      "(`arm1`–`arm4`) and on abandoned task variants (multitask, ablation, dream-length, "
      "stability, smoke tests) that predate the two reported bodies and are not conditions "
      "of this study. They are listed rather than summarised, so that the next register is "
      "checkable against this one.")
    A("")
    return "\n".join(L)


def main():
    text = render()
    print(text)
    if "--append" in sys.argv:
        with open(REC_PATH, encoding="utf-8") as f:
            cur = f.read()
        if MARKER in cur:
            print("\n[recompute_hand] marker already present — refusing to append.",
                  file=sys.stderr)
            return 0
        with open(REC_PATH, "a", encoding="utf-8", newline="") as f:
            f.write(text)
        print("\n[recompute_hand] appended Appendix G to RECOMPUTED.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
