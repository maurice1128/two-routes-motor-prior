r"""recompute_addendum.py -- Appendix F of RECOMPUTED.md.

Five analyses that the r11 draft asserted, implied or got wrong, and that had
no script behind them:

  F1  The paper's OWN Sec. 3.6 exclusion rule, applied to the literal 72.
      Sec. 3.6 excludes the per-checkpoint finger cells "because they are
      sub-divisions of windows already in the family and would double-count
      the same seeds", and separately states that "a late-window result and an
      endpoint result are one confirmation and not two".  ENDPOINT is a
      sub-division of LATE by that same rule, and the literal 72 enters both
      for all seven window-spanning pairs on both bodies.  Pruning the 14 LATE
      cells gives K = 58, under which the title's reversal SURVIVES.  The
      mirror prune (drop ENDPOINT instead) also gives K = 58 and removes the
      title's cell from the family altogether; both are printed.

  F2  The decay interaction under three scalings (absolute, log-error,
      blank-normalised).  Sec. 4.6 already adopts a scale-free CV ratio for
      dispersion "because reach error is bounded below by zero, so SD tracks
      the mean"; the identical reasoning was never applied to the interaction.

  F3  `results_myo/{warm,colearn,imag}_seed0..4` -- the un-frozen and
      co-learned versions of the paper's own model, run on the paper's own
      body at the paper's own budget and never scored.  The fifth recurrence
      of the run-but-never-scored species.

  F4  The correction to E5, whose "nothing else in the repository was found
      computed-but-unprinted" was false: F3 is exactly such a register entry.
      E5 is NOT edited; the corrected register is appended here.

  F5  The six Sec. 4.5 / soft-withdrawal cells Sec. 4.9 declined to enter in
      the literal family, all six computed.  Sec. 4.9 called 0.0078 "the
      largest"; it is the smallest.

Append-only, exactly like `movement_index_holm72.py` and `sweep_selective.py`:
this script never opens RECOMPUTED.md in write mode and refuses to append if
its marker heading is already present.

Usage:  python recompute_addendum.py            # print only
        python recompute_addendum.py --append   # print and append
"""

import hashlib
import json
import math
import os
import sys

import numpy as np

import movement_index_holm72 as MI

ROOT = os.path.dirname(os.path.abspath(__file__))
REC_PATH = os.path.join(ROOT, "RECOMPUTED.md")
MARKER = ("# Appendix F — the paper's own exclusion rule applied to the literal family, "
          "the interaction under three scalings, and the unscored `results_myo` arms")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# --------------------------------------------------------------------- stats
def t_crit(df, q=0.975):
    """Two-sided critical value, by bisection on MI.t_sf (no scipy needed)."""
    lo, hi = 0.0, 500.0
    target = 2.0 * (1.0 - q)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if MI.t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_n(d):
    """(n, mean, lo, hi, p) for a vector of paired differences, any n."""
    d = np.asarray(d, float)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1)) / math.sqrt(n)
    h = t_crit(n - 1) * se
    return n, m, m - h, m + h, MI.t_sf(m / se, n - 1)


def holm(cells, K=None):
    """Step-down Holm at alpha = 0.05 over `cells`, optionally with an
    inflated K.  Returns (ordered, survivors, stopping_row)."""
    rs = sorted(cells, key=lambda r: r["p"])
    KK = len(rs) if K is None else K
    surv, stop = [], None
    for i, r in enumerate(rs, 1):
        thr = 0.05 / (KK - i + 1)
        r["_rank"], r["_thr"] = i, thr
        r["_ok"] = stop is None and r["p"] <= thr
        if r["_ok"]:
            surv.append(r)
        elif stop is None:
            stop = (i, r, thr)
    return rs, surv, stop


def wsel(arr, w):
    return {"ENDPOINT": arr[-1], "EARLY": arr[0:3].mean(),
            "MID": arr[2:5].mean(), "LATE": arr[4:6].mean()}[w]


def row(r):
    return ("| %s | %s | %s | %s | %.7f | %.6f | %s | %s |"
            % (("**%d**" % r["_rank"]) if not r["_ok"] and r.get("_stop") else str(r["_rank"]),
               r["label"], r["w"], r["body"], r["p"], r["_thr"],
               "**yes**" if r["_ok"] else
               ("**no — procedure stops here**" if r.get("_stop") else "no"),
               "yes" if r["declared"] else "no"))


# ================================================================== F1
WINDOW_LABELS = [lab for lab, _a, _b in MI.WINDOW_PAIRS]


def f1():
    cells = MI.build_cells()
    base_rs, base_surv, base_stop = holm([dict(c) for c in cells])

    late = [c for c in cells if c["w"] == "LATE"]
    no_late = [dict(c) for c in cells if c["w"] != "LATE"]
    L_rs, L_surv, L_stop = holm(no_late)
    if L_stop:
        L_stop[1]["_stop"] = True

    no_end = [dict(c) for c in cells
              if not (c["w"] == "ENDPOINT" and c["label"] in WINDOW_LABELS)]
    E_rs, E_surv, E_stop = holm(no_end)
    if E_stop:
        E_stop[1]["_stop"] = True

    title = ("coach(const) − prior", "ENDPOINT", "finger")
    t_in_late = any((c["label"], c["w"], c["body"]) == title for c in no_late)
    t_in_end = any((c["label"], c["w"], c["body"]) == title for c in no_end)
    t_row = [r for r in L_rs if (r["label"], r["w"], r["body"]) == title][0]

    late_above = [r for r in base_rs[:base_stop[0] - 1] if r["w"] == "LATE"]
    cb_late_elbow = [c for c in cells if c["label"] == "coach(const) − blank"
                     and c["w"] == "LATE" and c["body"] == "elbow"][0]
    return dict(cells=cells, base=(base_rs, base_surv, base_stop), late=late,
                L=(L_rs, L_surv, L_stop), E=(E_rs, E_surv, E_stop),
                t_in_late=t_in_late, t_in_end=t_in_end, t_row=t_row,
                late_above=late_above, cb_late_elbow=cb_late_elbow)


# ================================================================== F2
def f2():
    out = {}
    for body in ("elbow", "finger"):
        co = MI.dists(MI.COACH_CONST, body)
        pr = MI.dists(MI.PRIOR_HO, body)
        bl = MI.dists(MI.BLANK_HO, body)
        seeds = sorted(set(co) & set(pr) & set(bl))
        absd, logd, nrmd, lE, lL = [], [], [], [], []
        for s in seeds:
            cE, cL = wsel(co[s], "EARLY"), wsel(co[s], "LATE")
            pE, pL = wsel(pr[s], "EARLY"), wsel(pr[s], "LATE")
            bE, bL = wsel(bl[s], "EARLY"), wsel(bl[s], "LATE")
            absd.append((cL - pL) - (cE - pE))
            logd.append((math.log(cL) - math.log(pL)) - (math.log(cE) - math.log(pE)))
            nrmd.append((cL / bL - pL / bL) - (cE / bE - pE / bE))
            lE.append(math.log(pE / cE))
            lL.append(math.log(pL / cL))
        out[body] = dict(
            absolute=paired_n(absd), log=paired_n(logd), norm=paired_n(nrmd),
            ratio_early=math.exp(float(np.mean(lE))),
            ratio_late=math.exp(float(np.mean(lL))))
    return out


# ================================================================== F3
MYO_ARMS = [
    ("blank", "model-free SAC, no model", "12k real"),
    ("prior", "**frozen** babble prior, Dyna (`freeze=True`) — the paper's method", "12k real"),
    ("warm", "the **same** babble prior with `freeze=False`, fine-tuned online", "12k real"),
    ("colearn", "world model trained online from scratch, MBPO-style, no babble", "12k real"),
    ("imag", "policy trained **inside** the frozen prior only", "**0 real**"),
]


def f3():
    pairing = []
    for s in range(5):
        a = os.path.join(ROOT, "results_myo", "blank_seed%d.json" % s)
        b = os.path.join(ROOT, "results_2x2_elbow", "blank_seed%d.json" % s)
        ha = hashlib.sha256(open(a, "rb").read()).hexdigest()
        hb = hashlib.sha256(open(b, "rb").read()).hexdigest()
        pairing.append((s, ha[:16], hb[:16], ha == hb))

    arms, meta = {}, {}
    for cond, _desc, _budget in MYO_ARMS:
        vals = {}
        for s in range(5):
            j = json.load(open(os.path.join(ROOT, "results_myo",
                                            "%s_seed%d.json" % (cond, s))))
            vals[s] = j[-1]["eval_dist"] * 1000.0
            if s == 0:
                meta[cond] = dict(n_ckpt=len(j), last=j[-1])
        arms[cond] = vals

    levels = []
    for cond, desc, budget in MYO_ARMS:
        v = np.array([arms[cond][s] for s in range(5)])
        levels.append(dict(cond=cond, desc=desc, budget=budget, mean=float(v.mean()),
                           sd=float(v.std(ddof=1)), per=[float(x) for x in v]))

    contrasts = []
    for a, b in (("prior", "warm"), ("prior", "colearn"), ("prior", "imag"),
                 ("warm", "colearn"), ("blank", "prior"), ("blank", "warm"),
                 ("blank", "colearn"), ("blank", "imag")):
        n, m, lo, hi, p = paired_n([arms[a][s] - arms[b][s] for s in range(5)])
        contrasts.append((a, b, n, m, lo, hi, p))
    return pairing, levels, contrasts, meta


# ================================================================== F5
def f5():
    specs = [("`prior soft − blank`", MI.PRIOR_SOFT, MI.BLANK_HO),
             ("`prior strict − blank`", MI.PRIOR_STRICT, MI.BLANK_HO),
             ("`prior soft − prior strict`", MI.PRIOR_SOFT, MI.PRIOR_STRICT)]
    out = []
    for body in ("elbow", "finger"):
        for name, A, B in specs:
            a, b = MI.ser(A, body, "ENDPOINT"), MI.ser(B, body, "ENDPOINT")
            common = sorted(set(a) & set(b))
            n, m, lo, hi, p = paired_n([a[s] - b[s] for s in common])
            out.append(dict(name=name, body=body, n=n, mean=m, lo=lo, hi=hi, p=p))
    return sorted(out, key=lambda r: r["p"])


# ============================================================== rendering
def render():
    L = []
    A = L.append
    A("")
    A("---")
    A("")
    A(MARKER)
    A("")
    A("_Generated by `recompute_addendum.py` (appended, never rewritten; it refuses to run "
      "twice). p-values are two-sided paired t from the primary estimator — df = 11 at "
      "n = 12, df = 4 for the n = 5 `results_myo` arms of F3. Windows are Sec. 3.6's: "
      "EARLY = mean{2k,4k,6k}, MID = mean{6k,8k,10k}, LATE = mean{10k,12k}, ENDPOINT = 12k._")
    A("")

    # ---------------------------------------------------------------- F1
    d = f1()
    base_rs, base_surv, base_stop = d["base"]
    L_rs, L_surv, L_stop = d["L"]
    E_rs, E_surv, E_stop = d["E"]

    A("## F1. The paper's own Sec. 3.6 exclusion rule, applied to the literal 72 (K = 58)")
    A("")
    A("Sec. 3.6 excludes the per-checkpoint finger contrasts from the family on a stated "
      "ground: they are *sub-divisions of windows already in the family and would "
      "double-count the same seeds*. The same subsection separately states that **ENDPOINT "
      "⊂ LATE**, so *a late-window result and an endpoint result are one confirmation and "
      "not two*. Put together, those two sentences say that the literal family must not "
      "contain both LATE and ENDPOINT for the same pair on the same body. **It contains "
      "both, for all seven window-spanning pairs on both bodies — 14 duplicated cells.** "
      "The rule was written down and never applied to the enumeration it governs.")
    A("")
    A("This is *not* a claim that the literal-72 verdict is wrong. Holm is valid under "
      "arbitrary dependence, so including correlated cells is conservative, not erroneous; "
      "K = 72 remains a legitimate family. What follows is that it is **one** family among "
      "several, and that **the paper's own stated exclusion criterion picks out a different "
      "one** — under which the verdict on the title reverses.")
    A("")
    A("### F1a. The 14 LATE cells")
    A("")
    A("| pair | body | LATE p | t-significant |")
    A("|---|---|---|---|")
    for c in sorted(d["late"], key=lambda r: r["p"]):
        A("| `%s` | %s | %.7f | %s |" % (c["label"], c["body"], c["p"],
                                         "yes" if c["sig"] else "no"))
    A("")
    A("**Exactly one of the fourteen sits above the stopping cell in the literal-72 "
      "ordering**: `%s` LATE %s at rank %d, p = %.7f. Every other LATE cell is at rank %d "
      "or below. So pruning LATE costs the ordering one rank above the reversal and "
      "thirteen below it, which is why the reversal moves up by exactly one."
      % (d["cb_late_elbow"]["label"], d["cb_late_elbow"]["body"],
         [r["_rank"] for r in base_rs
          if (r["label"], r["w"], r["body"]) == (d["cb_late_elbow"]["label"], "LATE",
                                                 d["cb_late_elbow"]["body"])][0],
         d["cb_late_elbow"]["p"], base_stop[0] + 1))
    A("")
    A("### F1b. Holm over the LATE-pruned family (K = %d)" % len(L_rs))
    A("")
    A("**K = %d, %d t-significant, %d surviving.** Threshold at rank *i* is "
      "0.05/(%d − i + 1)."
      % (len(L_rs), sum(1 for r in L_rs if r["sig"]), len(L_surv), len(L_rs)))
    A("")
    A("| # | contrast | window | body | uncorrected p | Holm threshold | survives | in declared 28? |")
    A("|---|---|---|---|---|---|---|---|")
    for r in L_rs[:18]:
        A(row(r))
    A("")
    tr = d["t_row"]
    A("**The title's cell moves from rank %d to rank %d and SURVIVES**: "
      "`coach(const) − prior` ENDPOINT finger, p = %.8f against a threshold of "
      "0.05/%d = %.8f. All %d cells above it clear their own thresholds, so the step-down "
      "reaches it; it then continues and stops at rank %d on `%s` %s %s "
      "(p = %.7f against %.6f)."
      % (base_stop[0], tr["_rank"], tr["p"], len(L_rs) - tr["_rank"] + 1, tr["_thr"],
         tr["_rank"] - 1, L_stop[0], L_stop[1]["label"], L_stop[1]["w"],
         L_stop[1]["body"], L_stop[1]["p"], L_stop[2]))
    A("")
    dsurv = [r for r in L_surv if r["declared"]]
    A("%d survive, %d of them among the declared 28:" % (len(L_surv), len(dsurv)))
    for r in dsurv:
        A("- `%s`, %s, %s (p = %.7f)" % (r["label"], r["w"], r["body"], r["p"]))
    A("")
    A("### F1c. The mirror consequence inside Sec. 4.2")
    A("")
    cb = d["cb_late_elbow"]
    A("`coach(const) − blank` **LATE elbow** (p = %.7f) is a **literal-72 survivor** — rank "
      "10 of the 72, threshold 0.000794 — and LATE contains the endpoint. Sec. 4.2's "
      "sentence *not one endpoint accuracy claim in Sec. 4.2 survives Holm over the literal "
      "family* is therefore true of the **ENDPOINT** cell (p = 0.0017642, rank 21, fails) "
      "and **false of the same confirmation read at LATE**. The two are one confirmation "
      "and not two, by Sec. 3.6's own rule — which means the family cannot both contain "
      "them and treat a failure at one of them as the whole verdict." % cb["p"])
    A("")
    A("### F1d. Pruning is not free: the mirror prune")
    A("")
    A("Dropping **ENDPOINT** instead of LATE for the same seven window-spanning pairs on "
      "both bodies also gives **K = %d** — and removes `coach(const) − prior` ENDPOINT "
      "finger, the title's own cell, from the family entirely (title cell present: %s). "
      "Under that family %d cells survive and the step-down stops at rank %d on `%s` %s %s "
      "(p = %.7f against %.6f). A reader who prefers ENDPOINT-pruning does not get a "
      "surviving reversal; they get no reversal cell to correct."
      % (len(E_rs), "yes" if d["t_in_end"] else "**no**", len(E_surv), E_stop[0],
         E_stop[1]["label"], E_stop[1]["w"], E_stop[1]["body"], E_stop[1]["p"], E_stop[2]))
    A("")
    A("| family | K | t-significant | survivors | verdict on `coach(const) − prior` ENDPOINT finger |")
    A("|---|---|---|---|---|")
    A("| literal (both LATE and ENDPOINT) | 72 | %d | %d | **fails** — the stopping cell, p = 0.0009005 vs 0.000833 |"
      % (sum(1 for r in base_rs if r["sig"]), len(base_surv)))
    A("| LATE pruned (Sec. 3.6's rule applied) | %d | %d | %d | **survives** — rank %d, p = %.7f vs %.8f |"
      % (len(L_rs), sum(1 for r in L_rs if r["sig"]), len(L_surv), tr["_rank"],
         tr["p"], tr["_thr"]))
    A("| ENDPOINT pruned (the mirror) | %d | %d | %d | **not in the family** |"
      % (len(E_rs), sum(1 for r in E_rs if r["sig"]), len(E_surv)))
    A("")

    # ---------------------------------------------------------------- F2
    A("## F2. The decay interaction under three scalings")
    A("")
    A("Sec. 4.6 adopts the scale-free CV ratio beside the raw SD ratio, and states the "
      "reason: *because reach error is bounded below by zero, SD tracks the mean*. The "
      "identical reasoning applies to `[coach(const) − prior]@LATE − @EARLY`, which is a "
      "difference of differences of the same bounded-below quantity measured at two very "
      "different error levels — and it was never applied. Three scalings of the same "
      "within-seed interaction, n = 12, held-out:")
    A("")
    A("- **absolute** — `(cL − pL) − (cE − pE)`, the paper's version, in mrad / mm;")
    A("- **log-error** — `(ln cL − ln pL) − (ln cE − ln pE)`, scale-free: a change in the "
      "*ratio* between the arms;")
    A("- **blank-normalised** — each arm divided by that seed's own `blank` in the same "
      "window before differencing, so the units are fractions of the untreated baseline.")
    A("")
    A("| body | scaling | mean | paired t 95% CI | p |")
    A("|---|---|---|---|---|")
    F2 = f2()
    for body in ("elbow", "finger"):
        for key, nm in (("absolute", "absolute (paper)"), ("log", "log-error"),
                        ("norm", "blank-normalised")):
            n, m, lo, hi, p = F2[body][key]
            star = " **← includes zero**" if lo <= 0 <= hi else ""
            fmt = "%+.3f" if key == "absolute" else "%+.4f"
            A("| %s | %s | %s | [%s, %s] | %.4f%s |"
              % (body, nm, fmt % m, fmt % lo, fmt % hi, p, star))
    A("")
    A("**On elbow the interaction is significant in raw units and in blank-normalised "
      "units, and NOT significant on a log reading — the CI includes zero (p = %.4f).** "
      "The reason is visible in the ratio: the elbow `prior`/`coach` error ratio runs "
      "**×%.3f (EARLY) → ×%.3f (LATE)**, so on elbow the coach's *proportional* lead "
      "shrinks by far less than its absolute lead, because both arms' errors fall steeply "
      "over the budget. **On finger the interaction is significant under all three "
      "scalings**, and it is not merely a shrinkage but a **sign change** — the ratio runs "
      "×%.3f (EARLY) → ×%.3f (LATE), crossing 1 — which is scale-invariant and therefore "
      "untouched by this analysis."
      % (F2["elbow"]["log"][4], F2["elbow"]["ratio_early"], F2["elbow"]["ratio_late"],
         F2["finger"]["ratio_early"], F2["finger"]["ratio_late"]))
    A("")
    A("So abstract clause (i) and Sec. 1.1 claim 1's *significant on each body* is, on "
      "**elbow**, a raw-units result that does not survive a scale-free reading. This is "
      "the same fault the paper corrected for dispersion, and it is now recorded as a "
      "correction item rather than left to the reader.")
    A("")

    # ---------------------------------------------------------------- F3
    pairing, levels, contrasts, meta = f3()
    A("## F3. The unscored `results_myo` arms — the fifth recurrence of run-but-never-scored")
    A("")
    A("`results_myo/` holds five arms × five seeds on **myoElbow**, 12k steps, **train** "
      "targets. Three of them — `warm`, `colearn`, `imag` — were never scored anywhere in "
      "this paper, in `RECOMPUTED.md`, or in any figure. They matter because **frozen-ness "
      "is a design commitment carried by the title, Sec. 1, Sec. 1.2, Sec. 2's positioning "
      "against co-learning methods, and Sec. 3.3**, and `warm` is *the same babble prior "
      "loaded with `freeze=False`* (`run_reach.py:103–104`): the un-frozen version of the "
      "identical model, on the paper's own body, at the paper's own budget.")
    A("")
    A("**Pairing check first.** The arms are only mutually paired if `results_myo/blank` is "
      "the same run as the `blank` the rest of the study uses. SHA-256 of the archived "
      "per-seed JSONs:")
    A("")
    A("| seed | `results_myo/blank_seed{s}.json` | `results_2x2_elbow/blank_seed{s}.json` | bit-identical |")
    A("|---|---|---|---|")
    for s, ha, hb, ok in pairing:
        A("| %d | `%s…` | `%s…` | **%s** |" % (s, ha, hb, "yes" if ok else "NO"))
    A("")
    A("All five are byte-identical, so seeds 0–4 of `results_myo` and of the main "
      "train-target study are the same runs and every contrast below is genuinely paired.")
    A("")
    A("| arm | what it is | real env steps | mean (mrad) | SD | per-seed |")
    A("|---|---|---|---|---|---|")
    for r in levels:
        A("| `%s` | %s | %s | **%.2f** | %.2f | %s |"
          % (r["cond"], r["desc"], r["budget"], r["mean"], r["sd"],
             ", ".join("%.1f" % x for x in r["per"])))
    A("")
    A("`imag` is the strongest of these to state plainly: its archived records carry "
      "`real_steps = 0` at every one of its %d checkpoints (`imagined_steps` %d at the "
      "last), so it is a policy trained **entirely inside the frozen babble model** and "
      "then evaluated in the real environment."
      % (meta["imag"]["n_ckpt"], meta["imag"]["last"]["imagined_steps"]))
    A("")
    A("Paired contrasts over the five shared seeds (df = 4; negative = the first arm has "
      "lower error):")
    A("")
    A("| contrast | n | mean | paired t 95% CI | p |")
    A("|---|---|---|---|---|")
    for a, b, n, m, lo, hi, p in contrasts:
        A("| `%s − %s` | %d | %+.2f | [%+.2f, %+.2f] | %.4f |" % (a, b, n, m, lo, hi, p))
    A("")
    A("**No numbered claim moves.** Every interval above contains zero at n = 5, including "
      "`blank − prior`, which is the paper's own headline effect and which this subsample "
      "cannot resolve either. What these data do NOT do is settle the frozen-versus-"
      "unfrozen question: `prior − warm` is %+.2f [%+.2f, %+.2f] and `prior − colearn` is "
      "%+.2f [%+.2f, %+.2f], both wide enough to accommodate a substantial advantage in "
      "either direction. **The honest statement is that the paper's central design "
      "commitment — that the body model is frozen — was tested on the paper's own body and "
      "the test is inconclusive at n = 5, and that this was never reported.**"
      % (contrasts[0][3], contrasts[0][4], contrasts[0][5],
         contrasts[1][3], contrasts[1][4], contrasts[1][5]))
    A("")

    # ---------------------------------------------------------------- F4
    A("## F4. Correction to E5's negative register")
    A("")
    A("E5 closes with: *Nothing else in the repository was found computed-but-unprinted: "
      "`results_plateau_*` duplicates `results_plateau2_*` for the steps both cover, and "
      "the seed-3 plateau runs `run_plateau2.ps1` asks for were never launched.* "
      "**That statement is false**, and F3 is the counterexample: three arms × five seeds, "
      "run to completion, archived as JSON with logs beside them, never scored.")
    A("")
    A("E5 is **not edited in place** — this appendix is append-only and the earlier text "
      "stands as written. The corrected register is:")
    A("")
    A("| repository artefact | status in E5 | corrected status |")
    A("|---|---|---|")
    A("| `results_plateau_*` | duplicate of `results_plateau2_*` | unchanged — still a duplicate |")
    A("| seed-3 plateau runs | never launched | unchanged — still absent |")
    A("| `results_myo/warm_seed0..4` | **not registered** | **computed, never scored** — F3 |")
    A("| `results_myo/colearn_seed0..4` | **not registered** | **computed, never scored** — F3 |")
    A("| `results_myo/imag_seed0..4` | **not registered** | **computed, never scored** — F3 |")
    A("| `results_myo/{blank,prior}_seed0..4` | not registered | duplicates of `results_2x2_elbow` seeds 0–4 (bit-identical, F3) |")
    A("")
    A("The species-level point is the one worth keeping: **a sweep's own negative register "
      "was wrong.** E5 was written to close the run-but-never-scored species and it "
      "asserted completeness over a repository it had not enumerated. A negative register "
      "is a claim like any other and needs the same evidence.")
    A("")

    # ---------------------------------------------------------------- F5
    A("## F5. The six cells Sec. 4.9 omitted from the literal family, all six computed")
    A("")
    A("Sec. 4.9 justified leaving `prior strict-withdrawn − blank` and the "
      "`prior soft-withdrawn` contrasts out of the literal enumeration with: *every one of "
      "them has a larger p than rank 13's 0.00090 (the largest, the elbow soft-withdrawal "
      "cell, is 0.0078)*. The parenthesis is wrong in the direction that flatters the "
      "omission: **0.0078 is the smallest of the six, not the largest.**")
    A("")
    A("| cell | body | mean | paired t 95% CI | p |")
    A("|---|---|---|---|---|")
    for r in f5():
        A("| %s | %s | %+.3f | [%+.3f, %+.3f] | **%.4f** |"
          % (r["name"], r["body"], r["mean"], r["lo"], r["hi"], r["p"]))
    A("")
    six = f5()
    A("Smallest **%.4f**, largest **%.4f**. The *argument* is unaffected — all six are "
      "still larger than 0.00090, so entering them would still leave the stopping cell's "
      "threshold untouched and would still lower every threshold in the table (D2b). Only "
      "the parenthetical superlative was wrong, and it is corrected in Sec. 4.9 to "
      "\"the smallest … is 0.0078; the largest is 0.53\"."
      % (six[0]["p"], six[-1]["p"]))
    A("")
    return "\n".join(L)


def main():
    text = render()
    print(text)
    if "--append" in sys.argv:
        with open(REC_PATH, encoding="utf-8") as f:
            cur = f.read()
        if MARKER in cur:
            print("\n[recompute_addendum] marker already present — refusing to append.",
                  file=sys.stderr)
            return 0
        with open(REC_PATH, "a", encoding="utf-8", newline="") as f:
            f.write(text)
        print("\n[recompute_addendum] appended Appendix F to RECOMPUTED.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
