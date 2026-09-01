"""movement_index_holm72.py -- regenerates the two load-bearing analyses that
the paper quoted but never archived:

  D1  the movement-index (M) table for every myoFinger arm, each scored
      against the referent measured on ITS OWN target set;
  D2  the LITERAL 72-cell Holm-Bonferroni table (the family Sec. 3.6's rule
      enumerates), its sensitivity to widening K, and the three-family split.

Both are written by APPENDING to RECOMPUTED.md -- this script never opens that
file in write mode and never truncates it.  Running it twice is a no-op: it
refuses to append if the marker heading is already present.

D3 (the eight-RNG-seed random-action referent) is produced by the separate
`random_baseline_M.py`, which needs MuJoCo + MyoSuite; its transcript is pasted
into the appendix by hand from that script's own stdout.

Definitions, unchanged from the paper:
  M = (-eval_return / 100) - eval_dist,  both terms in millimetres.
  Windows: EARLY = mean of checkpoints 2k/4k/6k, MID = 6k/8k/10k,
           LATE = 10k/12k, ENDPOINT = 12k.
  p-values are two-sided paired t with df = 11, from the primary estimator.

Usage:  python movement_index_holm72.py            # print only
        python movement_index_holm72.py --append   # print and append to RECOMPUTED.md
"""

import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REC_PATH = os.path.join(ROOT, "RECOMPUTED.md")
MARKER = "# Appendix D — the movement index M, and Holm over the literal 72-cell family"

# --------------------------------------------------------------------- stats
try:
    from scipy import stats as _st

    def t_sf(t, df):
        return float(2.0 * _st.t.sf(abs(t), df))

    T975_11 = float(_st.t.ppf(0.975, 11))
except Exception:                                     # pragma: no cover
    T975_11 = 2.200985160

    def _betacf(a, b, x):
        MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < EPS:
                break
        return h

    def _betainc(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        front = math.exp(lb + a * math.log(x) + b * math.log(1.0 - x))
        if x < (a + 1.0) / (a + b + 2.0):
            return front * _betacf(a, b, x) / a
        return 1.0 - front * _betacf(b, a, 1.0 - x) / b

    def t_sf(t, df):
        return float(_betainc(df / 2.0, 0.5, df / (df + t * t)))


def paired(d):
    """(n, mean, lo, hi, p) for a vector of paired differences."""
    d = np.asarray(d, float)
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    half = T975_11 * se
    return n, m, m - half, m + half, t_sf(m / se, n - 1)


# ---------------------------------------------------------------- data access
def _load(dirname, cond, body, field):
    d = os.path.join(ROOT, "%s_%s" % (dirname, body))
    out = {}
    for f in glob.glob(os.path.join(d, "%s_seed*.json" % cond)):
        base = os.path.basename(f)
        sid = int(base[len(cond) + len("_seed"):-len(".json")])
        with open(f) as fh:
            j = json.load(fh)
        out[sid] = np.array([c[field] for c in j], dtype=float)
    if not out:
        raise FileNotFoundError("%s cond=%s field=%s" % (d, cond, field))
    return out


def dists(spec, body):
    """seed -> 6 checkpoint distances, x1000 (mrad for elbow, mm for finger)."""
    return {s: v * 1000.0 for s, v in _load(spec[0], spec[1], body, "eval_dist").items()}


def window(arr, w):
    return {"ENDPOINT": arr[-1], "EARLY": arr[0:3].mean(),
            "MID": arr[2:5].mean(), "LATE": arr[4:6].mean()}[w]


def ser(spec, body, w):
    return {s: window(v, w) for s, v in dists(spec, body).items()}


BLANK_HO = ("results_heldout", "blank")
PRIOR_HO = ("results_heldout", "prior")
BLANK_TR = ("results_2x2", "blank")
PRIOR_TR = ("results_2x2", "prior")
COACH_CONST = ("results_noanneal", "coach")
COACH_ANN = ("results_cleancoach", "coach")
COACH_ABRUPT = ("results_abruptcoach", "coach")
PC_CONST = ("results_pcconst", "priorcoach")
PC_ANN = ("results_cleancoach", "priorcoach")
PRIOR_SOFT = ("results_prioroff", "prior")
PRIOR_STRICT = ("results_priorpurge", "prior")
RANDPRIOR_TR = ("results_randprior", "prior")

# --------------------------------------------------------------- D1: M table
# Referents scored under the identical protocol (Sec. 4.8 runs, re-scored for M
# in Sec. 4.6).  A TRAIN-target arm must be read against the TRAIN referents.
REFERENT = {
    "held-out": dict(noop=-40.12, teacher=-17.69),
    "train":    dict(noop=-26.89, teacher=-3.63),
}

ARMS = [
    ("blank (model-free SAC)",  BLANK_HO,     "held-out"),
    ("prior kept",              PRIOR_HO,     "held-out"),
    ("prior soft-withdrawn",    PRIOR_SOFT,   "held-out"),
    ("prior strict-withdrawn",  PRIOR_STRICT, "held-out"),
    ("coach constant",          COACH_CONST,  "held-out"),
    ("coach annealed",          COACH_ANN,    "held-out"),
    ("coach abrupt-withdrawn",  COACH_ABRUPT, "held-out"),
    ("prior + constant coach",  PC_CONST,     "held-out"),
    ("prior + annealed coach",  PC_ANN,       "held-out"),
    ("randprior",               RANDPRIOR_TR, "train"),
]


def m_table():
    rows = []
    for name, spec, tset in ARMS:
        ret = _load(spec[0], spec[1], "finger", "eval_return")
        dst = _load(spec[0], spec[1], "finger", "eval_dist")
        seeds = sorted(ret)
        M = np.array([(-ret[s][-1] / 100.0) * 1000.0 - dst[s][-1] * 1000.0 for s in seeds])
        D = np.array([dst[s][-1] * 1000.0 for s in seeds])
        r = REFERENT[tset]
        rows.append(dict(name=name, tset=tset, n=len(seeds), dist=float(D.mean()),
                         M=float(M.mean()), sd=float(M.std(ddof=1)),
                         dnoop=abs(float(M.mean()) - r["noop"]),
                         dteach=abs(float(M.mean()) - r["teacher"])))
    return rows


# ------------------------------------------------------- D2: the literal 72
WINDOW_PAIRS = [
    ("prior − blank (held-out)", PRIOR_HO, BLANK_HO),
    ("prior − blank (train)", PRIOR_TR, BLANK_TR),
    ("coach(const) − blank", COACH_CONST, BLANK_HO),
    ("coach(anneal) − blank", COACH_ANN, BLANK_HO),
    ("coach(const) − prior", COACH_CONST, PRIOR_HO),
    ("priorcoach(const) − prior", PC_CONST, PRIOR_HO),
    ("priorcoach(const) − coach(const)", PC_CONST, COACH_CONST),
]
ENDPOINT_PAIRS = [
    ("trained prior − randprior (train)", PRIOR_TR, RANDPRIOR_TR),
    ("randprior − blank (train)", RANDPRIOR_TR, BLANK_TR),
    ("coach abrupt − coach const", COACH_ABRUPT, COACH_CONST),
    ("prior strict − prior kept", PRIOR_STRICT, PRIOR_HO),
    ("coach abrupt − blank", COACH_ABRUPT, BLANK_HO),
    ("coach abrupt − prior strict", COACH_ABRUPT, PRIOR_STRICT),
]

# The 28 cells the paper DECLARED, as (label, window, body); used only to mark
# which literal-family survivors carry a claim.
DECLARED = {
    ("trained prior − randprior (train)", "ENDPOINT", "elbow"),
    ("trained prior − randprior (train)", "ENDPOINT", "finger"),
    ("randprior − blank (train)", "ENDPOINT", "elbow"),
    ("coach(const) − blank", "EARLY", "elbow"),
    ("coach(const) − blank", "EARLY", "finger"),
    ("coach(const) − blank", "ENDPOINT", "elbow"),
    ("coach(anneal) − blank", "EARLY", "elbow"),
    ("coach(anneal) − blank", "EARLY", "finger"),
    ("coach(const) − prior", "EARLY", "elbow"),
    ("coach(const) − prior", "MID", "elbow"),
    ("coach(const) − prior", "LATE", "elbow"),
    ("coach(const) − prior", "ENDPOINT", "elbow"),
    ("coach(const) − prior", "EARLY", "finger"),
    ("coach(const) − prior", "MID", "finger"),
    ("coach(const) − prior", "LATE", "finger"),
    ("coach(const) − prior", "ENDPOINT", "finger"),
    ("prior − blank (held-out)", "ENDPOINT", "elbow"),
    ("prior − blank (train)", "ENDPOINT", "elbow"),
    ("priorcoach(const) − prior", "ENDPOINT", "elbow"),
    ("priorcoach(const) − prior", "ENDPOINT", "finger"),
    ("priorcoach(const) − prior", "LATE", "finger"),
    ("priorcoach(const) − coach(const)", "EARLY", "elbow"),
    ("coach abrupt − coach const", "ENDPOINT", "elbow"),
    ("coach abrupt − coach const", "ENDPOINT", "finger"),
    ("coach abrupt − blank", "ENDPOINT", "finger"),
    ("coach abrupt − prior strict", "ENDPOINT", "finger"),
    ("decay interaction [coach(const)−prior]@LATE−@EARLY", "—", "elbow"),
    ("decay interaction [coach(const)−prior]@LATE−@EARLY", "—", "finger"),
}

# The three questions Sec. 4.9 names, used for the three-family split.
ROUTE = ("coach(const) − prior", "priorcoach(const) − prior",
         "priorcoach(const) − coach(const)",
         "decay interaction [coach(const)−prior]@LATE−@EARLY")
BEATS = ("prior − blank (held-out)", "prior − blank (train)",
         "coach(const) − blank", "coach(anneal) − blank",
         "trained prior − randprior (train)", "randprior − blank (train)")
REMOVAL = ("coach abrupt − coach const", "prior strict − prior kept",
           "coach abrupt − blank", "coach abrupt − prior strict",
           "coach cost − prior cost")


def build_cells():
    cells = []

    def add(label, w, body, d):
        n, m, lo, hi, p = paired(d)
        cells.append(dict(label=label, w=w, body=body, n=n, mean=m, lo=lo, hi=hi, p=p,
                          sig=(lo > 0 or hi < 0),
                          declared=(label, w, body) in DECLARED))

    for body in ("elbow", "finger"):
        for label, A, B in WINDOW_PAIRS:
            for w in ("EARLY", "MID", "LATE", "ENDPOINT"):
                a, b = ser(A, body, w), ser(B, body, w)
                common = sorted(set(a) & set(b))
                add(label, w, body, [a[s] - b[s] for s in common])
        for label, A, B in ENDPOINT_PAIRS:
            a, b = ser(A, body, "ENDPOINT"), ser(B, body, "ENDPOINT")
            common = sorted(set(a) & set(b))
            add(label, "ENDPOINT", body, [a[s] - b[s] for s in common])
        # withdrawal-cost difference
        a1, b1 = ser(COACH_ABRUPT, body, "ENDPOINT"), ser(COACH_CONST, body, "ENDPOINT")
        a2, b2 = ser(PRIOR_STRICT, body, "ENDPOINT"), ser(PRIOR_HO, body, "ENDPOINT")
        common = sorted(set(a1) & set(b1) & set(a2) & set(b2))
        add("coach cost − prior cost", "ENDPOINT", body,
            [(a1[s] - b1[s]) - (a2[s] - b2[s]) for s in common])
        # decay interaction
        aL, bL = ser(COACH_CONST, body, "LATE"), ser(PRIOR_HO, body, "LATE")
        aE, bE = ser(COACH_CONST, body, "EARLY"), ser(PRIOR_HO, body, "EARLY")
        common = sorted(set(aL) & set(bL))
        add("decay interaction [coach(const)−prior]@LATE−@EARLY", "—", body,
            [(aL[s] - bL[s]) - (aE[s] - bE[s]) for s in common])
    return cells


def holm(cells, K=None):
    """Step-down Holm at alpha = 0.05. Returns (survivors, stopping_row, thresholds)."""
    rs = sorted(cells, key=lambda r: r["p"])
    K = len(rs) if K is None else K
    surv, stop, thr = [], None, []
    for i, r in enumerate(rs, 1):
        t = 0.05 / (K - i + 1)
        thr.append(t)
        if stop is None and r["p"] <= t:
            surv.append(r)
        elif stop is None:
            stop = (i, r, t)
    return rs, surv, stop, thr


# ------------------------------------------------------------------ rendering
def render():
    L = []
    A = L.append
    A("")
    A("---")
    A("")
    A(MARKER)
    A("")
    A("_Generated by `movement_index_holm72.py` (appended, never rewritten). "
      "p-values are two-sided paired t, df = 11, from the primary estimator; "
      "M = (−`eval_return`/100) − `eval_dist`, both in millimetres, at the 12k "
      "checkpoint. The eight-RNG-seed random-action referent of D3 is produced "
      "by `random_baseline_M.py`._")
    A("")

    # ---------------- D1
    A("## D1. The movement index M for every myoFinger arm, against its OWN referent")
    A("")
    A("Referents, re-scored for M under the Sec. 4.8 protocol: held-out no-op **−40.12**, "
      "held-out clean teacher **−17.69**; train no-op **−26.89**, train teacher **−3.63**. "
      "`randprior` was run on **training targets only** (Sec. 4.4), so it is scored against "
      "the TRAIN referents; every other arm is held-out. Mixing the two is the error "
      "correction item 36 deleted once already.")
    A("")
    A("| arm | target set | dist (mm) | M (mm) | per-seed SD | \\|M − own no-op\\| | \\|M − own teacher\\| | n |")
    A("|---|---|---|---|---|---|---|---|")
    rows = m_table()
    for r in rows:
        A("| %s | %s | %.2f | %+.2f | %.2f | **%.2f** | %.2f | %d |"
          % (r["name"], r["tset"], r["dist"], r["M"], r["sd"], r["dnoop"], r["dteach"], r["n"]))
    A("")
    ho = [r for r in rows if r["tset"] == "held-out"]
    rp = [r for r in rows if r["tset"] == "train"][0]
    closest = min(rows, key=lambda r: r["dnoop"])
    furthest = max(rows, key=lambda r: r["dnoop"])
    A("- All ten arms lie in **[%+.2f, %+.2f]**; the universal is now computed over ten arms "
      "and not asserted over seven." % (min(r["M"] for r in rows), max(r["M"] for r in rows)))
    A("- Against **its own** referent the arm CLOSEST to the no-op is **%s at %.2f mm** "
      "(%.1f per-seed SDs), and the arm FURTHEST is **%s at %.2f mm**."
      % (closest["name"], closest["dnoop"], closest["dnoop"] / closest["sd"],
         furthest["name"], furthest["dnoop"]))
    A("- The nine **held-out** arms span **%.2f–%.2f mm** from the held-out no-op and "
      "**%.2f–%.2f mm** from the held-out teacher."
      % (min(r["dnoop"] for r in ho), max(r["dnoop"] for r in ho),
         min(r["dteach"] for r in ho), max(r["dteach"] for r in ho)))
    A("- `randprior` sits **%.2f mm** from the train no-op and **%.2f mm** from the train "
      "teacher. Read against the *held-out* no-op it would appear 42.92 mm away — the "
      "largest number in the table and the one an earlier draft reported. That comparison "
      "mixes target sets and is void." % (rp["dnoop"], rp["dteach"]))
    A("- No arm reproduces the no-op's movement signature: the smallest gap, %.2f mm, is "
      "%.1f per-seed SDs." % (closest["dnoop"], closest["dnoop"] / closest["sd"]))
    A("")

    # ---------------- D2
    cells = build_cells()
    rs, surv, stop, thr = holm(cells)
    K = len(cells)
    nsig = sum(1 for c in cells if c["sig"])
    A("## D2. Holm–Bonferroni over the LITERAL family (K = %d)" % K)
    A("")
    A("One paired-t contrast per (condition pair × window) among the directly compared "
      "conditions, both bodies, plus the two withdrawal-cost differences and the two decay "
      "interactions — Sec. 3.6's rule with no significance filter. **K = %d, %d t-significant, "
      "%d surviving.** Threshold at rank *i* is 0.05/(K − i + 1); the step-down stops at the "
      "first failure." % (K, nsig, len(surv)))
    A("")
    A("| # | contrast | window | body | uncorrected p | Holm threshold | survives | in declared 28? |")
    A("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rs[:22], 1):
        ok = i <= len(surv)
        A("| %s | %s | %s | %s | %.7f | %.6f | %s | %s |"
          % (("**%d**" % i) if (stop and i == stop[0]) else str(i),
             r["label"], r["w"], r["body"], r["p"], thr[i - 1],
             "**yes**" if ok else ("**no — procedure stops here**" if (stop and i == stop[0]) else "no"),
             "yes" if r["declared"] else "no"))
    A("")
    A("Ranks 23–%d fail and are t-significant; ranks %d–%d are the %d t-non-significant cells."
      % (nsig, nsig + 1, K, K - nsig))
    A("")
    dsurv = [r for r in surv if r["declared"]]
    A("**%d survive; the step-down stops at rank %d on `%s` %s %s (p = %.7f against %.6f).** "
      "%d of the survivors are among the declared 28:"
      % (len(surv), stop[0], stop[1]["label"], stop[1]["w"], stop[1]["body"],
         stop[1]["p"], stop[2], len(dsurv)))
    for r in dsurv:
        A("- %s, %s, %s (p = %.7f)" % (r["label"], r["w"], r["body"], r["p"]))
    A("")

    # ---------------- D2b sensitivity
    A("### D2b. The survivor count is NOT invariant to widening K")
    A("")
    A("Sec. 4.9 argues correctly that adding cells *more* significant than a given cell leaves "
      "its threshold unchanged. That argument settles the verdict on the **stopping** cell. It "
      "does **not** settle the **survivor count**, which falls as soon as enough *less* "
      "significant cells are added — and Sec. 4.9 names roughly eight such cells it left out "
      "(`prior strict-withdrawn − blank` on both bodies, the `prior soft-withdrawn` contrasts) "
      "with about 25 more in Appendix C. Holding the ordering fixed and raising K:")
    A("")
    A("| K | rank-12 cell (`%s` %s %s, p = %.8f) | threshold | verdict | survivors | of them declared |"
      % (rs[11]["label"], rs[11]["w"], rs[11]["body"], rs[11]["p"]))
    A("|---|---|---|---|---|---|")
    for Kn in (72, 76, 78, 79, 80, 84, 90):
        _r, s2, _st2, th2 = holm(cells, K=Kn)
        A("| %d | rank 12 | %.8f | %s | %d | %d |"
          % (Kn, 0.05 / (Kn - 12 + 1),
             "passes" if rs[11]["p"] <= 0.05 / (Kn - 12 + 1) else "**fails**",
             len(s2), sum(1 for x in s2 if x["declared"])))
    A("")
    A("**Break-even is K = 79 → 80: eight further less-significant cells anywhere in the "
      "family drop the declared-28 core from six to five**, the cell lost being "
      "`coach(const) − blank` EARLY **elbow**. The two finger `coach − blank` EARLY cells, "
      "both `trained prior − randprior` cells and `coach(const) − prior` EARLY elbow are the "
      "five that hold at every K in this range.")
    A("")

    # ---------------- D2c selection inside the literal family
    A("### D2c. The literal family itself is not selection-free")
    A("")
    A("Two matched pairs show it. `coach abrupt − blank` is entered for **both** bodies "
      "(finger significant at +51.54, elbow not), while `prior strict-withdrawn − blank` — the "
      "same contrast on the other route — is entered for **neither**, and both of its cells are "
      "non-significant. Four claim-free `priorcoach(const)` window cells are in; the claim-free "
      "`prior soft-withdrawn` cells are out. Every omission runs the same way: the excluded "
      "cells are non-significant, and excluding them is what keeps K at 72 rather than ~80. "
      "The literal family therefore reproduces in miniature the selection-on-significance it "
      "was constructed to expose, in the direction that preserves the six-cell count.")
    A("")
    A("Sec. 4.9's defence of the omissions — that *every one of them has a larger p than rank "
      "13's 0.00090* — is the right test for the **stopping cell** and the wrong test for the "
      "**survivor count**. Larger p is exactly the property that lowers every threshold, and "
      "the count is what it lowers.")
    A("")

    # ---------------- D2d three-family split
    A("### D2d. The three-family split Sec. 4.9 itself proposes")
    A("")
    A("Sec. 4.9 observes that these contrasts *span three distinct questions … that many "
      "readers would treat as three families rather than one, and under any such split more "
      "cells survive*. Splitting the literal 72 that way:")
    A("")
    A("| family | question | cells | survivors | verdict on `coach(const) − prior` ENDPOINT finger |")
    A("|---|---|---|---|---|")
    for nm, keys, q in (("route comparison", ROUTE, "which aid is faster / which leads"),
                        ("versus model-free", BEATS, "does either beat model-free RL"),
                        ("removal", REMOVAL, "what removal costs")):
        sub = [c for c in cells if c["label"] in keys]
        _r2, s3, st3, th3 = holm(sub)
        tgt = [c for c in sub if c["label"] == "coach(const) − prior"
               and c["w"] == "ENDPOINT" and c["body"] == "finger"]
        if tgt:
            rank = sorted(sub, key=lambda r: r["p"]).index(tgt[0]) + 1
            t = 0.05 / (len(sub) - rank + 1)
            verd = ("**survives** — rank %d, p = %.7f against %.5f (%.1f×)"
                    % (rank, tgt[0]["p"], t, t / tgt[0]["p"])) if tgt[0] in s3 else \
                   ("fails — rank %d, p = %.7f against %.5f" % (rank, tgt[0]["p"], t))
        else:
            verd = "not in this family"
        A("| %s | %s | %d | %d | %s |" % (nm, q, len(sub), len(s3), verd))
    A("")
    A("So the reversal's correction status **depends on the family definition**: it survives "
      "the declared 28-cell family, it survives a three-family split by question, and it fails "
      "the single literal 72-cell family, which is the one on whose rank 13 the step-down "
      "stops. No single one of those three is the uniquely defensible choice, and the paper "
      "states all three rather than picking the flattering one.")
    A("")

    # ---------------- D3
    A("## D3. The random-action referent under eight RNG seeds (archived transcript)")
    A("")
    A("Sec. 4.8 said the random-action referent *would move under a different RNG seed by an "
      "amount we have not measured* while Sec. 4.6 quoted a spread of `−45.8 ± 7.2` over eight "
      "RNG seeds. Neither statement had a script. `random_baseline_M.py` is that script; it "
      "reproduces the archived Sec. 4.8 row bit-for-bit at RNG seed 0 (train 219.956 mm / "
      "held-out 243.448 mm against the reported 220.0 / 243.4; train M −11.856 / held-out M "
      "−47.028 against −11.86 / −47.03) and then continues to seeds 1–7. Verbatim output of "
      "`WM_BODY=myofinger python random_baseline_M.py`:")
    A("")
    A("```")
    A("rng | train dist  train M | held-out dist  held-out M")
    A("  0 |    219.956  -11.856 |       243.448    -47.028")
    A("  1 |    254.765  -24.455 |       257.395    -62.125")
    A("  2 |    255.089  -71.016 |       244.734    -47.839")
    A("  3 |    255.219  -33.771 |       261.519    -52.279")
    A("  4 |    229.858  -44.495 |       238.771    -38.504")
    A("  5 |    252.823  -36.220 |       262.403    -45.207")
    A("  6 |    253.015  -45.680 |       233.220    -48.136")
    A("  7 |    258.479  -44.003 |       252.168    -64.038")
    A("mean +- SD (ddof=1) over 8 RNG seeds:")
    A("  train    dist = 247.4 +- 14.2 mm    M = -38.94 +- 17.35")
    A("  held-out dist = 249.2 +- 10.8 mm    M = -50.64 +- 8.60")
    A("")
    A("held-out set alone, fresh default_rng(j):")
    A("  0 | dist    241.714  M  -42.756")
    A("  1 | dist    249.225  M  -40.168")
    A("  2 | dist    246.983  M  -56.916")
    A("  3 | dist    263.572  M  -49.318")
    A("  4 | dist    245.930  M  -54.509")
    A("  5 | dist    252.254  M  -41.307")
    A("  6 | dist    243.212  M  -36.357")
    A("  7 | dist    257.820  M  -44.647")
    A("  held-out M (fresh rng) = -45.75 +- 7.21")
    A("```")
    A("")
    A("Three things follow, and each changes a sentence in the paper.")
    A("")
    A("1. **The quoted `−45.8 ± 7.2` is correct, and its protocol now has a name.** It is the "
      "held-out M under a *fresh* `default_rng(j)` per seed — the protocol whose j = 0 gives the "
      "−42.76 quoted alongside it. Under the *sequential* protocol that produced the archived "
      "Sec. 4.8 row (train scored first from the same generator) the eight-seed held-out M is "
      "**−50.64 ± 8.60** instead. Both are reported, because the protocol has to travel with "
      "the number.")
    A("2. **Sec. 4.8's \"an amount we have not measured\" is measured here and must be "
      "replaced.** Held-out random-action distance over eight RNG seeds is **249.2 ± 10.8 mm**; "
      "the archived referent of 243.4 mm is seed 0 of that set, 0.5 SD below the mean. Train is "
      "**247.4 ± 14.2 mm** against an archived 220.0 mm — **1.9 SD below** — so the *train* "
      "random-action referent is a favourable draw and the Sec. 4.8 finger-train comparison "
      "should be read against ~247 mm.")
    A("3. **No verdict moves.** The finger orderings Sec. 4.8 and Sec. 5 rest on hold at the "
      "eight-seed mean exactly as at the archived draw: the no-op (172.3 mm) still beats random "
      "action on finger by a wider margin than reported, and all three M referents stay 30+ mm "
      "below every student arm.")
    A("")
    return L


def main(argv):
    lines = render()
    print("\n".join(lines))
    if "--append" in argv:
        with open(REC_PATH, encoding="utf-8") as fh:
            existing = fh.read()
        if MARKER in existing:
            print("\n[skip] appendix already present in RECOMPUTED.md; not appending again.")
            return
        before = len(existing.splitlines())
        with open(REC_PATH, "a", encoding="utf-8") as fh:   # APPEND ONLY
            fh.write("\n".join(lines) + "\n")
        with open(REC_PATH, encoding="utf-8") as fh:
            after = len(fh.read().splitlines())
        assert after > before, "RECOMPUTED.md shrank -- restore from RECOMPUTED.GOLDEN.bak"
        print("\n[ok] appended %d lines to RECOMPUTED.md (%d -> %d)"
              % (after - before, before, after))


if __name__ == "__main__":
    main(sys.argv[1:])
