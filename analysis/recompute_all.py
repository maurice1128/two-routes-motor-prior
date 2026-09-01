"""
recompute_all.py -- authoritative recomputation of every statistical contrast
in the wm_prior study, using the CORRECT paired-t interval as primary and a
BCa bootstrap as a sensitivity check.

Method (see header of RECOMPUTED.md):
  * unit of analysis = seed (n = 12 matched seeds per condition)
  * paired differences d_i = A_i - B_i over matched seed ids
  * PRIMARY interval: paired t 95% CI, mean(d) +/- t_{0.975, n-1} * sd(d)/sqrt(n)
  * SENSITIVITY interval: BCa bootstrap 95% CI, 20000 resamples, seed 0,
    resampling SEEDS (i.e. the paired differences), not timesteps
  * eval_dist multiplied by 1000 (elbow -> milliradians, finger -> millimetres)
  * ENDPOINT = last checkpoint (index 5); EARLY = mean of indices 0,1,2;
    MID = mean of indices 2,3,4; LATE = mean of indices 4,5

Outputs RECOMPUTED.md.  No interpretation, numbers only.
"""

import json
import os
import glob
import numpy as np

try:
    from scipy import stats as _st

    def tcrit(df):
        return float(_st.t.ppf(0.975, df))

    def ndtri(p):
        return float(_st.norm.ppf(p))

    def ndtr(x):
        return float(_st.norm.cdf(x))
except Exception:  # pragma: no cover -- fallback
    import math
    import sys as _sys
    # LOUD, deliberately. Without scipy the t quantile comes from a 4-dp lookup
    # table (2.2010 against the true 2.200985...), so every t interval in the
    # output shifts in its last printed digit and NOTHING ELSE SIGNALS IT. That
    # divergence was traced only by regenerating an archived appendix under two
    # interpreters and diffing. The canonical environment is .venv_mm
    # (scipy present); anything produced without it is NOT byte-comparable to
    # the archived tables. The table also covers df 9-11 only, so any other n
    # raises KeyError below rather than returning a wrong number quietly.
    print("WARNING: scipy unavailable -- using a 4-dp t-quantile table. "
          "t intervals will differ from the archived tables in the last digit. "
          "Regenerate with .venv_mm (scipy) for byte-comparable output.",
          file=_sys.stderr)
    _TTAB = {11: 2.2010, 10: 2.2281, 9: 2.2622}

    def tcrit(df):
        if df not in _TTAB:
            raise KeyError(
                "no tabulated t critical value for df=%d and scipy is unavailable; "
                "run under .venv_mm rather than accepting an approximation" % df)
        return _TTAB[df]

    def ndtr(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def ndtri(p):
        # Acklam's inverse normal approximation
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        pl, ph = 0.02425, 1 - 0.02425
        if p < pl:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > ph:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


ROOT = os.path.dirname(os.path.abspath(__file__))
NBOOT = 20000
SEED = 0

# ---------------------------------------------------------------- data loading

_cache = {}


def load(dirname, cond, body):
    """Return dict seed_id -> np.array of 6 eval_dist values (x1000)."""
    key = (dirname, cond, body)
    if key in _cache:
        return _cache[key]
    d = os.path.join(ROOT, "%s_%s" % (dirname, body))
    out = {}
    for f in glob.glob(os.path.join(d, "%s_seed*.json" % cond)):
        base = os.path.basename(f)
        sid = int(base[len(cond) + len("_seed"):-len(".json")])
        with open(f) as fh:
            j = json.load(fh)
        out[sid] = np.array([c["eval_dist"] for c in j], dtype=float) * 1000.0
    if not out:
        raise FileNotFoundError("no data: %s cond=%s" % (d, cond))
    _cache[key] = out
    return out


def window(arr, w):
    if w == "ENDPOINT":
        return arr[-1]
    if w == "EARLY":
        return arr[0:3].mean()
    if w == "MID":
        return arr[2:5].mean()
    if w == "LATE":
        return arr[4:6].mean()
    raise ValueError(w)


def series(spec, body, w):
    """spec = (dirname, cond). Return (seeds_sorted, values)."""
    data = load(spec[0], spec[1], body)
    seeds = sorted(data)
    return seeds, np.array([window(data[s], w) for s in seeds])


# ------------------------------------------------------------------ statistics

def t_ci(d):
    n = len(d)
    m = float(np.mean(d))
    if n < 2:
        return m, float("nan"), float("nan")
    se = float(np.std(d, ddof=1)) / np.sqrt(n)
    h = tcrit(n - 1) * se
    return m, m - h, m + h


def bca_ci(d, nboot=NBOOT, seed=SEED):
    d = np.asarray(d, dtype=float)
    n = len(d)
    theta = float(np.mean(d))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(nboot, n))
    boots = d[idx].mean(axis=1)
    boots.sort()

    # bias correction
    prop = float(np.mean(boots < theta)) + 0.5 * float(np.mean(boots == theta))
    prop = min(max(prop, 1.0 / (2 * nboot)), 1.0 - 1.0 / (2 * nboot))
    z0 = ndtri(prop)

    # acceleration (jackknife)
    tot = d.sum()
    jack = (tot - d) / (n - 1.0)
    jm = jack.mean()
    diff = jm - jack
    num = float(np.sum(diff ** 3))
    den = 6.0 * (float(np.sum(diff ** 2)) ** 1.5)
    a = 0.0 if den == 0 else num / den

    out = []
    for q in (0.025, 0.975):
        zq = ndtri(q)
        adj = z0 + (z0 + zq) / (1.0 - a * (z0 + zq))
        p = ndtr(adj)
        p = min(max(p, 0.0), 1.0)
        k = int(np.floor(p * nboot))
        k = min(max(k, 0), nboot - 1)
        out.append(float(boots[k]))
    return theta, out[0], out[1]


def flag(lo_t, hi_t, lo_b, hi_b):
    t_excl = (lo_t > 0) or (hi_t < 0)
    b_excl = (lo_b > 0) or (hi_b < 0)
    if t_excl and b_excl:
        st = 1 if lo_t > 0 else -1
        sb = 1 if lo_b > 0 else -1
        return "ROBUST" if st == sb else "METHOD-DEPENDENT"
    if (not t_excl) and (not b_excl):
        return "ROBUST-NULL"
    return "METHOD-DEPENDENT"


ROWS = []          # (family, label, body, n, mean, tlo, thi, blo, bhi, flag)
FAMILIES = []      # ordered list of family names
SKIPPED = []


def add(family, label, body, d):
    if family not in FAMILIES:
        FAMILIES.append(family)
    n = len(d)
    m, tlo, thi = t_ci(d)
    _, blo, bhi = bca_ci(d)
    ROWS.append((family, label, body, n, m, tlo, thi, blo, bhi,
                 flag(tlo, thi, blo, bhi)))


def contrast(family, label, A, B, w, bodies=("elbow", "finger")):
    """A - B on window w."""
    for body in bodies:
        try:
            sa, va = series(A, body, w)
            sb, vb = series(B, body, w)
        except FileNotFoundError as e:
            SKIPPED.append("%s [%s]: %s" % (label, body, e))
            continue
        common = sorted(set(sa) & set(sb))
        if len(common) < 2:
            SKIPPED.append("%s [%s]: only %d matched seeds" % (label, body, len(common)))
            continue
        ma = dict(zip(sa, va))
        mb = dict(zip(sb, vb))
        d = np.array([ma[s] - mb[s] for s in common])
        add(family, label, body, d)


def contrast_dd(family, label, A1, B1, A2, B2, w, bodies=("elbow", "finger")):
    """(A1 - B1) - (A2 - B2), paired within seed."""
    for body in bodies:
        try:
            specs = [series(s, body, w) for s in (A1, B1, A2, B2)]
        except FileNotFoundError as e:
            SKIPPED.append("%s [%s]: %s" % (label, body, e))
            continue
        common = sorted(set.intersection(*[set(s[0]) for s in specs]))
        maps = [dict(zip(s[0], s[1])) for s in specs]
        d = np.array([(maps[0][s] - maps[1][s]) - (maps[2][s] - maps[3][s])
                      for s in common])
        add(family, label, body, d)


def contrast_win_interaction(family, label, A, B, w1, w2, bodies=("elbow", "finger")):
    """(A-B)@w1 - (A-B)@w2, paired within seed."""
    for body in bodies:
        try:
            sa1, va1 = series(A, body, w1)
            sb1, vb1 = series(B, body, w1)
            sa2, va2 = series(A, body, w2)
            sb2, vb2 = series(B, body, w2)
        except FileNotFoundError as e:
            SKIPPED.append("%s [%s]: %s" % (label, body, e))
            continue
        common = sorted(set(sa1) & set(sb1))
        m = [dict(zip(sa1, va1)), dict(zip(sb1, vb1)),
             dict(zip(sa2, va2)), dict(zip(sb2, vb2))]
        d = np.array([(m[0][s] - m[1][s]) - (m[2][s] - m[3][s]) for s in common])
        add(family, label, body, d)


# ------------------------------------------------------------- condition specs
BLANK_HO = ("results_heldout", "blank")
PRIOR_HO = ("results_heldout", "prior")
BLANK_TR = ("results_2x2", "blank")
PRIOR_TR = ("results_2x2", "prior")

COACH_CONST = ("results_noanneal", "coach")
COACH_ANN = ("results_cleancoach", "coach")
COACH_ANN_TR = ("results_cleancoach_train", "coach")
COACH_ABRUPT = ("results_abruptcoach", "coach")

PC_CONST = ("results_pcconst", "priorcoach")
PC_ANN = ("results_cleancoach", "priorcoach")
PC_ANN_TR = ("results_cleancoach_train", "priorcoach")

PRIOR_SOFT = ("results_prioroff", "prior")
PRIOR_STRICT = ("results_priorpurge", "prior")
RANDPRIOR_TR = ("results_randprior", "prior")

# ---------------------------------------------------------------- 1. prior vs blank
F = "1. prior vs blank"
contrast(F, "prior - blank (held-out, ENDPOINT)", PRIOR_HO, BLANK_HO, "ENDPOINT")
contrast(F, "prior - blank (train, ENDPOINT)", PRIOR_TR, BLANK_TR, "ENDPOINT")
contrast(F, "prior - blank (held-out, EARLY)", PRIOR_HO, BLANK_HO, "EARLY")

# ------------------------------------------------- 2. coach(CONSTANT) vs blank
F = "2. coach (CONSTANT) vs blank"
contrast(F, "coach(const) - blank (held-out, ENDPOINT)", COACH_CONST, BLANK_HO, "ENDPOINT")
contrast(F, "coach(const) - blank (held-out, EARLY)", COACH_CONST, BLANK_HO, "EARLY")

# ------------------------------------------------- 3. coach(CONSTANT) vs prior
F = "3. coach (CONSTANT) vs prior"
for w in ("ENDPOINT", "EARLY", "MID", "LATE"):
    contrast(F, "coach(const) - prior (held-out, %s)" % w, COACH_CONST, PRIOR_HO, w)

# ------------------------------------------------- 4. coach(ANNEALED) vs blank
F = "4. coach (ANNEALED) vs blank"
contrast(F, "coach(anneal) - blank (held-out, ENDPOINT)", COACH_ANN, BLANK_HO, "ENDPOINT")
contrast(F, "coach(anneal) - blank (held-out, EARLY)", COACH_ANN, BLANK_HO, "EARLY")

# ----------------------------------------------------------- 5. randprior (train)
F = "5. randprior (train targets)"
contrast(F, "trained prior - randprior (train, ENDPOINT)", PRIOR_TR, RANDPRIOR_TR, "ENDPOINT")
contrast(F, "randprior - blank (train, ENDPOINT)", RANDPRIOR_TR, BLANK_TR, "ENDPOINT")

# ------------------------------------------- 6. withdrawal costs (withdrawn - kept)
F = "6. withdrawal cost (withdrawn - kept; + = withdrawal hurt)"
contrast(F, "prior strict-withdrawn - prior kept (ENDPOINT)", PRIOR_STRICT, PRIOR_HO, "ENDPOINT")
contrast(F, "coach abrupt-withdrawn - coach const (ENDPOINT)", COACH_ABRUPT, COACH_CONST, "ENDPOINT")
contrast_dd(F, "coach cost - prior cost (ENDPOINT)",
            COACH_ABRUPT, COACH_CONST, PRIOR_STRICT, PRIOR_HO, "ENDPOINT")

# ------------------------------------------------- 7. post-withdrawal vs blank
F = "7. post-withdrawal vs blank"
contrast(F, "prior strict-withdrawn - blank (ENDPOINT)", PRIOR_STRICT, BLANK_HO, "ENDPOINT")
contrast(F, "coach abrupt-withdrawn - blank (ENDPOINT)", COACH_ABRUPT, BLANK_HO, "ENDPOINT")

# --------------------------------------------- 8. between-arm post-withdrawal
F = "8. between-arm post-withdrawal"
contrast(F, "coach abrupt-withdrawn - prior strict-withdrawn (ENDPOINT)",
         COACH_ABRUPT, PRIOR_STRICT, "ENDPOINT")

# ------------------------------------------------------- 9. soft withdrawal
F = "9. prior soft withdrawal"
contrast(F, "prior soft-withdrawn - blank (ENDPOINT)", PRIOR_SOFT, BLANK_HO, "ENDPOINT")
contrast(F, "prior soft-withdrawn - prior strict-withdrawn (ENDPOINT)",
         PRIOR_SOFT, PRIOR_STRICT, "ENDPOINT")

# ------------------------------------------------------------ 10. combination
F = "10. combination (priorcoach)"
for w in ("ENDPOINT", "EARLY", "MID", "LATE"):
    contrast(F, "priorcoach(const) - prior (held-out, %s)" % w, PC_CONST, PRIOR_HO, w)
for w in ("ENDPOINT", "EARLY", "MID", "LATE"):
    contrast(F, "priorcoach(const) - coach(const) (held-out, %s)" % w, PC_CONST, COACH_CONST, w)
contrast(F, "priorcoach(anneal) - prior (held-out, ENDPOINT)", PC_ANN, PRIOR_HO, "ENDPOINT")
contrast(F, "priorcoach(anneal) - prior (train, ENDPOINT)", PC_ANN_TR, PRIOR_TR, "ENDPOINT")

# ------------------------------------------------------------- 11. interaction
F = "11. EARLY-minus-ENDPOINT interaction"
contrast_win_interaction(F, "[priorcoach(const)-coach(const)]@EARLY - @ENDPOINT",
                         PC_CONST, COACH_CONST, "EARLY", "ENDPOINT")

# -------------------------------------------------------------- 12. dispersion
DISP_CONDS = [
    ("blank", BLANK_HO),
    ("prior kept", PRIOR_HO),
    ("prior soft-withdrawn", PRIOR_SOFT),
    ("prior strict-withdrawn", PRIOR_STRICT),
    ("coach constant", COACH_CONST),
    ("coach annealed", COACH_ANN),
    ("coach abrupt-withdrawn", COACH_ABRUPT),
]
DISP = []
for name, spec in DISP_CONDS:
    for body in ("elbow", "finger"):
        seeds, v = series(spec, body, "ENDPOINT")
        DISP.append((name, body, len(v), float(np.mean(v)),
                     float(np.std(v, ddof=1)), float(np.min(v)), float(np.max(v))))

# ------------------------------------------------------------------- reporting
UNIT = {"elbow": "mrad", "finger": "mm"}


def fmt(x):
    return "%.3f" % x


lines = []
A = lines.append
A("# RECOMPUTED — authoritative statistical recomputation")
A("")
A("Generated by `recompute_all.py`. Supersedes every previously reported interval in this study.")
A("")
A("## Method")
A("")
A("- **Unit of analysis:** one training seed. Conditions share matched seed ids "
  "(seed0..seed11); every contrast is computed on **paired differences over matched seeds**.")
A("- **Metric:** `eval_dist` from each run's eval-checkpoint JSON, multiplied by 1000. "
  "Elbow bodies are reported in **milliradians (mrad)**; finger bodies in **millimetres (mm)**. "
  "Bodies are never pooled in a single contrast.")
A("- **Windows:** ENDPOINT = last checkpoint (step 12000, index 5). "
  "EARLY = mean of checkpoint indices 0,1,2 (steps 2k/4k/6k). "
  "MID = mean of indices 2,3,4 (6k/8k/10k). LATE = mean of indices 4,5 (10k/12k).")
A("- **PRIMARY interval — paired t 95% CI:** `mean(d) ± t_{0.975, df} · sd(d)/√n` "
  "with **df = n − 1**. For n = 12, t_{0.975,11} = 2.20099. "
  "(The previously used normal-approximation multiplier z = 1.96 yields intervals that are "
  "only 89.1% of the correct half-width, i.e. the correct t interval is 12.3% wider.)")
A("- **SENSITIVITY interval — BCa bootstrap 95% CI:** bias-corrected and accelerated "
  "percentile bootstrap, **20 000 resamples, RNG seed 0**, resampling **seeds** "
  "(i.e. the vector of paired differences) — never timesteps. Acceleration from the "
  "leave-one-seed-out jackknife.")
A("- **Verdict flag:** `ROBUST` = both intervals exclude zero with the same sign; "
  "`ROBUST-NULL` = both intervals contain zero; `METHOD-DEPENDENT` = the two intervals disagree.")
A("- Difference-of-difference contrasts (family 6 direct contrast, family 11 interaction) "
  "are formed **within seed** before any interval is computed.")
A("")
A("## Condition → data mapping")
A("")
A("| label | directory | condition key |")
A("|---|---|---|")
for nm, sp in [("blank (held-out)", BLANK_HO), ("prior kept (held-out)", PRIOR_HO),
               ("blank (train)", BLANK_TR), ("prior kept (train)", PRIOR_TR),
               ("coach CONSTANT", COACH_CONST), ("coach ANNEALED (held-out)", COACH_ANN),
               ("coach ANNEALED (train)", COACH_ANN_TR),
               ("coach ABRUPT-withdrawn @8k", COACH_ABRUPT),
               ("priorcoach CONSTANT", PC_CONST),
               ("priorcoach ANNEALED (held-out)", PC_ANN),
               ("priorcoach ANNEALED (train)", PC_ANN_TR),
               ("prior SOFT-withdrawn @8k", PRIOR_SOFT),
               ("prior STRICT-withdrawn @8k (buffer purged)", PRIOR_STRICT),
               ("randprior (train)", RANDPRIOR_TR)]:
    A("| %s | `%s_{elbow,finger}/` | `%s` |" % (nm, sp[0], sp[1]))
A("")
A("## Results")
A("")
for fam in FAMILIES:
    A("### %s" % fam)
    A("")
    A("| contrast | body | unit | n | mean diff | paired t 95% CI (PRIMARY) | BCa 95% CI (sensitivity) | flag |")
    A("|---|---|---|---|---|---|---|---|")
    for r in ROWS:
        if r[0] != fam:
            continue
        _, lab, body, n, m, tlo, thi, blo, bhi, fl = r
        A("| %s | %s | %s | %d | %s | [%s, %s] | [%s, %s] | %s |"
          % (lab, body, UNIT[body], n, fmt(m), fmt(tlo), fmt(thi), fmt(blo), fmt(bhi), fl))
    A("")

# summary
rob = [r for r in ROWS if r[9] == "ROBUST"]
nul = [r for r in ROWS if r[9] == "ROBUST-NULL"]
md = [r for r in ROWS if r[9] == "METHOD-DEPENDENT"]

A("## Summary")
A("")
A("Total contrasts computed: **%d**  (ROBUST %d / ROBUST-NULL %d / METHOD-DEPENDENT %d)"
  % (len(ROWS), len(rob), len(nul), len(md)))
A("")
A("### (a) ROBUST contrasts with a non-zero effect")
A("")
if rob:
    A("| contrast | body | unit | n | mean diff | paired t 95% CI | BCa 95% CI |")
    A("|---|---|---|---|---|---|---|")
    for r in rob:
        _, lab, body, n, m, tlo, thi, blo, bhi, _f = r
        A("| %s | %s | %s | %d | %s | [%s, %s] | [%s, %s] |"
          % (lab, body, UNIT[body], n, fmt(m), fmt(tlo), fmt(thi), fmt(blo), fmt(bhi)))
else:
    A("_none_")
A("")
A("### (b) METHOD-DEPENDENT contrasts")
A("")
if md:
    A("| contrast | body | unit | n | mean diff | paired t 95% CI | BCa 95% CI | t excludes 0? | BCa excludes 0? |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in md:
        _, lab, body, n, m, tlo, thi, blo, bhi, _f = r
        te = "yes" if (tlo > 0 or thi < 0) else "no"
        be = "yes" if (blo > 0 or bhi < 0) else "no"
        A("| %s | %s | %s | %d | %s | [%s, %s] | [%s, %s] | %s | %s |"
          % (lab, body, UNIT[body], n, fmt(m), fmt(tlo), fmt(thi), fmt(blo), fmt(bhi), te, be))
else:
    A("_none_")
A("")
A("### (c) ROBUST-NULL contrasts")
A("")
A("| contrast | body | unit | n | mean diff | paired t 95% CI | BCa 95% CI |")
A("|---|---|---|---|---|---|---|")
for r in nul:
    _, lab, body, n, m, tlo, thi, blo, bhi, _f = r
    A("| %s | %s | %s | %d | %s | [%s, %s] | [%s, %s] |"
      % (lab, body, UNIT[body], n, fmt(m), fmt(tlo), fmt(thi), fmt(blo), fmt(bhi)))
A("")
A("### (d) Dispersion — per-condition ENDPOINT SD (not a contrast)")
A("")
A("Sample SD (ddof = 1) across the 12 seeds at the final checkpoint.")
A("")
A("| condition | body | unit | n | mean | SD | min | max |")
A("|---|---|---|---|---|---|---|---|")
for name, body, n, mu, sd, lo, hi in DISP:
    A("| %s | %s | %s | %d | %s | %s | %s | %s |"
      % (name, body, UNIT[body], n, fmt(mu), fmt(sd), fmt(lo), fmt(hi)))
A("")
A("### (e) Contrasts that could not be computed")
A("")
if SKIPPED:
    for s in SKIPPED:
        A("- %s" % s)
else:
    A("_none — every requested contrast was computed._")
A("")

# GUARD: importing this module must NOT overwrite RECOMPUTED.md.
# It has silently truncated the authoritative table twice (682 -> 262 lines both
# times), destroying the previously-legacy appendix, Appendix B (Wilcoxon / CV /
# SD ratios) and Appendix C (Holm-28, per-checkpoint) -- and NO script in this
# repo can rebuild Appendix C. figstats.py / make_figs.py / extra_stats.py all
# import this module for its estimators, which is a legitimate use; only a direct
# run may write. Do not remove this guard.
if __name__ == "__main__":
    out_path = os.path.join(ROOT, "RECOMPUTED.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", out_path)
print("contrasts: %d total | ROBUST %d | ROBUST-NULL %d | METHOD-DEPENDENT %d"
      % (len(ROWS), len(rob), len(nul), len(md)))
print("\nMETHOD-DEPENDENT:")
for r in md:
    print("  - %s [%s] mean=%.3f t=[%.3f,%.3f] bca=[%.3f,%.3f]"
          % (r[1], r[2], r[4], r[5], r[6], r[7], r[8]))
print("\nDISPERSION (endpoint SD):")
for name, body, n, mu, sd, lo, hi in DISP:
    print("  %-26s %-7s mean=%8.3f sd=%8.3f" % (name, body, mu, sd))
if SKIPPED:
    print("\nSKIPPED:")
    for s in SKIPPED:
        print("  -", s)
