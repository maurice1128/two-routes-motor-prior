"""
figstats.py -- shared statistical layer for make_figs.py.

Every interval drawn in the figure set comes from here, and every interval is
the PAPER'S DECLARED METHOD:

  PRIMARY      paired t 95% CI, mean(d) +/- t_{0.975, n-1} * sd(d)/sqrt(n), df = n-1 = 11
  SENSITIVITY  BCa bootstrap 95% CI (20000 resamples, RNG seed 0, resampling seeds)
  FLAG         ROBUST / ROBUST-NULL / METHOD-DEPENDENT

The three estimator functions (`t_ci`, `bca_ci`, `flag`) are IMPORTED from
`recompute_all.py`, never reimplemented, so the figures cannot drift from
`RECOMPUTED.md`.

NOTE on the import: executing `recompute_all.py` rewrites RECOMPUTED.md from
scratch (dropping the appendix that `recompute_legacy.py` appends).  We snapshot
the file before the import and restore it byte-for-byte afterwards, so importing
the estimators is side-effect free.
"""
import os
import sys
import importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC_PATH = os.path.join(HERE, "RECOMPUTED.md")
FLAGS = ("ROBUST", "ROBUST-NULL", "METHOD-DEPENDENT")
BODIES = ("elbow", "finger")
UNIT = {"elbow": "mrad", "finger": "mm"}


# ------------------------------------------------------- parse RECOMPUTED.md
def _interval(cell):
    a, b = cell.strip().strip("[]").split(",")
    return float(a), float(b)


def parse_recomputed(path=REC_PATH):
    """(contrast label, body) -> dict(mean, tlo, thi, blo, bhi, flag, n).

    Picks up every row of every RESULTS table (main + appendix); the summary
    tables have a different column layout and are skipped automatically because
    column 7 is not a flag there.
    """
    rec = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 8 or cells[1] not in BODIES or cells[7] not in FLAGS:
                continue
            tlo, thi = _interval(cells[5])
            blo, bhi = _interval(cells[6])
            rec[(cells[0], cells[1])] = dict(
                mean=float(cells[4]), tlo=tlo, thi=thi, blo=blo, bhi=bhi,
                flag=cells[7], n=int(cells[3]))
    return rec


def parse_dispersion(path=REC_PATH):
    """(condition, body) -> dict(n, mean, sd, min, max) from section (d)."""
    out = {}
    inside = False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("### (d)"):
                inside = True
                continue
            if inside and line.startswith("### "):
                break
            if not inside or not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 8 or cells[1] not in BODIES:
                continue
            try:
                out[(cells[0], cells[1])] = dict(
                    n=int(cells[3]), mean=float(cells[4]), sd=float(cells[5]),
                    min=float(cells[6]), max=float(cells[7]))
            except ValueError:
                continue
    return out


REC = parse_recomputed()
DISP = parse_dispersion()

# ------------------------------- import the estimators (side-effect free) ---
# NOTE: this used to snapshot RECOMPUTED.md and write it back after loading
# recompute_all, because that module wrote the file at import time. It now
# carries a __main__ guard, and exec_module runs its body under the name
# "recompute_all", so it writes nothing here. The restore was therefore
# defending against something that can no longer happen -- while being ITSELF
# an unguarded module-scope write to the tracked artefact, invisible to any
# content check because it restored the bytes exactly and only moved mtime.
# Removed. Do not reintroduce: verify by MTIME, not by checksum, since a
# writer that restores its own bytes is invisible to a checksum.
_spec = importlib.util.spec_from_file_location(
    "recompute_all", os.path.join(HERE, "recompute_all.py"))
RA = importlib.util.module_from_spec(_spec)
sys.modules["recompute_all"] = RA
_spec.loader.exec_module(RA)

t_ci = RA.t_ci
bca_ci = RA.bca_ci
flag = RA.flag
series = RA.series
load = RA.load

# ------------------------------------------------------------ condition specs
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


def vals(spec, body, w):
    return series(spec, body, w)[1]


def curve(spec, body):
    """(n_seed, n_step) matrix of eval_dist*1000, seeds sorted."""
    d = load(spec[0], spec[1], body)
    return np.array([d[s] for s in sorted(d)])


def _paired(specs, body, w):
    got = [series(s, body, w) for s in specs]
    common = sorted(set.intersection(*[set(g[0]) for g in got]))
    maps = [dict(zip(g[0], g[1])) for g in got]
    return common, maps


def d_diff(A, B, body, w):
    common, m = _paired([A, B], body, w)
    return np.array([m[0][s] - m[1][s] for s in common])


def d_dd(A1, B1, A2, B2, body, w):
    common, m = _paired([A1, B1, A2, B2], body, w)
    return np.array([(m[0][s] - m[1][s]) - (m[2][s] - m[3][s]) for s in common])


def d_mid(PC, PR, CO, body, w):
    """priorcoach - 0.5*(prior + coach), per seed."""
    common, m = _paired([PC, PR, CO], body, w)
    return np.array([m[0][s] - 0.5 * (m[1][s] + m[2][s]) for s in common])


def d_win(A, B, body, w1, w2):
    """(A-B)@w1 - (A-B)@w2, per seed."""
    c1, m1 = _paired([A, B], body, w1)
    c2, m2 = _paired([A, B], body, w2)
    common = sorted(set(c1) & set(c2))
    return np.array([(m1[0][s] - m1[1][s]) - (m2[0][s] - m2[1][s]) for s in common])


# ------------------------------------------------- registry + verification ---
RES = {}        # (label, body) -> result dict
CHECKS = []     # verification rows
MEAN_CHECKS = []

TOL_MEAN = 0.02
TOL_CI = 0.05


def register(label, body, d, in_rec=True):
    """Compute the paired t CI + BCa + flag, verify against RECOMPUTED.md."""
    m, tlo, thi = t_ci(d)
    _, blo, bhi = bca_ci(d)
    fl = flag(tlo, thi, blo, bhi)
    # the SUPERSEDED z interval, kept only to report which cells lose their
    # significance when the declared paired t interval is used instead
    se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
    zlo, zhi = m - 1.96 * se, m + 1.96 * se
    z_sig = (zlo > 0) or (zhi < 0)
    t_sig = (tlo > 0) or (thi < 0)
    r = dict(label=label, body=body, n=len(d), mean=m, tlo=tlo, thi=thi,
             blo=blo, bhi=bhi, flag=fl, in_rec=in_rec,
             zlo=zlo, zhi=zhi, z_sig=z_sig, t_sig=t_sig,
             changed=(z_sig and not t_sig))
    ref = REC.get((label, body))
    if in_rec:
        if ref is None:
            status = "MISSING-FROM-RECOMPUTED"
        else:
            ok = (abs(m - ref["mean"]) <= max(TOL_MEAN, abs(ref["mean"]) * 0.002)
                  and abs(tlo - ref["tlo"]) <= TOL_CI
                  and abs(thi - ref["thi"]) <= TOL_CI
                  and fl == ref["flag"])
            status = "MATCH" if ok else "MISMATCH"
    else:
        status = "NOT-IN-RECOMPUTED (computed here, same estimator)"
    r["status"] = status
    r["ref"] = ref
    CHECKS.append(r)
    RES[(label, body)] = r
    return r


def C(label, body):
    return RES[(label, body)]


def cond_mean(label, body, arr, rec_name=None, paper_mean=None):
    """One-sample t 95% CI (df = n-1) on a condition's raw endpoint values."""
    m, lo, hi = t_ci(arr)
    ref = DISP.get((rec_name, body)) if rec_name else None
    if ref is not None:
        status = "MATCH" if abs(m - ref["mean"]) <= max(TOL_MEAN, abs(ref["mean"]) * 0.002) \
            else "MISMATCH"
        exp = ref["mean"]
        src = "RECOMPUTED (d)"
    elif paper_mean is not None:
        status = "MATCH" if abs(m - paper_mean) <= max(0.15, abs(paper_mean) * 0.02) \
            else "MISMATCH"
        exp = paper_mean
        src = "PAPER point mean"
    else:
        status = "NOT-IN-RECOMPUTED (point mean)"
        exp = None
        src = "-"
    MEAN_CHECKS.append(dict(label=label, body=body, mean=m, lo=lo, hi=hi,
                            expected=exp, src=src, status=status))
    return m, lo, hi


def sd_ratio_ci(blank, cond, nboot=20000, seed=0):
    """Percentile bootstrap CI for SD(blank)/SD(cond); resamples seeds."""
    rng = np.random.default_rng(seed)
    i1 = rng.integers(0, len(blank), size=(nboot, len(blank)))
    i2 = rng.integers(0, len(cond), size=(nboot, len(cond)))
    r = blank[i1].std(axis=1, ddof=1) / cond[i2].std(axis=1, ddof=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return (float(blank.std(ddof=1) / cond.std(ddof=1)), float(lo), float(hi))


def n_bad():
    return sum(1 for c in CHECKS if c["status"] in ("MISMATCH", "MISSING-FROM-RECOMPUTED")) \
        + sum(1 for c in MEAN_CHECKS if c["status"] == "MISMATCH")
