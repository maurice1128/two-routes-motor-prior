r"""recompute_appendixh.py -- Appendix H of RECOMPUTED.md.

Five things the r13 draft asserted without a test, imposed asymmetrically, or
could not see at all:

  H1  **The elbow "soft-withdrawal inversion" was never tested.**  Sec. 4.6
      and the abstract report that the elbow SOFT prior withdrawal suppresses
      dispersion more than its own KEPT counterpart (SD ratio 3.10 vs 2.78)
      and say "we have no account of that inversion"; Sec. 4.6 adds that it
      "survives the correction".  Two SD point estimates compared with no
      within-axis test is the inference Sec. 1.2 forbids and item 69 had just
      corrected for the randprior/prior pair -- in the same abstract sentence
      that carries item 69's own correction.  The paper's own G3 machinery
      (Pitman-Morgan + paired bootstrap of the ratio-of-ratios) is run here
      one pair over, on the twelve shared held-out seeds.  **There is no
      inversion**, and nothing "survives the correction" because no
      correction was ever applied to these cells.

  H2  **"Neither arm learns myoHand in 60k steps" is a null asserted from
      levels alone.**  Asserted in the abstract, Sec. 4.11, Sec. 6 and Sec. 7
      from "102.5 at 2k and 106.3 at 60k".  Sec. 6's own uniformly-applied
      rule says an undetected change is not an absence, so the within-arm
      change is tested here.  The fix strengthens the claim.

  H3  **Sec. 4.11 inverted the paper's primary-statistic policy on the one
      body where the primary statistic is unfavourable.**  Sec. 3.6 refuses to
      correct per-checkpoint cells ("would need a correction scheme we have
      not run") and the finger 4k/6k/8k cells are asserted affirmatively with
      no correction -- while Sec. 4.11 BUILDS a 30-checkpoint grid and uses it
      to kill the myoHand 12k cell.  The grid is also unnecessary: the cell
      fails the paper's ORDINARY window family anyway (K = 5, Holm).

  H4  **A fourth MyoSuite body was instantiated and trained on.**
      `gonogo_myoarm.py` calls `gym.make("myoArmReachRandom-v0")` and trains
      this study's SAC on it; `gonogo_myoarm.log` archives the run.  `grep
      myoArm` returned zero hits in both documents while Sec. 4.11 asserted
      "Three MyoSuite bodies were run".  It carries no claim -- the defect is
      purely the count.  Seventh recurrence of run-but-unreported.

  H5  **The completeness register's METHOD is structurally blind.**  G5
      enumerates `results*` directories.  `gonogo_myoarm` writes stdout only
      and creates no directory, so the method cannot see it -- nor
      `scratch_buildprior.log`, which records `collected 300000 transitions`
      and `k=5 fingertip position error: mean 8.5 mm, median 7.5 mm` for
      `prior_hand.pt`, neither number appearing in either document, making
      Sec. 3.3's "10^5-2x10^5 transitions" false for the third body the paper
      now reports.  There is also NO build log anywhere for `prior_myo.pt`,
      so the elbow's ~19 mrad fidelity figure and its babble budget have no
      archived transcript -- and that figure is load-bearing in Sec. 5's
      fourth qualification.  Every completeness statement is rescoped to what
      the method can support: *no unreported artefact remains THAT WRITES A
      FILE INTO THIS TREE*.

Append-only, exactly like `recompute_hand.py`, `movement_index_holm72.py`,
`sweep_selective.py` and `recompute_addendum.py`: this script never opens
RECOMPUTED.md in write mode and refuses to append if its marker heading is
already present.  Importing it writes nothing (`__main__` guard below).
Verify by MTIME, not by checksum: a writer that restores its own bytes is
invisible to a checksum.

Usage:  python recompute_appendixh.py            # print only
        python recompute_appendixh.py --append   # print and append
"""

import glob
import math
import os
import re
import sys

import numpy as np

import movement_index_holm72 as MI
import recompute_hand as RH

ROOT = os.path.dirname(os.path.abspath(__file__))
REC_PATH = os.path.join(ROOT, "RECOMPUTED.md")
MARKER = ("# Appendix H — the alleged dispersion inversion tested, myoHand's no-learning "
          "claim tested, the myoHand window family, the fourth body, and the register's "
          "blind spot")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ------------------------------------------------------------------ estimators
def pitman_morgan(x, y):
    """Paired variance-ratio test: H0 var(x) = var(y).  Same routine as G3."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    u, v = x + y, x - y
    rho = float(np.corrcoef(u, v)[0, 1])
    t = rho * math.sqrt(n - 2) / math.sqrt(1.0 - rho * rho)
    return rho, t, MI.t_sf(t, n - 2), n


def rr_boot(x, y, seed=0, nboot=20000):
    """Paired percentile bootstrap of SD(x)/SD(y), resampling seeds JOINTLY."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    rng = np.random.default_rng(seed)          # fresh generator per pair, as in G3
    idx = rng.integers(0, n, size=(nboot, n))
    bs = x[idx].std(axis=1, ddof=1) / y[idx].std(axis=1, ddof=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(x.std(ddof=1) / y.std(ddof=1)), float(lo), float(hi)


# ================================================================== H1
# Every cell is the HELD-OUT ENDPOINT, the target set and window Sec. 4.6's
# dispersion table uses.  `soft` is the reference arm in every pair, so the
# ratio-of-ratios is SD(other)/SD(soft) = [SD(blank)/SD(soft)] / [SD(blank)/SD(other)],
# i.e. exactly the ratio of the two SD ratios Sec. 4.6 prints.
H1_PAIRS = [("prior kept", MI.PRIOR_HO, ("elbow", "finger")),
            ("prior strict-withdrawn", MI.PRIOR_STRICT, ("elbow", "finger")),
            ("coach abrupt-withdrawn", MI.COACH_ABRUPT, ("elbow", "finger"))]


def h1():
    out = []
    for name, spec, bodies in H1_PAIRS:
        for body in bodies:
            S = MI.ser(MI.PRIOR_SOFT, body, "ENDPOINT")
            O = MI.ser(spec, body, "ENDPOINT")
            Bl = MI.ser(MI.BLANK_HO, body, "ENDPOINT")
            seeds = sorted(set(S) & set(O) & set(Bl))
            s = np.array([S[k] for k in seeds])
            o = np.array([O[k] for k in seeds])
            b = np.array([Bl[k] for k in seeds])
            rho, t, p, n = pitman_morgan(o, s)
            rr, lo, hi = rr_boot(o, s)
            out.append(dict(other=name, body=body, n=n,
                            sd_blank=float(b.std(ddof=1)),
                            sd_other=float(o.std(ddof=1)),
                            sd_soft=float(s.std(ddof=1)),
                            ratio_other=float(b.std(ddof=1) / o.std(ddof=1)),
                            ratio_soft=float(b.std(ddof=1) / s.std(ddof=1)),
                            rr=rr, lo=lo, hi=hi, rho=rho, t=t, p=p,
                            excl=bool(lo > 1.0 or hi < 1.0)))
    return out


# ================================================================== H2 / H3
HAND_SEEDS = list(range(6))


def hand_tables():
    B = {s: RH.hand_load("blank", s) for s in HAND_SEEDS}
    P = {s: RH.hand_load("prior", s) for s in HAND_SEEDS}
    steps = sorted(B[0])
    first5, last5 = steps[:5], steps[-5:]

    def within(D, w2, w1):
        d = np.array([np.mean([D[s][w] for w in w2]) - np.mean([D[s][w] for w in w1])
                      for s in HAND_SEEDS])
        n, m, lo, hi, p = RH.paired(d)
        return dict(n=n, mean=m, lo=lo, hi=hi, p=p, pos=int((d > 0).sum()))

    h2 = [("blank", "60k − 2k", within(B, [60000], [2000])),
          ("prior", "60k − 2k", within(P, [60000], [2000])),
          ("blank", "mean(last 5 ckpt) − mean(first 5 ckpt)", within(B, last5, first5)),
          ("prior", "mean(last 5 ckpt) − mean(first 5 ckpt)", within(P, last5, first5))]

    def cell(ws):
        a = np.array([np.mean([P[s][w] for w in ws]) for s in HAND_SEEDS])
        b = np.array([np.mean([B[s][w] for w in ws]) for s in HAND_SEEDS])
        n, m, lo, hi, p = RH.paired(a - b)
        return dict(n=n, mean=m, lo=lo, hi=hi, p=p, pos=int(((a - b) > 0).sum()))

    windows = [("EARLY {2,4,6k}", [2000, 4000, 6000]),
               ("MID {6,8,10k}", [6000, 8000, 10000]),
               ("LATE {10,12k}", [10000, 12000]),
               ("ENDPOINT-of-main {12k}", [12000]),
               ("ENDPOINT-of-run {60k}", [60000])]
    rows = [dict(label=nm, **cell(ws)) for nm, ws in windows]
    rs, surv, stop = RH.holm([dict(r) for r in rows])

    # the per-checkpoint grid Sec. 4.11 built, for the disclosure
    per = [(w, cell([w])) for w in steps]
    grid_sig = [(w, c["p"]) for w, c in per if c["p"] < 0.05]
    gk = [dict(label="checkpoint %dk" % (w // 1000), p=c["p"]) for w, c in per]
    _, g_surv, g_stop = RH.holm(gk)
    return dict(rows=rs, surv=surv, stop=stop, h2=h2,
                grid_n=len(per), grid_sig=grid_sig, grid_surv=g_surv, grid_stop=g_stop,
                first5=first5, last5=last5)


# ================================================================== H4
def myoarm():
    """Parse the archived stdout transcript of the fourth body."""
    path = os.path.join(ROOT, "gonogo_myoarm.log")
    src = os.path.join(ROOT, "gonogo_myoarm.py")
    with open(path, encoding="utf-8", errors="replace") as fh:
        log = fh.read()
    with open(src, encoding="utf-8", errors="replace") as fh:
        code = fh.read()
    head = re.search(r"myoArm: obs (\d+) act (\d+) muscles (\d+)", log)
    rows = re.findall(r"step\s+(\d+)\s+eval tip-dist ([\d.]+) m \(init ~([\d.]+)\)", log)
    env = re.search(r'gym\.make\("([^"]+)"\)', code)
    hidden = re.search(r"hidden=(\d+)", code)
    total = re.search(r"total, start = (\d+), (\d+)", code)
    return dict(env=env.group(1) if env else "?",
                obs=head.group(1), act=head.group(2), muscles=head.group(3),
                rows=[(int(a), float(b)) for a, b, _ in rows],
                init=float(rows[0][2]) if rows else float("nan"),
                hidden=hidden.group(1) if hidden else "?",
                declared=int(total.group(1)) if total else -1,
                killed=int(rows[-1][0]) if rows else -1,
                exists_dir=os.path.isdir(os.path.join(ROOT, "results_myoarm")))


# ================================================================== H5
BUILD_TARGETS = [("prior_myo.pt", "myoElbow", "elbow ≈19 mrad (§3.3)"),
                 ("prior_finger.pt", "myoFinger", "finger 14.3 mm mean / 9.8 mm median (§3.3)"),
                 ("prior_hand.pt", "myoHand", "not stated in either document before this appendix")]


def build_logs():
    """Which frozen priors have an archived build transcript, and what it says."""
    found = {}
    for path in glob.glob(os.path.join(ROOT, "*.log")):
        with open(path, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        m = re.search(r"saved prior -> (\S+)", txt)
        if not m:
            continue
        n = re.search(r"collected (\d+) transitions", txt)
        f = re.search(r"k=5 fingertip position error: mean ([\d.]+) mm\s+median ([\d.]+) mm", txt)
        found[m.group(1)] = dict(log=os.path.basename(path),
                                 n=int(n.group(1)) if n else None,
                                 mean=float(f.group(1)) if f else None,
                                 median=float(f.group(2)) if f else None)
    rows = []
    for pt, body, quoted in BUILD_TARGETS:
        rows.append(dict(pt=pt, body=body, quoted=quoted,
                         on_disk=os.path.isfile(os.path.join(ROOT, pt)),
                         **(found.get(pt) or dict(log=None, n=None, mean=None, median=None))))
    return rows


def nondir_artefacts():
    """Artefacts a `results*`-directory enumeration cannot see, by construction."""
    return [("`gonogo_myoarm.log` + `gonogo_myoarm.err`",
             "a fourth MyoSuite body (`myoArmReachRandom-v0`) instantiated and trained "
             "on with this study's SAC; stdout only, **no directory is created**",
             "H4 / §4.12"),
            ("`scratch_buildprior.log`",
             "the `prior_hand.pt` build: **300 000** babble transitions and a k=5 "
             "fingertip error of **8.5 mm mean / 7.5 mm median**",
             "H5 / §3.3"),
            ("`build_prior_finger.log`",
             "the `prior_finger.pt` build: 200 000 transitions, 14.3 mm / 9.8 mm",
             "already in §3.3 (fidelity only; the transition count was not attributed)"),
            ("*(absent)* — no build log for `prior_myo.pt`",
             "the elbow prior's babble budget and its ≈19 mrad fidelity figure have "
             "**no archived transcript at all**",
             "H5 / §3.3, §5")]


# ================================================================== render
def f(x, nd=4):
    """Signed fixed-point with the typographic minus the rest of the file uses."""
    return (("%+." + str(nd) + "f") % x).replace("-", "−")


def render():
    L = []
    A = L.append
    H1 = h1()
    HD = hand_tables()
    MA = myoarm()
    BL = build_logs()

    A("")
    A(MARKER)
    A("")
    A("Produced by `recompute_appendixh.py`. Append-only; `recompute_hand.py`'s estimators "
      "and `movement_index_holm72.py`'s data layer are imported unchanged, and both carry "
      "`__main__` guards, so importing them writes nothing. **Verify by mtime, not by "
      "checksum** — a writer that restores its own bytes is invisible to a checksum.")
    A("")

    # ---------------------------------------------------------------- H1
    A("## H1. The elbow \"soft-withdrawal inversion\" — tested, and it is a null")
    A("")
    A("§4.6, abstract clause (v) and §1.1 claim 5 report that the elbow **soft** prior "
      "withdrawal is *\"the strongest suppressor of any withdrawn arm … and stronger than "
      "its own kept counterpart's 2.78\"*, that *\"we have **no account of that "
      "inversion**\"*, and — §4.6 only — that what is anomalous *\"**survives the "
      "correction**\"*. Both halves are defects, and they are the two species this "
      "appendix's immediate predecessor had just corrected elsewhere:")
    A("")
    A("- **3.10 against 2.78 is two SD point estimates compared with no within-axis "
      "test.** That is the inference §1.2 lists as one of the three things this paper "
      "commits to avoiding, and item 69 ran the missing test for the `randprior`/`prior` "
      "pair — **in the same abstract sentence that carries the inversion claim**. The "
      "machinery was already written, in G3; it was simply never pointed at this pair.")
    A("- **Nothing \"survives the correction\", because no correction was ever applied "
      "to these cells.** §3.6 excludes dispersion ratios from every Holm family in this "
      "paper — they are ratios of sample SDs, not paired-difference contrasts, and have "
      "no paired-t p-value the procedure could take — and G3's own closing paragraph "
      "says so of the Pitman–Morgan cells as well. The phrase asserts a correction that "
      "does not exist for this quantity.")
    A("")
    A("The test, run with G3's own routine one pair over: **Pitman–Morgan** on the "
      "twelve shared held-out seeds (u = x + y, v = x − y, H0: var(x) = var(y) ⇔ "
      "corr(u, v) = 0, t = r√(n−2)/√(1−r²) on n − 2 df), and a **paired percentile "
      "bootstrap of the ratio-of-ratios**, 20 000 resamples, a fresh `default_rng(0)` per "
      "pair, resampling the twelve seeds jointly. The soft arm is the reference in every "
      "row, so the ratio-of-ratios is SD(other)/SD(soft) = "
      "[SD(blank)/SD(soft)] ÷ [SD(blank)/SD(other)] — exactly the ratio of the two SD "
      "ratios §4.6 prints. All cells are held-out ENDPOINT, n = 12.")
    A("")
    A("| comparison | body | SD(blank) | SD(other) | SD(soft) | SD ratio, other | SD ratio, soft | ratio-of-ratios | paired bootstrap 95% CI | excludes 1? | Pitman–Morgan r | t (df 10) | **p** |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in H1:
        A("| soft vs **%s** | %s | %.2f | %.2f | %.2f | %.2f | %.2f | **%.3f** | [%.3f, %.3f] | %s | %s | %s | **%.5f** |"
          % (r["other"], r["body"], r["sd_blank"], r["sd_other"], r["sd_soft"],
             r["ratio_other"], r["ratio_soft"], r["rr"], r["lo"], r["hi"],
             "**yes**" if r["excl"] else "no", f(r["rho"], 4), f(r["t"], 4), r["p"]))
    A("")
    kept_e = [r for r in H1 if r["other"] == "prior kept" and r["body"] == "elbow"][0]
    kept_f = [r for r in H1 if r["other"] == "prior kept" and r["body"] == "finger"][0]
    strict_e = [r for r in H1 if r["other"].startswith("prior strict") and r["body"] == "elbow"][0]
    abrupt_e = [r for r in H1 if r["other"].startswith("coach abrupt") and r["body"] == "elbow"][0]
    A("**There is no inversion.** The elbow soft-versus-kept cell — the one the abstract, "
      "§1.1 claim 5 and §4.6 all carry — is **ratio-of-ratios %.3f [%.3f, %.3f], which "
      "includes 1, with Pitman–Morgan r = %s, t(10) = %s, p = %.5f**. The two arms' "
      "across-seed SDs (%.2f kept against %.2f soft) are not distinguishable at n = 12. "
      "The finger twin is the same verdict, **%.3f [%.3f, %.3f], p = %.5f**. A 3.10-versus-"
      "2.78 gap between two point estimates is what a null looks like on this axis, and "
      "the paper spent a paragraph, an abstract clause and a claim-block sentence "
      "declaring itself unable to explain it."
      % (kept_e["rr"], kept_e["lo"], kept_e["hi"], f(kept_e["rho"]), f(kept_e["t"]),
         kept_e["p"], kept_e["sd_other"], kept_e["sd_soft"],
         kept_f["rr"], kept_f["lo"], kept_f["hi"], kept_f["p"]))
    A("")
    A("**Half of the surrounding paragraph does hold under direct test, and it is the "
      "half worth keeping.** The soft arm is significantly tighter than **both** matched "
      "withdrawals on elbow: against the strict prior withdrawal **%.3f [%.3f, %.3f], "
      "p = %.5f**, and against the abrupt coach withdrawal **%.3f [%.3f, %.3f], "
      "p = %.5f**. So the defensible statement is *the soft-withdrawn arm is the tightest "
      "of the three withdrawn elbow arms, significantly tighter than either matched "
      "withdrawal, and not distinguishable from the kept arm it was withdrawn from* — "
      "which is not an inversion and needs no account. The corresponding finger cells are "
      "both null and are printed above for completeness."
      % (strict_e["rr"], strict_e["lo"], strict_e["hi"], strict_e["p"],
         abrupt_e["rr"], abrupt_e["lo"], abrupt_e["hi"], abrupt_e["p"]))
    A("")
    A("These cells are **post hoc**, are variance-ratio tests rather than paired-difference "
      "contrasts, and are therefore in no Holm family — the same scope G3's cells carry, "
      "and the reason the \"survives the correction\" phrase had nothing behind it.")
    A("")

    # ---------------------------------------------------------------- H2
    A("## H2. \"Neither arm learns myoHand in 60k steps\" — the null the paper asserted from levels")
    A("")
    A("The sentence appears in the abstract, §4.11, §6 and §7. Its evidence was a pair of "
      "levels — *\"`blank` runs 102.5 mm at 2k and 106.3 mm at 60k, `prior` 103.1 and "
      "99.6\"* — and no test. **§6's rule, which this paper says it applies uniformly, is "
      "that an undetected change is a failure to detect and not an absence**; the rule was "
      "honoured for every null in §4.5 and broken here, on a body where the absence is "
      "convenient. Tested, paired within arm over the six shared seeds, positive = error "
      "**rose** (the arm got worse):")
    A("")
    A("| arm | contrast | n | mean (mm) | paired t 95% CI | t p | seeds on the sign |")
    A("|---|---|---|---|---|---|---|")
    for nm, lab, r in HD["h2"]:
        A("| `%s` | %s | %d | **%s** | [%s, %s] | %.3f | %d/6 |"
          % (nm, lab, r["n"], f(r["mean"], 3), f(r["lo"], 3), f(r["hi"], 3), r["p"], r["pos"]))
    A("")
    A("Every interval contains zero and every interval is wide — the `blank` 60k−2k cell "
      "admits anything from a 7.1 mm improvement to a 14.9 mm deterioration. **The correct "
      "statement is therefore \"we detect no improvement on either arm over the budget\", "
      "with the intervals**, not \"neither arm learns\". The change is a strengthening, not "
      "a retreat: the claim now rests on a test rather than on two endpoints of an "
      "oscillating thirty-checkpoint series, and the verdict — that myoHand is "
      "uninformative about the forward-model route — is unchanged.")
    A("")
    A("Note that this is a **within-arm** contrast across time, and §6's own pairing "
      "bullet says the design pairs across conditions within a seed and *never across "
      "time*. These four cells are the only across-time contrasts in the paper; they are "
      "reported because the sentence they replace was an across-time claim already, made "
      "without an interval.")
    A("")

    # ---------------------------------------------------------------- H3
    A("## H3. The myoHand 12k cell under the paper's **ordinary** window family (K = 5)")
    A("")
    A("§3.6 states the primary-statistic policy — *\"a contrast is reported as significant "
      "only when the paired t interval excludes zero\"* — and refuses to correct the "
      "per-checkpoint cells, on the ground that doing so *\"would need a correction scheme "
      "we have not run\"*; the finger per-checkpoint cells at 4k, 6k and 8k are asserted "
      "affirmatively in the abstract and §1.1 claim 1 with no correction at all. **§4.11 "
      "then builds a 30-checkpoint grid and uses it to kill the one myoHand cell that "
      "reaches significance** (12k, `prior − blank` = +12.49 [+0.99, +23.99], Wilcoxon "
      "0.03125, 6/6 seeds on the sign). Per-checkpoint enumeration is thus exempted where "
      "it costs the paper a claim and imposed where it costs the paper an embarrassment, "
      "and the asymmetry was nowhere disclosed.")
    A("")
    A("**The grid is also unnecessary.** myoHand has exactly the same window structure "
      "every other body gets — EARLY, MID, LATE, the main study's 12k endpoint, and this "
      "run's own 60k endpoint — which is a five-cell family, and the cell fails it:")
    A("")
    A("| Holm rank | window | `prior − blank` (mm) | paired t 95% CI | **p** | threshold 0.05/(K−i+1) | verdict |")
    A("|---|---|---|---|---|---|---|")
    for r in HD["rows"]:
        A("| %d | %s | %s | [%s, %s] | **%.5f** | %.5f | **%s** |"
          % (r["_rank"], r["label"], f(r["mean"], 2), f(r["lo"], 2), f(r["hi"], 2),
             r["p"], r["_thr"], "SURVIVES" if r["_surv"] else "FAILS"))
    A("")
    st = HD["stop"]
    A("**Holm rank 1 is the 12k cell at p = %.5f against 0.05/5 = %.5f — it FAILS, and the "
      "step-down stops there, so nothing in the family survives.** The verdict §4.11 "
      "reached with a 30-cell grid it built for the purpose is reached by the family the "
      "paper applies to every other body, with a margin of %.1f×."
      % (HD["rows"][0]["p"], HD["rows"][0]["_thr"], HD["rows"][0]["p"] / HD["rows"][0]["_thr"]))
    A("")
    A("**The per-checkpoint grid is disclosed rather than deleted, and it gives the same "
      "verdict.** Over the %d archived checkpoints, %d reach p < 0.05 (%s) against a chance "
      "expectation of %.1f, and Holm over that grid stops at rank 1 as well. Both routes "
      "reach \"nothing survives\"; the window family is the one the paper's own policy "
      "licenses, and it is what §4.11 now uses. **What is stated plainly, in §4.11 and in "
      "§3.6, is the exemption/imposition asymmetry itself**: per-checkpoint cells are "
      "outside every correction family in this paper *except* on the one body where "
      "entering them removes an unfavourable result."
      % (HD["grid_n"], len(HD["grid_sig"]),
         ", ".join("%dk at %.4f" % (w // 1000, p) for w, p in HD["grid_sig"]),
         0.05 * HD["grid_n"]))
    A("")

    # ---------------------------------------------------------------- H4
    A("## H4. A **fourth** MyoSuite body was instantiated and trained on")
    A("")
    A("`gonogo_myoarm.py` calls `gym.make(\"%s\")` and trains **this study's SAC** "
      "(`from sac_dyna import SAC, Replay`) on it. `gonogo_myoarm.log` archives the run. "
      "Before this revision, `grep myoArm` returned **zero hits in both `PAPER_CLEAN.md` "
      "and `PAPER_TMLR.md`**, while §4.11 asserted *\"Three MyoSuite bodies were run\"*."
      % MA["env"])
    A("")
    A("| property | value |")
    A("|---|---|")
    A("| environment | `%s` |" % MA["env"])
    A("| observation / action dimension | %s / %s |" % (MA["obs"], MA["act"]))
    A("| muscles | %s |" % MA["muscles"])
    A("| reward | **MyoSuite's native reward**, not this study's reach reward |")
    A("| SAC width | `hidden=%s` (the study's own arms use 128) |" % MA["hidden"])
    A("| seeds | **1** (`seed=0`) |")
    A("| conditions | **`blank` only** — no `prior`, no `coach`, no Dyna |")
    A("| declared budget | %s steps |" % format(MA["declared"], ","))
    A("| actually reached | **%s steps** — truncated, `DONE` never printed |" % format(MA["killed"], ","))
    A("| metric | fingertip→target distance, metres |")
    A("| initial distance | %.2f m |" % MA["init"])
    A("| results directory | **none — the script writes stdout only** |")
    A("")
    A("| step | eval tip-distance (m) |")
    A("|---|---|")
    for s, d in MA["rows"]:
        A("| %s | %.3f |" % (format(s, ","), d))
    A("")
    A("**The defect is purely the count.** A single seed, one condition, a different "
      "network width, a different reward, no held-out target set and a truncated run "
      "support no contrast and no interval; nothing here can be entered into any family "
      "or compared with any cell in this paper. That is exactly what the paper already "
      "says of myoHand — and myoHand nevertheless gets a subsection, because *the "
      "sentences that count the bodies were false*. They were false by one more than the "
      "last correction made them. Every such sentence is corrected: **four MyoSuite "
      "bodies were instantiated and trained on, two carry the analysis, the third "
      "(myoHand) is reported and carries no claim, and the fourth (myoArm) is reported "
      "here and carries no claim.**")
    A("")
    A("This is the **seventh** recurrence of run-but-unreported — after `results_myo/`'s "
      "three arms (fifth, item 63, §4.10) and `results_hand/` (sixth, item 65, §4.11). "
      "The species has now recurred once per audit pass for three consecutive passes, and "
      "H5 is about why.")
    A("")

    # ---------------------------------------------------------------- H5
    A("## H5. The completeness register's **method** is structurally blind")
    A("")
    A("G5/F4b was produced by enumerating every `results*` **directory** on disk and "
      "checking each against the draft, and it closes with an unrestricted claim. "
      "**`gonogo_myoarm.py` writes stdout only and creates no directory**, so no run of "
      "that method could ever have found the fourth body — the method did not miss it, "
      "the method cannot see it. The same blindness covers every artefact that is a loose "
      "file rather than a results tree:")
    A("")
    A("| artefact invisible to a `results*` enumeration | what it records | now reported in |")
    A("|---|---|---|")
    for a, b, c in nondir_artefacts():
        A("| %s | %s | %s |" % (a, b, c))
    A("")
    A("**The build transcripts, enumerated.** Every frozen prior this paper loads, against "
      "the archived log that produced it:")
    A("")
    A("| checkpoint | body | on disk | build transcript | babble transitions | k=5 fingertip error | fidelity figure quoted in §3.3 |")
    A("|---|---|---|---|---|---|---|")
    for r in BL:
        A("| `%s` | %s | %s | %s | %s | %s | %s |"
          % (r["pt"], r["body"], "yes" if r["on_disk"] else "no",
             ("`%s`" % r["log"]) if r["log"] else "**none — no build log anywhere in the tree**",
             format(r["n"], ",") if r["n"] else "**unrecorded**",
             ("mean %.1f mm / median %.1f mm" % (r["mean"], r["median"])) if r["mean"] else "**unrecorded**",
             r["quoted"]))
    A("")
    hand = [r for r in BL if r["pt"] == "prior_hand.pt"][0]
    A("Two consequences, both load-bearing.")
    A("")
    A("1. **§3.3's \"10⁵–2×10⁵ transitions\" is false for the third body the paper now "
      "reports.** `prior_hand.pt` was built on **%s** transitions — 1.5× the top of that "
      "range — and its k=5 fingertip error is **%.1f mm mean, %.1f mm median**, better "
      "than myoFinger's 14.3 / 9.8. Neither number appeared in either document. §3.3's "
      "fidelity list gave elbow and finger only, while §4.11 reports a body built on a "
      "third prior whose budget the same subsection's range excludes. Both numbers are "
      "added to §3.3."
      % (format(hand["n"], ","), hand["mean"], hand["median"]))
    A("")
    A("2. **The elbow prior has no archived build transcript at all.** `prior_myo.pt` is "
      "on disk and is loaded by every elbow `prior`, `randprior` (via `build_randprior.py`) "
      "and `priorcoach` run in the study, and there is no log of its babbling or of its "
      "open-loop evaluation anywhere in the tree. So **§3.3's ≈19 mrad elbow fidelity "
      "figure, and the elbow babble budget that §3.5's 5×-more-raw-interaction argument "
      "rests on, are unarchived** — and that fidelity figure is load-bearing: §5's fourth "
      "qualification of the forward-model arm is *\"the elbow model's own ≈19 mrad "
      "open-loop prediction error is essentially the size of the −20.7 mrad effect "
      "attributed to its fidelity\"*, which is the paper's own strongest deflation of its "
      "own headline. It is stated in §3.3 that the figure has no archived transcript.")
    A("")
    A("**Every completeness statement is rescoped to what the method can support.** "
      "F4b/G5 close with an unrestricted claim over \"the repository\"; the enumeration "
      "behind it ranges over `results*` directories only. The supportable form is:")
    A("")
    A("> **No unreported artefact remains *that writes a file into this tree*** — where "
      "\"writes a file\" means a `results*` directory, a `*.log`, a `*.pt` or a `*.json` "
      "checked against both drafts. A run that writes only to a terminal is outside the "
      "method's reach, and a run whose transcript was never redirected to a file leaves "
      "no trace this or any future register can enumerate.")
    A("")
    A("That is weaker than what E5, F4 and F4b each asserted, and it is the first of the "
      "four registers whose scope matches its method. **The corrected register**, "
      "extending F4b to the non-directory artefacts:")
    A("")
    A("| artefact | kind | status in F4b/G5 | corrected status (H5) |")
    A("|---|---|---|---|")
    A("| every `results*` directory | directory | enumerated | unchanged — F4b's table stands |")
    A("| `gonogo_myoarm.py` / `.log` / `.err` | **stdout only** | **invisible to the method** | **a fourth MyoSuite body, trained on with this study's SAC — reported in H4 and §4.12** |")
    A("| `scratch_buildprior.log` | loose log | not enumerated | **the `prior_hand.pt` build: %s transitions, %.1f / %.1f mm — added to §3.3** |"
      % (format(hand["n"], ","), hand["mean"], hand["median"]))
    A("| `build_prior_finger.log` | loose log | not enumerated | fidelity was already in §3.3; the 200 000-transition count is now attributed |")
    A("| `prior_myo.pt` | checkpoint, **no log** | not enumerated | **no build transcript exists — §3.3's ≈19 mrad figure and the elbow babble budget are unarchived** |")
    A("| `prior3*.pt`, `prior_arm{1,4}.pt`, `prior_{1000..200000}.pt` | checkpoints | not enumerated | pre-study pilots on the toy planar arms and the babble-size sweep; not conditions of this study |")
    A("| `teacher_*.log` / `.pt` | logs + checkpoints | not enumerated | all scored — §3.4, §4.8, E3 |")
    A("| `eval_teacher_*_clean30k.log` | loose log | not enumerated | scored — §3.4 |")
    A("| `prior_sizes.log`, `prior3_sizes.log`, `prior_sizes.json`, `prior3_sizes.json` | loose logs | not enumerated | pre-study babble-size sweep on the toy arms; not a condition of this study |")
    A("")
    A("**The species-level finding is about method, not about instances.** E5 asserted "
      "completeness without enumerating; F4 enumerated the instance that had just been "
      "found; F4b enumerated `results*` directories and asserted completeness over the "
      "repository. Each correction widened the *instance* set and none widened the "
      "*method*, which is why the species recurred a seventh time on the first artefact "
      "that does not write a directory. The remedy is not a fifth register; it is the "
      "scope restriction above, stated wherever a completeness claim is made.")
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
