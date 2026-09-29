# -*- coding: utf-8 -*-
"""Generates the two large tables of the stage-B paper from raw per-seed JSON:
   tables/arms.tex   joint-count series (arm1..arm4, 100k, held-out)
   tables/sweep.tex  attachment sweep at 100k on both bodies (dip and endpoint components)
The audit (audits/audit_v3b.py) recomputes every cell with separate code."""
import json, os, math, glob
import numpy as np

W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
OUT = os.path.join(W, "tcds_v3", "tables"); os.makedirs(OUT, exist_ok=True)
TQ = {12: 2.201, 24: 2.069}
PRIOR_DIR = {"arm1": "prior2", "arm2": "prior", "arm3": "prior", "arm4": "prior2"}   # arm1/arm4: priors rebuilt by the recorded recipe (prior_arm{1,4}_200k.pt)


def load(d, c, seeds=None):
    r = {}
    for f in glob.glob(os.path.join(W, d, "%s_seed*.json" % c)):
        s = int(f.split("_seed")[1].split(".")[0])
        if seeds is None or s in seeds:
            r[s] = {int(x["step"]): x["eval_dist"] * 1000 for x in json.load(open(f))}
    return r


def ci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n); t = TQ[n]
    return m, m - t * se, m + t * se


def cell(v, bold=True, dec=2):
    m, lo, hi = ci(v); sig = lo > 0 or hi < 0
    s = "$%+.*f\\ [%+.*f, %+.*f]$" % (dec, m, dec, lo, dec, hi)
    return ("$\\mathbf{%+.*f}\\ [%+.*f, %+.*f]$" % (dec, m, dec, lo, dec, hi)) if (sig and bold) else s


def short(v, dec=1):
    m, lo, hi = ci(v); return "$%+.*f%s$" % (dec, m, "^*" if (lo > 0 or hi < 0) else "")


def slope_cell(per_seed_curves, xs):
    sl = [np.polyfit(xs, y, 1)[0] for y in per_seed_curves]
    m, lo, hi = ci(sl); sig = lo > 0 or hi < 0
    return ("$\\mathbf{%+.2f}$ $[%+.2f,\\,%+.2f]^*$" if sig else "$%+.2f$ $[%+.2f,\\,%+.2f]$") % (m, lo, hi)


# ---------------------------------------------------------------- joint-count series
rows = []
hdr = []
for b, nj in (("arm1", 1), ("arm2", 2), ("arm3", 3), ("arm4", 4)):
    A = {"blank": load("results_arms100k_%s_blank" % b, "blank"), "blank64": load("results_arms100k_%s_blank64" % b, "blank64"),
         "prior": load("results_arms100k_%s_%s" % (b, PRIOR_DIR[b]), "prior"), "randprior": load("results_arms100k_%s_randprior" % b, "prior")}
    seeds = sorted(set.intersection(*[set(v) for v in A.values()]))
    n = len(seeds); ST = 100000
    mean = lambda k: np.mean([A[k][s][ST] for s in seeds])
    diff = lambda x, y: [A[x][s][ST] - A[y][s][ST] for s in seeds]
    bl = mean("blank")
    hdr.append((nj, n))
    rows.append([
        "%.1f" % mean("blank"), "%.1f" % mean("blank64"), "%.1f" % mean("prior"), "%.1f" % mean("randprior"),
        cell(diff("prior", "blank")), "%+.0f\\%%" % (100 * np.mean(diff("prior", "blank")) / bl),
        cell(diff("prior", "blank64")), cell(diff("blank64", "blank")),
        cell(diff("randprior", "blank")), cell(diff("prior", "randprior")),
        cell(diff("prior", "prior")) if False else "",
    ])
lines = []
lines.append("\\begin{tabular}{@{}l" + "r" * 4 + "@{}}")
lines.append("\\toprule")
lines.append(" & " + " & ".join("%d joint%s ($n{=}%d$)" % (nj, "" if nj == 1 else "s", n) for nj, n in hdr) + " \\\\")
lines.append("\\midrule")
names = ["\\texttt{blank}", "\\texttt{blank64}", "\\texttt{prior}", "\\texttt{randprior}",
         "\\texttt{prior}$-$\\texttt{blank}", "\\quad as \\% of \\texttt{blank}",
         "\\texttt{prior}$-$\\texttt{blank64}", "\\texttt{blank64}$-$\\texttt{blank}",
         "\\texttt{randprior}$-$\\texttt{blank}", "\\texttt{prior}$-$\\texttt{randprior}"]
for i, nm in enumerate(names):
    if i == 4: lines.append("\\midrule")
    lines.append(nm + " & " + " & ".join(r[i] for r in rows) + " \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
open(os.path.join(OUT, "arms.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("arms.tex written; n per arm:", hdr)

# ---------------------------------------------------------------- attachment sweep at 100k
TWS = [2000, 3000, 4000, 5000, 6000, 7000, 8000]
xs = np.array(TWS) / 1000.0
S = range(12)


def sweep_arms(body):
    if body == "elbow":
        con = load("results_coach100k_elbow", "constant"); none = load("results_conv_elbow_blank", "blank")
        arms = {tw: (load("results_sweep100k_elbow/w%d" % tw, "abrupt"), load("results_sweep100k_elbow/w%d" % tw, "selfanchor")) for tw in TWS[:-1]}
        arms[8000] = (load("results_coach100k_elbow", "abrupt"), load("results_coach100k_elbow_off2", "selfanchor"))
        # dips need a 1k grid: the 12k sweep runs (read only) reproduce these runs seed for seed at 2k..12k
        Z = r"C:\Users\maurice\Desktop\world_model_zeroshot\b2_pilot"
        dip_arms = {tw: (load(os.path.join(Z, "results_attachment_myoelbow"), "abrupt_w%d" % tw), load(os.path.join(Z, "results_attachment_myoelbow"), "selfanchor_w%d" % tw)) for tw in TWS}
        dip_con = load(os.path.join(Z, "results_withdrawal_myoelbow"), "constant"); dip_none = load(os.path.join(Z, "results_withdrawal_myoelbow"), "none")
    else:
        con = load("results_sweep100k_finger_T2", "constant"); none = load("results_sweep100k_finger_T2", "none")
        arms = {tw: (load("results_sweep100k_finger_T2/w%d" % tw, "abrupt"), load("results_sweep100k_finger_T2/w%d" % tw, "selfanchor")) for tw in TWS[:-1]}
        arms[8000] = (load("results_coach100k_finger_T2", "abrupt"), load("results_coach100k_finger_T2", "selfanchor"))
        dip_arms, dip_con, dip_none = arms, con, none
    return con, none, arms, dip_arms, dip_con, dip_none


COMP = [("teacher", "sa", "con"), ("loss term", "ab", "sa"), ("total", "ab", "con"), ("abrupt $-$ none", "ab", "none")]
lines = ["\\begin{tabular}{@{}llrrrrrrr l@{}}", "\\toprule",
         " & & 2k & 3k & 4k & 5k & 6k & 7k & 8k & slope per 1k \\\\", "\\midrule"]
for body, unit in (("elbow", "myoElbow (mrad)"), ("finger", "myoFinger (mm), teacher T2")):
    con, none, arms, dip_arms, dip_con, dip_none = sweep_arms(body)
    lines.append("\\multicolumn{10}{@{}l}{\\emph{%s}}\\\\" % unit)
    for metric in ("dip", "endpoint"):
        for ci_, (name, x, y) in enumerate(COMP):
            per = []
            cells = []
            for tw in TWS:
                if metric == "dip":
                    a, sa = dip_arms[tw]; src = {"ab": a, "sa": sa, "con": dip_con, "none": dip_none}
                    f = lambda r, s: r[s][tw + 1000] - r[s][tw]
                else:
                    a, sa = arms[tw]; src = {"ab": a, "sa": sa, "con": con, "none": none}
                    f = lambda r, s: r[s][100000]
                v = [f(src[x], s) - f(src[y], s) for s in S]
                per.append(v); cells.append(short(v))
            per_seed = [[per[j][s] for j in range(len(TWS))] for s in S]
            lab = ("dip" if metric == "dip" else "100k") if ci_ == 0 else ""
            lines.append("%-5s & %-16s & %s & %s\\\\" % (lab, name, " & ".join(cells), slope_cell(per_seed, xs)))
    if body == "elbow": lines.append("\\midrule")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "sweep.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("sweep.tex written")
