# -*- coding: utf-8 -*-
"""Table V (replacing regime, both bodies) generated from raw per-seed JSON -> tables/replace.tex.
Elbow: teacher T1, none = results_conv_elbow_blank. Finger: teacher T2, none = the 1k-grid
never-guided runs of the finger sweep. post-4k = error 4,000 steps after withdrawal."""
import json, os, math, glob
import numpy as np

W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
OUT = os.path.join(W, "tcds_v3", "tables"); os.makedirs(OUT, exist_ok=True)
S = range(12)
TW4 = [2000, 4000, 6000, 8000]
XS = np.array(TW4) / 1000.0
REP = [("selfanchor", "constant"), ("gateon", "constant"), ("selfanchor", "gateon"),
       ("abrupt", "selfanchor"), ("abrupt", "constant"), ("abrupt", "none")]


def load(d, c):
    r = {}
    for f in glob.glob(os.path.join(W, d, "%s_seed*.json" % c)):
        s = int(f.split("_seed")[1].split(".")[0])
        r[s] = {int(x["step"]): x["eval_dist"] * 1000 for x in json.load(open(f))}
    return r


def ci(v):
    v = np.asarray(v, float); m = v.mean(); se = v.std(ddof=1) / math.sqrt(len(v))
    return m, m - 2.201 * se, m + 2.201 * se


def short(v):
    m, lo, hi = ci(v)
    return "$%+.1f%s$" % (m, "^*" if (lo > 0 or hi < 0) else "")


def slope_cell(per):
    sl = [np.polyfit(XS, [per[j][s] for j in range(4)], 1)[0] for s in S]
    m, lo, hi = ci(sl)
    if lo > 0 or hi < 0:
        return "$\\mathbf{%+.2f}$ $[%+.2f,\\,%+.2f]^*$" % (m, lo, hi)
    return "$%+.2f$ $[%+.2f,\\,%+.2f]$" % (m, lo, hi)


lines = ["\\begin{tabular}{@{}lrrrr l rrrr l@{}}", "\\toprule",
         " & \\multicolumn{5}{c}{post-4k} & \\multicolumn{5}{c}{endpoint 100k}\\\\",
         "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
         " & 2k & 4k & 6k & 8k & slope & 2k & 4k & 6k & 8k & slope\\\\", "\\midrule"]
done = []
for body, unit, rdir, none_d, none_c in (
        ("myoelbow", "myoElbow (mrad), teacher T1", "results_replace100k_myoelbow", "results_conv_elbow_blank", "blank"),
        ("myofinger", "myoFinger (mm), teacher T2", "results_replace100k_myofinger", "results_sweep100k_finger_T2", "none")):
    con = load(rdir, "constant"); none = load(none_d, none_c)
    arm = {a: {tw: load(rdir, "%s_w%d" % (a, tw)) for tw in TW4} for a in ("abrupt", "selfanchor", "gateon")}
    missing = [(a, tw) for a in arm for tw in TW4 if len(arm[a][tw]) < 12] + ([("constant", None)] if len(con) < 12 else [])
    if missing:
        print("%s incomplete (%d cells missing seeds): skipped" % (body, len(missing))); continue
    get = lambda n, tw: con if n == "constant" else (none if n == "none" else arm[n][tw])
    if done: lines.append("\\midrule")
    lines.append("\\multicolumn{11}{@{}l}{\\emph{%s}}\\\\" % unit)
    for x, y in REP:
        cells = []
        for metric in ("post4k", "end"):
            per = []
            for tw in TW4:
                st = tw + 4000 if metric == "post4k" else 100000
                v = [get(x, tw)[s][st] - get(y, tw)[s][st] for s in S]
                per.append(v); cells.append(short(v))
            cells.append(slope_cell(per))
        lines.append("%-24s & %s\\\\" % ("%s $-$ %s" % (x, y), " & ".join(cells)))
    done.append(body)
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "replace.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("replace.tex written for", done)
