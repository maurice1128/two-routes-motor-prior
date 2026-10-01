# -*- coding: utf-8 -*-
"""Every table of the v4 paper, generated from raw per-seed JSON with the late-mean endpoint
(common.py). Writes tables/*.tex and numbers.json (the quantities the prose quotes).
The audit (audits/audit_v4.py) recomputes every cell with separate code."""
import os, io, json, itertools
import numpy as np
from common import *

OUT = os.path.join(W, "tcds_v4", "tables"); os.makedirs(OUT, exist_ok=True)
NUM = {}


def wilcoxon(v):
    v = np.asarray(v, float); v = v[v != 0]; n = len(v); r = np.argsort(np.argsort(np.abs(v))) + 1.0
    wp = r[v > 0].sum(); cnt = 0
    for m in range(1 << n):
        cnt += sum(r[i] for i in range(n) if m >> i & 1) <= wp
    p1 = cnt / float(1 << n); return 2 * min(p1, 1 - p1)


def c_full(v, bold=True, wil=False):
    m, lo, hi = ci(v); s = sig((m, lo, hi))
    dag = "^\\ddagger" if (wil and s and len(v) == 12 and wilcoxon(v) >= 0.05) else ""
    if s and bold:
        return "$\\mathbf{%+.2f}%s\\ [%+.2f, %+.2f]$" % (m, dag, lo, hi)
    return "$%+.2f\\ [%+.2f, %+.2f]$" % (m, lo, hi)


def c_short(v, wil=False):
    m, lo, hi = ci(v); s = sig((m, lo, hi))
    dag = "\\ddagger" if (wil and s and len(v) == 12 and wilcoxon(v) >= 0.05) else ""
    return "$%+.2f%s$" % (m, ("^{*%s}" % dag) if s else "")


def c_slope(series, xs):
    sl = slope(series, xs); m, lo, hi = ci(sl)
    return ("$\\mathbf{%+.2f}$ $[%+.2f,\\,%+.2f]^*$" if sig((m, lo, hi)) else "$%+.2f$ $[%+.2f,\\,%+.2f]$") % (m, lo, hi), (m, lo, hi)


def write(name, lines):
    io.open(os.path.join(OUT, name), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def rec(key, v):
    m, lo, hi = ci(v); NUM[key] = [round(m, 2), round(lo, 2), round(hi, 2), round(upper1(v), 2)]


E, F = core()
SW = {"elbow": sweep("elbow"), "finger": sweep("finger")}
RP = {"elbow": replace("elbow"), "finger": replace("finger")}
BM = {"elbow": bodymodel("elbow"), "finger": bodymodel("finger")}

# ================================================================== headline: withdrawn vs never had
rows = []
def headline_row(label, withdrawn_by_tw, none, ref_a, ref_b, key):
    pooled = [np.mean([end(withdrawn_by_tw[tw], s) - end(none, s) for tw in withdrawn_by_tw]) for s in S12]
    refv = [end(ref_a, s) - end(ref_b, s) for s in S12]
    rec(key + "_pooled", pooled); rec(key + "_ref", refv)
    m, lo, hi = ci(pooled); up = upper1(pooled); rm = ci(refv)[0]
    worst = max(ci([end(withdrawn_by_tw[tw], s) - end(none, s) for s in S12])[0] for tw in withdrawn_by_tw)
    NUM[key + "_share"] = round(100 * up / abs(rm), 1) if up > 0 else None
    rows.append("%s & %d & %s & $%+.2f$ & $%+.2f$ \\\\" % (label, len(withdrawn_by_tw), c_full(pooled), up, rm))
for body, unit in (("elbow", "mrad"), ("finger", "mm")):
    sw = SW[body]
    headline_row("%s, additive" % ("Elbow" if body == "elbow" else "Finger"),
                 {tw: sw["arms"][tw][0] for tw in TWS}, sw["none"], sw["con"], sw["none"], "%s_add" % body)
for body in ("elbow", "finger"):
    con, none, arm = RP[body]
    headline_row("%s, replacing" % ("Elbow" if body == "elbow" else "Finger"), arm["abrupt"], none, con, none, "%s_rep" % body)
write("headline.tex", ["\\begin{tabular}{@{}lrlrr@{}}", "\\toprule",
                       "Body, regime & $n_{t_w}$ & withdrawn $-$ none & upper & worth\\\\", "\\midrule"] + rows +
      ["\\bottomrule", "\\end{tabular}"])

# ================================================================== decomposition at t_w = 8000
d = lambda r, s, tw: r[s][tw + 1000] - r[s][tw]
lines = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
         " & \\multicolumn{3}{c}{myoElbow} & \\multicolumn{3}{c}{myoFinger, T2}\\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Contrast & dip & 12k & end & dip & 12k & end\\\\", "\\midrule"]
for name, x, y in (("abrupt $-$ constant", "ab", "con"), ("selfanchor $-$ constant", "sa", "con"), ("abrupt $-$ selfanchor", "ab", "sa"),
                   ("randanchor $-$ selfanchor", "ra", "sa"), ("abrupt $-$ none", "ab", "none"), ("constant $-$ none", "con", "none")):
    cells = []
    for body in ("elbow", "finger"):
        sw = SW[body]; a8, s8 = sw["arms"][8000]; da8, ds8 = sw["dip"][8000]
        full = {"ab": a8, "sa": s8, "con": sw["con"], "none": sw["none"], "ra": sw["ra"]}
        dipr = {"ab": da8, "sa": ds8, "con": sw["dcon"], "none": sw["dnone"], "ra": sw["dra"]}
        if x == "con" and y == "none":
            cells.append("--")
        else:
            v = [d(dipr[x], s, 8000) - d(dipr[y], s, 8000) for s in S12]; cells.append(c_short(v, wil=True)); rec("dec_%s_%s_%s_dip" % (body, x, y), v)
        v = [full[x][s][12000] - full[y][s][12000] for s in S12]; cells.append(c_short(v, wil=True)); rec("dec_%s_%s_%s_12k" % (body, x, y), v)
        v = [end(full[x], s) - end(full[y], s) for s in S12]; cells.append(c_short(v, wil=True)); rec("dec_%s_%s_%s_end" % (body, x, y), v)
    lines.append("%-26s & %s\\\\" % (name, " & ".join(cells)))
write("decomp.tex", lines + ["\\bottomrule", "\\end{tabular}"])

# ================================================================== attachment sweep
xs = np.array(TWS) / 1000.0
COMP = [("teacher", "sa", "con"), ("loss term", "ab", "sa"), ("total", "ab", "con"), ("abrupt $-$ none", "ab", "none")]
lines = ["\\begin{tabular}{@{}llrrrrrrr l@{}}", "\\toprule", " & & 2k & 3k & 4k & 5k & 6k & 7k & 8k & slope per 1k \\\\", "\\midrule"]
for body, unit in (("elbow", "myoElbow (mrad), teacher T1"), ("finger", "myoFinger (mm), teacher T2")):
    sw = SW[body]
    if body == "finger": lines.append("\\midrule")
    lines.append("\\multicolumn{10}{@{}l}{\\emph{%s}}\\\\" % unit)
    for metric in ("dip", "end"):
        for k, (name, x, y) in enumerate(COMP):
            per = []
            for tw in TWS:
                if metric == "dip":
                    a, sa = sw["dip"][tw]; src = {"ab": a, "sa": sa, "con": sw["dcon"], "none": sw["dnone"]}
                    per.append([d(src[x], s, tw) - d(src[y], s, tw) for s in S12])
                else:
                    a, sa = sw["arms"][tw]; src = {"ab": a, "sa": sa, "con": sw["con"], "none": sw["none"]}
                    per.append([end(src[x], s) - end(src[y], s) for s in S12])
            cells = [c_short(v) for v in per]
            sc, st = c_slope([[per[j][s] for j in range(len(TWS))] for s in S12], xs)
            key = "sw_%s_%s_%s" % (body, metric, name.split()[0] if name != "abrupt $-$ none" else "abnone")
            NUM[key + "_slope"] = [round(v, 2) for v in st]
            NUM[key + "_cells"] = [[round(ci(v)[0], 2), bool(sig(ci(v)))] for v in per]
            lab = ("dip" if metric == "dip" else "end") if k == 0 else ""
            lines.append("%-4s & %-16s & %s & %s\\\\" % (lab, name, " & ".join(cells), sc))
write("sweep.tex", lines + ["\\bottomrule", "\\end{tabular}"])

# ================================================================== replacing regime
xs4 = np.array(TW4) / 1000.0
REP = [("selfanchor", "constant"), ("gateon", "constant"), ("selfanchor", "gateon"), ("abrupt", "selfanchor"), ("abrupt", "constant"), ("abrupt", "none")]
lines = ["\\begin{tabular}{@{}lrrrr l rrrr l@{}}", "\\toprule",
         " & \\multicolumn{5}{c}{post-4k} & \\multicolumn{5}{c}{end}\\\\", "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
         " & 2k & 4k & 6k & 8k & slope & 2k & 4k & 6k & 8k & slope\\\\", "\\midrule"]
for body, unit in (("elbow", "myoElbow (mrad), teacher T1"), ("finger", "myoFinger (mm), teacher T2")):
    con, none, arm = RP[body]
    get = lambda n, tw: con if n == "constant" else (none if n == "none" else arm[n][tw])
    if body == "finger": lines.append("\\midrule")
    lines.append("\\multicolumn{11}{@{}l}{\\emph{%s}}\\\\" % unit)
    for x, y in REP:
        cells = []
        for metric in ("post4k", "end"):
            per = []
            for tw in TW4:
                if metric == "post4k":
                    per.append([get(x, tw)[s][tw + 4000] - get(y, tw)[s][tw + 4000] for s in S12])
                else:
                    per.append([end(get(x, tw), s) - end(get(y, tw), s) for s in S12])
            cells += [c_short(v) for v in per]
            sc, st = c_slope([[per[j][s] for j in range(4)] for s in S12], xs4); cells.append(sc)
            NUM["rep_%s_%s_%s_%s_slope" % (body, x, y, metric)] = [round(v, 2) for v in st]
            NUM["rep_%s_%s_%s_%s_cells" % (body, x, y, metric)] = [[round(ci(v)[0], 2), bool(sig(ci(v)))] for v in per]
        lines.append("%-24s & %s\\\\" % ("%s $-$ %s" % (x, y), " & ".join(cells)))
    rec("rep_%s_constant_none" % body, [end(con, s) - end(none, s) for s in S12])
write("replace.tex", lines + ["\\bottomrule", "\\end{tabular}"])

# ================================================================== body-model withdrawal (batch-matched)
# After withdrawal at 8k, purge trains exactly as blank does (128 distinct real per update) and
# matched exactly as blank64 does (64 distinct real, each repeated), so each is compared with the
# never-had condition that trains the same way. keep = the prior condition of the same body.
AR0 = arms()
BMB = {}
for body in ("elbow", "finger"):
    M = BM[body]; B = E if body == "elbow" else F
    BMB[body] = (dict(keep=M["keep"], purge=M["purge"], matched=M["matched"], blank=M["blank"], blank64=B["blank64"]), S12)
A4, _ = AR0["arm4"]
BMB["arm4"] = (dict(keep=A4["prior"], purge=load("results_matched100k_arm4", "purge"), matched=load("results_matched100k_arm4", "matched"),
                    blank=A4["blank"], blank64=A4["blank64"]), list(range(24)))
lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r" & worth & \multicolumn{2}{c}{withdrawn $-$ never had} & shortfall\\", r"\cmidrule(lr){3-4}",
         r" & keep$-$blank & purge$-$blank & matched$-$blank64 & matched$-$keep\\", r"\midrule"]
for body, lab in (("elbow", "myoElbow (mrad), $n{=}12$"), ("finger", "myoFinger (mm), $n{=}12$"), ("arm4", "4-joint arm (mm), $n{=}24$")):
    M, seeds = BMB[body]; cells = []
    for x, y in (("keep", "blank"), ("purge", "blank"), ("matched", "blank64"), ("matched", "keep")):
        v = [end(M[x], s) - end(M[y], s) for s in seeds]; rec("bm_%s_%s_%s_end" % (body, x, y), v); cells.append(c_short(v))
    for x, y in (("keep", "blank64"),):
        rec("bm_%s_%s_%s_end" % (body, x, y), [end(M[x], s) - end(M[y], s) for s in seeds])
    if body != "arm4":
        v = [M["matched"][s][12000] - M["keep"][s][12000] for s in seeds]; rec("bm_%s_matched_keep_12k" % body, v)
    for a in ("keep", "purge", "matched", "blank", "blank64"):
        NUM["bm_%s_mean_%s" % (body, a)] = round(float(np.mean([end(M[a], s) for s in seeds])), 2)
    lines.append("%s & %s \\\\" % (lab, " & ".join(cells)))
write("bodymodel.tex", lines + [r"\bottomrule", r"\end{tabular}"])

# ================================================================== core grid
means = lambda B, k: float(np.mean([end(B[k], s) for s in S12]))
lines = ["\\begin{tabular}{@{}lrrr@{}}", "\\toprule", " & myoElbow (mrad) & myoFinger, T1 (mm) & myoFinger, T2 (mm) \\\\", "\\midrule",
         "\\texttt{blank} / \\texttt{blank64}   & %.2f / %.2f & %.2f / %.2f & \\\\" % (means(E, "blank"), means(E, "blank64"), means(F, "blank"), means(F, "blank64")),
         "\\texttt{prior} / \\texttt{randprior} & %.2f / %.2f & %.2f / %.2f & \\\\" % (means(E, "prior"), means(E, "randprior"), means(F, "prior"), means(F, "randprior")),
         "\\texttt{coach} / \\texttt{randcoach} & %.2f / %.2f & %.2f / %.2f & %.2f / %.2f \\\\" % (means(E, "coach"), means(E, "randcoach"), means(F, "coach"), means(F, "randcoach"), means(F, "coachT2"), means(F, "randcoach")),
         "\\texttt{priorcoach}                 & %.2f & %.2f & %.2f \\\\" % (means(E, "priorcoach"), means(F, "priorcoach"), means(F, "priorcoachT2")),
         "\\midrule"]
for k in E: NUM["core_elbow_mean_" + k] = round(means(E, k), 2)
for k in F: NUM["core_finger_mean_" + k] = round(means(F, k), 2)
dd = lambda B, x, y: [end(B[x], s) - end(B[y], s) for s in S12]
for x, y, t2 in (("prior", "blank", False), ("prior", "blank64", False), ("blank64", "blank", False), ("prior", "randprior", False), ("randprior", "blank", False),
                 ("coach", "blank", True), ("coach", "prior", True), ("coach", "randcoach", True), ("priorcoach", "coach", True)):
    cells = [c_full(dd(E, x, y), wil=True), c_full(dd(F, x, y), wil=True)]
    rec("core_elbow_%s_%s" % (x, y), dd(E, x, y)); rec("core_finger_%s_%s" % (x, y), dd(F, x, y))
    if t2:
        xx = "priorcoachT2" if x == "priorcoach" else "coachT2"; yy = "coachT2" if y == "coach" else y
        cells.append(c_full(dd(F, xx, yy), wil=True)); rec("core_finger_%s_%s" % (xx, yy), dd(F, xx, yy))
    else:
        cells.append("")
    lines.append("\\texttt{%s}$-$\\texttt{%s} & %s \\\\" % (x, y, " & ".join(cells)))
lines.append("\\texttt{coach}$_{\\mathrm{T2}}-$\\texttt{coach}$_{\\mathrm{T1}}$ & & \\multicolumn{2}{c}{%s} \\\\" % c_full(dd(F, "coachT2", "coach"), wil=True))
rec("core_finger_coachT2_coach", dd(F, "coachT2", "coach"))
for extra in ("blank64", "randprior", "priorcoach"):
    rec("core_finger_coachT2_%s" % extra, dd(F, "coachT2", extra))
write("core.tex", lines + ["\\bottomrule", "\\end{tabular}"])
for body, B in (("elbow", E), ("finger", F)):
    for st in (12000,):
        rec("core_%s_prior_blank_12k" % body, [B["prior"][s][st] - B["blank"][s][st] for s in S12])
        rec("core_%s_coach_blank_12k" % body, [B["coach"][s][st] - B["blank"][s][st] for s in S12])

# ================================================================== joint-count series
AR = arms(); hdr = []; rowsA = []
for b, nj in (("arm1", 1), ("arm2", 2), ("arm3", 3), ("arm4", 4)):
    A, seeds = AR[b]; n = len(seeds); hdr.append((nj, n))
    mean = lambda k: float(np.mean([end(A[k], s) for s in seeds]))
    diff = lambda x, y: [end(A[x], s) - end(A[y], s) for s in seeds]
    for x, y in (("prior", "blank"), ("prior", "blank64"), ("blank64", "blank"), ("randprior", "blank"), ("prior", "randprior")):
        rec("arm_%s_%s_%s" % (b, x, y), diff(x, y))
    rowsA.append(["%.1f" % mean("blank"), "%.1f" % mean("blank64"), "%.1f" % mean("prior"), "%.1f" % mean("randprior"),
                  c_full(diff("prior", "blank")), "%+.0f\\%%" % (100 * np.mean(diff("prior", "blank")) / mean("blank")),
                  c_full(diff("prior", "blank64")), c_full(diff("blank64", "blank")), c_full(diff("randprior", "blank")), c_full(diff("prior", "randprior"))])
    NUM["arm_%s_pct" % b] = round(100 * float(np.mean(diff("prior", "blank"))) / mean("blank"))
names = ["\\texttt{blank}", "\\texttt{blank64}", "\\texttt{prior}", "\\texttt{randprior}", "\\texttt{prior}$-$\\texttt{blank}", "\\quad as \\% of \\texttt{blank}",
         "\\texttt{prior}$-$\\texttt{blank64}", "\\texttt{blank64}$-$\\texttt{blank}", "\\texttt{randprior}$-$\\texttt{blank}", "\\texttt{prior}$-$\\texttt{randprior}"]
lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", " & " + " & ".join("%d joint%s ($n{=}%d$)" % (nj, "" if nj == 1 else "s", n) for nj, n in hdr) + " \\\\", "\\midrule"]
for i, nm in enumerate(names):
    if i == 4: lines.append("\\midrule")
    lines.append(nm + " & " + " & ".join(r[i] for r in rowsA) + " \\\\")
write("arms.tex", lines + ["\\bottomrule", "\\end{tabular}"])
# trend across joints (seeds 0-11 on all four arms), relative to each arm's blank mean
rel = {}
for key, x, y in (("pb", "prior", "blank"), ("rb", "randprior", "blank")):
    rel[key] = {}
    for b in ("arm1", "arm2", "arm3", "arm4"):
        A, seeds = AR[b]; bl = float(np.mean([end(A["blank"], s) for s in seeds]))
        rel[key][b] = {s: (end(A[x], s) - end(A[y], s)) / bl for s in seeds}
    sl = [float(np.polyfit([1, 2, 3, 4], [rel[key][b][s] for b in ("arm1", "arm2", "arm3", "arm4")], 1)[0]) for s in S12]
    rec("arm_trend_" + key, sl); NUM["arm_trend_%s_wilcoxon" % key] = round(wilcoxon(sl), 3)

io.open(os.path.join(W, "tcds_v4", "numbers.json"), "w", encoding="utf-8").write(json.dumps(NUM, indent=1))
print("tables written:", sorted(os.listdir(OUT)), "| numbers:", len(NUM))
