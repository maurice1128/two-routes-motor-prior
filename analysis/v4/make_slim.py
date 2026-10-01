# -*- coding: utf-8 -*-
"""Slim v4 (2026-10-01): only claims that hold under BOTH endpoints (late mean 90-100k and the
single 100k evaluation). Writes tables/slim_*.tex and merges slim_* keys into numbers.json.

Message 1: what a body model has learned matters more the more joints the body has
           (trained minus random model, identical machinery).
Message 2: a competent teacher can be taken away without leaving the learner dependent
           (withdrawn minus never guided; retained fraction of the teacher's benefit), and at long
           attachment the drop at withdrawal is the deleted imitation term (self-anchor control)."""
import io, json, math, os
import numpy as np
from common import *

EP = {"late": end, "100k": lambda r, s: r[s][100000]}
NUM = json.load(open("numbers.json"))
OUTT = os.path.join(W, "tcds_v4", "tables")


def rec(key, v):
    c = ci(v); NUM[key] = [round(c[0], 2), round(c[1], 2), round(c[2], 2), round(upper1(v), 2)]; return c


def cell(c, robust):
    s = "$\\mathbf{%+.1f}$" % c[0] if sig(c) else "$%+.1f$" % c[0]
    return s + ("$^{\\dagger}$" if robust else "") + " {\\scriptsize$[%+.1f, %+.1f]$}" % (c[1], c[2])


E, F = core(); AR = arms()
BODIES = [("myoElbow (mrad)", 1, E, S12, "elbow"), ("myoFinger (mm)", 4, F, S12, "finger")]
for b, nj in (("arm1", 1), ("arm2", 2), ("arm3", 3), ("arm4", 4)):
    A, s = AR[b]; BODIES.append(("%d-joint arm (mm)" % nj, nj, A, s, b))

# ---------------------------------------------------------------- Table: the body model, six bodies
rows = []
for lab, nj, B, seeds, key in BODIES:
    cells = []
    for x, y in (("prior", "randprior"), ("prior", "blank"), ("prior", "blank64")):
        cs = {}
        for e, f in EP.items():
            cs[e] = rec("slim_%s_%s_%s_%s" % (key, x, y, e), [f(B[x], s) - f(B[y], s) for s in seeds])
        robust = sig(cs["late"]) and sig(cs["100k"]) and np.sign(cs["late"][0]) == np.sign(cs["100k"][0])
        NUM["slim_%s_%s_%s_robust" % (key, x, y)] = bool(robust)
        cells.append(cell(cs["late"], robust))
    for k in ("blank", "blank64", "prior", "randprior"):
        NUM["slim_%s_mean_%s" % (key, k)] = round(float(np.mean([end(B[k], s) for s in seeds])), 1)
    rows.append("%s & %d & %s \\\\" % (lab, len(seeds), " & ".join(cells)))
io.open(os.path.join(OUTT, "slim_model.tex"), "w", encoding="utf-8").write("\n".join(
    [r"\begin{tabular}{@{}lrlll@{}}", r"\toprule",
     r"Body & $n$ & trained $-$ random model & trained $-$ \texttt{blank} & trained $-$ \texttt{blank64}\\", r"\midrule"]
    + rows + [r"\bottomrule", r"\end{tabular}"]) + "\n")

# joint-count trend of the trained-vs-random gap (arms), both endpoints
for e, f in EP.items():
    rel = {b: {s: (f(AR[b][0]["prior"], s) - f(AR[b][0]["randprior"], s)) / np.mean([f(AR[b][0]["blank"], t) for t in AR[b][1]]) for s in AR[b][1]} for b in AR}
    sl = [float(np.polyfit([1, 2, 3, 4], [rel[b][s] for b in ("arm1", "arm2", "arm3", "arm4")], 1)[0]) for s in S12]
    c = ci(sl); NUM["slim_trend_seed_" + e] = [round(z, 2) for z in c]
    for kind in ("mm", "rel"):
        ys, ws = [], []
        for b in ("arm1", "arm2", "arm3", "arm4"):
            A, s = AR[b]; bl = np.mean([f(A["blank"], t) for t in s]) if kind == "rel" else 1.0
            v = np.array([(f(A["prior"], t) - f(A["randprior"], t)) / bl for t in s]); ys.append(v.mean()); ws.append(len(v) / v.var(ddof=1))
        X = np.vstack([np.ones(4), [1, 2, 3, 4.]]).T; Wm = np.diag(ws); cov = np.linalg.inv(X.T @ Wm @ X); beta = cov @ X.T @ Wm @ np.array(ys)
        se = math.sqrt(cov[1, 1]); NUM["slim_trend_meta_%s_%s" % (kind, e)] = [round(float(beta[1]), 2), round(se, 2), round(float(beta[1] / se), 1)]
    for b in ("arm1", "arm2", "arm3", "arm4"):
        A, s = AR[b]; NUM["slim_gap_rel_%s_%s" % (b, e)] = round(float(np.mean([(f(A["prior"], t) - f(A["randprior"], t)) for t in s]) / np.mean([f(A["blank"], t) for t in s])), 2)

# ---------------------------------------------------------------- Table: taking the teacher away
rows = []
for body, unit in (("elbow", "myoElbow (mrad), T1"), ("finger", "myoFinger (mm), T2")):
    sw = sweep(body); cells = []
    for e, f in EP.items():
        pooled = [np.mean([f(sw["arms"][tw][0], s) - f(sw["none"], s) for tw in TWS]) for s in S12]
        worth = [f(sw["con"], s) - f(sw["none"], s) for s in S12]
        cw = rec("slim_wd_%s_worth_%s" % (body, e), worth); cp = rec("slim_wd_%s_pooled_%s" % (body, e), pooled)
        kept = None
        for r in np.arange(1.0, 0.0, -0.01):
            if upper1([p - r * w for p, w in zip(pooled, worth)]) < 0:
                kept = int(round(100 * r)); break
        NUM["slim_wd_%s_kept_%s" % (body, e)] = kept
        # paired non-inferiority: deficit smaller than a fraction lam of the teacher's worth
        NUM["slim_wd_%s_ni_%s" % (body, e)] = [[lam, round(upper1([p + lam * w for p, w in zip(pooled, worth)]), 2)] for lam in (1.0, 0.5)]
        cells += [cell(cw, False), cell(cp, False), ("$\\geq %d\\%%$" % kept) if kept else "--"]
    # per-t_w withdrawn-none, both endpoints
    NUM["slim_wd_%s_cells" % body] = {e: [[round(ci([f(sw["arms"][tw][0], s) - f(sw["none"], s) for s in S12])[0], 2),
                                           bool(sig(ci([f(sw["arms"][tw][0], s) - f(sw["none"], s) for s in S12])))] for tw in TWS] for e, f in EP.items()}
    NUM["slim_wd_%s_total_cells" % body] = {e: [[round(ci([f(sw["arms"][tw][0], s) - f(sw["con"], s) for s in S12])[0], 2),
                                                 bool(sig(ci([f(sw["arms"][tw][0], s) - f(sw["con"], s) for s in S12])))] for tw in TWS] for e, f in EP.items()}
    rows.append("%s & %s \\\\" % (unit, " & ".join(cells)))
io.open(os.path.join(OUTT, "slim_teacher.tex"), "w", encoding="utf-8").write("\n".join(
    [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
     r" & \multicolumn{3}{c}{late mean (90--100k)} & \multicolumn{3}{c}{single 100k evaluation}\\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     r" & worth & withdrawn$-$never & kept & worth & withdrawn$-$never & kept\\", r"\midrule"]
    + rows + [r"\bottomrule", r"\end{tabular}"]) + "\n")

# 200k elbow: non-inferiority at 190-200k
D = "results_long200k_elbow"; L200 = list(range(190000, 200001, 2000))
none = load(D, "none"); con = load(D, "constant"); ab = {tw: load(D + "/w%d" % tw, "abrupt") for tw in TWS}
pooled = [np.mean([win(ab[tw], s, L200) - win(none, s, L200) for tw in TWS]) for s in S12]
worth = [win(con, s, L200) - win(none, s, L200) for s in S12]
NUM["slim_200k_ni"] = [[lam, round(upper1([p + lam * w for p, w in zip(pooled, worth)]), 2)] for lam in (1.0, 0.5, 0.25)]

# ---------------------------------------------------------------- Table: the drop at withdrawal, by t_w
d = lambda r, s, tw: r[s][tw + 1000] - r[s][tw]
lines = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule", r" & & 2k & 3k & 4k & 5k & 6k & 7k & 8k\\", r"\midrule"]
for body, unit in (("elbow", "myoElbow (mrad)"), ("finger", "myoFinger (mm)")):
    sw = sweep(body)
    if body == "finger": lines.append(r"\midrule")
    for k, (name, x, y) in enumerate((("total", "ab", "con"), ("teacher", "sa", "con"), ("loss term", "ab", "sa"))):
        cs = []
        for tw in TWS:
            a, sa = sw["dip"][tw]; src = {"ab": a, "sa": sa, "con": sw["dcon"]}
            v = [d(src[x], s, tw) - d(src[y], s, tw) for s in S12]; c = ci(v)
            cs.append("$%+.1f%s$" % (c[0], "^{*}" if sig(c) else ""))
            NUM["slim_dip_%s_%s_%d" % (body, name.split()[0], tw)] = [round(z, 2) for z in c]
        lines.append("%s & %s & %s\\\\" % (unit if k == 0 else "", name, " & ".join(cs)))
    # loss-term component minus teacher component, per t_w (paired): abrupt - 2*selfanchor + constant
    for tw in TWS:
        a, sa = sw["dip"][tw]
        v = [d(a, s, tw) - 2 * d(sa, s, tw) + d(sw["dcon"], s, tw) for s in S12]
        NUM["slim_dip_%s_lossminusteacher_%d" % (body, tw)] = [round(z, 2) for z in ci(v)]
io.open(os.path.join(OUTT, "slim_dip.tex"), "w", encoding="utf-8").write("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n")

json.dump(NUM, open("numbers.json", "w"), indent=1)
for k in sorted(NUM):
    if k.startswith("slim_") and ("robust" not in k): print(k, NUM[k])
