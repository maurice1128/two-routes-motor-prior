# -*- coding: utf-8 -*-
"""Stage-B statistics: the four v3 experiment groups, from per-seed JSON only.

1 blank-64 (both bodies, 100k)             -> is the body model's effect real-data halving?
2 elbow randanchor at 100k                  -> anchor content at convergence
3 replacing regime at 100k (elbow)          -> does the 12k replacing-regime story survive convergence?
4 finger with a competent teacher T2 (100k) -> does "guidance is only as good as its teacher" hold
                                               when the teacher is good; decomposition at t_w=8k
Also reproduction checks: new runs vs the runs they must equal before any divergence.
"""
import json, os, math, itertools, io
import numpy as np

W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
Z = r"C:\Users\maurice\Desktop\world_model_zeroshot"
rng = np.random.default_rng(0)
out = []


def P(s=""):
    print(s); out.append(s)


def load(d, c, n=12, root=W):
    res = {}
    for s in range(n):
        p = os.path.join(root, d, "%s_seed%d.json" % (c, s))
        res[s] = {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(p))}
    return res


def pt(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n)
    return m, m - 2.201 * se, m + 2.201 * se


def wil(d):
    d = np.asarray(d, float); d = d[d != 0]; n = len(d)
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    wpos = ranks[d > 0].sum(); tot = n * (n + 1) / 2
    dist = np.zeros(int(tot) + 1)
    for signs in itertools.product([0, 1], repeat=n):
        dist[int(sum(r for r, s in zip(ranks, signs) if s))] += 1
    dist /= dist.sum()
    return min(1.0, 2 * dist[:int(min(wpos, tot - wpos)) + 1].sum())


def fmt(t3):
    m, lo, hi = t3
    return "%+7.2f [%+7.2f, %+7.2f]%s" % (m, lo, hi, "*" if lo > 0 or hi < 0 else " ")


def C(A, a, B_, b, st):
    return [A[s][st] - B_[s][st] for s in range(12)]


def row(name, d):
    P("  %-44s %s  W p=%.4f" % (name, fmt(pt(d)), wil(d)))


def mean(A, st):
    return np.mean([A[s][st] for s in range(12)])


# ------------------------------------------------------------------ 1 blank-64
P("== 1. blank-64: model-free with 64 distinct real transitions per update (same as the Dyna arms)")
for tag in ("elbow", "finger"):
    bl = load("results_conv_%s_blank" % tag, "blank"); b64 = load("results_conv_%s_blank64" % tag, "blank64")
    pr = load("results_conv_%s_prior" % tag, "prior"); rp = load("results_conv_%s_randprior" % tag, "prior")
    P(" %s means 12k/50k/100k: blank %.2f/%.2f/%.2f  blank64 %.2f/%.2f/%.2f  prior %.2f/%.2f/%.2f  randprior %.2f/%.2f/%.2f" % (
        tag, *[mean(bl, s) for s in (12000, 50000, 100000)], *[mean(b64, s) for s in (12000, 50000, 100000)],
        *[mean(pr, s) for s in (12000, 50000, 100000)], *[mean(rp, s) for s in (12000, 50000, 100000)]))
    for st in (12000, 50000, 100000):
        row("%s %dk blank64 - blank (halved real data)" % (tag, st // 1000), C(b64, 0, bl, 0, st))
        row("%s %dk prior - blank64 (model at equal real data)" % (tag, st // 1000), C(pr, 0, b64, 0, st))
        row("%s %dk randprior - blank64" % (tag, st // 1000), C(rp, 0, b64, 0, st))
    row("%s plateau blank64 100k-80k" % tag, [b64[s][100000] - b64[s][80000] for s in range(12)])

# ------------------------------------------------------------------ 2 elbow randanchor 100k
P("\n== 2. elbow randanchor at 100k (teacher clean30k, t_w=8000)")
con = load("results_coach100k_elbow", "constant"); ab = load("results_coach100k_elbow", "abrupt")
sa = load("results_coach100k_elbow_off2", "selfanchor"); ra = load("results_coach100k_elbow", "randanchor")
zra = load("b2_pilot/results_withdrawal_myoelbow", "randanchor", root=Z)
steps = sorted(ra[0].keys())
P("  randanchor eval steps: %d..%d every %d" % (steps[0], steps[-1], steps[1] - steps[0]))
rep = sum(1 for s in range(12) if abs(ra[s][12000] - zra[s][12000]) > 0.005)
P("  reproduction vs 12k study randanchor at 12k: %d/12 seeds differ" % rep)
P("  means 100k: constant %.2f abrupt %.2f selfanchor %.2f randanchor %.2f" % tuple(mean(x, 100000) for x in (con, ab, sa, ra)))
for st in (12000, 100000):
    row("randanchor - selfanchor %dk" % (st // 1000), C(ra, 0, sa, 0, st))
    row("randanchor - constant %dk" % (st // 1000), C(ra, 0, con, 0, st))

# ------------------------------------------------------------------ 3 replacing regime 100k
P("\n== 3. replacing regime at 100k (elbow)")
R = {"constant": load("results_replace100k_myoelbow", "constant")}
TW = [2000, 4000, 6000, 8000]
for a in ("abrupt", "selfanchor", "gateon"):
    for tw in TW:
        R[(a, tw)] = load("results_replace100k_myoelbow", "%s_w%d" % (a, tw))
R12 = {"constant": load("b2_pilot/results_replace_myoelbow", "constant", root=Z)}
for a in ("abrupt", "selfanchor", "gateon"):
    for tw in TW:
        R12[(a, tw)] = load("b2_pilot/results_replace_myoelbow", "%s_w%d" % (a, tw), root=Z)
diff = 0; tot = 0
for k in R:
    for s in range(12):
        for st in (6000, 12000):
            tot += 1; diff += abs(R[k][s][st] - R12[k][s][st]) > 0.005
P("  reproduction vs the 12k replacing-regime runs at 6k and 12k: %d/%d cells differ" % (diff, tot))
none = load("results_conv_elbow_blank", "blank")
g = lambda k: R["constant"] if k == "constant" else R[k]
P("  endpoint means at 100k: constant %.2f | abrupt %s | selfanchor %s | gateon %s | never-guided blank %.2f" % (
    mean(R["constant"], 100000), " ".join("%.1f" % mean(R[("abrupt", tw)], 100000) for tw in TW),
    " ".join("%.1f" % mean(R[("selfanchor", tw)], 100000) for tw in TW), " ".join("%.1f" % mean(R[("gateon", tw)], 100000) for tw in TW),
    mean(none, 100000)))
row("constant(pure distillation) - blank 100k", C(R["constant"], 0, none, 0, 100000))
xs = np.array(TW) / 1000.0
for name, a, b in (("selfanchor - constant", "selfanchor", "constant"), ("gateon - constant", "gateon", "constant"),
                   ("selfanchor - gateon", "selfanchor", "gateon"), ("abrupt - selfanchor", "abrupt", "selfanchor"),
                   ("abrupt - constant", "abrupt", "constant"), ("abrupt - none", "abrupt", "none")):
    for mname in ("post-4k", "endpoint100k"):
        cells = []; per = {s: [] for s in range(12)}
        for tw in TW:
            X = R[(a, tw)]; Y = R["constant"] if b == "constant" else (none if b == "none" else R[(b, tw)])
            st = tw + 4000 if mname == "post-4k" else 100000
            v = [X[s][st] - Y[s][st] for s in range(12)]
            for s in range(12): per[s].append(v[s])
            m, lo, hi = pt(v); cells.append("%+.1f%s" % (m, "*" if lo > 0 or hi < 0 else ""))
        sl = pt([np.polyfit(xs, per[s], 1)[0] for s in range(12)])
        P("  %-22s %-12s %s   slope %s" % (name, mname, "  ".join(cells), fmt(sl)))
P("  plateau constant(replace) 100k-80k: %s" % fmt(pt([R["constant"][s][100000] - R["constant"][s][80000] for s in range(12)])))
for tw in TW:
    for a in ("abrupt", "selfanchor", "gateon"):
        v = [R[(a, tw)][s][100000] - R[(a, tw)][s][80000] for s in range(12)]
        m, lo, hi = pt(v)
        if lo > 0 or hi < 0: P("  NOT plateaued: %s w%d %s" % (a, tw, fmt((m, lo, hi))))
rec = {a: [sum(1 for s in range(12) if R[(a, tw)][s][100000] <= R[(a, tw)][s][tw]) for tw in TW] for a in ("abrupt", "selfanchor", "gateon")}
P("  seeds whose 100k error is at or below their error at t_w: %s" % rec)

# ------------------------------------------------------------------ 4 finger with competent teacher T2
P("\n== 4. myoFinger with a competent teacher T2 (actor of a 100k prior run, held-out 35.6 mm)")
bl = load("results_conv_finger_blank", "blank"); pr = load("results_conv_finger_prior", "prior")
c1 = load("results_conv_finger_coach", "coach"); pc1 = load("results_conv_finger_priorcoach", "priorcoach")
c2 = load("results_conv_finger_coachT2", "coach"); pc2 = load("results_conv_finger_priorcoachT2", "priorcoach")
b64 = load("results_conv_finger_blank64", "blank64")
P("  means 12k/50k/100k: coachT2 %.2f/%.2f/%.2f  priorcoachT2 %.2f/%.2f/%.2f  | coachT1 %.2f/%.2f/%.2f  prior %.2f/%.2f/%.2f  blank %.2f/%.2f/%.2f" % (
    *[mean(c2, s) for s in (12000, 50000, 100000)], *[mean(pc2, s) for s in (12000, 50000, 100000)],
    *[mean(c1, s) for s in (12000, 50000, 100000)], *[mean(pr, s) for s in (12000, 50000, 100000)], *[mean(bl, s) for s in (12000, 50000, 100000)]))
for st in (12000, 50000, 100000):
    row("%dk coachT2 - blank" % (st // 1000), C(c2, 0, bl, 0, st))
    row("%dk coachT2 - prior" % (st // 1000), C(c2, 0, pr, 0, st))
    row("%dk coachT2 - coachT1 (teacher quality)" % (st // 1000), C(c2, 0, c1, 0, st))
    row("%dk priorcoachT2 - coachT2" % (st // 1000), C(pc2, 0, c2, 0, st))
    row("%dk priorcoachT2 - prior" % (st // 1000), C(pc2, 0, pr, 0, st))
    row("%dk priorcoachT2 - blank" % (st // 1000), C(pc2, 0, bl, 0, st))
row("plateau coachT2 100k-80k", [c2[s][100000] - c2[s][80000] for s in range(12)])
row("plateau priorcoachT2 100k-80k", [pc2[s][100000] - pc2[s][80000] for s in range(12)])
P("  coachT2 student vs its teacher (35.6 mm): student 100k mean %.2f" % mean(c2, 100000))
# checkpoint counts
CK = list(range(2000, 100001, 2000))
for nm, X, Y in (("coachT2-blank", c2, bl), ("coachT2-prior", c2, pr)):
    sig = [st for st in CK if (lambda t: t[1] > 0 or t[2] < 0)(pt(C(X, 0, Y, 0, st)))]
    neg = [st for st in CK if pt(C(X, 0, Y, 0, st))[2] < 0]; pos = [st for st in CK if pt(C(X, 0, Y, 0, st))[1] > 0]
    P("  %s: sig<0 at %d/50 (first %s), sig>0 at %d/50" % (nm, len(neg), neg[0] // 1000 if neg else "-", len(pos)))

P("\n  decomposition at t_w=8000 with T2 (constant = coachT2 run)")
abT = load("results_coach100k_finger_T2", "abrupt"); saT = load("results_coach100k_finger_T2", "selfanchor"); raT = load("results_coach100k_finger_T2", "randanchor")
stp_c = sorted(c2[0].keys()); stp_a = sorted(abT[0].keys())
P("  eval steps: constant every %d, withdrawn arms every %d" % (stp_c[1] - stp_c[0], stp_a[1] - stp_a[0]))
pre = sum(1 for s in range(12) for st in (2000, 4000, 6000) if abs(abT[s][st] - c2[s][st]) > 0.005 or abs(saT[s][st] - c2[s][st]) > 0.005)
P("  pre-withdrawal identity (abrupt/selfanchor == constant at 2k,4k,6k): %d/72 cells differ" % pre)
dip2 = lambda A: [A[s][10000] - A[s][8000] for s in range(12)]
dd = lambda A, B_: [a - b for a, b in zip(dip2(A), dip2(B_))]
row("dip(8k->10k) abrupt - constant (total)", dd(abT, c2))
row("dip(8k->10k) selfanchor - constant (teacher)", dd(saT, c2))
row("dip(8k->10k) abrupt - selfanchor (loss term)", dd(abT, saT))
row("dip(8k->10k) randanchor - selfanchor", dd(raT, saT))
for st in (12000, 50000, 100000):
    row("%dk abrupt - constant (total)" % (st // 1000), C(abT, 0, c2, 0, st))
    row("%dk selfanchor - constant (teacher)" % (st // 1000), C(saT, 0, c2, 0, st))
    row("%dk abrupt - selfanchor (loss term)" % (st // 1000), C(abT, 0, saT, 0, st))
    row("%dk randanchor - selfanchor" % (st // 1000), C(raT, 0, saT, 0, st))
    row("%dk abrupt - none (literature's measure)" % (st // 1000), C(abT, 0, bl, 0, st))
P("  means 100k: constantT2 %.2f abrupt %.2f selfanchor %.2f randanchor %.2f blank %.2f" % tuple(mean(x, 100000) for x in (c2, abT, saT, raT, bl)))
for nm, X in (("abrupt", abT), ("selfanchor", saT), ("randanchor", raT)):
    row("plateau %s 100k-80k" % nm, [X[s][100000] - X[s][80000] for s in range(12)])

io.open(os.path.join(W, "tcds_v3", "v3b_stats.txt"), "w", encoding="utf-8").write("\n".join(out))
