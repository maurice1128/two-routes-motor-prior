# -*- coding: utf-8 -*-
"""Statistics the v3 revision adds, all from the per-seed JSON already on disk.

D2  plateau: paired change over the last 20k (100k - 80k) with a 95% CI, every arm, both bodies
D1  multiplicity: Holm-corrected checkpoint counts for the three core contrasts, plus the
    three pre-specified endpoints (12k, 50k, 100k)
D4  robustness of every starred / bold contrast: exact Wilcoxon signed-rank p (n=12) and a
    BCa bootstrap 95% interval
A1  abrupt - none (the literature's dependence measure) at every t_w, both bodies
C1  myoHand prior - blank at 60k, n=6 (the 12k study's third body)
cost: wall-clock from the run logs

No scipy on this machine: Wilcoxon is enumerated exactly over the 2^12 sign patterns,
the bootstrap is plain numpy.
"""
import json, os, glob, math, re, io, itertools
import numpy as np

W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
Z = r"C:\Users\maurice\Desktop\world_model_zeroshot"
T = 2.201
CK = list(range(2000, 100001, 2000))
rng = np.random.default_rng(0)
out = []


def P(s=""):
    print(s); out.append(s)


def load(root, d, c, n=12):
    return {s: {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(os.path.join(root, d, "%s_seed%d.json" % (c, s))))} for s in range(n)}


def pt(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n)
    tq = {12: 2.201, 6: 2.571}[n]
    return m, m - tq * se, m + tq * se


def wilcoxon_exact(d):
    """Two-sided exact signed-rank p for n<=12 with no ties (ties broken by tiny noise)."""
    d = np.asarray(d, float); d = d[d != 0]; n = len(d)
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    wpos = ranks[d > 0].sum()
    tot = n * (n + 1) / 2
    # distribution of W+ under H0: all sign patterns equally likely
    dist = np.zeros(int(tot) + 1)
    for signs in itertools.product([0, 1], repeat=n):
        dist[int(sum(r for r, s in zip(ranks, signs) if s))] += 1
    dist /= dist.sum()
    stat = min(wpos, tot - wpos)
    p = 2 * dist[:int(stat) + 1].sum()
    return min(1.0, p)


def bca(d, B=20000):
    d = np.asarray(d, float); n = len(d); m = d.mean()
    boots = np.array([rng.choice(d, n, replace=True).mean() for _ in range(B)])
    z0 = _ndtri((boots < m).mean())
    jack = np.array([np.delete(d, i).mean() for i in range(n)])
    jm = jack.mean(); num = ((jm - jack) ** 3).sum(); den = 6 * (((jm - jack) ** 2).sum()) ** 1.5
    a = num / den if den else 0.0
    def q(alpha):
        z = _ndtri(alpha); adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        return np.quantile(boots, _ndtr(adj))
    return q(0.025), q(0.975)


def _ndtr(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _ndtri(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _ndtr(mid) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def holm(pvals, alpha=0.05):
    idx = np.argsort(pvals); m = len(pvals); rej = np.zeros(m, bool)
    for k, i in enumerate(idx):
        if pvals[i] <= alpha / (m - k): rej[i] = True
        else: break
    return rej


def t_p(d):
    d = np.asarray(d, float); n = len(d); t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    return 2 * (1 - _tcdf(abs(t), n - 1))


def _tcdf(t, df):
    # Student t CDF via incomplete beta (numerical integration, fine for df=11)
    x = df / (df + t * t)
    # regularized incomplete beta I_x(df/2, 1/2) by Simpson integration
    a, b = df / 2, 0.5
    f = lambda u: u ** (a - 1) * (1 - u) ** (b - 1)
    N = 20000; us = np.linspace(1e-12, x, N + 1); ys = f(us)
    integ = (x / N / 3) * (ys[0] + ys[-1] + 4 * ys[1:-1:2].sum() + 2 * ys[2:-1:2].sum())
    B = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    return 1 - 0.5 * integ / B


ARMS = [("blank", "blank"), ("prior", "prior"), ("randprior", "prior"), ("coach", "coach"), ("randcoach", "coach"), ("priorcoach", "priorcoach")]
D = {tag: {lab: load(W, "results_conv_%s_%s" % (tag, lab), c) for lab, c in ARMS} for tag in ("elbow", "finger")}

P("== D2 plateau: paired change 100k-80k, mean [95% CI], per arm")
for tag in ("elbow", "finger"):
    for lab, _ in ARMS:
        v = [D[tag][lab][s][100000] - D[tag][lab][s][80000] for s in range(12)]
        m, lo, hi = pt(v); P("  %-6s %-10s %+6.2f [%+6.2f, %+6.2f]%s" % (tag, lab, m, lo, hi, "  *" if lo > 0 or hi < 0 else ""))
B = {a: load(W, "results_matched100k_elbow", a) for a in ("keep", "soft", "purge", "matched")}
C = {"constant": load(W, "results_coach100k_elbow", "constant"), "abrupt": load(W, "results_coach100k_elbow", "abrupt"), "selfanchor": load(W, "results_coach100k_elbow_off2", "selfanchor")}
for name, src in (("withdraw-body", B), ("withdraw-teacher", C)):
    for a, v in src.items():
        vv = [v[s][100000] - v[s][80000] for s in range(12)]
        m, lo, hi = pt(vv); P("  elbow  %-10s %+6.2f [%+6.2f, %+6.2f]  (%s)" % (a, m, lo, hi, name))

P("\n== D1 multiplicity over 50 checkpoints (paired t p-values, Holm at 0.05)")
for tag in ("elbow", "finger"):
    for a, b in (("prior", "blank"), ("coach", "blank"), ("coach", "prior")):
        ps = [t_p([D[tag][a][s][st] - D[tag][b][s][st] for s in range(12)]) for st in CK]
        raw = sum(p < 0.05 for p in ps); rej = holm(np.array(ps))
        surv = [CK[i] for i in range(50) if rej[i]]
        ends = {st: pt([D[tag][a][s][st] - D[tag][b][s][st] for s in range(12)]) for st in (12000, 50000, 100000)}
        P("  %-6s %s-%s: raw sig %d/50, Holm-surviving %d/50 %s" % (tag, a, b, raw, len(surv), ("(first %dk, last %dk)" % (surv[0] // 1000, surv[-1] // 1000)) if surv else ""))
        P("         pre-specified endpoints: " + "; ".join("%dk %+.2f [%+.2f, %+.2f]" % (st // 1000, *ends[st]) for st in (12000, 50000, 100000)))

P("\n== D4 robustness: exact Wilcoxon p and BCa 95% for starred contrasts")
def rob(name, d):
    m, lo, hi = pt(d); p = wilcoxon_exact(d); blo, bhi = bca(d)
    P("  %-40s t %+7.2f [%+7.2f, %+7.2f]  Wilcoxon p=%.4f  BCa [%+7.2f, %+7.2f]%s" % (name, m, lo, hi, p, blo, bhi, "" if p < 0.05 else "   <-- Wilcoxon n.s."))
end = lambda A, a, b, st=100000: [A[a][s][st] - A[b][s][st] for s in range(12)]
for tag in ("elbow", "finger"):
    for a, b in (("coach", "prior"), ("prior", "blank"), ("coach", "blank"), ("priorcoach", "blank"), ("priorcoach", "coach"), ("prior", "randprior"), ("coach", "randcoach")):
        d = end(D[tag], a, b); m, lo, hi = pt(d)
        if lo > 0 or hi < 0: rob("%s %s-%s 100k" % (tag, a, b), d)
M12 = {a: load(W, "results_matched_elbow", a) for a in ("keep", "soft", "purge", "matched")}
rob("elbow matched-keep 12k", end(M12, "matched", "keep", 12000)); rob("elbow soft-matched 12k", end(M12, "soft", "matched", 12000))
rob("elbow soft-keep 100k", end(B, "soft", "keep")); rob("elbow abrupt-constant 100k", end(C, "abrupt", "constant")); rob("elbow selfanchor-constant 100k", end(C, "selfanchor", "constant"))
ZW = {tag: {a: load(Z, "b2_pilot/results_withdrawal_%s" % body, a) for a in ("none", "constant", "abrupt", "selfanchor", "randanchor")} for tag, body in (("elbow", "myoelbow"), ("finger", "myofinger"))}
dip = lambda A, a, b: [(A[a][s][9000] - A[a][s][8000]) - (A[b][s][9000] - A[b][s][8000]) for s in range(12)]
for tag in ("elbow", "finger"):
    rob("%s abrupt-constant dip" % tag, dip(ZW[tag], "abrupt", "constant")); rob("%s abrupt-constant 12k" % tag, end(ZW[tag], "abrupt", "constant", 12000))
    rob("%s selfanchor-constant 12k" % tag, end(ZW[tag], "selfanchor", "constant", 12000)); rob("%s abrupt-selfanchor dip" % tag, dip(ZW[tag], "abrupt", "selfanchor"))

P("\n== A1 abrupt - none (the literature's dependence measure) by t_w; dip / post-4k / endpoint-12k; slope per 1k")
TWS = [2000, 3000, 4000, 5000, 6000, 7000, 8000]
for tag, body in (("elbow", "myoelbow"), ("finger", "myofinger")):
    none = ZW[tag]["none"]; ab = {tw: load(Z, "b2_pilot/results_attachment_%s" % body, "abrupt_w%d" % tw) for tw in TWS}
    for mname, f in (("dip", lambda c, tw: c[tw + 1000] - c[tw]), ("post-4k", lambda c, tw: c[tw + 4000]), ("endpoint", lambda c, tw: c[12000])):
        cells = []; per = {s: [] for s in range(12)}
        for tw in TWS:
            v = [f(ab[tw][s], tw) - f(none[s], tw) for s in range(12)]
            for s in range(12): per[s].append(v[s])
            m, lo, hi = pt(v); cells.append("%+.1f%s" % (m, "*" if lo > 0 or hi < 0 else ""))
        xs = np.array(TWS) / 1000.0
        slopes = [np.polyfit(xs, per[s], 1)[0] for s in range(12)]
        m, lo, hi = pt(slopes)
        P("  %-6s %-8s " % (tag, mname) + "  ".join(cells) + "   slope %+.2f [%+.2f, %+.2f]%s" % (m, lo, hi, "*" if lo > 0 or hi < 0 else ""))
    v = [ZW[tag]["abrupt"][s][12000] - none[s][12000] for s in range(12)]
    P("  %-6s abrupt(8k)-none at 12k from the withdrawal dir: %+.2f [%+.2f, %+.2f]" % ((tag,) + pt(v)))

P("\n== C1 myoHand (12k study's third body), n=6, 60k")
H = {a: load(W, "results_hand", a, 6) for a in ("blank", "prior")}
last = max(H["blank"][0].keys())
for st in (12000, 30000, last):
    v = [H["prior"][s][st] - H["blank"][s][st] for s in range(6)]
    P("  prior-blank at %dk: %+.2f [%+.2f, %+.2f] mm  (blank %.1f, prior %.1f)" % ((st // 1000,) + pt(v) + (np.mean([H["blank"][s][st] for s in range(6)]), np.mean([H["prior"][s][st] for s in range(6)]))))

P("\n== compute: wall-clock from run logs (last '(Ns)' per log)")
tot = 0; nlog = 0
for d in glob.glob(os.path.join(W, "results_conv_*")) + glob.glob(os.path.join(W, "results_matched*")) + glob.glob(os.path.join(W, "results_coach100k*")):
    for f in glob.glob(os.path.join(d, "*.log")):
        txt = io.open(f, encoding="utf-8", errors="ignore").read()
        m = re.findall(r"\((\d+)s\)", txt)
        if m: tot += int(m[-1]); nlog += 1
P("  own 100k-era logs: %d runs, %.0f CPU-hours (single-thread runs), mean %.1f min/run" % (nlog, tot / 3600, tot / nlog / 60 if nlog else 0))
tot2 = 0; n2 = 0
for d in glob.glob(os.path.join(Z, "b2_pilot", "logs*")) + [os.path.join(Z, "b1_pilot")]:
    for f in glob.glob(os.path.join(d, "*.log")):
        txt = io.open(f, encoding="utf-8", errors="ignore").read()
        for m in re.findall(r"done in (\d+)s", txt): tot2 += int(m); n2 += 1
P("  carried 12k-era logs: %d cells with 'done in', %.0f CPU-hours" % (n2, tot2 / 3600))
io.open(os.path.join(W, "tcds_v3", "v3_stats.txt"), "w", encoding="utf-8").write("\n".join(out))
