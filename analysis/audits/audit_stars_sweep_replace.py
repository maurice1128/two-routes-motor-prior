# -*- coding: utf-8 -*-
"""Significance stars in the sweep table (tab:sweep) and the replacing-regime
table (tab:replace): every cell's * must match a paired-t interval excluding
zero, recomputed from the other draft's JSON (read only)."""
import json, os, re, math, io
Z = r"C:\Users\maurice\Desktop\world_model_zeroshot"
TEX = r"C:\Users\maurice\Desktop\robotic_research\wm_prior\tcds_merged\paper.tex"
T = 2.201
tex = io.open(TEX, encoding="utf-8").read()
bad = checks = 0


def load(d, stem):
    return {s: {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(os.path.join(Z, d, "%s_seed%d.json" % (stem, s))))} for s in range(12)}


def pt(v):
    n = len(v); m = sum(v) / n; se = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1) / n)
    return m, m - T * se, m + T * se


def sig(t3):
    return not (t3[1] < 0 < t3[2])


def cmp(name, star, t3):
    global bad, checks
    checks += 1
    if star != sig(t3):
        bad += 1; print("  MISMATCH %-40s star=%s ci=%.2f [%.2f,%.2f]" % (name, star, *t3))


def block(label):
    return re.search(r"\\label\{" + re.escape(label) + r"\}(.*?)\\end\{tabular\}", tex, re.S).group(1)


# ---------------- sweep ----------------
TWS = [2000, 3000, 4000, 5000, 6000, 7000, 8000]
def sweep_rows(body_label):
    blk = re.search(r"\\emph\{" + body_label + r" \((?:mrad|mm)\)\}\}\\\\(.*?)(?:\\midrule|\\bottomrule)", block("tab:sweep"), re.S).group(1)
    rows = {}; cur = None
    for line in blk.strip().splitlines():
        if "&" not in line: continue
        parts = [p.strip() for p in line.split("&")]
        if parts[0]: cur = parts[0]
        rows[(cur, parts[1])] = ([("^*" in p) for p in parts[2:9]], "^*" in parts[9])
    return rows


for body, label in (("myoelbow", "myoElbow"), ("myofinger", "myoFinger")):
    dw = "b2_pilot/results_withdrawal_%s" % body; da = "b2_pilot/results_attachment_%s" % body
    const = load(dw, "constant")
    ab = {tw: load(da, "abrupt_w%d" % tw) for tw in TWS}; sa = {tw: load(da, "selfanchor_w%d" % tw) for tw in TWS}
    metrics = {"dip": lambda c, tw: c[tw + 1000] - c[tw], "post-4k": lambda c, tw: c[tw + 4000], "endpoint": lambda c, tw: c[12000]}
    comps = {"teacher": (sa, const), "loss term": (ab, sa), "total": (ab, const)}
    want = sweep_rows(label)
    for (mname, cname), (stars, sstar) in want.items():
        mf = metrics[mname]; X, Y = comps[cname]
        per_seed = {s: [] for s in range(12)}
        for i, tw in enumerate(TWS):
            Ya = Y if 0 in Y else Y[tw]
            vals = [mf(X[tw][s], tw) - mf(Ya[s], tw) for s in range(12)]
            for s in range(12): per_seed[s].append(vals[s])
            cmp("sweep %s %s/%s tw=%d" % (body, mname, cname, tw), stars[i], pt(vals))
        xs = [tw / 1000.0 for tw in TWS]; xm = sum(xs) / 7
        slopes = []
        for s in range(12):
            ys = per_seed[s]; ym = sum(ys) / 7
            slopes.append(sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs))
        cmp("sweep %s %s/%s slope" % (body, mname, cname), sstar, pt(slopes))

# ---------------- replace ----------------
dr = "b2_pilot/results_replace_myoelbow"; RT = [2000, 4000, 6000, 8000]
R = {"constant": load(dr, "constant")}
for a in ("abrupt", "selfanchor", "gateon"):
    for tw in RT: R[(a, tw)] = load(dr, "%s_w%d" % (a, tw))
blk = block("tab:replace")
for row, (a, b) in (("selfanchor $-$ constant", ("selfanchor", "constant")), ("gateon $-$ constant", ("gateon", "constant")),
                    ("selfanchor $-$ gateon", ("selfanchor", "gateon")), ("abrupt $-$ selfanchor", ("abrupt", "selfanchor")),
                    ("abrupt $-$ constant", ("abrupt", "constant"))):
    line = [l for l in blk.splitlines() if row in l.split("&")[0]][0]
    cells = [p.strip() for p in line.split("&")[1:]]
    ps = {s: [] for s in range(12)}; es = {s: [] for s in range(12)}
    for i, tw in enumerate(RT):
        Xa = R[(a, tw)] if a != "constant" else R["constant"]; Ya = R[(b, tw)] if b != "constant" else R["constant"]
        d = [(Xa[s][tw + 1000] - Xa[s][tw]) - (Ya[s][tw + 1000] - Ya[s][tw]) for s in range(12)]
        cmp("replace %s dip tw=%d" % (row, tw), "^*" in cells[i], pt(d))
        for s in range(12):
            ps[s].append(Xa[s][tw + 4000] - Ya[s][tw + 4000]); es[s].append(Xa[s][12000] - Ya[s][12000])
    xs = [tw / 1000.0 for tw in RT]; xm = sum(xs) / 4
    def slopes(dd):
        out = []
        for s in range(12):
            ys = dd[s]; ym = sum(ys) / 4
            out.append(sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs))
        return pt(out)
    cmp("replace %s slope post-4k" % row, "^*" in cells[4], slopes(ps))
    cmp("replace %s slope endpoint" % row, "^*" in cells[5], slopes(es))
print("\nstars checked %d, mismatches %d" % (checks, bad))
