# -*- coding: utf-8 -*-
"""Audit the tables carried into the merged draft from the second draft, against
that draft's raw per-seed JSON (read only; nothing under world_model_zeroshot is
written).

Tables: decomposition at tw=8000 (both bodies), attachment sweep (both bodies,
three metrics, per-seed OLS slopes), replacing regime (elbow), budget matching
(both bodies). Same statistics as the paper: paired t, df=11, T=2.201; slopes
are per-seed OLS of a component against tw in thousands, then a paired-t
interval across seeds.

Table cells are printed to 1 decimal in the sweep/replace tables and 2 in the
others; the comparison tolerance follows the printed precision.
"""
import json, os, re, math, io

Z = r"C:\Users\maurice\Desktop\world_model_zeroshot"
TEX = r"C:\Users\maurice\Desktop\robotic_research\wm_prior\tcds_merged\paper.tex"
T = 2.201
tex = io.open(TEX, encoding="utf-8").read().replace("\\mathbf{", "").replace("\\ci{", "CI{")

fails = checks = 0


def load(d, stem):
    out = {}
    for s in range(12):
        p = os.path.join(Z, d, "%s_seed%d.json" % (stem, s))
        if os.path.exists(p):
            out[s] = {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(p))}
    assert len(out) == 12, (d, stem, len(out))
    return out


def pt(vals):
    n = len(vals); m = sum(vals) / n
    se = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1) / n)
    return m, m - T * se, m + T * se


def contrast(a, b, key):
    return pt([key(a[s]) - key(b[s]) for s in range(12)])


def cmp(name, got, want, tol):
    global fails, checks
    checks += 1
    ok = all(abs(g - w) <= tol for g, w in zip(got, want)) and len(got) == len(want)
    if not ok:
        fails += 1
        print("  MISMATCH %-40s tex=%s raw=%s" % (name, want, [round(g, 3) for g in got]))


def nums(s):
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+", s)]


def block(label):
    """The tabular body of the table with this \\label, so row searches cannot
    stray into another table that happens to use the same row names."""
    m = re.search(r"\\label\{" + re.escape(label) + r"\}(.*?)\\end\{tabular\}", tex, re.S)
    assert m, label
    return m.group(1)


DECOMP = block("tab:decomp"); REPL = block("tab:replace"); BUDG = block("tab:budget")


# ---------------- Table: decomposition at tw=8000 ----------------
for tag, body in (("elbow", "myoelbow"), ("finger", "myofinger")):
    d = "b2_pilot/results_withdrawal_%s" % body
    A = {a: load(d, a) for a in ("none", "constant", "abrupt", "selfanchor", "randanchor")}
    dip = lambda c: c[9000] - c[8000]; end = lambda c: c[12000]
    col = 1 if tag == "elbow" else 3
    for row, (a, b) in (("abrupt \\$-\\$ constant", ("abrupt", "constant")),
                        ("selfanchor \\$-\\$ constant", ("selfanchor", "constant")),
                        ("abrupt \\$-\\$ selfanchor", ("abrupt", "selfanchor")),
                        ("randanchor \\$-\\$ selfanchor", ("randanchor", "selfanchor"))):
        m = re.search(r"(?m)^" + row + r"[^&\n]*&(.*?)\\\\", DECOMP)
        if not m:
            print("  decomp row not found:", row); fails += 1; continue
        cells = [c.strip() for c in m.group(1).split("&")]
        cmp("decomp %s %s dip" % (tag, row[:10]), [contrast(A[a], A[b], dip)[0]], nums(cells[col - 1]), 0.006)
        cmp("decomp %s %s end" % (tag, row[:10]), [contrast(A[a], A[b], end)[0]], nums(cells[col]), 0.006)
    m = re.search(r"(?m)^constant \$-\$ none[^&\n]*&(.*?)\\\\", DECOMP)
    if not m:
        print("  decomp row not found: constant - none"); fails += 1; continue
    cells = [c.strip() for c in m.group(1).split("&")]
    cmp("decomp %s constant-none end" % tag, [contrast(A["constant"], A["none"], end)[0]], nums(cells[col]), 0.006)

# ---------------- Table: attachment sweep ----------------
TWS = [2000, 3000, 4000, 5000, 6000, 7000, 8000]


def sweep_table(tag, body):
    dw = "b2_pilot/results_withdrawal_%s" % body
    da = "b2_pilot/results_attachment_%s" % body
    const = load(dw, "constant"); none = load(dw, "none")
    ab = {tw: load(da, "abrupt_w%d" % tw) for tw in TWS}
    sa = {tw: load(da, "selfanchor_w%d" % tw) for tw in TWS}
    metrics = {"dip": lambda c, tw: c[tw + 1000] - c[tw],
               "post-4k": lambda c, tw: c[tw + 4000],
               "endpoint": lambda c, tw: c[12000]}
    comps = {"teacher": (sa, const), "loss term": (ab, sa), "total": (ab, const)}
    out = {}
    for mname, mf in metrics.items():
        for cname, (X, Y) in comps.items():
            per_seed = {s: [] for s in range(12)}
            cells = []
            for tw in TWS:
                Xa = X[tw]; Ya = Y if isinstance(Y, dict) and 0 in Y else Y[tw]
                vals = [mf(Xa[s], tw) - mf(Ya[s], tw) for s in range(12)]
                for s in range(12):
                    per_seed[s].append(vals[s])
                cells.append(pt(vals)[0])
            # per-seed OLS slope vs tw (thousands)
            xs = [tw / 1000.0 for tw in TWS]; xm = sum(xs) / len(xs)
            slopes = []
            for s in range(12):
                ys = per_seed[s]; ym = sum(ys) / len(ys)
                slopes.append(sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs))
            out[(mname, cname)] = (cells, pt(slopes))
    return out


# parse the sweep table rows from the tex: within the body block, rows "metric & comp & 7 cells & slope"
def sweep_rows(tex, body_label):
    blk = re.search(r"\\emph\{" + body_label + r" \((?:mrad|mm)\)\}\}\\\\(.*?)(?:\\midrule|\\bottomrule)", tex, re.S).group(1)
    rows = {}; cur = None
    for line in blk.strip().splitlines():
        if "&" not in line: continue
        parts = [p.strip() for p in line.split("&")]
        if parts[0]: cur = parts[0]
        comp = parts[1]
        vals = [nums(p)[0] for p in parts[2:9]]
        sl = nums(parts[9])
        rows[(cur, comp)] = (vals, sl)
    return rows


for tag, body, label in (("elbow", "myoelbow", "myoElbow"), ("finger", "myofinger", "myoFinger")):
    got = sweep_table(tag, body); want = sweep_rows(tex, label)
    for k, (wcells, wsl) in want.items():
        if k not in got:
            print("  sweep %s row %s not computed" % (tag, k)); fails += 1; continue
        gcells, gsl = got[k]
        cmp("sweep %s %s/%s cells" % (tag, k[0], k[1]), gcells, wcells, 0.06)
        cmp("sweep %s %s/%s slope" % (tag, k[0], k[1]), list(gsl), wsl, 0.006)

# ---------------- Table: replacing regime (elbow) ----------------
dr = "b2_pilot/results_replace_myoelbow"
RT = [2000, 4000, 6000, 8000]
R = {"constant": load(dr, "constant"), "none": load(dr, "none")}
for a in ("abrupt", "selfanchor", "gateon"):
    for tw in RT:
        R[(a, tw)] = load(dr, "%s_w%d" % (a, tw))


def rep_row(a, b):
    dips = []; ps = {s: [] for s in range(12)}; es = {s: [] for s in range(12)}
    for tw in RT:
        Xa = R[(a, tw)] if a != "constant" else R["constant"]
        Ya = R[(b, tw)] if b != "constant" else R["constant"]
        d = [(Xa[s][tw + 1000] - Xa[s][tw]) - (Ya[s][tw + 1000] - Ya[s][tw]) for s in range(12)]
        dips.append(pt(d)[0])
        for s in range(12):
            ps[s].append(Xa[s][tw + 4000] - Ya[s][tw + 4000]); es[s].append(Xa[s][12000] - Ya[s][12000])
    xs = [tw / 1000.0 for tw in RT]; xm = sum(xs) / 4
    def slopes(dd):
        out = []
        for s in range(12):
            ys = dd[s]; ym = sum(ys) / 4
            out.append(sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs))
        return pt(out)
    return dips, slopes(ps), slopes(es)


for row, (a, b) in (("selfanchor \\$-\\$ constant", ("selfanchor", "constant")),
                    ("gateon \\$-\\$ constant", ("gateon", "constant")),
                    ("selfanchor \\$-\\$ gateon", ("selfanchor", "gateon")),
                    ("abrupt \\$-\\$ selfanchor", ("abrupt", "selfanchor")),
                    ("abrupt \\$-\\$ constant", ("abrupt", "constant"))):
    m = re.search(r"(?m)^(?:\\textbf\{)?" + row + r"\}?[^&\n]*&(.*?)\\\\", REPL)
    if not m:
        print("  replace row not found:", row); fails += 1; continue
    cells = [c.strip() for c in m.group(1).split("&")]
    dips, sp, se_ = rep_row(a, b)
    cmp("replace %s dips" % row[:12], dips, [nums(c)[0] for c in cells[:4]], 0.06)
    cmp("replace %s slope post4k" % row[:12], list(sp), nums(cells[4]), 0.006)
    cmp("replace %s slope endpoint" % row[:12], list(se_), nums(cells[5]), 0.006)
# endpoints quoted in the caption: constant 27.86; abrupt 107.7,133.0,168.2,180.8; selfanchor 45.9,59.8,110.8,86.7
cmp("replace constant endpoint", [sum(c[12000] for c in R["constant"].values()) / 12], [27.86], 0.006)
cmp("replace abrupt endpoints", [sum(R[("abrupt", tw)][s][12000] for s in range(12)) / 12 for tw in RT], [107.7, 133.0, 168.2, 180.8], 0.06)
cmp("replace selfanchor endpoints", [sum(R[("selfanchor", tw)][s][12000] for s in range(12)) / 12 for tw in RT], [45.9, 59.8, 110.8, 86.7], 0.06)

# ---------------- Table: budget matching ----------------
for tag, body, col in (("elbow", "myoelbow", 0), ("finger", "myofinger", 1)):
    db = "b1_pilot/results_budget_%s" % body
    B = {a: load(db, a) for a in ("blank", "coach", "prior_matched", "prior_full")}
    end = lambda c: c[12000]
    means = {a: sum(c[12000] for c in v.values()) / 12 for a, v in B.items()}
    sds = {a: math.sqrt(sum((c[12000] - means[a]) ** 2 for c in v.values()) / 11) for a, v in B.items()}
    for row, a in (("none", "blank"), ("coach, 30k rewarded steps", "coach"),
                   ("prior, 30k babble", "prior_matched"), ("prior, full budget", "prior_full")):
        m = re.search(r"(?m)^" + re.escape(row) + r"\s*&(.*?)\\\\", BUDG)
        if not m:
            print("  budget mean row not found:", row); fails += 1; continue
        cells = [c.strip() for c in m.group(1).split("&")]
        cmp("budget %s %s mean/sd" % (tag, a), [means[a], sds[a]], nums(cells[col]), 0.006)
    for row, (a, b) in (("prior\\$_\\{30\\\\mathrm\\{k\\}\\}\\$ \\$-\\$ prior\\$_\\{\\\\mathrm\\{full\\}\\}\\$", ("prior_matched", "prior_full")),
                        ("prior\\$_\\{\\\\mathrm\\{full\\}\\}\\$ \\$-\\$ coach", ("prior_full", "coach")),
                        ("prior\\$_\\{30\\\\mathrm\\{k\\}\\}\\$ \\$-\\$ coach", ("prior_matched", "coach"))):
        m = re.search(r"(?m)^" + row + r"\s*&(.*?)\\\\", BUDG)
        if not m:
            print("  budget row not found:", a, b); fails += 1; continue
        cells = [c.strip() for c in m.group(1).split("&")]
        cmp("budget %s %s-%s" % (tag, a, b), list(contrast(B[a], B[b], end)), nums(cells[col]), 0.006)

print("\nchecked %d items, mismatches %d" % (checks, fails))
