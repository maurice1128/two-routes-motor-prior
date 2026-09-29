# -*- coding: utf-8 -*-
"""Stage-B audit: recomputes every number in the six tables and in the prose of
tcds_v3/paper.tex from raw per-seed JSON, with code written independently of
make_tables_v3.py, and reports mismatches. Exit code = number of mismatches.

Tables: I core (hand-written), II arms (generated), III decomposition (hand),
IV sweep (generated), V replacing (hand), VI body-model withdrawal (hand).
Prose: every \\ci{...} macro and the bare numbers listed in PROSE below."""
import json, os, re, sys, math, glob, itertools
import numpy as np

W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
Z = r"C:\Users\maurice\Desktop\world_model_zeroshot\b2_pilot"
V3 = os.path.join(W, "tcds_v3")
T95 = {11: 2.201, 23: 2.069}
bad = []; nitems = 0


def L(d, c, seeds=None, root=W):
    r = {}
    for f in glob.glob(os.path.join(root, d, "%s_seed*.json" % c)):
        s = int(re.search(r"_seed(\d+)\.json$", f).group(1))
        if seeds is None or s in seeds:
            r[s] = {int(x["step"]): x["eval_dist"] * 1000.0 for x in json.load(open(f))}
    return r


def tci(v):
    v = np.asarray(v, float); n = len(v); m = float(v.mean()); se = float(v.std(ddof=1)) / math.sqrt(n)
    t = T95[n - 1]; return m, m - t * se, m + t * se


def check(label, got, want, tol=0.0051):
    global nitems
    nitems += 1
    if abs(got - want) > tol:
        bad.append("%s: paper %s, recomputed %.4f" % (label, want, got))


def check3(label, v, want3, tol=0.0051):
    m, lo, hi = tci(v)
    for name, g, w in (("mean", m, want3[0]), ("lo", lo, want3[1]), ("hi", hi, want3[2])):
        check(label + " " + name, g, w, tol)


def check_sig(label, v, starred):
    global nitems
    nitems += 1
    m, lo, hi = tci(v); sig = lo > 0 or hi < 0
    if sig != starred:
        bad.append("%s: paper %s, recomputed %s [%.2f, %.2f]" % (label, "sig" if starred else "n.s.", "sig" if sig else "n.s.", lo, hi))


def wilcoxon(v):
    v = np.asarray(v, float); v = v[v != 0]; n = len(v); r = np.argsort(np.argsort(np.abs(v))) + 1.0
    wp = r[v > 0].sum(); cnt = tot = 0
    for m in range(1 << n):
        s = sum(r[i] for i in range(n) if m >> i & 1); tot += 1; cnt += (s <= wp)
    p1 = cnt / tot; return 2 * min(p1, 1 - p1)


tex = open(os.path.join(V3, "paper.tex"), encoding="utf-8").read()
tex = tex.replace("\\input{tables/arms}", open(os.path.join(V3, "tables", "arms.tex"), encoding="utf-8").read())
tex = tex.replace("\\input{tables/sweep}", open(os.path.join(V3, "tables", "sweep.tex"), encoding="utf-8").read())
tex = tex.replace("\\input{tables/replace}", open(os.path.join(V3, "tables", "replace.tex"), encoding="utf-8").read())


def table_block(label):
    i = tex.index("\\label{%s}" % label); j = tex.index("\\end{tabular}", i)
    return tex[i:j]


def parse_ci_cells(line):
    """all '+x [lo, hi]' cells in a table line, with bold flag"""
    out = []
    for m in re.finditer(r"(\\mathbf\{)?([+-]\d+\.\d+)\}?\\?\s*\[([+-]\d+\.\d+),\s*([+-]\d+\.\d+)\]", line):
        out.append((m.group(1) is not None, float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return out


def strip_slopes(line):
    """remove slope cells ('$+x$ $[lo,\\,hi]^*$' or the bold form) so the plain star-cell parser sees only the t_w cells"""
    return re.sub(r"(\\mathbf\{[+-]\d+\.\d+\}|\$[+-]\d+\.\d+)\$ \$\[[^\]]*\](\^\*)?\$", "SLOPE", line)


def parse_star_cells(line):
    out = []
    for m in re.finditer(r"\$([+-]\d+\.\d+)(\^\*|\^\{\*\\ddagger\})?\$", line):
        out.append((m.group(2) is not None, float(m.group(1))))
    return out


S12 = range(12)
# ------------------------------------------------------------------ data
E = {k: L("results_conv_elbow_%s" % k, c) for k, c in (("blank", "blank"), ("blank64", "blank64"), ("prior", "prior"), ("randprior", "prior"), ("coach", "coach"), ("randcoach", "coach"), ("priorcoach", "priorcoach"))}
F = {k: L("results_conv_finger_%s" % k, c) for k, c in (("blank", "blank"), ("blank64", "blank64"), ("prior", "prior"), ("randprior", "prior"), ("coach", "coach"), ("randcoach", "coach"), ("priorcoach", "priorcoach"), ("coachT2", "coach"), ("priorcoachT2", "priorcoach"))}
d100 = lambda A, x, y, st=100000: [A[x][s][st] - A[y][s][st] for s in S12]

# ------------------------------------------------------------------ Table I
blk = table_block("tab:conv")
rows = [l for l in blk.split("\n") if l.strip().endswith("\\\\")]
# means
def means_row(prefix):
    l = [r for r in rows if r.startswith(prefix)][0]
    return [float(x) for x in re.findall(r"(\d+\.\d+)", l)]
m = means_row("\\texttt{blank} / \\texttt{blank64}")
for got, want in zip([np.mean([E["blank"][s][100000] for s in S12]), np.mean([E["blank64"][s][100000] for s in S12]), np.mean([F["blank"][s][100000] for s in S12]), np.mean([F["blank64"][s][100000] for s in S12])], m): check("T1 mean blank/blank64", got, want)
m = means_row("\\texttt{prior} / \\texttt{randprior}")
for got, want in zip([np.mean([E["prior"][s][100000] for s in S12]), np.mean([E["randprior"][s][100000] for s in S12]), np.mean([F["prior"][s][100000] for s in S12]), np.mean([F["randprior"][s][100000] for s in S12])], m): check("T1 mean prior/randprior", got, want)
m = means_row("\\texttt{coach} / \\texttt{randcoach}")
for got, want in zip([np.mean([E["coach"][s][100000] for s in S12]), np.mean([E["randcoach"][s][100000] for s in S12]), np.mean([F["coach"][s][100000] for s in S12]), np.mean([F["randcoach"][s][100000] for s in S12]), np.mean([F["coachT2"][s][100000] for s in S12]), np.mean([F["randcoach"][s][100000] for s in S12])], m): check("T1 mean coach/randcoach", got, want)
m = means_row("\\texttt{priorcoach} ")
for got, want in zip([np.mean([E["priorcoach"][s][100000] for s in S12]), np.mean([F["priorcoach"][s][100000] for s in S12]), np.mean([F["priorcoachT2"][s][100000] for s in S12])], m): check("T1 mean priorcoach", got, want)
CONTR = {"prior}$-$\\texttt{blank}": ("prior", "blank"), "prior}$-$\\texttt{blank64}": ("prior", "blank64"), "blank64}$-$\\texttt{blank}": ("blank64", "blank"),
         "prior}$-$\\texttt{randprior}": ("prior", "randprior"), "randprior}$-$\\texttt{blank}": ("randprior", "blank"),
         "coach}$-$\\texttt{blank}": ("coach", "blank"), "coach}$-$\\texttt{prior}": ("coach", "prior"), "coach}$-$\\texttt{randcoach}": ("coach", "randcoach"), "priorcoach}$-$\\texttt{coach}": ("priorcoach", "coach")}
for key, (x, y) in CONTR.items():
    l = [r for r in rows if r.startswith("\\texttt{" + key)][0]
    cells = parse_ci_cells(l)
    vs = [d100(E, x, y), d100(F, x, y)]
    if x in ("coach", "priorcoach"):
        vs.append(d100(F, x + "T2" if x == "priorcoach" else "coachT2", ("coachT2" if y == "coach" else y)))
    assert len(cells) == len(vs), (key, len(cells), len(vs))
    for (bold, mm, lo, hi), v in zip(cells, vs):
        check3("T1 %s" % key, v, (mm, lo, hi)); check_sig("T1 %s bold" % key, v, bold)
l = [r for r in rows if "coach}$_{\\mathrm{T2}}" in r][0]
(bold, mm, lo, hi), = parse_ci_cells(l)
v = d100(F, "coachT2", "coach"); check3("T1 coachT2-coachT1", v, (mm, lo, hi)); check_sig("T1 coachT2-coachT1 bold", v, bold)
# Wilcoxon flags in Table I: exactly the two ddagger cells fail at 0.05
for key, (x, y) in CONTR.items():
    l = [r for r in rows if r.startswith("\\texttt{" + key)][0]
    for bi, body in enumerate((E, F)):
        v = d100(body, x, y); m_, lo, hi = tci(v)
        if lo > 0 or hi < 0:
            p = wilcoxon(v); flagged = ("^\\ddagger" in l.split("&")[bi + 1])
            nitems += 1
            if (p >= 0.05) != flagged: bad.append("T1 Wilcoxon flag %s body%d: p=%.4f flagged=%s" % (key, bi, p, flagged))

# ------------------------------------------------------------------ Table II (arms)
PRIOR_DIR = {"arm1": "prior", "arm2": "prior", "arm3": "prior", "arm4": "prior"}
for line in open(os.path.join(V3, "make_tables_v3.py"), encoding="utf-8"):
    if line.startswith("PRIOR_DIR"):
        PRIOR_DIR = eval(line.split("=", 1)[1].split("#")[0])
blk = table_block("tab:arms"); rows = [l for l in blk.split("\n") if l.strip().endswith("\\\\")]
ARMS = {}
for b in ("arm1", "arm2", "arm3", "arm4"):
    A = {"blank": L("results_arms100k_%s_blank" % b, "blank"), "blank64": L("results_arms100k_%s_blank64" % b, "blank64"),
         "prior": L("results_arms100k_%s_%s" % (b, PRIOR_DIR[b]), "prior"), "randprior": L("results_arms100k_%s_randprior" % b, "prior")}
    seeds = sorted(set(A["blank"]) & set(A["blank64"]) & set(A["prior"]) & set(A["randprior"])); ARMS[b] = (A, seeds)
hdr = [r for r in rows if "joint" in r][0]
for (b, (A, seeds)), n_paper in zip(ARMS.items(), [int(x) for x in re.findall(r"n\{=\}(\d+)", hdr)]):
    check("T2 n %s" % b, len(seeds), n_paper, 0)
for name in ("blank", "blank64", "prior", "randprior"):
    l = [r for r in rows if r.startswith("\\texttt{%s} &" % name)][0]
    for (b, (A, seeds)), want in zip(ARMS.items(), [float(x) for x in re.findall(r"& (\d+\.\d+)", l)]):
        check("T2 mean %s %s" % (name, b), np.mean([A[name][s][100000] for s in seeds]), want, 0.051)
for key, (x, y) in {"prior}$-$\\texttt{blank}": ("prior", "blank"), "prior}$-$\\texttt{blank64}": ("prior", "blank64"), "blank64}$-$\\texttt{blank}": ("blank64", "blank"),
                    "randprior}$-$\\texttt{blank}": ("randprior", "blank"), "prior}$-$\\texttt{randprior}": ("prior", "randprior")}.items():
    l = [r for r in rows if r.startswith("\\texttt{" + key)][0]
    for (b, (A, seeds)), (bold, mm, lo, hi) in zip(ARMS.items(), parse_ci_cells(l)):
        v = [A[x][s][100000] - A[y][s][100000] for s in seeds]
        check3("T2 %s %s" % (key, b), v, (mm, lo, hi)); check_sig("T2 %s %s bold" % (key, b), v, bold)
l = [r for r in rows if "as \\% of" in r][0]
for (b, (A, seeds)), want in zip(ARMS.items(), [float(x) for x in re.findall(r"([+-]\d+)\\%", l)]):
    v = [A["prior"][s][100000] - A["blank"][s][100000] for s in seeds]
    check("T2 pct %s" % b, round(100 * np.mean(v) / np.mean([A["blank"][s][100000] for s in seeds])), want, 0)

# ------------------------------------------------------------------ Table III (decomposition at 8k)
Ecoach = L("results_coach100k_elbow", "constant"); Eab = L("results_coach100k_elbow", "abrupt"); Esa = L("results_coach100k_elbow_off2", "selfanchor"); Era = L("results_coach100k_elbow", "randanchor")
zc = L("results_withdrawal_myoelbow", "constant", root=Z); zab = L("results_withdrawal_myoelbow", "abrupt", root=Z); zsa = L("results_withdrawal_myoelbow", "selfanchor", root=Z); zra = L("results_withdrawal_myoelbow", "randanchor", root=Z); zn = L("results_withdrawal_myoelbow", "none", root=Z)
Fc = L("results_sweep100k_finger_T2", "constant"); Fn = L("results_sweep100k_finger_T2", "none"); Fab = L("results_coach100k_finger_T2", "abrupt"); Fsa = L("results_coach100k_finger_T2", "selfanchor"); Fra = L("results_coach100k_finger_T2", "randanchor")
dip = lambda r, s: r[s][9000] - r[s][8000]
ROWS3 = {"abrupt $-$ constant": [("d", zab, zc), (12000, Eab, Ecoach), (100000, Eab, Ecoach), ("d", Fab, Fc), (12000, Fab, Fc), (100000, Fab, Fc)],
         "selfanchor $-$ constant": [("d", zsa, zc), (12000, Esa, Ecoach), (100000, Esa, Ecoach), ("d", Fsa, Fc), (12000, Fsa, Fc), (100000, Fsa, Fc)],
         "abrupt $-$ selfanchor": [("d", zab, zsa), (12000, Eab, Esa), (100000, Eab, Esa), ("d", Fab, Fsa), (12000, Fab, Fsa), (100000, Fab, Fsa)],
         "randanchor $-$ selfanchor": [("d", zra, zsa), (12000, Era, Esa), (100000, Era, Esa), ("d", Fra, Fsa), (12000, Fra, Fsa), (100000, Fra, Fsa)],
         "abrupt $-$ none": [("d", zab, zn), (12000, Eab, E["blank"]), (100000, Eab, E["blank"]), ("d", Fab, Fn), (12000, Fab, Fn), (100000, Fab, Fn)],
         "constant $-$ none": [None, (12000, Ecoach, E["blank"]), (100000, Ecoach, E["blank"]), None, (12000, Fc, Fn), (100000, Fc, Fn)]}
blk = table_block("tab:decomp"); rows = [l for l in blk.split("\n") if l.strip().endswith("\\\\")]
for name, spec in ROWS3.items():
    l = [r for r in rows if r.startswith(name)][0]
    cells = parse_star_cells(l); spec = [x for x in spec if x is not None]
    assert len(cells) == len(spec), (name, len(cells), len(spec))
    for (star, val), (st, a, b) in zip(cells, spec):
        v = [dip(a, s) - dip(b, s) for s in S12] if st == "d" else [a[s][st] - b[s][st] for s in S12]
        check("T3 %s %s" % (name, st), tci(v)[0], val); check_sig("T3 %s %s star" % (name, st), v, star)
        if star:
            nitems += 1
            flagged = ("ddagger" in l) and st == 100000 and name.startswith("constant") and a is Ecoach
            if (wilcoxon(v) >= 0.05) != flagged: bad.append("T3 %s %s: starred, Wilcoxon p=%.4f, flagged=%s" % (name, st, wilcoxon(v), flagged))

# ------------------------------------------------------------------ Table IV (sweep)
TWS = [2000, 3000, 4000, 5000, 6000, 7000, 8000]
def sweep_data(body):
    if body == "elbow":
        con, none = Ecoach, E["blank"]
        arms = {tw: (L("results_sweep100k_elbow/w%d" % tw, "abrupt"), L("results_sweep100k_elbow/w%d" % tw, "selfanchor")) for tw in TWS[:-1]}; arms[8000] = (Eab, Esa)
        darms = {tw: (L("results_attachment_myoelbow", "abrupt_w%d" % tw, root=Z), L("results_attachment_myoelbow", "selfanchor_w%d" % tw, root=Z)) for tw in TWS}
        return con, none, arms, darms, zc, zn
    con, none = Fc, Fn
    arms = {tw: (L("results_sweep100k_finger_T2/w%d" % tw, "abrupt"), L("results_sweep100k_finger_T2/w%d" % tw, "selfanchor")) for tw in TWS[:-1]}; arms[8000] = (Fab, Fsa)
    return con, none, arms, arms, con, none
blk = table_block("tab:sweep"); rows = [l for l in blk.split("\n") if l.strip().endswith("\\\\") and "&" in l and "2k &" not in l]
COMP = ["teacher", "loss term", "total", "abrupt $-$ none"]
ri = 0
for body in ("elbow", "finger"):
    con, none, arms, darms, dcon, dnone = sweep_data(body)
    for metric in ("dip", "100k"):
        for comp in COMP:
            l = rows[ri]; ri += 1
            assert comp in l, (body, metric, comp, l)
            cells = parse_star_cells(strip_slopes(l))[:7]
            sl = re.search(r"([+-]\d+\.\d+)\}?\$ \$\[([+-]\d+\.\d+),\\,([+-]\d+\.\d+)\](\^\*)?", l)
            per = []
            for tw, (star, val) in zip(TWS, cells):
                if metric == "dip":
                    a, sa = darms[tw]; src = {"teacher": (sa, dcon), "loss term": (a, sa), "total": (a, dcon), "abrupt $-$ none": (a, dnone)}[comp]
                    v = [(src[0][s][tw + 1000] - src[0][s][tw]) - (src[1][s][tw + 1000] - src[1][s][tw]) for s in S12]
                else:
                    a, sa = arms[tw]; src = {"teacher": (sa, con), "loss term": (a, sa), "total": (a, con), "abrupt $-$ none": (a, none)}[comp]
                    v = [src[0][s][100000] - src[1][s][100000] for s in S12]
                per.append(v); check("T4 %s %s %s tw%d" % (body, metric, comp, tw), tci(v)[0], val, 0.051); check_sig("T4 %s %s %s tw%d star" % (body, metric, comp, tw), v, star)
            slopes = [np.polyfit(np.array(TWS) / 1000.0, [per[j][s] for j in range(7)], 1)[0] for s in S12]
            check3("T4 %s %s %s slope" % (body, metric, comp), slopes, (float(sl.group(1)), float(sl.group(2)), float(sl.group(3)))); check_sig("T4 %s %s %s slope star" % (body, metric, comp), slopes, sl.group(4) is not None)

# ------------------------------------------------------------------ Table V (replacing)
R = lambda c: L("results_replace100k_myoelbow", c)
Rc = R("constant"); TW4 = [2000, 4000, 6000, 8000]
Rarm = {a: {tw: R("%s_w%d" % (a, tw)) for tw in TW4} for a in ("abrupt", "selfanchor", "gateon")}
RFd = lambda c: L("results_replace100k_myofinger", c)
RFc = RFd("constant"); RFarm = {a: {tw: RFd("%s_w%d" % (a, tw)) for tw in TW4} for a in ("abrupt", "selfanchor", "gateon")}
RFnone = L("results_sweep100k_finger_T2", "none")
blk = table_block("tab:replace")
SPEC5 = {"selfanchor $-$ constant": ("selfanchor", "constant"), "gateon $-$ constant": ("gateon", "constant"), "selfanchor $-$ gateon": ("selfanchor", "gateon"),
         "abrupt $-$ selfanchor": ("abrupt", "selfanchor"), "abrupt $-$ constant": ("abrupt", "constant"), "abrupt $-$ none": ("abrupt", "none")}
BODY5 = {"myoElbow": (Rc, Rarm, E["blank"]), "myoFinger": (RFc, RFarm, RFnone)}
segs = re.split(r"\\multicolumn\{11\}\{@\{\}l\}\{\\emph\{(myoElbow|myoFinger)[^}]*\}\}", blk)
bodies_in_table = []
for bi in range(1, len(segs), 2):
    bname = segs[bi]; bodies_in_table.append(bname); con_, arm_, none_ = BODY5[bname]
    rows = [l for l in segs[bi + 1].split("\n") if l.strip().endswith("\\\\") and "$" in l]
    get = lambda name, tw: con_ if name == "constant" else (none_ if name == "none" else arm_[name][tw])
    for name, (x, y) in SPEC5.items():
        l = [r for r in rows if r.startswith(name)][0]
        cells = parse_star_cells(strip_slopes(l)); slopes_tex = re.findall(r"([+-]\d+\.\d+)\}?\$ \$\[([+-]\d+\.\d+),\\,([+-]\d+\.\d+)\](\^\*)?", l)
        assert len(cells) == 8 and len(slopes_tex) == 2, (bname, name, len(cells), len(slopes_tex))
        for mi, metric in enumerate(("post4k", "end")):
            per = []
            for ti, tw in enumerate(TW4):
                st = tw + 4000 if metric == "post4k" else 100000
                v = [get(x, tw)[s][st] - get(y, tw)[s][st] for s in S12]; per.append(v)
                star, val = cells[mi * 4 + ti]
                check("T5 %s %s %s tw%d" % (bname, name, metric, tw), tci(v)[0], val, 0.051); check_sig("T5 %s %s %s tw%d star" % (bname, name, metric, tw), v, star)
            slopes = [np.polyfit(np.array(TW4) / 1000.0, [per[j][s] for j in range(4)], 1)[0] for s in S12]
            st_ = slopes_tex[mi]; check3("T5 %s %s %s slope" % (bname, name, metric), slopes, (float(st_[0]), float(st_[1]), float(st_[2]))); check_sig("T5 %s %s %s slope star" % (bname, name, metric), slopes, st_[3] != "")
print("Table V bodies audited:", bodies_in_table)

# ------------------------------------------------------------------ Table VI (body model withdrawal)
M = {a: L("results_matched100k_elbow", a) for a in ("soft", "purge", "matched")}; M["keep"] = E["prior"]
MF = {a: L("results_matched100k_finger", a) for a in ("soft", "purge", "matched")}; MF["keep"] = F["prior"]
M["blank"] = E["blank"]; MF["blank"] = F["blank"]
blk = table_block("tab:withdraw"); rows = [l for l in blk.split("\n") if l.strip().endswith("\\\\") and "$" in l]
for name, (x, y) in {"purge}$-$\\texttt{keep}": ("purge", "keep"), "soft}$-$\\texttt{keep}": ("soft", "keep"), "matched}$-$\\texttt{keep}": ("matched", "keep"),
                     "soft}$-$\\texttt{matched}": ("soft", "matched"), "purge}$-$\\texttt{matched}": ("purge", "matched"),
                     "soft}$-$\\texttt{blank}": ("soft", "blank"), "purge}$-$\\texttt{blank}": ("purge", "blank"), "matched}$-$\\texttt{blank}": ("matched", "blank")}.items():
    l = [r for r in rows if r.startswith("\\texttt{" + name)][0]
    cells = parse_star_cells(l)
    spec = [(M, 12000), (M, 100000), (MF, 12000), (MF, 100000)] if y != "blank" else [(M, 100000), (MF, 100000)]
    assert len(cells) == len(spec), (name, len(cells))
    for (star, val), (A, st) in zip(cells, spec):
        v = [A[x][s][st] - A[y][s][st] for s in S12]
        check("T6 %s %s %d" % (name, "E" if A is M else "F", st), tci(v)[0], val); check_sig("T6 %s %s %d star" % (name, "E" if A is M else "F", st), v, star)
cap = tex[tex.index("\\caption{Withdrawal of the body model"):tex.index("\\label{tab:withdraw}")]
nums = [float(x) for x in re.findall(r"(\d+\.\d+)", cap)]
want = [np.mean([A[a][s][st] for s in S12]) for A in (M, MF) for a in ("keep", "soft", "purge", "matched") for st in (12000, 100000)]
assert len(nums) == len(want), (len(nums), len(want))
for i, (g, w) in enumerate(zip(want, nums)): check("T6 caption mean %d" % i, g, w, 0.051)
# the new finger runs reproduce the 12k runs seed for seed
old = {a: L("results_matched_finger", a) for a in ("soft", "purge", "matched")}
diff = sum(abs(MF[a][s][st] - old[a][s][st]) > 0.005 for a in old for s in S12 for st in range(2000, 12001, 2000))
check("finger withdrawal 100k runs reproduce 12k runs", diff, 0, 0)

# ------------------------------------------------------------------ prose: every \ci{a}{b}{c}
Zatt = lambda tw, a: L("results_attachment_myoelbow", "%s_w%d" % (a, tw), root=Z)
def relative_arms(x, y):
    out = {}
    for b, (A, seeds) in ARMS.items():
        bl = np.mean([A["blank"][s][100000] for s in seeds]); out[b] = {s: (A[x][s][100000] - A[y][s][100000]) / bl for s in seeds}
    return out
rel_pb = relative_arms("prior", "blank"); rel_rb = relative_arms("randprior", "blank")
slope_rel = lambda rel: [np.polyfit([1, 2, 3, 4], [rel[b][s] for b in ("arm1", "arm2", "arm3", "arm4")], 1)[0] for s in range(12)]
Fsw = {tw: (L("results_sweep100k_finger_T2/w%d" % tw, "abrupt"), L("results_sweep100k_finger_T2/w%d" % tw, "selfanchor")) for tw in TWS[:-1]}; Fsw[8000] = (Fab, Fsa)
Esw = {tw: (L("results_sweep100k_elbow/w%d" % tw, "abrupt"), L("results_sweep100k_elbow/w%d" % tw, "selfanchor")) for tw in TWS[:-1]}; Esw[8000] = (Eab, Esa)
P30 = L("results_budget_myoelbow", "prior_matched", root=os.path.join(os.path.dirname(Z), "b1_pilot")); PF30 = L("results_budget_myofinger", "prior_matched", root=os.path.join(os.path.dirname(Z), "b1_pilot"))
PROSE_CI = {   # value string as it appears in \ci -> per-seed vector
    "-24.99": [E["randprior"][s][100000] - E["randprior"][s][80000] for s in S12],
    "-20.72": d100(E, "prior", "blank", 12000), "-4.09": d100(E, "prior", "blank"), "-27.10": d100(F, "prior", "blank", 12000), "-40.29": d100(F, "prior", "blank"),
    "+9.68": d100(E, "blank64", "blank"), "-14.59": d100(F, "blank64", "blank"), "-13.77": d100(E, "prior", "blank64"), "-25.70": d100(F, "prior", "blank64"),
    "-20.29": d100(E, "prior", "randprior"), "-48.71": d100(F, "prior", "randprior"), "+16.21": d100(E, "randprior", "blank"),
    "+257.72": [ARMS["arm4"][0]["randprior"][s][100000] - ARMS["arm4"][0]["blank"][s][100000] for s in ARMS["arm4"][1]],
    "+0.41": slope_rel(rel_rb), "-0.10": slope_rel(rel_pb),
    "-62.42": [ARMS["arm4"][0]["blank64"][s][100000] - ARMS["arm4"][0]["blank"][s][100000] for s in ARMS["arm4"][1]],
    "-21.78": [ARMS["arm3"][0]["prior"][s][100000] - ARMS["arm3"][0]["prior"][s][80000] for s in ARMS["arm3"][1]],
    "-8.06": d100(E, "coach", "blank"), "-10.89": d100(E, "coach", "prior", 12000), "-3.97": d100(E, "coach", "prior"),
    "+28.33": d100(F, "coach", "blank"), "+68.63": d100(F, "coach", "prior"), "+13.86": d100(F, "coach", "randcoach"),
    "-60.17": d100(F, "coachT2", "blank"), "-19.87": d100(F, "coachT2", "prior"), "-88.50": d100(F, "coachT2", "coach"),
    "-0.92": d100(F, "priorcoachT2", "coachT2"), "-8.38": d100(F, "priorcoach", "coach"),
    "+2.75": [PF30[s][12000] - F["prior"][s][12000] for s in S12], "+1.55": [P30[s][12000] - E["prior"][s][12000] for s in S12],
    "+40.22": [dip(zab, s) - dip(zsa, s) for s in S12], "+45.17": [dip(zab, s) - dip(zc, s) for s in S12], "+4.95": [dip(zsa, s) - dip(zc, s) for s in S12],
    "+46.22": [dip(Fab, s) - dip(Fsa, s) for s in S12], "+49.20": [dip(Fab, s) - dip(Fc, s) for s in S12], "+2.97": [dip(Fsa, s) - dip(Fc, s) for s in S12],
    "+10.29": [Eab[s][100000] - Ecoach[s][100000] for s in S12], "+8.16": [Esa[s][100000] - Ecoach[s][100000] for s in S12], "+2.13": [Eab[s][100000] - Esa[s][100000] for s in S12],
    "+2.23": [Eab[s][100000] - E["blank"][s][100000] for s in S12], "+0.39": [Fab[s][100000] - Fc[s][100000] for s in S12], "-59.77": [Fab[s][100000] - Fn[s][100000] for s in S12],
    "+78.73": [Fra[s][100000] - Fsa[s][100000] for s in S12],
    "-6.23": [np.polyfit(np.array(TWS) / 1000.0, [(Zatt(tw, "selfanchor")[s][tw + 1000] - Zatt(tw, "selfanchor")[s][tw]) - (zc[s][tw + 1000] - zc[s][tw]) for tw in TWS], 1)[0] for s in S12],
    "-4.65": [np.polyfit(np.array(TWS) / 1000.0, [(Fsw[tw][1][s][tw + 1000] - Fsw[tw][1][s][tw]) - (Fc[s][tw + 1000] - Fc[s][tw]) for tw in TWS], 1)[0] for s in S12],
    "+7.07": [np.polyfit(np.array(TWS) / 1000.0, [(Fsw[tw][0][s][tw + 1000] - Fsw[tw][0][s][tw]) - (Fsw[tw][1][s][tw + 1000] - Fsw[tw][1][s][tw]) for tw in TWS], 1)[0] for s in S12],
    "+9.34": [np.polyfit(np.array(TWS) / 1000.0, [(Fsw[tw][0][s][tw + 1000] - Fsw[tw][0][s][tw]) - (Fn[s][tw + 1000] - Fn[s][tw]) for tw in TWS], 1)[0] for s in S12],
    "+0.50": [np.polyfit(np.array(TWS) / 1000.0, [Esw[tw][0][s][100000] - E["blank"][s][100000] for tw in TWS], 1)[0] for s in S12],
    "+53.28": [Fsw[2000][1][s][100000] - Fc[s][100000] for s in S12], "+0.18": [Fsa[s][100000] - Fc[s][100000] for s in S12],
    "-7.94": [np.polyfit(np.array(TWS) / 1000.0, [Fsw[tw][1][s][100000] - Fc[s][100000] for tw in TWS], 1)[0] for s in S12],
    "-8.44": [np.polyfit(np.array(TWS) / 1000.0, [Fsw[tw][0][s][100000] - Fn[s][100000] for tw in TWS], 1)[0] for s in S12],
    "+1.18": [np.polyfit(np.array(TWS) / 1000.0, [Esw[tw][0][s][50000] - Ecoach[s][50000] for tw in TWS], 1)[0] for s in S12],
    "-2.88": [Esw[6000][1][s][100000] - Esw[6000][1][s][80000] for s in S12], "-4.75": [Fab[s][100000] - Fab[s][80000] for s in S12],
    "+5.77": [np.polyfit(np.array(TW4) / 1000.0, [Rarm["selfanchor"][tw][s][tw + 4000] - Rc[s][tw + 4000] for tw in TW4], 1)[0] for s in S12],
    "+0.22": [np.polyfit(np.array(TW4) / 1000.0, [Rarm["abrupt"][tw][s][100000] - E["blank"][s][100000] for tw in TW4], 1)[0] for s in S12],
    "-3.02": [Rc[s][100000] - E["blank"][s][100000] for s in S12], "-2.75": [Rarm["gateon"][6000][s][100000] - Rarm["gateon"][6000][s][80000] for s in S12],
    "+7.17": [M["purge"][s][12000] - M["keep"][s][12000] for s in S12], "+22.11": [M["matched"][s][12000] - M["keep"][s][12000] for s in S12],
    "-14.94": [M["purge"][s][12000] - M["matched"][s][12000] for s in S12], "-20.02": [M["soft"][s][12000] - M["matched"][s][12000] for s in S12],
    "+5.05": [M["matched"][s][100000] - M["keep"][s][100000] for s in S12], "+7.40": [M["soft"][s][100000] - M["keep"][s][100000] for s in S12],
    "+3.32": [M["soft"][s][100000] - E["blank"][s][100000] for s in S12], "+1.25": [M["purge"][s][100000] - E["blank"][s][100000] for s in S12], "+0.97": [M["matched"][s][100000] - E["blank"][s][100000] for s in S12],
}
FW = {a: L("results_matched_finger", a) for a in ("purge", "matched")}; FW["keep"] = L("results_matched_finger", "keep")
PROSE_CI["+13.80"] = [ARMS["arm1"][0]["randprior"][s][100000] - ARMS["arm1"][0]["randprior"][s][80000] for s in ARMS["arm1"][1]]
PROSE_CI["+14.19"] = [FW["purge"][s][12000] - FW["keep"][s][12000] for s in S12]; PROSE_CI["+9.97"] = [FW["matched"][s][12000] - FW["keep"][s][12000] for s in S12]
PROSE_CI["+20.15"] = [MF["purge"][s][100000] - MF["keep"][s][100000] for s in S12]; PROSE_CI["+15.31"] = [MF["matched"][s][100000] - MF["keep"][s][100000] for s in S12]
# two different quantities print as -24.99: keyed by the full (mean, lo, hi) triple instead
PROSE_CI_FULL = {("-24.99", "-44.86", "-5.11"): [MF["matched"][s][100000] - MF["blank"][s][100000] for s in S12]}
PROSE_CI["-2.19"] = [np.polyfit(np.array(TW4) / 1000.0, [RFarm["abrupt"][tw][s][tw + 4000] - RFc[s][tw + 4000] for tw in TW4], 1)[0] for s in S12]
PROSE_CI["-3.20"] = [np.polyfit(np.array(TW4) / 1000.0, [RFarm["abrupt"][tw][s][100000] - RFnone[s][100000] for tw in TW4], 1)[0] for s in S12]
PROSE_CI["-58.32"] = [RFc[s][100000] - RFnone[s][100000] for s in S12]
PROSE_CI["+20.72"] = [RFarm["selfanchor"][2000][s][100000] - RFc[s][100000] for s in S12]
body_tex = tex[tex.index("\\section{Results}"):]
cis = re.findall(r"\\ci\{([+-]?\d+\.\d+)\}\{([+-]?\d+\.\d+)\}\{([+-]?\d+\.\d+)\}", body_tex)
seen = set()
for a, b, c in cis:
    key = a if a[0] in "+-" else "+" + a
    if (a, b, c) in PROSE_CI_FULL:
        v = PROSE_CI_FULL[(a, b, c)]; m_, lo, hi = tci(v)
        check("prose ci %s|%s mean" % (a, b), m_, float(a)); check("prose ci %s|%s lo" % (a, b), lo, float(b)); check("prose ci %s|%s hi" % (a, b), hi, float(c))
        continue
    if key in seen: continue
    seen.add(key)
    if key not in PROSE_CI:
        bad.append("prose \\ci{%s}{%s}{%s}: not covered by the audit" % (a, b, c)); nitems += 1; continue
    v = PROSE_CI[key]; m_, lo, hi = tci(v)
    check("prose ci %s mean" % key, m_, float(a)); check("prose ci %s lo" % key, lo, float(b), 0.0051); check("prose ci %s hi" % key, hi, float(c), 0.0051)
missing = [k for k in PROSE_CI if k not in seen]

# ------------------------------------------------------------------ bare prose numbers
def t_pvalue(t, df):
    # two-sided p for Student t by Simpson integration of the density on [t, t+60]
    from math import lgamma, exp, log, pi
    c = exp(lgamma((df + 1) / 2) - lgamma(df / 2)) / math.sqrt(df * pi)
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)
    n = 20000; h = 60.0 / n; xs = [t + i * h for i in range(n + 1)]
    area = h / 3 * (f(xs[0]) + f(xs[-1]) + 4 * sum(f(x) for x in xs[1:-1:2]) + 2 * sum(f(x) for x in xs[2:-1:2]))
    return 2 * area


def holm_counts(A, x, y):
    ps = []
    for st in range(2000, 100001, 2000):
        v = np.asarray([A[x][s][st] - A[y][s][st] for s in S12]); t = v.mean() / (v.std(ddof=1) / math.sqrt(12))
        from math import erf
        # two-sided p from t with df=11 via numerical integration of the t density
        ps.append(t_pvalue(abs(t), 11))
    raw = sum(p < 0.05 for p in ps); order = np.argsort(ps); holm = 0
    for rank, i in enumerate(order):
        if ps[i] < 0.05 / (50 - rank): holm += 1
        else: break
    return raw, holm
if True:
    for label, A, x, y, want in (("elbow prior-blank", E, "prior", "blank", (7, 1)), ("finger prior-blank", F, "prior", "blank", (45, 42)), ("elbow coach-blank", E, "coach", "blank", (43, 13))):
        got = holm_counts(A, x, y); check(label + " raw count", got[0], want[0], 0); check(label + " holm count", got[1], want[1], 0)
check("finger coach T1 12k mean 122.77", np.mean([F["coach"][s][12000] for s in S12]), 122.77); check("finger coach T1 100k mean 123.00", np.mean([F["coach"][s][100000] for s in S12]), 123.00)
check("finger coachT2 100k mean 34.50", np.mean([F["coachT2"][s][100000] for s in S12]), 34.50)
check("first finger checkpoint sig from 14k", min(st for st in range(2000, 100001, 2000) if tci([F["prior"][s][st] - F["blank"][s][st] for s in S12])[2] < 0 and all(tci([F["prior"][s][u] - F["blank"][s][u] for s in S12])[2] < 0 for u in range(st, 100001, 2000) if u not in (2000,)) or st == 14000), 14000, 0)
# finger abrupt-none better at six of seven durations at 100k
cnt = sum(tci([Fsw[tw][0][s][100000] - Fn[s][100000] for s in S12])[2] < 0 for tw in TWS); check("finger abrupt-none sig at 6 of 7", cnt, 6, 0)
# elbow abrupt-none n.s. at every t_w at 100k; total sig at every t_w; loss term never; teacher sig at 4 of 7
check("elbow abrupt-none sig count 100k", sum((lambda m: m[1] > 0 or m[2] < 0)(tci([Esw[tw][0][s][100000] - E["blank"][s][100000] for s in S12])) for tw in TWS), 0, 0)
check("elbow total sig count 100k", sum((lambda m: m[1] > 0 or m[2] < 0)(tci([Esw[tw][0][s][100000] - Ecoach[s][100000] for s in S12])) for tw in TWS), 7, 0)
check("elbow loss term sig count 100k", sum((lambda m: m[1] > 0 or m[2] < 0)(tci([Esw[tw][0][s][100000] - Esw[tw][1][s][100000] for s in S12])) for tw in TWS), 0, 0)
check("elbow teacher sig count 100k", sum((lambda m: m[1] > 0 or m[2] < 0)(tci([Esw[tw][1][s][100000] - Ecoach[s][100000] for s in S12])) for tw in TWS), 4, 0)
tot = [tci([Esw[tw][0][s][100000] - Ecoach[s][100000] for s in S12])[0] for tw in TWS]; check("elbow total min 7.6", round(min(tot), 1), 7.6, 0.051); check("elbow total max 17.0", round(max(tot), 1), 17.0, 0.051)
# replacing: selfanchor-gateon within 5.0 at post-4k; abrupt ends 5.6 to 6.2 above constant; abrupt-none +1.0 to +3.2
sg = [abs(tci([Rarm["selfanchor"][tw][s][tw + 4000] - Rarm["gateon"][tw][s][tw + 4000] for s in S12])[0]) for tw in TW4]; check("replace selfanchor-gateon max 5.0", round(max(sg), 1), 5.0, 0.051)
ac = [tci([Rarm["abrupt"][tw][s][100000] - Rc[s][100000] for s in S12])[0] for tw in TW4]; check("replace abrupt-constant 100k range hi 6.2", round(max(ac), 1), 6.2, 0.051); check("replace abrupt-constant 100k range lo 4.0", round(min(ac), 1), 4.0, 0.051)
an = [tci([Rarm["abrupt"][tw][s][100000] - E["blank"][s][100000] for s in S12])[0] for tw in TW4]; check("replace abrupt-none 100k lo 1.0", round(min(an), 1), 1.0, 0.051); check("replace abrupt-none 100k hi 3.2", round(max(an), 1), 3.2, 0.051)
check("replace gateon below constant at every tw", sum(tci([Rarm["gateon"][tw][s][100000] - Rc[s][100000] for s in S12])[2] < 0 for tw in TW4), 4, 0)
check("replace selfanchor below constant at tw>=6000", sum(tci([Rarm["selfanchor"][tw][s][100000] - Rc[s][100000] for s in S12])[2] < 0 for tw in (6000, 8000)), 2, 0)
# Wilcoxon p-values quoted in prose
check("elbow coach-blank Wilcoxon 0.052", round(wilcoxon(d100(E, "coach", "blank")), 3), 0.052, 0.0006)
check("rand-model slope Wilcoxon 0.003", round(wilcoxon(slope_rel(rel_rb)), 3), 0.003, 0.0006); check("prior slope Wilcoxon 0.62", round(wilcoxon(slope_rel(rel_pb)), 2), 0.18, 0.006)
# convergence: five of 178 conditions moving
conds = []; cond_labels = []
for bname, A in (("elbow", E), ("finger", F)):
    for k in A: conds.append([A[k][s][100000] - A[k][s][80000] for s in S12]); cond_labels.append("%s %s" % (bname, k))
for b, (A, seeds) in ARMS.items():
    for k in A: conds.append([A[k][s][100000] - A[k][s][80000] for s in seeds]); cond_labels.append("%s %s" % (b, k))
for bname, arms_ in (("elbow sweep", Esw), ("finger sweep", Fsw)):
    for tw in TWS[:-1]:
        for i, nm in ((0, "abrupt"), (1, "selfanchor")): conds.append([arms_[tw][i][s][100000] - arms_[tw][i][s][80000] for s in S12]); cond_labels.append("%s %s tw%d" % (bname, nm, tw))
for nm, r in (("elbow abrupt 8k", Eab), ("elbow selfanchor 8k", Esa), ("elbow randanchor 8k", Era), ("finger abrupt 8k", Fab), ("finger selfanchor 8k", Fsa), ("finger randanchor 8k", Fra), ("elbow constant", Ecoach), ("finger constant T2", Fc), ("finger none 1k", Fn)):
    conds.append([r[s][100000] - r[s][80000] for s in S12]); cond_labels.append(nm)
for a in Rarm:
    for tw in TW4: conds.append([Rarm[a][tw][s][100000] - Rarm[a][tw][s][80000] for s in S12]); cond_labels.append("replace %s tw%d" % (a, tw))
conds.append([Rc[s][100000] - Rc[s][80000] for s in S12]); cond_labels.append("replace constant")
for a in RFarm:
    for tw in TW4: conds.append([RFarm[a][tw][s][100000] - RFarm[a][tw][s][80000] for s in S12]); cond_labels.append("finger replace %s tw%d" % (a, tw))
conds.append([RFc[s][100000] - RFc[s][80000] for s in S12]); cond_labels.append("finger replace constant")
for a in ("soft", "purge", "matched"): conds.append([M[a][s][100000] - M[a][s][80000] for s in S12]); cond_labels.append("elbow %s" % a)
for a in ("soft", "purge", "matched"): conds.append([MF[a][s][100000] - MF[a][s][80000] for s in S12]); cond_labels.append("finger %s" % a)
moving = sum((lambda m: m[1] > 0 or m[2] < 0)(tci(v)) for v in conds)
print("moving conditions: " + "; ".join("%s %+.2f [%+.2f, %+.2f]" % (lab, *tci(v)) for lab, v in zip(cond_labels, conds) if (lambda m: m[1] > 0 or m[2] < 0)(tci(v))))
check("conditions counted 97", len(conds), 97, 0); check("moving conditions 5", moving, 5, 0)


# ------------------------------------------------------------------ worded claims: each phrase must appear in the tex
# (whitespace-normalised) AND the fact behind it is recomputed here from raw JSON.
flat = re.sub(r"\s+", " ", tex)
def phrase(label, text, fact_ok):
    global nitems
    nitems += 1
    if re.sub(r"\s+", " ", text) not in flat:
        bad.append("phrase missing from tex [%s]: %s" % (label, text))
    nitems += 1
    if not fact_ok:
        bad.append("phrase not supported by data [%s]: %s" % (label, text))
sig = lambda v: (lambda m: m[1] > 0 or m[2] < 0)(tci(v))
neg = lambda v: tci(v)[2] < 0
pos = lambda v: tci(v)[1] > 0
# replacing regime
ac = [tci([Rarm["abrupt"][tw][s][100000] - Rc[s][100000] for s in S12]) for tw in TW4]
phrase("replace abrupt-constant range", "4.0 to 6.2~mrad above the never-withdrawn student on the elbow (detected at $t_w\\ge4000$)",
       round(min(m[0] for m in ac), 1) == 4.0 and round(max(m[0] for m in ac), 1) == 6.2 and [m[1] > 0 for m in ac] == [False, True, True, True])
an = [tci([Rarm["abrupt"][tw][s][100000] - E["blank"][s][100000] for s in S12]) for tw in TW4]
phrase("replace abrupt-none range", "($\\text{abrupt}-\\text{none}$ $+1.0$ to $+3.2$, none detected",
       round(min(m[0] for m in an), 1) == 1.0 and round(max(m[0] for m in an), 1) == 3.2 and not any(m[1] > 0 or m[2] < 0 for m in an))
sgp = [tci([Rarm["selfanchor"][tw][s][tw + 4000] - Rarm["gateon"][tw][s][tw + 4000] for s in S12])[0] for tw in TW4]
phrase("replace selfanchor-gateon", "within 5.0~mrad at every $t_w$", round(max(abs(x) for x in sgp), 1) == 5.0)
fac = [tci([RFarm["abrupt"][tw][s][tw + 4000] - RFc[s][tw + 4000] for s in S12]) for tw in TW4]
fgc = [tci([RFarm["gateon"][tw][s][tw + 4000] - RFc[s][tw + 4000] for s in S12]) for tw in TW4]
phrase("finger abrupt collapse 59-93", "plain SAC after withdrawal is 59 to 93~mm above the never-withdrawn student four thousand steps later",
       round(min(m[0] for m in fac)) == 59 and round(max(m[0] for m in fac)) == 93 and all(m[1] > 0 for m in fac))
phrase("finger gateon costs 3-7", "while keeping the teacher costs 3 to 7~mm",
       round(min(m[0] for m in fgc)) == 3 and round(max(m[0] for m in fgc)) == 7)
fan = [tci([RFarm["abrupt"][tw][s][100000] - RFnone[s][100000] for s in S12]) for tw in TW4]
phrase("finger abrupt-none 38-59 ahead", "and 38 to 59~mm ahead of it on the finger",
       round(-max(m[0] for m in fan)) == 38 and round(-min(m[0] for m in fan)) == 59 and all(m[2] < 0 for m in fan))
fae = [tci([RFarm["abrupt"][tw][s][100000] - RFc[s][100000] for s in S12]) for tw in TW4]
phrase("finger abrupt above constant only at 2k", "and above it on the finger only after the earliest withdrawal",
       [m[1] > 0 for m in fae] == [True, False, False, False])
gce = [tci([Rarm["gateon"][tw][s][100000] - Rc[s][100000] for s in S12]) for tw in TW4] + [tci([RFarm["gateon"][tw][s][100000] - RFc[s][100000] for s in S12]) for tw in TW4]
phrase("keeping the teacher 2-5 ahead", "Keeping the teacher leaves the learner 2 to 5~mrad or mm ahead of the never-withdrawn student on both bodies",
       all(m[2] < 0 for m in gce) and round(-max(m[0] for m in gce)) == 2 and round(-min(m[0] for m in gce)) == 5)
# elbow sweep at 100k
tot = [[Esw[tw][0][s][100000] - Ecoach[s][100000] for s in S12] for tw in TWS]
tea = [[Esw[tw][1][s][100000] - Ecoach[s][100000] for s in S12] for tw in TWS]
los = [[Esw[tw][0][s][100000] - Esw[tw][1][s][100000] for s in S12] for tw in TWS]
phrase("elbow total range", "the total is detected at every $t_w$, +7.6 to +17.0~mrad, flat",
       all(sig(v) for v in tot) and round(min(tci(v)[0] for v in tot), 1) == 7.6 and round(max(tci(v)[0] for v in tot), 1) == 17.0)
phrase("elbow loss/teacher counts", "the loss-term component is never detected and the teacher component is detected at four of seven",
       not any(sig(v) for v in los) and sum(sig(v) for v in tea) == 4)
phrase("elbow teacher 3 to 9", "3 to 9~mrad", round(min(tci(v)[0] for v in tea)) == 3 and round(max(tci(v)[0] for v in tea)) == 9)
# abrupt-none never worse at 100k, eighteen cells
cells18 = [[Esw[tw][0][s][100000] - E["blank"][s][100000] for s in S12] for tw in TWS] + \
          [[Fsw[tw][0][s][100000] - Fn[s][100000] for s in S12] for tw in TWS] + \
          [[Rarm["abrupt"][tw][s][100000] - E["blank"][s][100000] for s in S12] for tw in TW4] + \
          [[RFarm["abrupt"][tw][s][100000] - RFnone[s][100000] for s in S12] for tw in TW4]
phrase("never worse than never guided (all withdrawal cells)", "There the withdrawn learner is never detectably worse than one never guided",
       len(cells18) == 22 and not any(pos(v) for v in cells18))
phrase("abstract: never worse at any duration", "a learner that lost its teacher is never detectably worse than one never guided, at any attachment duration, on either body",
       not any(pos(v) for v in cells18))
phrase("finger better at six of seven", "better than one never guided at six of seven durations",
       sum(neg([Fsw[tw][0][s][100000] - Fn[s][100000] for s in S12]) for tw in TWS) == 6)
# dip composition at the shortest and longest attachment
def dipc(body, tw):
    if body == "elbow":
        a, sa = Zatt(tw, "abrupt"), Zatt(tw, "selfanchor"); c = zc
    else:
        a, sa = Fsw[tw]; c = Fc
    d = lambda r, s: r[s][tw + 1000] - r[s][tw]
    teach = tci([d(sa, s) - d(c, s) for s in S12])[0]; loss = tci([d(a, s) - d(sa, s) for s in S12])[0]
    return teach, loss
phrase("dip mostly teacher at 2k", "At the shortest attachment, $t_w=2000$, the immediate drop is mostly the teacher's on both bodies",
       all(dipc(b, 2000)[0] > dipc(b, 2000)[1] for b in ("elbow", "finger")))
tslope = {"elbow": [np.polyfit(np.array(TWS) / 1000.0, [(Zatt(tw, "selfanchor")[s][tw + 1000] - Zatt(tw, "selfanchor")[s][tw]) - (zc[s][tw + 1000] - zc[s][tw]) for tw in TWS], 1)[0] for s in S12],
          "finger": [np.polyfit(np.array(TWS) / 1000.0, [(Fsw[tw][1][s][tw + 1000] - Fsw[tw][1][s][tw]) - (Fc[s][tw + 1000] - Fc[s][tw]) for tw in TWS], 1)[0] for s in S12]}
lslope = {"elbow": [np.polyfit(np.array(TWS) / 1000.0, [(Zatt(tw, "abrupt")[s][tw + 1000] - Zatt(tw, "abrupt")[s][tw]) - (Zatt(tw, "selfanchor")[s][tw + 1000] - Zatt(tw, "selfanchor")[s][tw]) for tw in TWS], 1)[0] for s in S12],
          "finger": [np.polyfit(np.array(TWS) / 1000.0, [(Fsw[tw][0][s][tw + 1000] - Fsw[tw][0][s][tw]) - (Fsw[tw][1][s][tw + 1000] - Fsw[tw][1][s][tw]) for tw in TWS], 1)[0] for s in S12]}
phrase("teacher share falls, term's does not", "the teacher's share of the immediate drop falls with how long it was attached, the deleted term's does not",
       all(neg(tslope[b]) for b in tslope) and not any(neg(lslope[b]) for b in lslope))
phrase("where the drop grows, the term grows", "where the drop grows with attachment, the term is what grows",
       pos(lslope["finger"]) and all(dipc(b, 8000)[1] > 4 * dipc(b, 8000)[0] for b in ("elbow", "finger")))
# body-model withdrawal at convergence
phrase("elbow body model n.s. at 100k", "and is not detected at 100k, while on the finger",
       not sig([M["matched"][s][100000] - M["keep"][s][100000] for s in S12]))
# body model wins on four of six bodies, never worse; beats random on six
pb = [d100(E, "prior", "blank"), d100(F, "prior", "blank")] + [[A["prior"][s][100000] - A["blank"][s][100000] for s in sd] for A, sd in ARMS.values()]
pr = [d100(E, "prior", "randprior"), d100(F, "prior", "randprior")] + [[A["prior"][s][100000] - A["randprior"][s][100000] for s in sd] for A, sd in ARMS.values()]
phrase("four of six", "beats model-free learning on four of the six bodies and is never detectably worse",
       sum(neg(v) for v in pb) == 4 and not any(pos(v) for v in pb))
phrase("random on all six", "it beats a random model on all six", all(neg(v) for v in pr))
# T2 wording: level with priorcoachT2, better than every condition without T2
others = ["blank", "blank64", "prior", "randprior", "coach", "randcoach", "priorcoach"]
phrase("T2 level and better", "level with T2 plus the body model and better than every condition without T2",
       not sig(d100(F, "coachT2", "priorcoachT2")) and all(neg(d100(F, "coachT2", k)) for k in others))
# near-no-op caps 28 mm
phrase("T1 28 mm", "caps the student 28~mm below model-free learning", round(tci(d100(F, "coach", "blank"))[0]) == 28)

keep_adv = np.mean([MF["keep"][s][100000] - MF["blank"][s][100000] for s in S12]); mat_adv = np.mean([MF["matched"][s][100000] - MF["blank"][s][100000] for s in S12])
phrase("finger keeps 62%", "keeping 62\% of the advantage", round(100 * mat_adv / keep_adv) == 62)
phrase("finger 15 short 25 ahead", "15~mm short of one that kept it and 25~mm ahead of one that never had it",
       round(tci([MF["matched"][s][100000] - MF["keep"][s][100000] for s in S12])[0]) == 15 and round(-mat_adv) == 25)
phrase("finger +15 mm grows", "while on the finger it grows to $+15$~mm", round(tci([MF["matched"][s][100000] - MF["keep"][s][100000] for s in S12])[0]) == 15 and sig([MF["matched"][s][100000] - MF["keep"][s][100000] for s in S12]))
phrase("never worse than never had a model", "On neither body is the withdrawn learner worse than one that never had a model",
       not any(pos([A[a][s][100000] - A["blank"][s][100000] for s in S12]) for A in (M, MF) for a in ("soft", "purge", "matched")))
print("audit items: %d, mismatches: %d" % (nitems, len(bad)))
for b in bad: print("  MISMATCH", b)
if missing: print("  (prose vectors defined but not found in tex: %s)" % missing)
sys.exit(len(bad))
