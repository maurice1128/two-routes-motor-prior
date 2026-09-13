# -*- coding: utf-8 -*-
"""Re-derive every number in the two TCDS tables from raw per-seed JSON.

The tables were filled by hand from analysis printouts over two days. This is the
check the audit SOP asks for: recompute each cell from the run files, then
compare against what the .tex actually says, so a transcription slip cannot ride
into the submission. Paired t, df=11, same as the paper.
"""
import json, os, re, math, io

W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
TEX = os.path.join(W, "tcds_merged", "paper.tex")
T = 2.201
tex = io.open(TEX, encoding="utf-8").read()
tex = tex.replace("\\mathbf{", "").replace("\\ ", " ")


def load(d, c):
    return {s: {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(os.path.join(W, d, "%s_seed%d.json" % (c, s))))}
            for s in range(12) if os.path.exists(os.path.join(W, d, "%s_seed%d.json" % (c, s)))}


def ci(a, b, key):
    d = [key(a[s]) - key(b[s]) for s in sorted(set(a) & set(b))]
    n = len(d); m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return n, m, m - T * se, m + T * se


def row_cells(label_regex):
    """Return the table row's cells (after the label) as strings, for the first matching row."""
    m = re.search(r"(?m)^\\texttt\{" + label_regex + r"\}[^&\n]*&(.*?)\\\\", tex)
    if not m:
        return None
    return [c.strip() for c in m.group(1).split("&")]


def nums(cell):
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+", cell.replace("$", ""))]


fails = 0; checks = 0


def cmp(name, got, cell):
    global fails, checks
    want = nums(cell)
    checks += 1
    ok = len(want) == len(got) and all(abs(g - w) < 0.006 for g, w in zip(got, want))
    if not ok:
        fails += 1
        print("  MISMATCH %-34s tex=%s  raw=%s" % (name, want, [round(g, 2) for g in got]))


# ---------------- Table 2: convergence ----------------
ARMS = [("blank", "blank"), ("prior", "prior"), ("randprior", "prior"),
        ("coach", "coach"), ("randcoach", "coach"), ("priorcoach", "priorcoach")]
data = {}
for tag in ("elbow", "finger"):
    for label, cond in ARMS:
        data[(tag, label)] = load("results_conv_%s_%s" % (tag, label), cond)
        assert len(data[(tag, label)]) == 12, (tag, label)

for label, _ in ARMS:
    cells = row_cells(re.escape(label))
    if cells is None or len(cells) < 4:
        print("  row not found:", label); fails += 1; continue
    got = []
    for tag in ("elbow", "finger"):
        v = data[(tag, label)]
        got += [sum(c[12000] for c in v.values()) / 12, sum(c[100000] for c in v.values()) / 12]
    for i, (cell, g) in enumerate(zip(cells[:4], got)):
        cmp("T2 %s mean[%d]" % (label, i), [g], cell)

CONTRASTS = [("coach", "prior"), ("prior", "blank"), ("coach", "blank"),
             ("priorcoach", "blank"), ("priorcoach", "coach"),
             ("prior", "randprior"), ("coach", "randcoach")]
end = lambda c: c[100000]; k12 = lambda c: c[12000]
for a, b in CONTRASTS:
    cells = row_cells(re.escape(a) + r"\}\$-\$\\texttt\{" + re.escape(b))
    if cells is None:
        print("  contrast row not found: %s-%s" % (a, b)); fails += 1; continue
    for j, tag in enumerate(("elbow", "finger")):
        _, m12, _, _ = ci(data[(tag, a)], data[(tag, b)], k12)
        _, m, lo, hi = ci(data[(tag, a)], data[(tag, b)], end)
        cmp("T2 %s-%s %s 12k" % (a, b, tag), [m12], cells[2 * j])
        cmp("T2 %s-%s %s 100k" % (a, b, tag), [m, lo, hi], cells[2 * j + 1])

# ---------------- Table 1: withdrawal ----------------
wd = {}
for tag in ("elbow", "finger"):
    for arm in ("keep", "soft", "purge", "matched"):
        wd[(tag, arm)] = load("results_matched_%s" % tag, arm)
        assert len(wd[(tag, arm)]) == 12, (tag, arm)
e12 = lambda c: c[12000]
for arm in ("keep", "soft", "purge", "matched"):
    cells = row_cells(re.escape(arm) + r"\}\s*&")  # plain mean rows
    m = re.search(r"(?m)^\\texttt\{" + arm + r"\}\s*&\s*([\d.]+)\s*&\s*([\d.]+)", tex)
    if not m:
        print("  T1 mean row not found:", arm); fails += 1; continue
    for j, tag in enumerate(("elbow", "finger")):
        g = sum(c[12000] for c in wd[(tag, arm)].values()) / 12
        cmp("T1 %s mean %s" % (arm, tag), [g], m.group(j + 1))
for a, b in (("purge", "keep"), ("soft", "keep"), ("matched", "keep"),
             ("soft", "matched"), ("purge", "matched")):
    m = re.search(r"(?m)^\\texttt\{" + a + r"\}\$-\$\\texttt\{" + b + r"\}[^&\n]*&(.*?)\\\\", tex)
    if not m:
        print("  T1 contrast row not found: %s-%s" % (a, b)); fails += 1; continue
    cells = [c.strip() for c in m.group(1).split("&")]
    for j, tag in enumerate(("elbow", "finger")):
        _, mm, lo, hi = ci(wd[(tag, a)], wd[(tag, b)], e12)
        cmp("T1 %s-%s %s" % (a, b, tag), [mm, lo, hi], cells[j])

print("\nchecked %d cells, mismatches %d" % (checks, fails))
