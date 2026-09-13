# -*- coding: utf-8 -*-
"""Audit Table VII (both withdrawal decompositions at 100k) and the numbers the
merged draft quotes from it, against raw per-seed JSON.

Body-model side: results_matched100k_elbow (keep/soft/purge/matched).
Teacher side: results_coach100k_elbow (constant, abrupt) and
results_coach100k_elbow_off2 (selfanchor with the swap on loop step t_w-2, the
arm that reproduces the 12k table seed for seed).
"""
import json, os, re, math, io

W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
TEX = os.path.join(W, "tcds_merged", "paper.tex")
T = 2.201
tex = io.open(TEX, encoding="utf-8").read().replace("\\mathbf{", "")
blk = re.search(r"\\label\{tab:w100k\}(.*?)\\end\{tabular\}", tex, re.S).group(1)


def load(d, c):
    out = {s: {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(os.path.join(W, d, "%s_seed%d.json" % (c, s))))} for s in range(12)}
    return out


def ci(a, b, st):
    d = [a[s][st] - b[s][st] for s in range(12)]; m = sum(d) / 12
    se = math.sqrt(sum((x - m) ** 2 for x in d) / 11 / 12); return m, m - T * se, m + T * se


def nums(s):
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+", s.replace("$", ""))]


B = {a: load("results_matched100k_elbow", a) for a in ("keep", "soft", "purge", "matched")}
Cc = {"constant": load("results_coach100k_elbow", "constant"), "abrupt": load("results_coach100k_elbow", "abrupt"),
      "selfanchor": load("results_coach100k_elbow_off2", "selfanchor")}
fails = checks = 0


def row(a, b):
    m = re.search(r"\\texttt\{" + a + r"\}\$-\$\\texttt\{" + b + r"\}\s*&(.*?)\\\\", blk)
    return [c.strip() for c in m.group(1).split("&")] if m else None


def cmp(name, got, want, tol=0.006):
    global fails, checks
    checks += 1
    ok = len(got) == len(want) and all(abs(g - w) <= tol for g, w in zip(got, want))
    if not ok:
        fails += 1; print("  MISMATCH %-28s tex=%s raw=%s" % (name, want, [round(g, 3) for g in got]))


for a, b in (("purge", "keep"), ("soft", "keep"), ("matched", "keep"), ("soft", "matched"), ("purge", "matched")):
    cells = row(a, b)
    if not cells: print("  row not found", a, b); fails += 1; continue
    cmp("%s-%s 12k" % (a, b), [ci(B[a], B[b], 12000)[0]], nums(cells[0]))
    cmp("%s-%s 100k" % (a, b), list(ci(B[a], B[b], 100000)), nums(cells[1]))
for a, b in (("abrupt", "constant"), ("selfanchor", "constant"), ("abrupt", "selfanchor")):
    cells = row(a, b)
    if not cells: print("  row not found", a, b); fails += 1; continue
    cmp("%s-%s 12k" % (a, b), [ci(Cc[a], Cc[b], 12000)[0]], nums(cells[0]))
    cmp("%s-%s 100k" % (a, b), list(ci(Cc[a], Cc[b], 100000)), nums(cells[1]))

# caption means
cap = re.search(r"Body-model\s+arm means at 100k: keep (\d+\.\d+), soft (\d+\.\d+), purge (\d+\.\d+), matched (\d+\.\d+)", tex)
if cap:
    cmp("caption means", [sum(c[100000] for c in B[a].values()) / 12 for a in ("keep", "soft", "purge", "matched")],
        [float(x) for x in cap.groups()])
else:
    print("  caption means not found"); fails += 1

# 12k reproduction of the teacher arms against the 12k runs used in the 12k table
Z = r"C:\Users\maurice\Desktop\world_model_zeroshot\b2_pilot\results_withdrawal_myoelbow"
zs = {s: {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(os.path.join(Z, "selfanchor_seed%d.json" % s)))} for s in range(12)}
mm = sum(abs(Cc["selfanchor"][s][12000] - zs[s][12000]) >= 0.005 for s in range(12))
checks += 1
if mm: fails += 1; print("  selfanchor(off2) 12k does not reproduce the 12k arm: %d/12 differ" % mm)
print("\nchecked %d items, mismatches %d" % (checks, fails))
