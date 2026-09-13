# -*- coding: utf-8 -*-
"""Audit the numbers quoted in the Results PROSE that are not in a table.

Table cells are checked by audit_tcds.py. The prose additionally quotes 12k
intervals, a post-withdrawal change, checkpoint counts ("45 of 50"), the last
significant checkpoint, and "every checkpoint from Xk on" claims. One table cell
was already found to carry a hand-copied interval; the prose is the other place
that could.
"""
import json, os, math
W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
T = 2.201
CK = list(range(2000, 100001, 2000))


def load(d, c):
    return {s: {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(os.path.join(W, d, "%s_seed%d.json" % (c, s))))} for s in range(12)}


def ci(a, b, st):
    d = [a[s][st] - b[s][st] for s in range(12)]; m = sum(d) / 12
    se = math.sqrt(sum((x - m) ** 2 for x in d) / 11 / 12); return m, m - T * se, m + T * se


def cipost(a, b):
    d = [(a[s][10000] - a[s][8000]) - (b[s][10000] - b[s][8000]) for s in range(12)]; m = sum(d) / 12
    se = math.sqrt(sum((x - m) ** 2 for x in d) / 11 / 12); return m, m - T * se, m + T * se


E = {k: load("results_conv_elbow_" + k, c) for k, c in (("blank", "blank"), ("prior", "prior"), ("coach", "coach"))}
F = {k: load("results_conv_finger_" + k, c) for k, c in (("blank", "blank"), ("prior", "prior"), ("coach", "coach"))}
M = {k: load("results_matched_elbow", k) for k in ("keep", "matched")}
bad = 0

intervals = [
    ("elbow prior-blank 12k", (-20.72, -35.8, -5.6), ci(E["prior"], E["blank"], 12000)),
    ("elbow coach-blank 12k", (-31.61, -48.6, -14.6), ci(E["coach"], E["blank"], 12000)),
    ("elbow coach-prior 12k", (-10.89, -19.3, -2.5), ci(E["coach"], E["prior"], 12000)),
    ("finger prior-blank 12k", (-27.10, -66.3, 12.1), ci(F["prior"], F["blank"], 12000)),
    ("finger coach-prior 12k", (32.27, 16.5, 48.0), ci(F["coach"], F["prior"], 12000)),
    ("finger coach-blank 12k", (5.17, -26.5, 36.8), ci(F["coach"], F["blank"], 12000)),
    ("matched-keep post 8k->10k", (21.47, 4.54, 38.40), cipost(M["matched"], M["keep"])),
]
# random controls against model-free, quoted in prose with intervals (100k)
E["randprior"] = load("results_conv_elbow_randprior", "prior"); E["randcoach"] = load("results_conv_elbow_randcoach", "coach")
F["randprior"] = load("results_conv_finger_randprior", "prior"); F["randcoach"] = load("results_conv_finger_randcoach", "coach")
intervals += [
    ("elbow randprior-blank 100k", (16.21, 6.82, 25.59), ci(E["randprior"], E["blank"], 100000)),
    ("elbow randcoach-blank 100k", (9.63, -8.07, 27.34), ci(E["randcoach"], E["blank"], 100000)),
    ("finger randprior-blank 100k", (8.41, -14.17, 30.99), ci(F["randprior"], F["blank"], 100000)),
    ("finger randcoach-blank 100k", (14.48, 0.74, 28.21), ci(F["randcoach"], F["blank"], 100000)),
]
for name, tex, got in intervals:
    ok = all(abs(t - g) < 0.06 for t, g in zip(tex, got)); bad += 0 if ok else 1
    print("%-28s tex=%s raw=%s %s" % (name, tex, tuple(round(g, 2) for g in got), "OK" if ok else "MISMATCH"))


def cnt(a, b, neg=True):
    return sum(1 for st in CK if (ci(a, b, st)[2] < 0 if neg else ci(a, b, st)[1] > 0))


counts = [("elbow prior-blank sig<0", cnt(E["prior"], E["blank"]), 7),
          ("elbow coach-blank sig<0", cnt(E["coach"], E["blank"]), 43),
          ("elbow coach-prior sig<0", cnt(E["coach"], E["prior"]), 38),
          ("finger prior-blank sig<0", cnt(F["prior"], F["blank"]), 45),
          ("finger coach-prior sig>0", cnt(F["coach"], F["prior"], False), 45)]
for name, g, t in counts:
    ok = g == t; bad += 0 if ok else 1
    print("%-28s tex=%d raw=%d %s" % (name, t, g, "OK" if ok else "MISMATCH"))

last = max(st for st in CK if ci(E["coach"], E["prior"], st)[2] < 0)
ok = last == 96000; bad += 0 if ok else 1
print("elbow coach-prior last sig ckpt   tex=96000 raw=%d %s" % (last, "OK" if ok else "MISMATCH"))

ok = all(ci(F["prior"], F["blank"], st)[2] < 0 for st in range(14000, 100001, 2000)); bad += 0 if ok else 1
print("finger prior-blank sig at every ckpt from 14k: %s" % ("OK" if ok else "MISMATCH"))
ok = all(ci(F["coach"], F["prior"], st)[1] > 0 for st in range(12000, 100001, 2000)); bad += 0 if ok else 1
print("finger coach-prior sig>0 at every ckpt from 12k: %s" % ("OK" if ok else "MISMATCH"))
# prose: "From 40k on ... significantly worse ... at 27 of 31 checkpoints"
from40 = [st for st in CK if st >= 40000]
n40 = sum(1 for st in from40 if ci(F["coach"], F["blank"], st)[1] > 0)
ok = (n40, len(from40)) == (27, 31); bad += 0 if ok else 1
print("finger coach-blank sig>0 from 40k    tex=27/31 raw=%d/%d %s" % (n40, len(from40), "OK" if ok else "MISMATCH"))

# prose: "significant at seven: 8k, 10k and 12k, and four isolated later checkpoints with no two adjacent"
sig = [st for st in CK if ci(E["prior"], E["blank"], st)[2] < 0]
later = [st for st in sig if st > 12000]
isolated = all(abs(a - b) > 2000 for i, a in enumerate(later) for b in later[i + 1:])
ok = sig[:3] == [8000, 10000, 12000] and len(later) == 4 and isolated; bad += 0 if ok else 1
print("elbow prior-blank sig ckpts %s: first three 8k-12k, four isolated later %s" % (sig, "OK" if ok else "MISMATCH"))

for tag, D in (("elbow", E), ("finger", F)):
    for k in ("blank", "prior", "coach"):
        xs = [c[100000] - c[80000] for c in D[k].values()]; m = sum(xs) / 12
        print("  plateau %-6s %-6s %+5.2f sd %5.2f" % (tag, k, m, math.sqrt(sum((x - m) ** 2 for x in xs) / 11)))

print("\nprose mismatches: %d" % bad)
