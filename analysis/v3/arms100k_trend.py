# -*- coding: utf-8 -*-
"""Does the body model's advantage grow with joint count? Seed is the unit; seeds 0-11 exist on all
four arms, so each seed gives a 4-point curve (advantage vs joints) and a per-seed OLS slope.
Advantage is relative (mm / mean never-guided endpoint of that body) because the four arms have
different reach scales. Also: the random model's harm vs joints, and the extra-seed (n=24) 3-vs-4 contrast."""
import json, os, glob, math
import numpy as np
W = "C:/Users/maurice/Desktop/robotic_research/wm_prior"
TQ = {12: 2.201, 24: 2.069}
def load(d, c):
    r = {}
    for f in glob.glob(os.path.join(W, d, "%s_seed*.json" % c)):
        s = int(f.split("_seed")[1].split(".")[0]); r[s] = {int(x["step"]): x["eval_dist"] * 1000 for x in json.load(open(f))}
    return r
def pt(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n); t = TQ[n]
    return "%+7.3f [%+7.3f, %+7.3f]%s" % (m, m - t * se, m + t * se, "*" if (m - t * se > 0 or m + t * se < 0) else " ")
def wilcoxon_exact(v):
    v = np.asarray(v, float); v = v[v != 0]; n = len(v); r = np.argsort(np.argsort(np.abs(v))) + 1.0
    wp = r[v > 0].sum(); tot = 0; cnt = 0
    for m in range(1 << n):
        s = sum(r[i] for i in range(n) if m >> i & 1); tot += 1; cnt += (s <= wp)
    p1 = cnt / tot; return 2 * min(p1, 1 - p1)
A = {}
for b, nj in (("arm1", 1), ("arm2", 2), ("arm3", 3), ("arm4", 4)):
    A[nj] = {k: load("results_arms100k_%s_%s" % (b, d), c) for k, d, c in (("blank", "blank", "blank"), ("prior", "prior", "prior"), ("rand", "randprior", "prior"), ("b64", "blank64", "blank64"))}
ST = 100000
for label, ref in (("prior-blank", "blank"), ("prior-blank64", "b64"), ("rand-blank", "blank")):
    src = "rand" if label == "rand-blank" else "prior"
    print("== relative %s at 100k (unit: fraction of body's mean never-guided endpoint)" % label)
    rel = {}
    for nj in (1, 2, 3, 4):
        bl = np.mean([A[nj]["blank"][s][ST] for s in A[nj]["blank"]])
        rel[nj] = {s: (A[nj][src][s][ST] - A[nj][ref][s][ST]) / bl for s in range(24) if s in A[nj][src] and s in A[nj][ref]}
        print("   %dj  n=%2d  %s" % (nj, len(rel[nj]), pt(list(rel[nj].values()))))
    sl = [np.polyfit([1, 2, 3, 4], [rel[nj][s] for nj in (1, 2, 3, 4)], 1)[0] for s in range(12)]
    print("   per-seed slope vs joints (seeds 0-11, per joint): %s   exact Wilcoxon p=%.3f" % (pt(sl), wilcoxon_exact(sl)))
    d41 = [rel[4][s] - rel[1][s] for s in range(12)]
    print("   4j-1j (seeds 0-11): %s   exact Wilcoxon p=%.3f" % (pt(d41), wilcoxon_exact(d41)))
    d43 = [rel[4][s] - rel[3][s] for s in range(24)]
    print("   4j-3j (seeds 0-23): %s" % pt(d43))
    print()
