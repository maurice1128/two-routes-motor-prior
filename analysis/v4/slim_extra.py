# -*- coding: utf-8 -*-
"""Slim v4 extras: p-values and Holm correction over the primary family (both endpoints), and the
drift (70-80k -> 90-100k) of the trained-minus-random gap on each arm. Merges into numbers.json."""
import json, math
import numpy as np
from scipy import stats
from common import *
NUM = json.load(open("numbers.json")); AR = arms()
EP = {"late": end, "100k": lambda r, s: r[s][100000]}
fam = {}
for e, f in EP.items():
    rel = {b: {s: (f(AR[b][0]["prior"], s) - f(AR[b][0]["randprior"], s)) / np.mean([f(AR[b][0]["blank"], t) for t in AR[b][1]]) for s in AR[b][1]} for b in AR}
    fam["trend_" + e] = [float(np.polyfit([1, 2, 3, 4], [rel[b][s] for b in ("arm1", "arm2", "arm3", "arm4")], 1)[0]) for s in S12]
    for b in ("arm2", "arm3", "arm4"):
        A, s = AR[b]; fam["%s_prior_randprior_%s" % (b, e)] = [f(A["prior"], t) - f(A["randprior"], t) for t in s]
    sw = sweep("finger")
    pooled = [np.mean([f(sw["arms"][tw][0], s) - f(sw["none"], s) for tw in TWS]) for s in S12]
    worth = [f(sw["con"], s) - f(sw["none"], s) for s in S12]
    fam["finger_withdrawn_never_" + e] = pooled
    fam["finger_kept60_" + e] = [p - 0.6 * w for p, w in zip(pooled, worth)]   # < 0  <=> keeps >= 60%
ps = {}
for k, v in fam.items():
    t, p = stats.ttest_1samp(v, 0.0)
    ps[k] = float(p / 2) if k.startswith("finger_kept60") else float(p)   # kept60 is one-sided
order = sorted(ps, key=ps.get); m = len(order); holm = {}; run = 0.0
for i, k in enumerate(order):
    run = max(run, min(1.0, (m - i) * ps[k])); holm[k] = run
for k in order: print("%-36s p %.2e  Holm %.2e" % (k, ps[k], holm[k]))
NUM["slim_primary_family"] = {k: [float("%.3g" % ps[k]), float("%.3g" % holm[k])] for k in order}
NUM["slim_primary_max_holm"] = float("%.3g" % max(holm.values()))
for b in ("arm1", "arm2", "arm3", "arm4"):
    A, s = AR[b]
    v = [(win(A["prior"], t, LATE) - win(A["randprior"], t, LATE)) - (win(A["prior"], t, EARLY_WINDOW) - win(A["randprior"], t, EARLY_WINDOW)) for t in s]
    c = ci(v); NUM["slim_gap_drift_" + b] = [round(z, 2) for z in c]; print(b, "gap drift", [round(z, 2) for z in c])
json.dump(NUM, open("numbers.json", "w"), indent=1)
