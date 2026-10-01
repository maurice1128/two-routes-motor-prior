# -*- coding: utf-8 -*-
"""Plateau check for the v4 endpoint: per condition, the paired change between the late window
(90-100k) and the window twenty thousand steps earlier (70-80k); and the same drift for the
headline contrasts (withdrawn minus never had, pooled over attachment)."""
import json, os
import numpy as np
from common import *

rows = []
def cond(label, run, seeds=S12):
    v = [win(run, s, LATE) - win(run, s, EARLY_WINDOW) for s in seeds]; c = ci(v); rows.append((label, c))

E, F = core()
for k in E: cond("elbow " + k, E[k])
for k in F: cond("finger " + k, F[k])
for b, (A, seeds) in arms().items():
    for k in A: cond("%s %s" % (b, k), A[k], seeds)
for body in ("elbow", "finger"):
    sw = sweep(body)
    for tw in TWS:
        cond("%s sweep abrupt w%d" % (body, tw), sw["arms"][tw][0]); cond("%s sweep selfanchor w%d" % (body, tw), sw["arms"][tw][1])
    cond("%s constant" % body, sw["con"]); cond("%s randanchor w8000" % body, sw["ra"])
    if body == "finger": cond("finger none", sw["none"])
    con, none, arm = replace(body)
    cond("%s replace constant" % body, con)
    for a in arm:
        for tw in TW4: cond("%s replace %s w%d" % (body, a, tw), arm[a][tw])
    M = bodymodel(body)
    for a in ("purge", "matched"): cond("%s bodymodel %s" % (body, a), M[a])
for a in ("purge", "matched"): cond("arm4 bodymodel %s" % a, load("results_matched100k_arm4", a), list(range(24)))
moving = [(l, c) for l, c in rows if sig(c)]
print("conditions: %d, moving: %d (expected by chance at 5%%: %.1f)" % (len(rows), len(moving), 0.05 * len(rows)))
for l, c in moving: print("  moving: %-32s %+.2f [%+.2f, %+.2f]" % (l, *c))
# drift of the headline contrasts
DRIFT = {}
print("\nheadline contrast drift (late window minus 70-80k window):")
for body in ("elbow", "finger"):
    sw = sweep(body)
    v = [np.mean([(win(sw["arms"][tw][0], s, LATE) - win(sw["none"], s, LATE)) - (win(sw["arms"][tw][0], s, EARLY_WINDOW) - win(sw["none"], s, EARLY_WINDOW)) for tw in TWS]) for s in S12]
    print("  %s additive abrupt-none pooled drift %+.2f [%+.2f, %+.2f]" % (body, *ci(v))); DRIFT[body + "_add"] = [round(x, 2) for x in ci(v)]
    con, none, arm = replace(body)
    v = [np.mean([(win(arm["abrupt"][tw], s, LATE) - win(none, s, LATE)) - (win(arm["abrupt"][tw], s, EARLY_WINDOW) - win(none, s, EARLY_WINDOW)) for tw in TW4]) for s in S12]
    print("  %s replacing abrupt-none pooled drift %+.2f [%+.2f, %+.2f]" % (body, *ci(v))); DRIFT[body + "_rep"] = [round(x, 2) for x in ci(v)]
json.dump({"n_conditions": len(rows), "headline_drift": DRIFT, "moving": [[l, [round(x, 2) for x in c]] for l, c in moving]},
          open(os.path.join(W, "tcds_v4", "plateau_v4.json"), "w"), indent=1)
