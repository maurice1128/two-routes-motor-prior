# -*- coding: utf-8 -*-
"""Elbow teacher withdrawal run to 200k (results_long200k_elbow): (1) checks that steps <= 100k
reproduce the paper's 100k runs; (2) the headline contrasts with the late mean taken over
190-200k; (3) plateau check 170-180k -> 190-200k. Appends to numbers.json."""
import json, os
import numpy as np
from common import *

D = "results_long200k_elbow"
L200 = list(range(190000, 200001, 2000)); E200 = list(range(170000, 180001, 2000))
none = load(D, "none"); con = load(D, "constant"); ab = {tw: load(D + "/w%d" % tw, "abrupt") for tw in TWS}
sw = sweep("elbow")
ref = {"none": sw["none"], "constant": sw["con"]}; ref.update({tw: sw["arms"][tw][0] for tw in TWS})
new = {"none": none, "constant": con}; new.update(ab)
mm = 0; tot = 0
for k in new:
    for s in S12:
        for st in range(2000, 100001, 2000):
            if st in ref[k][s]:
                tot += 1; mm += abs(new[k][s][st] - ref[k][s][st]) > 1e-9
print("reproduction of the 100k runs: %d / %d checkpoints differ" % (mm, tot))
NUM = json.load(open("numbers.json"))
e = lambda r, s: win(r, s, L200)
pooled = [np.mean([e(ab[tw], s) - e(none, s) for tw in TWS]) for s in S12]
worth = [e(con, s) - e(none, s) for s in S12]
NUM["long_elbow_pooled"] = [round(x, 2) for x in list(ci(pooled)) + [upper1(pooled)]]
NUM["long_elbow_worth"] = [round(x, 2) for x in ci(worth)]
NUM["long_elbow_abnone_cells"] = [[round(ci([e(ab[tw], s) - e(none, s) for s in S12])[0], 2), bool(sig(ci([e(ab[tw], s) - e(none, s) for s in S12])))] for tw in TWS]
NUM["long_elbow_total_cells"] = [[round(ci([e(ab[tw], s) - e(con, s) for s in S12])[0], 2), bool(sig(ci([e(ab[tw], s) - e(con, s) for s in S12])))] for tw in TWS]
NUM["long_elbow_abnone_slope"] = [round(x, 2) for x in ci(slope([[e(ab[tw], s) - e(none, s) for tw in TWS] for s in S12], np.array(TWS) / 1000.0))]
for k in ("none", "constant"):
    NUM["long_elbow_mean_" + k] = round(float(np.mean([e(new[k], s) for s in S12])), 2)
drift = [np.mean([(e(ab[tw], s) - e(none, s)) - (win(ab[tw], s, E200) - win(none, s, E200)) for tw in TWS]) for s in S12]
NUM["long_elbow_pooled_drift"] = [round(x, 2) for x in ci(drift)]
mov = [k for k in new if sig(ci([e(new[k], s) - win(new[k], s, E200) for s in S12]))]
NUM["long_elbow_moving"] = [str(k) for k in mov]
for k in sorted(NUM):
    if k.startswith("long_"): print(k, NUM[k])
json.dump(NUM, open("numbers.json", "w"), indent=1)
