# -*- coding: utf-8 -*-
"""Quantities the v4 prose quotes that the table generator does not write:
(1) the slope over t_w of constant-none's own change across each dip window (finger, additive),
(2) plateau drift of the two arm4 withdrawal conditions. Appends to numbers.json."""
import json, os
import numpy as np
from common import *
NUM = json.load(open("numbers.json"))
xs = np.array(TWS) / 1000.0
for body in ("elbow", "finger"):
    sw = SW = sweep(body)
    d = lambda r, s, tw: r[s][tw + 1000] - r[s][tw]
    per = [[d(sw["dcon"], s, tw) - d(sw["dnone"], s, tw) for tw in TWS] for s in S12]
    sl = slope(per, xs); NUM["sw_%s_dip_connone_window_slope" % body] = [round(v, 2) for v in ci(sl)]
    print(body, "constant-none window-change slope", NUM["sw_%s_dip_connone_window_slope" % body])
S24 = list(range(24))
for a in ("purge", "matched"):
    r = load("results_matched100k_arm4", a)
    c = ci([win(r, s, LATE) - win(r, s, EARLY_WINDOW) for s in S24]); NUM["plateau_arm4_%s" % a] = [round(v, 2) for v in c]
    print("arm4", a, "drift", NUM["plateau_arm4_%s" % a])
NUM["share_finger_add"] = round(100 * NUM["finger_add_pooled"][0] / NUM["finger_add_ref"][0])
NUM["share_finger_rep"] = round(100 * NUM["finger_rep_pooled"][0] / NUM["finger_rep_ref"][0])
NUM["share_finger_bm_purge"] = round(100 * NUM["bm_finger_purge_blank_end"][0] / NUM["bm_finger_keep_blank_end"][0])
NUM["share_finger_bm_matched"] = round(100 * NUM["bm_finger_matched_blank64_end"][0] / NUM["bm_finger_keep_blank64_end"][0])
print({k: NUM[k] for k in NUM if k.startswith("share")})
json.dump(NUM, open("numbers.json", "w"), indent=1)
