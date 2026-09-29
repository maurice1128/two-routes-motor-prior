# -*- coding: utf-8 -*-
"""Extra numbers for stage B: plateau checks of the 100k sweep arms and replacing arms, run counts and CPU hours."""
import json, os, glob, math, re
import numpy as np
W = "C:/Users/maurice/Desktop/robotic_research/wm_prior"
def load(d, c):
    return {s: {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(os.path.join(W, d, "%s_seed%d.json" % (c, s))))} for s in range(12)}
def pt(v):
    v = np.asarray(v, float); m = v.mean(); se = v.std(ddof=1) / math.sqrt(len(v)); lo, hi = m - 2.201 * se, m + 2.201 * se
    return "%+7.2f [%+7.2f, %+7.2f]%s" % (m, lo, hi, "*" if (lo > 0 or hi < 0) else " ")
print("== plateau 100k-80k, sweep arms")
for body, base, tws in (("finger", "results_sweep100k_finger_T2", [2000, 3000, 4000, 5000, 6000, 7000]), ("elbow", "results_sweep100k_elbow", [2000, 3000, 4000, 5000, 6000, 7000])):
    for tw in tws:
        for arm in ("abrupt", "selfanchor"):
            a = load("%s/w%d" % (base, tw), arm)
            print("  %-6s tw=%d %-10s %s" % (body, tw, arm, pt([a[s][100000] - a[s][80000] for s in range(12)])))
for name, d, c in (("finger constant T2 (1k grid)", "results_sweep100k_finger_T2", "constant"), ("finger none (1k grid)", "results_sweep100k_finger_T2", "none")):
    a = load(d, c); print("  %-30s %s" % (name, pt([a[s][100000] - a[s][80000] for s in range(12)])))
print("== plateau, replacing regime arms at 100k")
for d in sorted(glob.glob(os.path.join(W, "results_replace100k_myoelbow", "*"))):
    if not os.path.isdir(d): continue
    for f in sorted(set(re.sub(r"_seed\d+\.json$", "", os.path.basename(x)) for x in glob.glob(os.path.join(d, "*_seed*.json")))):
        try:
            a = load(os.path.relpath(d, W), f); print("  %-30s %-12s %s" % (os.path.basename(d), f, pt([a[s][100000] - a[s][80000] for s in range(12)])))
        except Exception as e: print("  skip", d, f, e)
print("== sanity: finger sweep constant vs coachT2 run at shared checkpoints")
c1 = load("results_sweep100k_finger_T2", "constant"); c2 = load("results_conv_finger_coachT2", "coach") if os.path.isdir(os.path.join(W, "results_conv_finger_coachT2")) else None
if c2:
    bad = sum(abs(c1[s][st] - c2[s][st]) > 0.005 for s in range(12) for st in range(2000, 100001, 2000)); print("  cells differing:", bad, "/", 12 * 50)
else:
    print("  coachT2 dir:", [x for x in os.listdir(W) if "coachT2" in x or "T2" in x])
