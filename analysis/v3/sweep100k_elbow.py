# -*- coding: utf-8 -*-
"""Attachment sweep at 100k, myoElbow (teacher clean30k).

Endpoint at 100k comes from the new runs (results_sweep100k_elbow/w<tw>, 2k eval grid) plus the
existing t_w=8000 arms (results_coach100k_elbow abrupt, results_coach100k_elbow_off2 selfanchor).
constant = results_coach100k_elbow/constant, none = results_conv_elbow_blank/blank.
Reproduction: every new run is compared with the 12k sweep (read only) at 2k..12k.
"""
import json, os, math
import numpy as np

W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
Z = r"C:\Users\maurice\Desktop\world_model_zeroshot\b2_pilot\results_attachment_myoelbow"
TWS = [2000, 3000, 4000, 5000, 6000, 7000, 8000]


def load(d, c):
    return {s: {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(os.path.join(d, "%s_seed%d.json" % (c, s))))} for s in range(12)}


def pt(v):
    v = np.asarray(v, float); m = v.mean(); se = v.std(ddof=1) / math.sqrt(len(v))
    return m, m - 2.201 * se, m + 2.201 * se


def star(t3):
    return "*" if (t3[1] > 0 or t3[2] < 0) else " "


con = load(os.path.join(W, "results_coach100k_elbow"), "constant")
none = load(os.path.join(W, "results_conv_elbow_blank"), "blank")
arms = {}
for tw in TWS[:-1]:
    d = os.path.join(W, "results_sweep100k_elbow", "w%d" % tw)
    arms[tw] = (load(d, "abrupt"), load(d, "selfanchor"))
arms[8000] = (load(os.path.join(W, "results_coach100k_elbow"), "abrupt"), load(os.path.join(W, "results_coach100k_elbow_off2"), "selfanchor"))

# reproduction vs the 12k sweep
bad = tot = 0
for tw in TWS[:-1]:
    for ai, name in enumerate(("abrupt", "selfanchor")):
        z = load(Z, "%s_w%d" % (name, tw))
        for s in range(12):
            for st in range(2000, 12001, 2000):
                tot += 1; bad += abs(arms[tw][ai][s][st] - z[s][st]) > 0.005
print("reproduction vs the 12k sweep at 2k..12k: %d/%d cells differ" % (bad, tot))
print("constant mean at 100k %.2f mrad | never-guided mean at 100k %.2f mrad\n" % (
    np.mean([con[s][100000] for s in range(12)]), np.mean([none[s][100000] for s in range(12)])))

COMP = {"teacher": ("sa", "con"), "loss term": ("ab", "sa"), "total": ("ab", "con"), "abrupt-none": ("ab", "none")}
for st_name, st in (("endpoint100k", 100000), ("at 50k", 50000)):
    for cname, (x, y) in COMP.items():
        cells, per = [], {s: [] for s in range(12)}
        for tw in TWS:
            a, sa = arms[tw]; src = {"ab": a, "sa": sa, "con": con, "none": none}
            v = [src[x][s][st] - src[y][s][st] for s in range(12)]
            for s in range(12): per[s].append(v[s])
            t3 = pt(v); cells.append("%+6.2f%s" % (t3[0], star(t3)))
        xs = np.array(TWS) / 1000.0
        sl = pt([np.polyfit(xs, per[s], 1)[0] for s in range(12)])
        print("%-12s %-12s %s   slope %+5.2f [%+5.2f, %+5.2f]%s" % (st_name, cname, " ".join(cells), *sl, star(sl)))
print("\narm means at 100k: t_w, abrupt, selfanchor")
for tw in TWS:
    a, sa = arms[tw]
    print("  %5d  %6.2f  %6.2f" % (tw, np.mean([a[s][100000] for s in range(12)]), np.mean([sa[s][100000] for s in range(12)])))
