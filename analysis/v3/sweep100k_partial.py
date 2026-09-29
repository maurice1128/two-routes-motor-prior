# -*- coding: utf-8 -*-
"""Attachment sweep at 100k, myoFinger with the competent teacher T2: whatever
attachment points are finished so far.

Components at each t_w, all paired on the seed:
  teacher   = selfanchor - constant      (the part dependence would have to live in)
  loss term = abrupt - selfanchor
  total     = abrupt - constant
  literature's measure = abrupt - none
Metrics: dip (t_w+1000 minus t_w), post-4k (error at t_w+4000), endpoint at 100k.
t_w = 8000 comes from results_coach100k_finger_T2 with constant/none from the
1k-grid control runs.
"""
import json, os, math, glob
import numpy as np

W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
F = os.path.join(W, "results_sweep100k_finger_T2")
TWS = [2000, 3000, 4000, 5000, 6000, 7000, 8000]


def load(d, c, n=12):
    out = {}
    for s in range(n):
        p = os.path.join(d, "%s_seed%d.json" % (c, s))
        if not os.path.exists(p):
            return None
        out[s] = {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(p))}
    return out


def pt(v):
    v = np.asarray(v, float); m = v.mean(); se = v.std(ddof=1) / math.sqrt(len(v))
    return m, m - 2.201 * se, m + 2.201 * se


def star(t3):
    return "*" if (t3[1] > 0 or t3[2] < 0) else " "


con = load(F, "constant"); none = load(F, "none")
arms = {}
for tw in TWS:
    d = os.path.join(F, "w%d" % tw) if tw != 8000 else os.path.join(W, "results_coach100k_finger_T2")
    a = load(d, "abrupt"); s = load(d, "selfanchor")
    if a and s:
        arms[tw] = (a, s)
done = sorted(arms)
print("myoFinger, competent teacher T2, 100k. Attachment points complete: %s" % [t // 1000 for t in done])
print("constant mean at 100k %.2f mm | never-guided mean at 100k %.2f mm\n" % (
    np.mean([con[s][100000] for s in range(12)]), np.mean([none[s][100000] for s in range(12)])))

M = {"dip": lambda c, tw: c[tw + 1000] - c[tw], "post-4k": lambda c, tw: c[tw + 4000], "endpoint100k": lambda c, tw: c[100000]}
COMP = {"teacher": ("sa", "con"), "loss term": ("ab", "sa"), "total": ("ab", "con"), "abrupt-none": ("ab", "none")}
rows = {}
for mname, f in M.items():
    for cname, (x, y) in COMP.items():
        cells, per = [], {s: [] for s in range(12)}
        for tw in done:
            a, sa = arms[tw]
            src = {"ab": a, "sa": sa, "con": con, "none": none}
            v = [f(src[x][s], tw) - f(src[y][s], tw) for s in range(12)]
            for s in range(12):
                per[s].append(v[s])
            t3 = pt(v); cells.append("%+7.2f%s" % (t3[0], star(t3)))
        xs = np.array(done) / 1000.0
        sl = pt([np.polyfit(xs, per[s], 1)[0] for s in range(12)]) if len(done) > 2 else (float("nan"),) * 3
        rows[(mname, cname)] = (cells, sl)
        print("%-12s %-12s %s   slope %+6.2f [%+6.2f, %+6.2f]%s" % (mname, cname, "  ".join(cells), *sl, star(sl)))

print("\nper-t_w arm means (mm): t_w, constant@100k, abrupt@100k, selfanchor@100k")
for tw in done:
    a, sa = arms[tw]
    print("  %5d  %6.2f  %6.2f  %6.2f" % (tw, np.mean([con[s][100000] for s in range(12)]),
                                          np.mean([a[s][100000] for s in range(12)]), np.mean([sa[s][100000] for s in range(12)])))
