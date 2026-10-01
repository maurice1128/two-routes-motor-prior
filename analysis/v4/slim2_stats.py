# -*- coding: utf-8 -*-
"""Statistics for the slim-2 revision (third council, 2026-10-01). Merges slim2_* keys into numbers.json.

Teacher (primary): finger withdrawn-minus-never, retention >= 50% (threshold fixed before testing,
replacing the data-derived 60%), per-t_w retention, the t_w slope, all under both endpoints.
Drop at withdrawal: loss-term minus teacher component per t_w with Holm across the seven t_w.
Body model (secondary): trained-minus-random gap compared BETWEEN bodies (arm3 vs arm1, arm4 vs
arm1; Welch t on independent seeds, body as the unit), both endpoints; the random model's
sensitivity to the action; the one-joint arm's reach floor is computed separately (reach floor
script) and entered as a constant.
Primary family (Holm): 14 tests listed in FAMILY below."""
import json, math
import numpy as np
from scipy import stats
from common import *

NUM = json.load(open("numbers.json")); AR = arms()
EP = {"late": end, "100k": lambda r, s: r[s][100000]}
P = {}


def tp(v, one_sided_less=False):
    t, p = stats.ttest_1samp(v, 0.0)
    if one_sided_less:
        return float(p / 2) if t < 0 else float(1 - p / 2)
    return float(p)


sw = sweep("finger")
for e, f in EP.items():
    per_tw = {tw: [f(sw["arms"][tw][0], s) - f(sw["none"], s) for s in S12] for tw in TWS}
    worth = [f(sw["con"], s) - f(sw["none"], s) for s in S12]
    pooled = [np.mean([per_tw[tw][i] for tw in TWS]) for i in range(12)]
    P["finger_withdrawn_never_" + e] = tp(pooled)
    P["finger_kept50_" + e] = tp([p - 0.5 * w for p, w in zip(pooled, worth)], one_sided_less=True)
    wm = float(np.mean(worth))
    NUM["slim2_kept_per_tw_" + e] = [int(round(100 * np.mean(per_tw[tw]) / wm)) for tw in TWS]
    sl = slope([[per_tw[tw][i] for tw in TWS] for i in range(12)], np.array(TWS) / 1000.0)
    NUM["slim2_tw_slope_" + e] = [round(z, 2) for z in ci(sl)]; P["finger_tw_slope_" + e] = tp(sl)
    # kept >= 50% one-sided upper bound
    NUM["slim2_kept50_upper_" + e] = round(upper1([p - 0.5 * w for p, w in zip(pooled, worth)]), 2)

# elbow totals at both endpoints (range of withdrawn-minus-kept)
swe = sweep("elbow")
for e, f in EP.items():
    cells = [float(np.mean([f(swe["arms"][tw][0], s) - f(swe["con"], s) for s in S12])) for tw in TWS]
    NUM["slim2_elbow_total_range_" + e] = [round(min(cells), 1), round(max(cells), 1)]

# body model: trained-minus-random gap, between bodies (Welch), both endpoints
for e, f in EP.items():
    g = {b: [f(AR[b][0]["prior"], t) - f(AR[b][0]["randprior"], t) for t in AR[b][1]] for b in AR}
    for b in ("arm3", "arm4"):
        t, p = stats.ttest_ind(g[b], g["arm1"], equal_var=False)
        d = float(np.mean(g[b]) - np.mean(g["arm1"]))
        se = math.sqrt(np.var(g[b], ddof=1) / len(g[b]) + np.var(g["arm1"], ddof=1) / len(g["arm1"]))
        NUM["slim2_gapdiff_%s_arm1_%s" % (b, e)] = [round(d, 2), round(float(p), 6)]
        P["gapdiff_%s_arm1_%s" % (b, e)] = float(p)
    # random minus blank, per arm, and trained minus blank64 relative (for the text)
    for b in AR:
        A, s = AR[b]
        NUM["slim2_rb_%s_%s" % (b, e)] = [round(z, 2) for z in ci([f(A["randprior"], t) - f(A["blank"], t) for t in s])]

# the drop at withdrawal: loss-term minus teacher component, Holm across t_w within each body
d1 = lambda r, s, tw: r[s][tw + 1000] - r[s][tw]
for body in ("elbow", "finger"):
    sb = sweep(body); ps = {}
    for tw in TWS:
        a, sa = sb["dip"][tw]
        v = [d1(a, s, tw) - 2 * d1(sa, s, tw) + d1(sb["dcon"], s, tw) for s in S12]
        ps[tw] = (float(np.mean(v)), tp(v))
    order = sorted(ps, key=lambda k: ps[k][1]); run = 0.0; holm = {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k][1])); holm[k] = run
    NUM["slim2_dip_lmt_holm_" + body] = {str(tw): [round(ps[tw][0], 2), float("%.3g" % ps[tw][1]), float("%.3g" % holm[tw])] for tw in TWS}
    # share of the total drop carried by the loss term, point estimates
    NUM["slim2_dip_loss_share_" + body] = {str(tw): int(round(100 * NUM["slim_dip_%s_loss_%d" % (body, tw)][0] / NUM["slim_dip_%s_total_%d" % (body, tw)][0])) for tw in TWS}

# primary family
order = sorted(P, key=P.get); run = 0.0; holm = {}
for i, k in enumerate(order):
    run = max(run, min(1.0, (len(order) - i) * P[k])); holm[k] = run
NUM["slim2_family"] = {k: [float("%.3g" % P[k]), float("%.3g" % holm[k])] for k in order}
NUM["slim2_family_size"] = len(P); NUM["slim2_family_max_holm"] = float("%.3g" % max(holm.values()))
json.dump(NUM, open("numbers.json", "w"), indent=1)
for k in sorted(NUM):
    if k.startswith("slim2_"): print(k, NUM[k])
