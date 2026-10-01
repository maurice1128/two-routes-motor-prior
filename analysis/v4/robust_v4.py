# -*- coding: utf-8 -*-
"""Robustness screen for the slimmed v4 story: every candidate claim under BOTH endpoints
(late mean 90-100k, and the single 100k evaluation). A claim is kept only if it holds under both.
Also the joint-count trend of trained-minus-random model, by (a) per-seed slope on seeds 0-11 and
(b) a meta-regression of the four arm contrasts (all seeds) on joint count, weighted by 1/SE^2."""
import json, math
import numpy as np
from common import *

EP = {"late": end, "100k": lambda r, s: r[s][100000]}
OUT = {}

def show(label, x, y, seeds, key=None):
    row = []
    for e, f in EP.items():
        v = [f(x, s) - f(y, s) for s in seeds]; c = ci(v); row.append((e, c))
    ok = all(sig(c) for _, c in row); same = len({np.sign(c[0]) for _, c in row}) == 1
    print("%-38s " % label + " | ".join("%s %+8.2f [%+8.2f, %+8.2f]%s" % (e, *c, "*" if sig(c) else " ") for e, c in row)
          + ("   ROBUST" if ok and same else ""))
    if key: OUT[key] = {e: [round(z, 2) for z in c] for e, c in row}
    return row

E, F = core(); AR = arms()
print("== trained model minus random model (only the model's content differs)")
show("elbow prior-randprior", E["prior"], E["randprior"], S12, "pr_elbow")
show("finger prior-randprior", F["prior"], F["randprior"], S12, "pr_finger")
for b in ("arm1", "arm2", "arm3", "arm4"):
    A, s = AR[b]; show("%s prior-randprior" % b, A["prior"], A["randprior"], s, "pr_" + b)
print("== random model minus blank")
for b in ("arm1", "arm2", "arm3", "arm4"):
    A, s = AR[b]; show("%s randprior-blank" % b, A["randprior"], A["blank"], s, "rb_" + b)
print("== trained model minus model-free (both baselines)")
for nm, B in (("elbow", E), ("finger", F)):
    show("%s prior-blank" % nm, B["prior"], B["blank"], S12); show("%s prior-blank64" % nm, B["prior"], B["blank64"], S12)
for b in ("arm1", "arm2", "arm3", "arm4"):
    A, s = AR[b]; show("%s prior-blank" % b, A["prior"], A["blank"], s); show("%s prior-blank64" % b, A["prior"], A["blank64"], s)

print("== joint-count trend of the trained-vs-random gap")
for e, f in EP.items():
    # (a) relative to each arm's blank mean, per-seed slope, seeds 0-11
    rel = {b: {s: (f(AR[b][0]["prior"], s) - f(AR[b][0]["randprior"], s)) / np.mean([f(AR[b][0]["blank"], t) for t in AR[b][1]]) for s in AR[b][1]} for b in AR}
    sl = [float(np.polyfit([1, 2, 3, 4], [rel[b][s] for b in ("arm1", "arm2", "arm3", "arm4")], 1)[0]) for s in S12]
    c = ci(sl); print("  %s per-seed slope (fraction of blank per joint) %+.3f [%+.3f, %+.3f]%s" % (e, *c, "*" if sig(c) else ""))
    OUT["trend_seed_" + e] = [round(z, 3) for z in c]
    # (b) meta-regression on joint count, all seeds, weighted by 1/SE^2, absolute mm and relative
    for kind in ("mm", "rel"):
        ys, ws = [], []
        for b in ("arm1", "arm2", "arm3", "arm4"):
            A, s = AR[b]; bl = np.mean([f(A["blank"], t) for t in s]) if kind == "rel" else 1.0
            v = np.array([(f(A["prior"], t) - f(A["randprior"], t)) / bl for t in s]); ys.append(v.mean()); ws.append(len(v) / v.var(ddof=1))
        x = np.array([1, 2, 3, 4.]); ys = np.array(ys); ws = np.array(ws)
        X = np.vstack([np.ones(4), x]).T; Wm = np.diag(ws)
        cov = np.linalg.inv(X.T @ Wm @ X); beta = cov @ X.T @ Wm @ ys
        se = math.sqrt(cov[1, 1]); z = beta[1] / se
        print("  %s meta-regression (%s) slope %+.3f, SE %.3f, z %.1f" % (e, kind, beta[1], se, z))
        OUT["trend_meta_%s_%s" % (kind, e)] = [round(float(beta[1]), 3), round(se, 3), round(float(z), 2)]

print("== teacher withdrawal: withdrawn minus never guided, pooled over t_w")
for body in ("elbow", "finger"):
    sw = sweep(body)
    for e, f in EP.items():
        pooled = [np.mean([f(sw["arms"][tw][0], s) - f(sw["none"], s) for tw in TWS]) for s in S12]
        worth = [f(sw["con"], s) - f(sw["none"], s) for s in S12]
        c = ci(pooled); w = ci(worth)
        # paired non-inferiority: deficit - lam * |worth| < 0 ; worth is negative (help lowers error)
        res = []
        for lam in (1.0, 0.5, 0.25):
            v = [p + lam * wv for p, wv in zip(pooled, worth)]   # p - lam*|w| with w<0
            res.append((lam, upper1(v)))
        kept = None
        if body == "finger":
            # largest retained fraction r such that pooled - r*worth has upper bound < 0  (pooled/worth >= r)
            for r in np.arange(1.0, 0.0, -0.01):
                v = [p - r * wv for p, wv in zip(pooled, worth)]
                if upper1(v) < 0: kept = round(float(r), 2); break
        print("  %s %-5s withdrawn-none %+.2f [%+.2f, %+.2f]%s  worth %+.2f [%+.2f, %+.2f]  NI upper: %s%s" % (
            body, e, *c, "*" if sig(c) else " ", *w, ", ".join("lam %.2f %+.2f" % t for t in res),
            ("  retained >= %.2f" % kept) if kept else ""))
        OUT["wd_%s_%s" % (body, e)] = {"pooled": [round(z, 2) for z in c], "worth": [round(z, 2) for z in w],
                                      "ni": [[l, round(u, 2)] for l, u in res], "kept_lower": kept}
json.dump(OUT, open("robust_v4.json", "w"), indent=1)
