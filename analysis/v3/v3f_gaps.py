# -*- coding: utf-8 -*-
"""Analysis of the gap-closing batch (run_v3c_gaps.ps1) and the arm1/arm4 prior rerun.
Works on partial data: every contrast uses the seeds present on both sides and prints n.
 1. finger body-model withdrawal at 100k (+ seed-for-seed check of the 12k runs)
 2. model-free SAC at batch 64 without duplication vs blank and blank64
 3. finger replacing regime with T2 (same layout as the elbow Table V)
 4. arm1/arm4 prior rebuilt (prior2) vs the unreproducible prior, and the Table II contrasts"""
import json, os, math, glob
import numpy as np
W = "C:/Users/maurice/Desktop/robotic_research/wm_prior"
TQ = {n: t for n, t in zip(range(2, 31), [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
                                          2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045])}
def load(d, c):
    r = {}
    for f in glob.glob(os.path.join(W, d, "%s_seed*.json" % c)):
        s = int(f.split("_seed")[1].split(".")[0]); r[s] = {int(x["step"]): x["eval_dist"] * 1000 for x in json.load(open(f))}
    return r
def pt(a, b, st, st_b=None):
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2: return "   (n=%d)" % len(seeds)
    v = np.array([a[s][st] - b[s][st_b or st] for s in seeds]); n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n); t = TQ[n]
    return "%+8.2f [%+8.2f, %+8.2f]%s n=%d" % (m, m - t * se, m + t * se, "*" if (m - t * se > 0 or m + t * se < 0) else " ", n)
def diffv(a, b, f):
    seeds = sorted(set(a) & set(b)); return seeds, [f(a, s) - f(b, s) for s in seeds]
def pv(seeds_v):
    seeds, v = seeds_v
    if len(v) < 2: return "   (n=%d)" % len(v)
    v = np.array(v); n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n); t = TQ[n]
    return "%+8.2f [%+8.2f, %+8.2f]%s n=%d" % (m, m - t * se, m + t * se, "*" if (m - t * se > 0 or m + t * se < 0) else " ", n)

# ---------------------------------------------------------------- 1
print("== 1. myoFinger body-model withdrawal at 8k, scored at 12k and 100k (mm)")
keep = load("results_conv_finger_prior", "prior"); fb = load("results_conv_finger_blank", "blank")
M = {a: load("results_matched100k_finger", a) for a in ("soft", "purge", "matched")}; M["keep"] = keep
old = {a: load("results_matched_finger", a) for a in ("soft", "purge", "matched", "keep")}
bad = tot = 0
for a in ("soft", "purge", "matched"):
    for s in set(M[a]) & set(old[a]):
        for st in range(2000, 12001, 2000):
            if st in old[a][s]: tot += 1; bad += abs(M[a][s][st] - old[a][s][st]) > 0.005
print("  reproduction of the 12k runs at 2k..12k: %d/%d cells differ" % (bad, tot))
for x, y in (("purge", "keep"), ("soft", "keep"), ("matched", "keep"), ("soft", "matched"), ("purge", "matched")):
    print("  %-8s - %-8s 12k %s | 100k %s" % (x, y, pt(M[x], M[y], 12000), pt(M[x], M[y], 100000)))
for a in ("soft", "purge", "matched", "keep"):
    print("  %-8s - blank    100k %s" % (a, pt(M[a], fb, 100000)))
for a in ("soft", "purge", "matched"):
    print("  plateau %-8s 100k-80k %s" % (a, pt(M[a], M[a], 100000, 80000)))

# ---------------------------------------------------------------- 2
print("\n== 2. model-free SAC at batch 64 (distinct), 100k")
for b, d in (("myoelbow", "elbow"), ("myofinger", "finger"), ("arm4", None)):
    b64 = load("results_batch64_%s" % b, "blank")
    if d:
        bl = load("results_conv_%s_blank" % d, "blank"); dup = load("results_conv_%s_blank64" % d, "blank64"); pr = load("results_conv_%s_prior" % d, "prior")
    else:
        bl = load("results_arms100k_arm4_blank", "blank"); dup = load("results_arms100k_arm4_blank64", "blank64"); pr = load("results_arms100k_arm4_prior2", "prior")
    print("  %-9s batch64 - blank   %s" % (b, pt(b64, bl, 100000)))
    print("  %-9s blank64 - batch64 %s" % (b, pt(dup, b64, 100000)))
    print("  %-9s blank64 - blank   %s" % (b, pt(dup, bl, 100000)))
    print("  %-9s prior - batch64   %s" % (b, pt(pr, b64, 100000)))
    print("  %-9s plateau batch64   %s" % (b, pt(b64, b64, 100000, 80000)))

# ---------------------------------------------------------------- 3
print("\n== 3. myoFinger replacing regime, teacher T2 (mm)")
R = "results_replace100k_myofinger"; TW4 = [2000, 4000, 6000, 8000]
rc = load(R, "constant"); none = load("results_sweep100k_finger_T2", "none")
arm = {a: {tw: load(R, "%s_w%d" % (a, tw)) for tw in TW4} for a in ("abrupt", "selfanchor", "gateon")}
get = lambda n, tw: rc if n == "constant" else (none if n == "none" else arm[n][tw])
print("  constant(pure distillation) - none 100k %s" % pt(rc, none, 100000))
for x, y in (("selfanchor", "constant"), ("gateon", "constant"), ("selfanchor", "gateon"), ("abrupt", "selfanchor"), ("abrupt", "constant"), ("abrupt", "none")):
    for metric in ("post4k", "end100k"):
        cells = []; per = {}
        for tw in TW4:
            st = tw + 4000 if metric == "post4k" else 100000
            seeds, v = diffv(get(x, tw), get(y, tw), lambda r, s: r[s][st])
            cells.append(pv((seeds, v)).split(" n=")[0].strip()[:9]); per[tw] = dict(zip(seeds, v))
        common = sorted(set.intersection(*[set(per[tw]) for tw in TW4])) if all(per.values()) else []
        sl = [np.polyfit(np.array(TW4) / 1000.0, [per[tw][s] for tw in TW4], 1)[0] for s in common]
        print("  %-10s - %-10s %-7s %s | slope %s" % (x, y, metric, "  ".join(cells), pv((common, sl))))
for a in arm:
    for tw in TW4:
        if arm[a][tw]:
            print("  plateau %-10s w%d %s" % (a, tw, pt(arm[a][tw], arm[a][tw], 100000, 80000)))
print("  seeds whose 100k error <= error at t_w:", {a: [sum(arm[a][tw][s][100000] <= arm[a][tw][s][tw] for s in arm[a][tw]) for tw in TW4] for a in arm})

# ---------------------------------------------------------------- 4
print("\n== 4. arm1 / arm4: rebuilt prior (prior2) vs the unreproducible prior, 100k")
for b in ("arm1", "arm4"):
    p_old = load("results_arms100k_%s_prior" % b, "prior"); p_new = load("results_arms100k_%s_prior2" % b, "prior")
    bl = load("results_arms100k_%s_blank" % b, "blank"); b64 = load("results_arms100k_%s_blank64" % b, "blank64"); rp = load("results_arms100k_%s_randprior" % b, "prior")
    print("  %s prior2 - prior(old) %s" % (b, pt(p_new, p_old, 100000)))
    for name, other in (("blank", bl), ("blank64", b64), ("randprior", rp)):
        print("  %s prior2 - %-9s %s   | old prior - %-9s %s" % (b, name, pt(p_new, other, 100000), name, pt(p_old, other, 100000)))
    print("  %s plateau prior2 %s" % (b, pt(p_new, p_new, 100000, 80000)))
