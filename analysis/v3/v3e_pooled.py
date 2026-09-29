# -*- coding: utf-8 -*-
"""Weakness checks that need no new runs.
 A. abrupt - none at 100k pooled over attachment durations (per seed: mean over the t_w cells),
    so the 'not worse than never guided' claim rests on one tight interval, not on 7 wide ones.
 B. T2 shares seed 1 with one student: every finger-T2 contrast recomputed without seed 1."""
import json, os, math, glob
import numpy as np
W = "C:/Users/maurice/Desktop/robotic_research/wm_prior"
TQ = {10: 2.228, 11: 2.201, 12: 2.201}
TQ = {11: 2.228, 12: 2.201}
def load(d, c):
    r = {}
    for f in glob.glob(os.path.join(W, d, "%s_seed*.json" % c)):
        s = int(f.split("_seed")[1].split(".")[0]); r[s] = {int(x["step"]): x["eval_dist"] * 1000 for x in json.load(open(f))}
    return r
def pt(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n); t = TQ[n]
    return "%+7.2f [%+7.2f, %+7.2f]%s  (n=%d)" % (m, m - t * se, m + t * se, "*" if (m - t * se > 0 or m + t * se < 0) else " ", n)
TWS = [2000, 3000, 4000, 5000, 6000, 7000, 8000]; TW4 = [2000, 4000, 6000, 8000]; S = range(12)
eb = load("results_conv_elbow_blank", "blank"); econ = load("results_coach100k_elbow", "constant")
esw = {tw: load("results_sweep100k_elbow/w%d" % tw, "abrupt") for tw in TWS[:-1]}; esw[8000] = load("results_coach100k_elbow", "abrupt")
fn = load("results_sweep100k_finger_T2", "none"); fcon = load("results_sweep100k_finger_T2", "constant")
fsw = {tw: load("results_sweep100k_finger_T2/w%d" % tw, "abrupt") for tw in TWS[:-1]}; fsw[8000] = load("results_coach100k_finger_T2", "abrupt")
rc = load("results_replace100k_myoelbow", "constant"); rab = {tw: load("results_replace100k_myoelbow", "abrupt_w%d" % tw) for tw in TW4}
print("== A. abrupt - none at 100k, pooled over attachment durations (per-seed mean)")
print("  elbow additive  ", pt([np.mean([esw[tw][s][100000] - eb[s][100000] for tw in TWS]) for s in S]))
print("  elbow replacing ", pt([np.mean([rab[tw][s][100000] - eb[s][100000] for tw in TW4]) for s in S]))
print("  finger additive ", pt([np.mean([fsw[tw][s][100000] - fn[s][100000] for tw in TWS]) for s in S]))
print("  relative to the never-guided mean: elbow %.1f mrad, finger %.1f mm" % (np.mean([eb[s][100000] for s in S]), np.mean([fn[s][100000] for s in S])))
print("  for scale, the teacher's own advantage (constant - none) at 100k: elbow", pt([econ[s][100000] - eb[s][100000] for s in S]))
print("\n== B. finger-T2 contrasts without seed 1 (T2 is the prior run of seed 1)")
S11 = [s for s in S if s != 1]
c2 = load("results_conv_finger_coachT2", "coach"); pc2 = load("results_conv_finger_priorcoachT2", "priorcoach"); fb = load("results_conv_finger_blank", "blank"); fp = load("results_conv_finger_prior", "prior"); fc1 = load("results_conv_finger_coach", "coach"); frc = load("results_conv_finger_randcoach", "coach")
for name, a, b in (("coachT2-blank", c2, fb), ("coachT2-prior", c2, fp), ("coachT2-coachT1", c2, fc1), ("coachT2-randcoach", c2, frc), ("priorcoachT2-coachT2", pc2, c2)):
    print("  %-22s all %s | without seed 1 %s" % (name, pt([a[s][100000] - b[s][100000] for s in S]), pt([a[s][100000] - b[s][100000] for s in S11])))
fsa = {tw: load("results_sweep100k_finger_T2/w%d" % tw, "selfanchor") for tw in TWS[:-1]}; fsa[8000] = load("results_coach100k_finger_T2", "selfanchor")
for label, fn_ in (("teacher comp. slope 100k", lambda s, tw: fsa[tw][s][100000] - fcon[s][100000]), ("abrupt-none slope 100k", lambda s, tw: fsw[tw][s][100000] - fn[s][100000]),
                   ("loss-term dip slope", lambda s, tw: (fsw[tw][s][tw + 1000] - fsw[tw][s][tw]) - (fsa[tw][s][tw + 1000] - fsa[tw][s][tw])),
                   ("teacher dip slope", lambda s, tw: (fsa[tw][s][tw + 1000] - fsa[tw][s][tw]) - (fcon[s][tw + 1000] - fcon[s][tw]))):
    sl = lambda seeds: [np.polyfit(np.array(TWS) / 1000.0, [fn_(s, tw) for tw in TWS], 1)[0] for s in seeds]
    print("  %-24s all %s | without seed 1 %s" % (label, pt(sl(S)), pt(sl(S11))))
cnt = lambda seeds: sum((lambda v: np.mean(v) + TQ[len(v)] * np.std(v, ddof=1) / math.sqrt(len(v)) < 0)([fsw[tw][s][100000] - fn[s][100000] for s in seeds]) for tw in TWS)
print("  finger abrupt-none better at %d/7 (all) and %d/7 (without seed 1)" % (cnt(S), cnt(S11)))
