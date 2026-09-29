# -*- coding: utf-8 -*-
"""Stage-B extras: finger-T2 randanchor dip at t_w+1000, coachT2 vs randcoach, withdrawn body-model arms vs blank,
Wilcoxon for the elbow abrupt-none dip at 8k, and 12k contrasts kept in the prose."""
import json, os, math, itertools
import numpy as np
W = "C:/Users/maurice/Desktop/robotic_research/wm_prior"
def load(d, c):
    return {s: {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(os.path.join(W, d, "%s_seed%d.json" % (c, s))))} for s in range(12)}
def wilcoxon(v):
    v = np.asarray(v, float); v = v[v != 0]; n = len(v); r = np.argsort(np.argsort(np.abs(v))) + 1.0
    wp = r[v > 0].sum(); cnt = tot = 0
    for m in range(1 << n):
        s = sum(r[i] for i in range(n) if m >> i & 1); tot += 1; cnt += (s <= wp)
    p1 = cnt / tot; return 2 * min(p1, 1 - p1)
def pt(v, wil=True):
    v = np.asarray(v, float); m = v.mean(); se = v.std(ddof=1) / math.sqrt(len(v)); lo, hi = m - 2.201 * se, m + 2.201 * se
    return "%+7.2f [%+7.2f, %+7.2f]%s%s" % (m, lo, hi, "*" if (lo > 0 or hi < 0) else " ", ("  W p=%.4f" % wilcoxon(v)) if wil else "")
S = range(12)
# finger T2 decomposition arms at t_w = 8000 (1k grid) and constant (2k grid), none (blank, 2k grid) and the 1k-grid none from the sweep
ab = load("results_coach100k_finger_T2", "abrupt"); sa = load("results_coach100k_finger_T2", "selfanchor"); ra = load("results_coach100k_finger_T2", "randanchor")
con = load("results_sweep100k_finger_T2", "constant"); none1k = load("results_sweep100k_finger_T2", "none"); blank = load("results_conv_finger_blank", "blank")
rc = load("results_conv_finger_randcoach", "coach"); c2 = load("results_conv_finger_coachT2", "coach"); pr = load("results_conv_finger_prior", "prior")
print("== finger T2, t_w = 8000, dip = error at 9000 minus error at 8000")
d = lambda a: [a[s][9000] - a[s][8000] for s in S]
print("  abrupt - constant    ", pt(np.subtract(d(ab), d(con))))
print("  selfanchor - constant", pt(np.subtract(d(sa), d(con))))
print("  abrupt - selfanchor  ", pt(np.subtract(d(ab), d(sa))))
print("  randanchor - selfanchor", pt(np.subtract(d(ra), d(sa))))
print("  abrupt - none        ", pt(np.subtract(d(ab), d(none1k))))
print("  none(1k grid) vs blank(2k grid) identical at 2k..100k:", sum(abs(none1k[s][st] - blank[s][st]) > 0.005 for s in S for st in range(2000, 100001, 2000)), "cells differ")
print("== finger coachT2 vs randcoach")
for st in (12000, 100000):
    print("  %3dk coachT2 - randcoach" % (st // 1000), pt([c2[s][st] - rc[s][st] for s in S]))
print("  100k randanchor(T2) - blank", pt([ra[s][100000] - blank[s][100000] for s in S]))
print("  100k randanchor(T2) - randcoach", pt([ra[s][100000] - rc[s][100000] for s in S]))
print("== elbow: withdrawn body-model arms vs never-modelled blank at 100k")
eb = load("results_conv_elbow_blank", "blank")
for arm in ("soft", "purge", "matched"):
    a = load("results_matched100k_elbow", arm); print("  %-8s - blank 100k" % arm, pt([a[s][100000] - eb[s][100000] for s in S]))
ek = load("results_conv_elbow_prior", "prior"); print("  keep     - blank 100k", pt([ek[s][100000] - eb[s][100000] for s in S]))
print("== elbow abrupt - none dip at t_w = 8000 (from the 12k sweep runs, read only)")
Z = "C:/Users/maurice/Desktop/world_model_zeroshot/b2_pilot"
za = load(Z + "/results_attachment_myoelbow", "abrupt_w8000"); zn = load(Z + "/results_withdrawal_myoelbow", "none")
print("  abrupt - none dip 8k", pt(np.subtract([za[s][9000] - za[s][8000] for s in S], [zn[s][9000] - zn[s][8000] for s in S])))
print("== 12k contrasts kept in prose (elbow / finger prior-blank, coach-blank)")
for body, unit in (("elbow", "mrad"), ("finger", "mm")):
    b = load("results_conv_%s_blank" % body, "blank"); p = load("results_conv_%s_prior" % body, "prior"); c = load("results_conv_%s_coach" % body, "coach")
    print("  %s 12k prior-blank" % body, pt([p[s][12000] - b[s][12000] for s in S], False), "| coach-blank", pt([c[s][12000] - b[s][12000] for s in S], False))
