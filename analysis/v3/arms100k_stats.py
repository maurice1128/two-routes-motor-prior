# -*- coding: utf-8 -*-
"""Joint-count series: planar muscle arms with 1..4 joints, 100k steps, held-out.
Uses the seeds present in ALL four arms of a body (so a partial batch is analysed honestly).
Question: does the babble body model's advantage over model-free learning persist to
convergence, and does that depend on the number of joints?"""
import json, os, math, glob, io
import numpy as np

W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
# two-sided 95% t quantiles, keyed by n (df = n-1)
TQ = {10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110,
      19: 2.101, 20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069}
out = []


def P(s=""):
    print(s); out.append(s)


def load(d, c):
    r = {}
    for f in glob.glob(os.path.join(W, d, "%s_seed*.json" % c)):
        s = int(f.split("_seed")[1].split(".")[0])
        r[s] = {int(x["step"]): x["eval_dist"] * 1000 for x in json.load(open(f))}
    return r


def pt(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n)
    t = TQ[n]; return m, m - t * se, m + t * se


def fmt(t3):
    return "%+8.2f [%+8.2f, %+8.2f]%s" % (t3[0], t3[1], t3[2], "*" if (t3[1] > 0 or t3[2] < 0) else " ")


CK = list(range(2000, 100001, 2000))
summary = {}
for b, nj in (("arm1", 1), ("arm2", 2), ("arm3", 3), ("arm4", 4)):
    A = {"blank": load("results_arms100k_%s_blank" % b, "blank"), "prior": load("results_arms100k_%s_prior" % b, "prior"),
         "randprior": load("results_arms100k_%s_randprior" % b, "prior"), "blank64": load("results_arms100k_%s_blank64" % b, "blank64")}
    seeds = sorted(set.intersection(*[set(v) for v in A.values()]))
    P("== %s (%d joint%s), n=%d seeds %s" % (b, nj, "" if nj == 1 else "s", len(seeds), seeds if seeds != list(range(len(seeds))) else "0-%d" % (len(seeds) - 1)))
    P("   means mm  12k / 50k / 100k:  " + "  ".join("%s %.1f/%.1f/%.1f" % (k, *[np.mean([A[k][s][st] for s in seeds]) for st in (12000, 50000, 100000)]) for k in A))
    for st in (12000, 50000, 100000):
        P("   %3dk prior-blank      %s   | %% of blank %+.0f%%" % (st // 1000, fmt(pt([A["prior"][s][st] - A["blank"][s][st] for s in seeds])),
          100 * np.mean([A["prior"][s][st] - A["blank"][s][st] for s in seeds]) / np.mean([A["blank"][s][st] for s in seeds])))
    P("   100k prior-blank64    %s" % fmt(pt([A["prior"][s][100000] - A["blank64"][s][100000] for s in seeds])))
    P("   100k prior-randprior  %s" % fmt(pt([A["prior"][s][100000] - A["randprior"][s][100000] for s in seeds])))
    P("   100k randprior-blank  %s" % fmt(pt([A["randprior"][s][100000] - A["blank"][s][100000] for s in seeds])))
    P("   100k blank64-blank    %s" % fmt(pt([A["blank64"][s][100000] - A["blank"][s][100000] for s in seeds])))
    sig = [st for st in CK if pt([A["prior"][s][st] - A["blank"][s][st] for s in seeds])[2] < 0]
    P("   prior-blank significant (uncorrected) at %d/50 checkpoints%s" % (len(sig), (", last %dk" % (sig[-1] // 1000)) if sig else ""))
    for k in ("blank", "prior"):
        P("   plateau %-6s 100k-80k %s" % (k, fmt(pt([A[k][s][100000] - A[k][s][80000] for s in seeds]))))
    summary[nj] = (np.mean([A["prior"][s][100000] - A["blank"][s][100000] for s in seeds]), np.mean([A["blank"][s][100000] for s in seeds]))
    P()
P("relative advantage of the body model at 100k by joint count: " + "  ".join("%dj %+.0f%%" % (nj, 100 * d / bl) for nj, (d, bl) in summary.items()))
io.open(os.path.join(W, "tcds_v3", "arms100k_stats.txt"), "w", encoding="utf-8").write("\n".join(out))
