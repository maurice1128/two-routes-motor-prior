"""randcoach - blank, the adversarial control for the guidance route.

Windows and estimator are the paper's, quoted from RECOMPUTED.md line 9:
ENDPOINT = index 5 (12k); EARLY = mean of indices 0,1,2 (2k/4k/6k);
MID = 2,3,4; LATE = 4,5.  Primary statistic is the paired t 95% CI at n=12,
df=11.  Elbow in mrad, finger in mm; the two bodies are never compared.
"""
import json, os, glob
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
W = {"EARLY": [0, 1, 2], "MID": [2, 3, 4], "LATE": [4, 5], "ENDPOINT": [5]}


def curve(path):
    d = json.load(open(path))
    return np.array([r["eval_dist"] for r in d]) * 1000.0     # metres -> mrad / mm


def series(d, stem):
    out = {}
    for f in glob.glob(os.path.join(ROOT, d, "%s_seed*.json" % stem)):
        s = int(os.path.basename(f).split("seed")[1].split(".")[0])
        out[s] = curve(f)
    return out


def paired(a, b, idx):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s][idx].mean() - b[s][idx].mean() for s in seeds])
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    p = 2 * (1 - stats.t.cdf(abs(m / se), n - 1))
    return n, m, m - t * se, m + t * se, p


for body, unit, rc, bl, co in [
        ("elbow", "mrad", "results_randcoach_elbow", "results_heldout_elbow", "results_noanneal_elbow"),
        ("finger", "mm", "results_randcoach_finger", "results_heldout_finger", "results_noanneal_finger")]:
    R, B, C = series(rc, "coach"), series(bl, "blank"), series(co, "coach")
    print("\n=== myo%s (%s), held-out ===" % (body.capitalize(), unit))
    print("  n: randcoach %d | blank %d | coach(const) %d" % (len(R), len(B), len(C)))
    for w, idx in W.items():
        for tag, X, Y in (("randcoach - blank", R, B),
                          ("coach(const) - blank", C, B),
                          ("randcoach - coach(const)", R, C)):
            if not (X and Y):
                continue
            n, m, lo, hi, p = paired(X, Y, idx)
            star = "  *" if (lo > 0) == (hi > 0) else ""
            print("  %-9s %-26s n=%2d  %+8.2f  [%+8.2f, %+8.2f]  p=%.4f%s"
                  % (w, tag, n, m, lo, hi, p, star))
