# -*- coding: utf-8 -*-
"""Analysis for the two experiments this paper adds: matched withdrawal, convergence.

Protocol follows the original study exactly, because these numbers have to sit
beside its tables: the seed is the unit of analysis, every comparison between two
arms is a direct paired contrast on the seed rather than an inference from two
separate verdicts against a third arm, and intervals are paired t 95% intervals.

Both bodies store eval_dist in the environment's own unit -- radians on myoElbow,
metres on myoFinger -- so both scale by 1000, to milliradians and millimetres
respectively. The two are never compared in magnitude.

Reports partial grids rather than refusing to run, because the convergence grid
takes about a day and a half-filled grid is still worth looking at; every table
prints the n it used.
"""
import json, math, os, sys, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}


def load(dirname, cond):
    """seed -> {step: error in mrad or mm}, for one arm."""
    out = {}
    d = os.path.join(ROOT, dirname)
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        if not (fn.startswith(cond + "_seed") and fn.endswith(".json")):
            continue
        seed = int(fn[len(cond) + 5:-5])
        try:
            rows = json.load(open(os.path.join(d, fn)))
        except (ValueError, OSError):
            continue                      # a run still writing its file
        out[seed] = {int(r["step"]): r["eval_dist"] * 1000.0 for r in rows}
    return out


def paired(a, b, key):
    """Paired contrast a-b over the seeds both arms have. None if too few."""
    seeds = sorted(set(a) & set(b))
    d = []
    for s in seeds:
        try:
            d.append(key(a[s]) - key(b[s]))
        except KeyError:
            pass                          # checkpoint not reached yet
    n = len(d)
    if n < 2 or (n - 1) not in T95:
        return None
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    t = T95[n - 1]
    return n, m, m - t * se, m + t * se


def show(label, r, unit):
    if r is None:
        print("  %-34s (not enough seeds yet)" % label)
        return
    n, m, lo, hi = r
    sig = "sig " if (lo > 0 or hi < 0) else "n.s."
    print("  %-34s n=%2d  %+8.2f %s  [%+8.2f, %+8.2f]  %s"
          % (label, n, m, unit, lo, hi, sig))


def matched_withdrawal():
    """Does the strict withdrawal cost survive holding real-data content fixed?"""
    print("\n" + "=" * 78)
    print("MATCHED PRIOR WITHDRAWAL  (withdraw at 8000, endpoint 12000)")
    print("=" * 78)
    for tag, unit in (("elbow", "mrad"), ("finger", "mm")):
        arms = {a: load("results_matched_%s" % tag, a)
                for a in ("keep", "soft", "purge", "matched")}
        have = {a: len(v) for a, v in arms.items()}
        print("\n%s   seeds present: %s" % (tag, have))
        if not any(have.values()):
            continue
        for a, v in arms.items():
            done = [c for c in v.values() if 12000 in c]
            if done:
                print("    %-8s endpoint mean %7.2f %s  (n=%d)"
                      % (a, sum(c[12000] for c in done) / len(done), unit, len(done)))
        end = lambda c: c[12000]
        post = lambda c: c[10000] - c[8000]     # eval cadence is every 2000 steps
        # Real-data content per update is 64 distinct in keep, soft and matched and
        # 128 in purge, so only contrasts among the first three are single-factor.
        pairs = [("purge", "keep", "as the paper measured it"),
                 ("soft", "keep", "as the paper measured it"),
                 ("matched", "keep", "real-data content held"),
                 ("soft", "matched", "worth of stale imagination"),
                 ("keep", "soft", "worth of fresh over stale"),
                 ("purge", "matched", "the real-data doubling")]
        for label, key in (("endpoint:", end),
                           ("post-withdrawal change (8000 -> 10000):", post)):
            print("  " + label)
            for a, b, note in pairs:
                show("%s - %s  (%s)" % (a, b, note),
                     paired(arms[a], arms[b], key), unit)


ARMS = [("blank", "blank"), ("prior", "prior"), ("randprior", "prior"),
        ("coach", "coach"), ("randcoach", "coach"), ("priorcoach", "priorcoach")]


def convergence(endpoint):
    """Do the orderings at 12k survive to a converged budget?"""
    print("\n" + "=" * 78)
    print("CONVERGENCE GRID  (endpoint %d, held-out)" % endpoint)
    print("=" * 78)
    for tag, unit in (("elbow", "mrad"), ("finger", "mm")):
        arms = {label: load("results_conv_%s_%s" % (tag, label), cond)
                for label, cond in ARMS}
        print("\n%s   seeds present: %s"
              % (tag, {k: len(v) for k, v in arms.items()}))
        for label, _ in ARMS:
            done = [c for c in arms[label].values() if endpoint in c]
            if not done:
                continue
            m12 = [c[12000] for c in arms[label].values() if 12000 in c]
            print("    %-11s %s %7.2f %s (n=%d)   at 12k %7.2f (n=%d)"
                  % (label, "endpoint", sum(done and [c[endpoint] for c in done]) / len(done),
                     unit, len(done),
                     (sum(m12) / len(m12)) if m12 else float("nan"), len(m12)))
        end = lambda c: c[endpoint]
        print("  the two routes, and each against model-free:")
        for a, b in [("coach", "prior"), ("prior", "blank"), ("coach", "blank"),
                     ("priorcoach", "blank"), ("prior", "randprior"),
                     ("coach", "randcoach")]:
            show("%s - %s" % (a, b), paired(arms[a], arms[b], end), unit)
        # Whether the budget was long enough is itself a measurement: if the last
        # fifth of training still moves the arm, the endpoint is not an asymptote.
        print("  plateau check (endpoint minus the checkpoint at 80%% of budget):")
        near = int(endpoint * 0.8)
        for label, _ in ARMS:
            v = arms[label]
            xs = [c[endpoint] - c[near] for c in v.values()
                  if endpoint in c and near in c]
            if len(xs) >= 2:
                m = sum(xs) / len(xs)
                sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
                print("    %-11s %+7.2f %s  (sd %5.2f, n=%d)"
                      % (label, m, unit, sd, len(xs)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", type=int, default=100000)
    ap.add_argument("--only", choices=["matched", "convergence"])
    a = ap.parse_args()
    if a.only != "convergence":
        matched_withdrawal()
    if a.only != "matched":
        convergence(a.endpoint)
