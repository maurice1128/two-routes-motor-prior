"""Budget-matched comparison of the two routes (B1).

Reads the curves written by run_budget_matched.py and reports the endpoint error per
arm, and the paired-t 95% CIs over seeds for the contrasts that carry the argument:

  prior_matched - prior_full   does cutting the prior's babble budget to the coach's
                               30k cost anything?          (the new number)
  prior_full    - coach        the TMLR comparison           (reproduction check)
  prior_matched - coach        the two routes at equal transition count
  prior_matched - blank        is a 30k prior still worth having?
  coach         - blank        value of the coach

Conventions follow the TMLR study: seed is the unit, arms are paired within seed,
distances are scaled by 1000 (mrad on myoElbow, mm on myoFinger).

Usage:
  python analyze_budget.py --dir results_budget_myoelbow
  python analyze_budget.py --dir results_budget_myofinger
"""
import argparse
import json
import math
import os
from collections import defaultdict

ARM_ORDER = ["blank", "coach", "prior_matched", "prior_full"]
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def t_crit(df):
    return _T95.get(df, _T95[min(_T95, key=lambda k: abs(k - df))])


def paired_ci(diffs):
    n = len(diffs)
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    half = t_crit(n - 1) * math.sqrt(var / n)
    return m, m - half, m + half, n


def load_dir(d):
    curves = defaultdict(dict)
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json") or fn == "meta.json":
            continue
        arm, _, rest = fn[:-5].partition("_seed")
        if not rest.isdigit():
            continue
        with open(os.path.join(d, fn)) as f:
            rows = json.load(f)
        curves[arm][int(rest)] = {int(r["step"]): float(r["eval_dist"]) * 1000.0
                                  for r in rows}
    return curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_budget_myoelbow")
    args = ap.parse_args()
    d = args.dir if os.path.isabs(args.dir) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.dir)
    meta_path = os.path.join(d, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    unit = "mrad" if meta.get("body") == "myoelbow" else "mm"
    curves = load_dir(d)
    if not curves:
        print(f"no curves in {d}")
        return
    arms = [a for a in ARM_ORDER if a in curves] + [a for a in sorted(curves) if a not in ARM_ORDER]
    steps = sorted({s for a in arms for c in curves[a].values() for s in c})
    print(f"body={meta.get('body','?')}  unit={unit}  seeds per arm: "
          + ", ".join(f"{a}={len(curves[a])}" for a in arms) + "\n")

    print("=" * 78)
    print(f"ENDPOINT (step {steps[-1]}) per arm, mean and sd over seeds; plus mean curve")
    print("=" * 78)
    print(f"{'arm':<15}{'n':>4}{'endpoint':>11}{'sd':>9}   " + "".join(f"{s:>8}" for s in steps))
    end = {}
    for a in arms:
        vals = [c[max(c)] for c in curves[a].values()]
        n = len(vals)
        m = sum(vals) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else float("nan")
        end[a] = {s: c[max(c)] for s, c in curves[a].items()}
        mean_curve = []
        for st in steps:
            xs = [c[st] for c in curves[a].values() if st in c]
            mean_curve.append(sum(xs) / len(xs) if xs else float("nan"))
        print(f"{a:<15}{n:>4}{m:>11.2f}{sd:>9.2f}   " + "".join(f"{v:>8.1f}" for v in mean_curve))

    contrasts = [
        ("prior_matched", "prior_full", "cost of cutting the prior's babble to the coach's budget"),
        ("prior_full", "coach", "the TMLR comparison (reproduction check)"),
        ("prior_matched", "coach", "the two routes at equal transition count"),
        ("prior_matched", "blank", "is a budget-matched prior still worth having?"),
        ("coach", "blank", "value of the coach"),
        ("prior_full", "blank", "value of the full prior"),
    ]
    print("\n" + "=" * 78)
    print(f"PAIRED CONTRASTS on the endpoint ({unit}); negative favours the first arm")
    print("=" * 78)
    for a, b, why in contrasts:
        if a not in end or b not in end:
            continue
        common = sorted(set(end[a]) & set(end[b]))
        if len(common) < 2:
            print(f"  {a} - {b}: n={len(common)}")
            continue
        m, lo, hi, n = paired_ci([end[a][s] - end[b][s] for s in common])
        sig = "significant" if (lo > 0 or hi < 0) else "n.s."
        print(f"  {a:<14} - {b:<14} {m:+9.2f}  [{lo:+8.2f}, {hi:+8.2f}]  n={n:<3} {sig}")
        print(f"      {why}")


if __name__ == "__main__":
    main()
