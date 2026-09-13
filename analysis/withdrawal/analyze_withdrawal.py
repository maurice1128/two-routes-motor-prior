"""Decompose the cost of withdrawing a coach.

Reads the curves written by run_withdrawal.py and reports, per arm:
  - the endpoint error at the final checkpoint,
  - the DIP: error immediately after the withdrawal step minus error immediately before,
  - RECOVERY: the first checkpoint after withdrawal at which the arm is back to its own
    pre-withdrawal level, reported as steps, or as censored if it never gets back.

Then the four contrasts that carry the argument, as paired-t 95% CIs over seeds:

  abrupt     - constant     losing the teacher AND the loss term   (the TMLR number)
  selfanchor - constant     losing the teacher only
  abrupt     - selfanchor   losing the loss term only              (the confound)
  randanchor - selfanchor   pulled somewhere arbitrary vs pulled where you already are

Conventions follow the TMLR study so the numbers are comparable: seed is the unit of
analysis, arms are paired within seed, distances are scaled by 1000 (mrad on myoElbow,
mm on myoFinger), and an interval that excludes zero is called significant.

Usage:
  python analyze_withdrawal.py --dir results_withdrawal_myoelbow
"""
import argparse
import json
import math
import os
from collections import defaultdict

ARM_ORDER = ["none", "constant", "abrupt", "selfanchor", "randanchor"]
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def t_crit(df):
    if df in _T95:
        return _T95[df]
    return _T95[min(_T95, key=lambda k: abs(k - df))]


def paired_ci(diffs):
    n = len(diffs)
    if n < 2:
        return (diffs[0] if n else float("nan"), float("nan"), float("nan"), n)
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    half = t_crit(n - 1) * math.sqrt(var / n)
    return m, m - half, m + half, n


def load_dir(d):
    """-> curves[arm][seed] = {step: dist_scaled}"""
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


def dip_and_recovery(series, withdraw_at):
    """(pre, post, dip, recovery_steps) for one seed's curve. recovery None if censored."""
    steps = sorted(series)
    pre_steps = [s for s in steps if s <= withdraw_at]
    post_steps = [s for s in steps if s > withdraw_at]
    if not pre_steps or not post_steps:
        return (float("nan"),) * 3 + (None,)
    pre = series[pre_steps[-1]]
    post = series[post_steps[0]]
    recovery = None
    for s in post_steps:
        if series[s] <= pre:
            recovery = s - withdraw_at
            break
    return pre, post, post - pre, recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_withdrawal_myoelbow")
    ap.add_argument("--withdraw-at", type=int, default=None)
    args = ap.parse_args()

    d = args.dir if os.path.isabs(args.dir) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.dir)
    meta_path = os.path.join(d, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    withdraw_at = args.withdraw_at or meta.get("withdraw_at", 8000)
    unit = "mrad" if meta.get("body") == "myoelbow" else "mm"

    curves = load_dir(d)
    if not curves:
        print(f"no curves in {d}")
        return
    arms = [a for a in ARM_ORDER if a in curves] + \
           [a for a in sorted(curves) if a not in ARM_ORDER]
    print(f"body={meta.get('body','?')}  withdrawal at {withdraw_at}  unit={unit}\n")

    # ---- per-arm summary ----------------------------------------------------
    print("=" * 78)
    print("PER-ARM SUMMARY (mean over seeds)")
    print("=" * 78)
    print(f"{'arm':<14}{'n':>4}{'endpoint':>11}{'pre':>10}{'post':>10}{'dip':>10}"
          f"{'recovered':>11}{'median rec':>12}")
    stats = {}
    for arm in arms:
        seeds = sorted(curves[arm])
        endp, pres, posts, dips, recs = [], [], [], [], []
        for s in seeds:
            ser = curves[arm][s]
            endp.append(ser[max(ser)])
            pre, post, dip, rec = dip_and_recovery(ser, withdraw_at)
            pres.append(pre); posts.append(post); dips.append(dip)
            recs.append(rec)
        got = [r for r in recs if r is not None]
        med = sorted(got)[len(got) // 2] if got else float("nan")
        stats[arm] = {"seeds": seeds, "endpoint": endp, "dip": dips, "rec": recs}
        print(f"{arm:<14}{len(seeds):>4}{sum(endp)/len(endp):>11.2f}"
              f"{sum(pres)/len(pres):>10.2f}{sum(posts)/len(posts):>10.2f}"
              f"{sum(dips)/len(dips):>10.2f}{len(got):>7}/{len(recs):<3}"
              f"{med:>12.0f}")

    # ---- contrasts ----------------------------------------------------------
    contrasts = [
        ("abrupt", "constant",
         "losing the teacher AND the loss term (reproduces the TMLR withdrawal cost)"),
        ("selfanchor", "constant",
         "losing the teacher ONLY (loss term kept, pointing at the student itself)"),
        ("abrupt", "selfanchor",
         "losing the LOSS TERM only -- the confound the TMLR paper could not separate"),
        ("randanchor", "selfanchor",
         "anchored to noise vs anchored to self, both with the loss term present"),
        ("constant", "none",
         "value of a maintained coach at the endpoint"),
    ]
    for metric in ("endpoint", "dip"):
        print("\n" + "=" * 78)
        print(f"PAIRED CONTRASTS on {metric.upper()} ({unit}); negative favours the first arm")
        print("=" * 78)
        for a, b, why in contrasts:
            if a not in stats or b not in stats:
                continue
            common = sorted(set(stats[a]["seeds"]) & set(stats[b]["seeds"]))
            if len(common) < 2:
                print(f"  {a} - {b}: n={len(common)}, need >=2 seeds")
                continue
            ia = {s: v for s, v in zip(stats[a]["seeds"], stats[a][metric])}
            ib = {s: v for s, v in zip(stats[b]["seeds"], stats[b][metric])}
            diffs = [ia[s] - ib[s] for s in common if ia[s] == ia[s] and ib[s] == ib[s]]
            if len(diffs) < 2:
                print(f"  {a} - {b}: too few finite pairs")
                continue
            m, lo, hi, n = paired_ci(diffs)
            sig = "significant" if (lo > 0 or hi < 0) else "n.s."
            print(f"  {a:<12} - {b:<12} {m:+9.2f}  [{lo:+8.2f}, {hi:+8.2f}]  n={n:<3} {sig}")
            print(f"      {why}")

    # ---- the reading --------------------------------------------------------
    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("If (abrupt - selfanchor) is large and significant, then most of what the TMLR")
    print("study measured as the cost of withdrawing a coach is the optimization shock of")
    print("deleting a weight-1 loss term, not the loss of guidance. In that case the")
    print("humanoid study must withdraw by swapping the teacher, not by zeroing the loss,")
    print("and the retention claim has to be stated against the selfanchor arm.")
    print("If it is near zero, simple withdrawal is clean and can be used as-is.")


if __name__ == "__main__":
    main()
