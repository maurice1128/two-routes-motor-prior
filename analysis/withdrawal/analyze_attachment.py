"""Does the cost of withdrawing a coach grow with how long it was attached?

Reads the curves written by run_attachment_sweep.py (abrupt / selfanchor at several
withdrawal steps t_w) plus the shared never-withdrawn `constant` arm and the `none` arm
from the single-point withdrawal run, and reports, at every t_w:

  dip             err(t_w + 1000) - err(t_w)            comparable across t_w
  excess@+k       err(t_w + k) - constant_err(t_w + k)  equal footing: k steps after withdrawal,
                                                        relative to the arm that never withdrew
  endpoint        err(12000)                             NOT comparable across t_w (fewer
                                                        post-withdrawal steps for late t_w)

and the three components, as paired-t 95% CIs over seeds:

  abrupt     - constant    total cost         (what the TMLR study reports)
  selfanchor - constant    guidance only      (should GROW with t_w if dependence accumulates)
  abrupt     - selfanchor  loss-term only     (should NOT grow: the loss term is the same size
                                               whenever it is deleted)

Finally a trend test: for each seed, the OLS slope of each component against t_w
(per 1000 steps of attachment), then a paired-t CI of that slope over seeds. Seeds are the
unit of analysis throughout, as in the TMLR study.

Usage:
  python analyze_attachment.py --dir results_attachment_myoelbow \
        --shared results_withdrawal_myoelbow
"""
import argparse
import json
import math
import os
from collections import defaultdict

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def t_crit(df):
    return _T95.get(df, _T95[min(_T95, key=lambda k: abs(k - df))])


def paired_ci(diffs):
    n = len(diffs)
    if n < 2:
        return (diffs[0] if n else float("nan"), float("nan"), float("nan"), n)
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    half = t_crit(n - 1) * math.sqrt(var / n)
    return m, m - half, m + half, n


def fmt_ci(diffs):
    m, lo, hi, n = paired_ci(diffs)
    sig = "*" if (lo > 0 or hi < 0) else " "
    return f"{m:+8.2f} [{lo:+7.2f},{hi:+7.2f}] n={n:<2}{sig}"


def ols_slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def load_curve(path):
    with open(path) as f:
        rows = json.load(f)
    return {int(r["step"]): float(r["eval_dist"]) * 1000.0 for r in rows}


def load_sweep(d):
    """-> curves[arm][t_w][seed] = {step: err}"""
    curves = defaultdict(lambda: defaultdict(dict))
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json") or fn == "meta.json":
            continue
        stem = fn[:-5]
        try:
            arm, w, s = stem.split("_")
            t_w, seed = int(w[1:]), int(s[4:])
        except ValueError:
            continue
        curves[arm][t_w][seed] = load_curve(os.path.join(d, fn))
    return curves


def load_shared(d, arm):
    out = {}
    if not d or not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        if fn.startswith(arm + "_seed") and fn.endswith(".json"):
            out[int(fn[len(arm) + 5:-5])] = load_curve(os.path.join(d, fn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_attachment_myoelbow")
    ap.add_argument("--shared", default="results_withdrawal_myoelbow",
                    help="dir holding constant_seed*.json and none_seed*.json")
    ap.add_argument("--k", type=int, default=4000,
                    help="steps after withdrawal for the equal-footing comparison")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    d = args.dir if os.path.isabs(args.dir) else os.path.join(here, args.dir)
    sd = args.shared if os.path.isabs(args.shared) else os.path.join(here, args.shared)
    meta_path = os.path.join(d, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    unit = "mrad" if meta.get("body", "myoelbow") == "myoelbow" else "mm"

    sweep = load_sweep(d)
    constant = load_shared(sd, "constant")
    none = load_shared(sd, "none")
    if not sweep:
        print(f"no curves in {d}")
        return
    tws = sorted({t for arm in sweep for t in sweep[arm]})
    arms = [a for a in ("abrupt", "selfanchor", "gateon") if a in sweep]
    K = args.k
    print(f"body={meta.get('body','?')}  unit={unit}  t_w={tws}  "
          f"constant n={len(constant)}  none n={len(none)}\n")

    # per (arm, t_w, seed) metrics
    M = defaultdict(dict)   # M[(arm,t_w)][seed] = dict(pre, post, dip, exk, end, recovery)
    for arm in arms:
        for t_w in tws:
            for seed, ser in sweep[arm].get(t_w, {}).items():
                if t_w not in ser or t_w + 1000 not in ser:
                    continue
                pre, post = ser[t_w], ser[t_w + 1000]
                rec = None
                for s in sorted(x for x in ser if x > t_w):
                    if ser[s] <= pre:
                        rec = s - t_w
                        break
                c = constant.get(seed)
                exk = (ser[t_w + K] - c[t_w + K]) if (c and t_w + K in ser) else float("nan")
                M[(arm, t_w)][seed] = dict(pre=pre, post=post, dip=post - pre, exk=exk,
                                          end=ser[max(ser)], rec=rec,
                                          errk=ser.get(t_w + K, float("nan")))
    # constant arm treated as "withdrawn at t_w" for dip/errk comparisons
    for t_w in tws:
        for seed, ser in constant.items():
            if t_w + 1000 in ser:
                M[("constant", t_w)][seed] = dict(pre=ser[t_w], post=ser[t_w + 1000],
                                                  dip=ser[t_w + 1000] - ser[t_w], exk=0.0,
                                                  end=ser[max(ser)], rec=None,
                                                  errk=ser.get(t_w + K, float("nan")))

    # ---- per-arm table --------------------------------------------------------
    print("=" * 96)
    print(f"PER ARM x t_w (mean over seeds).  excess@+{K} = err(t_w+{K}) - constant's err at the same step")
    print("=" * 96)
    print(f"{'arm':<11}{'t_w':>6}{'n':>4}{'pre':>9}{'post':>9}{'dip':>9}{f'err@+{K}':>11}"
          f"{f'excess@+{K}':>13}{'endpoint':>10}{'recovered':>11}")
    for arm in arms + (["constant"] if constant else []):
        for t_w in tws:
            rows = M.get((arm, t_w), {})
            if not rows:
                continue
            vals = list(rows.values())
            n = len(vals)
            mean = lambda k: sum(v[k] for v in vals) / n  # noqa: E731
            got = [v["rec"] for v in vals if v["rec"] is not None]
            print(f"{arm:<11}{t_w:>6}{n:>4}{mean('pre'):>9.2f}{mean('post'):>9.2f}"
                  f"{mean('dip'):>9.2f}{mean('errk'):>11.2f}{mean('exk'):>13.2f}"
                  f"{mean('end'):>10.2f}{len(got):>7}/{n:<3}")
        print()

    # ---- components at each t_w --------------------------------------------
    comps = [("abrupt", "constant", "total     (abrupt - constant)"),
             ("selfanchor", "constant", "guidance  (selfanchor - constant)"),
             ("abrupt", "selfanchor", "loss-term (abrupt - selfanchor)")]
    if "gateon" in sweep:   # replacing regime only
        comps += [("gateon", "constant", "own-signal (gateon - constant)"),
                  ("selfanchor", "gateon", "teacher-only (selfanchor - gateon)")]
    trend = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # [metric][comp][seed] = {t_w: diff}
    for metric, label in (("dip", "DIP = err(t_w+1000) - err(t_w)"),
                          ("errk", f"ERROR {K} STEPS AFTER WITHDRAWAL (equal footing across t_w)"),
                          ("end", "ENDPOINT err(12000)  (NOT equal footing: late t_w has fewer post steps)")):
        print("=" * 96)
        print(f"{label}  [{unit}]  paired-t 95% CI over seeds, * = excludes 0")
        print("=" * 96)
        print(f"{'component':<36}" + "".join(f"{'t_w=' + str(t):>30}" for t in tws))
        for a, b, name in comps:
            line = f"{name:<36}"
            for t_w in tws:
                A, B = M.get((a, t_w), {}), M.get((b, t_w), {})
                common = sorted(set(A) & set(B))
                diffs = [A[s][metric] - B[s][metric] for s in common
                         if A[s][metric] == A[s][metric] and B[s][metric] == B[s][metric]]
                for s, dd in zip([s for s in common if A[s][metric] == A[s][metric]
                                  and B[s][metric] == B[s][metric]], diffs):
                    trend[metric][name][s][t_w] = dd
                line += f"{fmt_ci(diffs) if len(diffs) >= 2 else 'n<2':>30}"
            print(line)
        print()

    # ---- trend test -----------------------------------------------------------
    print("=" * 96)
    print("TREND: per-seed OLS slope of each component vs t_w, in " + unit +
          " per 1000 attached steps; paired-t CI over seeds")
    print("  (guidance hypothesis: guidance slope > 0; shock slope ~ 0)")
    print("=" * 96)
    for metric in ("dip", "errk", "end"):
        print(f"-- {metric}")
        for _, _, name in comps:
            per_seed = trend[metric][name]
            slopes, firstlast = [], []
            for s, byw in per_seed.items():
                ws = sorted(byw)
                if len(ws) < 3:
                    continue
                slopes.append(ols_slope([w / 1000.0 for w in ws], [byw[w] for w in ws]))
                firstlast.append(byw[ws[-1]] - byw[ws[0]])
            if len(slopes) >= 2:
                print(f"  {name:<36} slope {fmt_ci(slopes)}   "
                      f"last-first {fmt_ci(firstlast)}")
            else:
                print(f"  {name:<36} (need >=3 t_w per seed)")
        print()

    # ---- constant vs none, as context ------------------------------------
    if constant and none:
        common = sorted(set(constant) & set(none))
        print("context: constant - none at 12000 (value of a maintained coach): " +
              fmt_ci([constant[s][12000] - none[s][12000] for s in common
                      if 12000 in constant[s] and 12000 in none[s]]))
        print("context: constant - none at each t_w (how much the coach was helping when withdrawn):")
        for t_w in tws:
            print(f"   t_w={t_w}: " + fmt_ci([constant[s][t_w] - none[s][t_w] for s in common
                                              if t_w in constant[s] and t_w in none[s]]))


if __name__ == "__main__":
    main()
