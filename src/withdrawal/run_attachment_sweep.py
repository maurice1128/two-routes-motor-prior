"""B2 main experiment: does the withdrawal cost grow with how long the coach was attached?

Why this is the experiment that makes it a paper
------------------------------------------------
The decomposition already run (run_withdrawal.py) shows WHAT the withdrawal cost is made
of at one withdrawal point. The guidance hypothesis (Salmoni, Schmidt & Walter 1984) makes
a sharper prediction than that: the longer and stronger the aid is attached, the more the
learner comes to depend on it, so the cost of removing it should GROW with attachment
duration. That monotone relationship is the actual content of the hypothesis, it has never
been tested on a robot, and a single withdrawal point cannot show it.

So this sweeps the withdrawal step and asks whether

    dip(t_w)   and   endpoint_cost(t_w)

increase with t_w. Because the decomposition is run at every t_w, the sweep also answers a
second question the literature cannot: does the OPTIMIZATION-SHOCK component grow with
attachment (it should not, the loss term is the same size whenever you delete it) while the
GENUINE-GUIDANCE component does (it should, if dependence accumulates)? If those two
components separate along the attachment axis, that is a clean mechanistic result.

Arms per withdrawal point: abrupt (delete the loss term) and selfanchor (swap the teacher
for a frozen copy of the student, keeping the loss term). `constant` is shared across all
withdrawal points because it never withdraws, so it is run once.

A caution on the endpoint. Later withdrawal leaves fewer post-withdrawal steps before the
12k budget ends, so endpoint costs at different t_w are not on equal footing. The dip is
comparable across t_w; the endpoint is not, unless you also report steps-since-withdrawal.
Both are recorded so the write-up can say which is which.

Usage
-----
  python run_attachment_sweep.py --body myoelbow --seeds 12
  python run_attachment_sweep.py --body myoelbow --seeds 12 --withdraw-at 4000 --arms abrupt
"""
import argparse
import json
import os
import sys
import time

_BODY = None
for i, tok in enumerate(sys.argv):
    if tok == "--body" and i + 1 < len(sys.argv):
        _BODY = sys.argv[i + 1]
    elif tok.startswith("--body="):
        _BODY = tok.split("=", 1)[1]
os.environ["WM_BODY"] = _BODY or os.environ.get("WM_BODY", "myoelbow")

import anchor_sac  # noqa: E402  (chdirs into wm_prior)
import run_reach  # noqa: E402

OUT_ROOT = os.path.dirname(os.path.abspath(__file__))
BODIES = {
    "myoelbow": {"teacher": "teacher_elbow_clean30k.pt"},
    "myofinger": {"teacher": "teacher_finger_clean30k.pt"},
}
DEFAULT_WITHDRAW = [2000, 4000, 6000, 8000]
ARMS = {
    "abrupt":     dict(abrupt=True,  anchor="none"),
    "selfanchor": dict(abrupt=False, anchor="self"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="myoelbow", choices=sorted(BODIES))
    ap.add_argument("--withdraw-at", type=str,
                    default=",".join(str(x) for x in DEFAULT_WITHDRAW))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed-list", type=str, default=None,
                    help="comma-separated seeds to run (subset of range(seeds)); lets one sweep be split across processes")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--utd", type=int, default=2)
    ap.add_argument("--ntargets", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seed_list = ([int(x) for x in args.seed_list.split(",")] if args.seed_list
                 else list(range(args.seeds)))

    body = BODIES[args.body]
    if not os.path.exists(body["teacher"]):
        raise SystemExit(f"teacher not found: {body['teacher']} (cwd {os.getcwd()})")
    withdraws = [int(x) for x in args.withdraw_at.split(",")]
    arms = [a.strip() for a in args.arms.split(",")]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}")

    out_dir = args.out or os.path.join(OUT_ROOT, f"results_attachment_{args.body}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"body": args.body, "withdraw_at": withdraws, "arms": arms,
                   "seeds": args.seeds, "steps": args.steps,
                   "eval_every": args.eval_every, "hidden": args.hidden,
                   "utd": args.utd, "teacher": body["teacher"],
                   "eval_heldout": True}, f, indent=2)

    original_sac = run_reach.SAC
    total = len(withdraws) * len(arms) * len(seed_list)
    n = 0
    t_start = time.time()
    for t_w in withdraws:
        for arm in arms:
            cfg = ARMS[arm]
            for seed in seed_list:
                path = os.path.join(out_dir, f"{arm}_w{t_w}_seed{seed}.json")
                if os.path.exists(path):
                    n += 1
                    continue
                run_reach.SAC = (
                    anchor_sac.make_anchor_sac(t_w, cfg["anchor"], args.hidden)
                    if cfg["anchor"] != "none" else original_sac
                )
                # abrupt cuts the weight at t_w; selfanchor keeps a constant weight
                # (anneal far beyond the budget) and swaps the teacher at t_w instead.
                anneal = t_w if cfg["abrupt"] else 10 ** 8
                t0 = time.time()
                print(f"[{arm} w{t_w} seed{seed}] starting ({n + 1}/{total})", flush=True)
                curve = run_reach.run(
                    condition="coach", seed=seed, total_steps=args.steps,
                    hidden=args.hidden, utd=args.utd, n_train_targets=args.ntargets,
                    eval_every=args.eval_every, eval_heldout=True,
                    coach_path=body["teacher"], coach0=1.0,
                    coach_anneal=anneal, coach_abrupt=cfg["abrupt"],
                )
                with open(path, "w") as f:
                    json.dump(curve, f)
                n += 1
                print(f"[{arm} w{t_w} seed{seed}] done in {time.time() - t0:.0f}s "
                      f"({n}/{total}, {(time.time() - t_start) / 60:.1f} min)", flush=True)

    run_reach.SAC = original_sac
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
