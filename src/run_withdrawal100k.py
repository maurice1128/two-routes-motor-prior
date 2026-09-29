# -*- coding: utf-8 -*-
"""Teacher-withdrawal arms at the converged budget with a chosen teacher.

Thin wrapper over the second draft's anchor_sac.py (read-only, imported from
world_model_zeroshot/b2_pilot) so that the finger decomposition can be re-run
with a competent teacher and to 100,000 steps, one cell per process:

  python run_withdrawal100k.py --body myofinger --arm abrupt --seed 0 --steps 100000 \
      --withdraw-at 8000 --teacher teacher_finger_prior100k.pt --out results_coach100k_finger_T2

Arms: constant, abrupt, selfanchor, randanchor (none = the blank arm of the
converged grid). The swap fires on loop step t_w-2 exactly as in the 12k study
(anchor_sac counts the two pre-loop coach_coef assignments), so the 12k
checkpoints of these runs reproduce that study's arms when the same teacher is
used.
"""
import argparse, json, os, sys, time
_BODY = None
for i, tok in enumerate(sys.argv):
    if tok == "--body" and i + 1 < len(sys.argv):
        _BODY = sys.argv[i + 1]
os.environ["WM_BODY"] = _BODY or os.environ.get("WM_BODY", "myofinger")
sys.path.insert(0, r"C:\Users\maurice\Desktop\world_model_zeroshot\b2_pilot")
import anchor_sac  # noqa: E402  (chdirs into wm_prior)
import run_reach   # noqa: E402

ARMS = {
    "constant":   dict(anneal=10 ** 8, abrupt=False, anchor="none"),
    "abrupt":     dict(anneal=None,    abrupt=True,  anchor="none"),
    "selfanchor": dict(anneal=10 ** 8, abrupt=False, anchor="self"),
    "randanchor": dict(anneal=10 ** 8, abrupt=False, anchor="random"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, choices=["myoelbow", "myofinger"])
    ap.add_argument("--arm", required=True, choices=sorted(ARMS) + ["none"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--withdraw-at", type=int, default=8000)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-every", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "%s_seed%d.json" % (args.arm, args.seed))
    if os.path.exists(dest):
        print("SKIP", dest); return 0
    if args.arm == "none":
        # never-guided learner on the same evaluation grid as the withdrawn arms
        t0 = time.time()
        curve = run_reach.run(condition="blank", seed=args.seed, total_steps=args.steps, hidden=128,
                              utd=2, n_train_targets=8, eval_every=args.eval_every, eval_heldout=True)
        with open(dest, "w") as f:
            json.dump(curve, f)
        print("saved %s in %.0fs" % (dest, time.time() - t0))
        return 0
    if not args.teacher or not os.path.exists(args.teacher):
        sys.exit("teacher not found: %s" % args.teacher)
    cfg = ARMS[args.arm]
    run_reach.SAC = anchor_sac.make_anchor_sac(args.withdraw_at, cfg["anchor"], 128)
    anneal = args.withdraw_at if cfg["abrupt"] else cfg["anneal"]
    t0 = time.time()
    curve = run_reach.run(condition="coach", seed=args.seed, total_steps=args.steps, hidden=128,
                          utd=2, n_train_targets=8, eval_every=args.eval_every, eval_heldout=True,
                          coach_path=args.teacher, coach0=1.0, coach_anneal=anneal, coach_abrupt=cfg["abrupt"])
    with open(dest, "w") as f:
        json.dump(curve, f)
    print("saved %s in %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
