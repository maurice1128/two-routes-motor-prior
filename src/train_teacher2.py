"""Train a CLEAN teacher policy for the coach conditions.

Why this exists: the original train_teacher.py called run("prior", ...), i.e. the teacher
was itself trained WITH the frozen babble body model in a Dyna loop. That contaminated the
2x2 -- the 'coach only' cell inherited the body prior laundered through the teacher's weights,
so 'the source of prior structure is the only variable' was false as implemented.

This version trains the teacher from a specified condition (default: blank = model-free SAC,
no body prior anywhere), so the coach route carries NO body-prior information.
Budget is explicit and reported, since teacher pretraining cost is part of the coach route.
"""
import argparse
from run_reach import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)              # myoelbow / myofinger (also set WM_BODY)
    ap.add_argument("--cond", default="blank", choices=["blank", "prior"],
                    help="condition used to TRAIN the teacher. 'blank' = clean (no body prior).")
    ap.add_argument("--prior", default=None, help="only used if --cond prior")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=100)      # distinct from student seeds 0-11
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(f"training teacher: body={args.body} cond={args.cond} steps={args.steps} seed={args.seed} -> {args.out}")
    run(args.cond, seed=args.seed, total_steps=args.steps, utd=2, hidden=128,
        n_train_targets=8, task="reach",
        prior_path=(args.prior or "prior.pt"), save_actor_path=args.out)
