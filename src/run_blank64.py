# -*- coding: utf-8 -*-
"""blank-64: model-free SAC whose every update sees 64 DISTINCT real transitions
repeated to fill the 128 batch, for the whole run.

Why: the Dyna arms train on 64 real + 64 imagined transitions per update where
model-free arms train on 128 real, so prior-vs-blank and randprior-vs-blank mix
"has a model" with "half the fresh real data per update". The matched arm
(run_prior_matched.py) controls this only after withdrawal. blank-64 is the
same manipulation applied from step 0 with no model at all: blank-64 vs blank
prices the halved real-data signal; prior vs blank-64 is the body model at
equal distinct real transitions per update.

Same runtime-patch approach as run_prior_matched.py (nothing in run_reach.py is
edited): run_reach.SAC is rebound to a subclass whose update() duplicates the
first half of the batch.

usage:  WM_BODY=myofinger python run_blank64.py --body myofinger --seed 0 --steps 100000 --out results_conv_finger_blank64
"""
import argparse, json, os, sys, torch
import sac_dyna, run_reach


class Blank64SAC(sac_dyna.SAC):
    def update(self, batch):
        n = batch[0].shape[0]; half = n // 2
        batch = tuple(torch.cat([b[:half], b[:half]], 0) for b in batch)
        return super().update(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, choices=["myoelbow", "myofinger", "arm1", "arm2", "arm3", "arm4"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.environ.get("WM_BODY") != args.body:
        sys.exit("WM_BODY=%r but --body=%r" % (os.environ.get("WM_BODY"), args.body))
    run_reach.SAC = Blank64SAC
    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "blank64_seed%d.json" % args.seed)
    if os.path.exists(dest):
        print("SKIP", dest); return 0
    curve = run_reach.run("blank", seed=args.seed, total_steps=args.steps, utd=2, hidden=128,
                          n_train_targets=8, eval_heldout=True)
    with open(dest, "w") as fh:
        json.dump(curve, fh)
    print("saved", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
