# -*- coding: utf-8 -*-
"""The control this paper says it did not run: a replay-matched prior withdrawal.

Section 3.4 discloses that a strict prior withdrawal is not single-factor. Purging
the imagined buffer removes imagination AND doubles the number of distinct real
transitions per update, because the guard `model_buf.n < batch` then sends every
update down the 128-real path. So the measured withdrawal cost mixes "lost the
body model" with "got twice the real data", and the two push in opposite
directions -- the confound pushes the measured cost toward zero.

The arm added here withdraws the prior exactly as the strict arm does, then holds
the number of DISTINCT real transitions per update at 64 by drawing 64 and
repeating them to fill the batch. Gradient magnitude and batch shape are
unchanged; only the amount of fresh real information per update is held fixed.

What this matches and what it does not: it matches distinct real transitions per
update, which is the quantity Section 3.4 names. It does not reproduce the
pre-withdrawal loss composition, where the imagined half carried its own
(model-generated) targets at half the batch weight. No control can, since the
whole point is that those targets are gone.

Implementation deliberately touches nothing in the frozen tree. run_reach.py is
imported by other work in this repository, so this file patches at runtime:

  * `sac_dyna.Replay.add` on the 200k-capacity instance is called exactly once per
    environment step by run_reach (line 121), which makes it an exact step counter.
  * `run_reach.SAC` is a module-level name, so binding a subclass there swaps the
    agent for the run without editing the module.

Usage mirrors run_reach.py:
  python run_prior_matched.py --body myofinger --arm matched --seed 0 --steps 12000
"""
import argparse, json, os, sys, torch

import sac_dyna
import run_reach

REAL_CAP = 200_000          # run_reach builds the real buffer at this capacity
_state = {"step": 0, "off": 10 ** 9, "match": False}

_orig_add = sac_dyna.Replay.add


def _counting_add(self, o, a, r, o2, d):
    # Only the real buffer is written once per environment step; the imagined
    # buffer is filled in bursts by add_batch and must not advance the counter.
    if getattr(self, "cap", None) == REAL_CAP:
        _state["step"] += 1
    return _orig_add(self, o, a, r, o2, d)


class MatchedSAC(sac_dyna.SAC):
    """SAC that, after withdrawal, keeps distinct real samples per update at 64."""

    def update(self, batch):
        if _state["match"] and _state["step"] >= _state["off"]:
            n = batch[0].shape[0]
            half = n // 2
            if half > 0:
                batch = tuple(torch.cat([b[:half], b[:half]], 0) for b in batch)
        return super().update(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, choices=["myoelbow", "myofinger"])
    ap.add_argument("--arm", required=True,
                    choices=["keep", "soft", "purge", "matched"],
                    help="keep: prior throughout, fresh imagination. soft: stop "
                         "imagining at the cut but keep the stale buffer (the "
                         "paper's soft arm). purge: strict withdrawal, buffer "
                         "emptied (the paper's strict arm). matched: strict "
                         "withdrawal with distinct real samples per update held "
                         "at 64. soft and matched differ only in whether stale "
                         "imagined transitions remain in the mix, so their "
                         "contrast isolates what stale imagination is worth.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--withdraw-at", type=int, default=8000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.environ.get("WM_BODY") != args.body:
        sys.exit("WM_BODY=%r but --body=%r; arm_env dispatches on the environment "
                 "variable, so they must agree." % (os.environ.get("WM_BODY"), args.body))

    prior = {"myoelbow": "prior_myo.pt", "myofinger": "prior_finger.pt"}[args.body]
    off = 10 ** 9 if args.arm == "keep" else args.withdraw_at
    _state["off"] = off
    _state["match"] = (args.arm == "matched")

    sac_dyna.Replay.add = _counting_add
    run_reach.SAC = MatchedSAC

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "%s_seed%d.json" % (args.arm, args.seed))
    if os.path.exists(dest):
        print("SKIP", dest); return 0

    curve = run_reach.run("prior", seed=args.seed, total_steps=args.steps, utd=2,
                          hidden=128, n_train_targets=8, prior_path=prior,
                          eval_heldout=True, prior_off=off,
                          prior_purge=(args.arm in ("purge", "matched")))
    # A silent counter failure would make `matched` identical to `purge`, which is
    # exactly the sort of null that looks like a finding. Assert it advanced.
    if _state["step"] < args.steps:
        sys.exit("step counter saw %d of %d environment steps; the Replay.add hook "
                 "did not fire as expected" % (_state["step"], args.steps))
    with open(dest, "w") as fh:
        json.dump(curve, fh)
    print("saved", dest, "(counter saw %d steps)" % _state["step"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
