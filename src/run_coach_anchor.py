# -*- coding: utf-8 -*-
"""Coach-side withdrawal arms at an arbitrary budget, without editing run_reach.py.

Three arms, all against the paper's constant-weight coach (clean30k teacher):
  constant   never withdraws.
  abrupt     distillation weight cut to zero at --withdraw-at (the literature's
             withdrawal, and the paper's Table 5 arm).
  selfanchor at --withdraw-at the teacher is replaced by a frozen copy of the
             student's own actor; the loss term survives at the same weight but
             carries no external information.

selfanchor − constant prices losing the teacher's information; abrupt −
selfanchor prices deleting the loss term. Both were measured at a 12k endpoint;
this file exists so they can be measured at 100k, where the arms have stopped
moving, because a paper that argues 12k endpoints sit in transients cannot rest
its own "persistent cost" on one.

Hook: run_reach.run assigns agent.coach_coef exactly once per environment step
(lines 111/113), so a property setter is an exact step counter. Assignments made
before the loop (SAC.__init__, set_teacher) are subtracted by recording the count
at the end of set_teacher, after which env_step == step.
"""
import argparse, copy, json, os, sys, torch

import sac_dyna
import run_reach

_cfg = {"swap_at": None}

# --swap-offset exists to reproduce another implementation of the same control
# exactly. That implementation counted every coach_coef assignment from object
# construction without subtracting the two made before the training loop
# (SAC.__init__ and set_teacher), and fired on >=, so its swap lands on loop step
# t_w - 2. Its 12k arms are the ones tabulated at 12k; to extend the same arm to
# 100k the swap has to land on the same step, so offset -2 reproduces it seed for
# seed, while offset 0 is the nominal t_w. Both are kept; the difference is a
# robustness note, not a result.


class AnchorSAC(sac_dyna.SAC):
    def __init__(self, *a, **k):
        self._n = 0; self._base = 0; self._swapped = False
        super().__init__(*a, **k)

    @property
    def coach_coef(self):
        return self._coach_coef

    @coach_coef.setter
    def coach_coef(self, v):
        self._coach_coef = v
        self._n += 1
        step = self._n - self._base
        if (_cfg["swap_at"] is not None and not self._swapped
                and self.teacher is not None
                and step == _cfg["swap_at"] + _cfg.get("offset", 0)):
            anchor = copy.deepcopy(self.actor)
            for p in anchor.parameters():
                p.requires_grad_(False)
            anchor.eval()
            self.teacher = anchor
            self._swapped = True
            _cfg["swapped_at"] = step
            print("    [anchor] env-step %d: teacher -> frozen self snapshot" % step, flush=True)

    def set_teacher(self, teacher_actor, coef=1.0):
        super().set_teacher(teacher_actor, coef)
        self._base = self._n      # everything counted so far happened before the loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, choices=["myoelbow", "myofinger"])
    ap.add_argument("--arm", required=True, choices=["constant", "abrupt", "selfanchor"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--withdraw-at", type=int, default=8000)
    ap.add_argument("--swap-offset", type=int, default=0,
                    help="selfanchor only: fire the swap at withdraw-at + offset "
                         "loop steps. -2 reproduces the other implementation.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    _cfg["offset"] = args.swap_offset
    if os.environ.get("WM_BODY") != args.body:
        sys.exit("WM_BODY=%r but --body=%r" % (os.environ.get("WM_BODY"), args.body))

    teacher = {"myoelbow": "teacher_elbow_clean30k.pt",
               "myofinger": "teacher_finger_clean30k.pt"}[args.body]
    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "%s_seed%d.json" % (args.arm, args.seed))
    if os.path.exists(dest):
        print("SKIP", dest); return 0

    _cfg["swap_at"] = args.withdraw_at if args.arm == "selfanchor" else None
    run_reach.SAC = AnchorSAC
    curve = run_reach.run("coach", seed=args.seed, total_steps=args.steps, utd=2,
                          hidden=128, n_train_targets=8, eval_heldout=True,
                          coach_path=teacher, coach0=1.0,
                          coach_anneal=(args.withdraw_at if args.arm == "abrupt" else 10 ** 8),
                          coach_abrupt=(args.arm == "abrupt"))
    # A swap that never fired would make selfanchor identical to constant -- a
    # null that reads as "losing the teacher costs nothing". Refuse to write it.
    want = args.withdraw_at + args.swap_offset
    if args.arm == "selfanchor" and _cfg.get("swapped_at") != want:
        sys.exit("selfanchor swap did not fire at %d (fired at %r); not saving"
                 % (want, _cfg.get("swapped_at")))
    with open(dest, "w") as fh:
        json.dump(curve, fh)
    print("saved", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
