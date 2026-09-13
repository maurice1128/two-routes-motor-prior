"""B2 pilot on MyoSuite: decompose the cost of withdrawing a coach.

Arms (all identical except for what happens at the withdrawal step)
-------------------------------------------------------------------
  constant     teacher kept at full weight for the whole run          (TMLR: results_noanneal_*)
  abrupt       weight cut to zero at the withdrawal step              (TMLR: results_abruptcoach_*)
  selfanchor   teacher swapped for a frozen copy of the student's own actor; weight unchanged
  randanchor   teacher swapped for a freshly initialised actor;        weight unchanged
  none         no teacher at all                                       (TMLR: blank)

The first two reproduce cells the TMLR paper already reports, so they double as a
reproduction check on this harness. The last three are new.

What the contrasts buy
----------------------
  abrupt     - constant   = losing the teacher AND the loss term   (the paper's number)
  selfanchor - constant   = losing the teacher only
  abrupt     - selfanchor = losing the loss term only              (the unmeasured confound)
  randanchor - selfanchor = being pulled somewhere arbitrary vs being pulled where you are

If `abrupt - selfanchor` is large, the paper's withdrawal cost is substantially an
optimization shock rather than a loss of guidance, and the humanoid study must withdraw
by swapping rather than by deleting. If it is near zero, the simple withdrawal is clean
and the humanoid study can use it, which is a result worth having before spending GPU
time on Isaac Lab.

Everything else is held at the TMLR settings: one SAC backbone, hidden 128, update-to-data
ratio 2, 8 training targets, 20 disjoint held-out targets, withdrawal at 8k, scored at 12k.

Usage
-----
  python run_withdrawal.py --body myoelbow --seeds 12
  python run_withdrawal.py --body myofinger --seeds 12 --arms constant,abrupt,selfanchor
"""
import argparse
import json
import os
import sys
import time

# The body is chosen by an environment variable read at import time in arm_env, so it has
# to be set before anything from the TMLR package is imported.
_BODY = None
for i, tok in enumerate(sys.argv):
    if tok == "--body" and i + 1 < len(sys.argv):
        _BODY = sys.argv[i + 1]
    elif tok.startswith("--body="):
        _BODY = tok.split("=", 1)[1]
os.environ["WM_BODY"] = _BODY or os.environ.get("WM_BODY", "myoelbow")

import anchor_sac  # noqa: E402  (also chdirs into the wm_prior directory)
import run_reach  # noqa: E402

OUT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Teacher and prior checkpoints, per body, as used by the TMLR study.
BODIES = {
    "myoelbow": {"teacher": "teacher_elbow_clean30k.pt", "prior": "prior_myo.pt"},
    "myofinger": {"teacher": "teacher_finger_clean30k.pt", "prior": "prior_finger.pt"},
}

ARMS = {
    #  name          coach   anneal      abrupt  anchor_mode
    "constant":     dict(coach=True,  anneal=10 ** 8, abrupt=False, anchor="none"),
    "abrupt":       dict(coach=True,  anneal=None,    abrupt=True,  anchor="none"),
    "selfanchor":   dict(coach=True,  anneal=10 ** 8, abrupt=False, anchor="self"),
    "randanchor":   dict(coach=True,  anneal=10 ** 8, abrupt=False, anchor="random"),
    "none":         dict(coach=False, anneal=10 ** 8, abrupt=False, anchor="none"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="myoelbow", choices=sorted(BODIES))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed-list", type=str, default=None,
                    help="comma-separated seeds to run (subset of range(seeds)); lets one sweep be split across processes")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--withdraw-at", type=int, default=8000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--utd", type=int, default=2)
    ap.add_argument("--ntargets", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1000,
                    help="TMLR used 2000; 1000 gives a tighter view of the dip and of the "
                         "recovery, which is the whole point of this pilot")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seed_list = ([int(x) for x in args.seed_list.split(",")] if args.seed_list
                 else list(range(args.seeds)))

    body = BODIES[args.body]
    if not os.path.exists(body["teacher"]):
        raise SystemExit(f"teacher checkpoint not found: {body['teacher']} "
                         f"(cwd is {os.getcwd()})")

    out_dir = args.out or os.path.join(OUT_ROOT, f"results_withdrawal_{args.body}")
    os.makedirs(out_dir, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",")]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; expected some of {sorted(ARMS)}")

    meta = {"body": args.body, "arms": arms, "seeds": args.seeds, "steps": args.steps,
            "withdraw_at": args.withdraw_at, "hidden": args.hidden, "utd": args.utd,
            "ntargets": args.ntargets, "eval_every": args.eval_every,
            "teacher": body["teacher"], "eval_heldout": True}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    original_sac = run_reach.SAC
    total = len(arms) * len(seed_list)
    n = 0
    t_start = time.time()
    for arm in arms:
        cfg = ARMS[arm]
        for seed in seed_list:
            path = os.path.join(out_dir, f"{arm}_seed{seed}.json")
            if os.path.exists(path):
                n += 1
                continue
            # Install or remove the anchor subclass for this run.
            run_reach.SAC = (
                anchor_sac.make_anchor_sac(args.withdraw_at, cfg["anchor"], args.hidden)
                if cfg["anchor"] != "none" else original_sac
            )
            anneal = args.withdraw_at if cfg["anneal"] is None else cfg["anneal"]
            t0 = time.time()
            print(f"[{arm} seed{seed}] starting ({n + 1}/{total})", flush=True)
            # run_reach.run returns the eval curve and writes nothing; the file naming is
            # done by its __main__ block, which we are bypassing, so we write it here
            # under the ARM name rather than the condition name.
            curve = run_reach.run(
                condition="coach" if cfg["coach"] else "blank",
                seed=seed,
                total_steps=args.steps,
                hidden=args.hidden,
                utd=args.utd,
                n_train_targets=args.ntargets,
                eval_every=args.eval_every,
                eval_heldout=True,
                coach_path=body["teacher"] if cfg["coach"] else None,
                coach0=1.0,
                coach_anneal=anneal,
                coach_abrupt=cfg["abrupt"],
            )
            with open(path, "w") as f:
                json.dump(curve, f)
            n += 1
            print(f"[{arm} seed{seed}] done in {time.time() - t0:.0f}s "
                  f"({n}/{total}, {(time.time() - t_start) / 60:.1f} min elapsed)", flush=True)

    run_reach.SAC = original_sac
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
