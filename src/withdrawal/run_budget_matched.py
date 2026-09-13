"""B1 pilot: what happens to the two routes when their data budgets are actually matched.

The gap this fills
------------------
The TMLR study compares a frozen body model against a distilled coach, and its own
Limitations say plainly that there is no budget matching: neither aid is matched to the
unaided baseline, and the two routes are not matched to each other. The coach was trained
on 30,000 rewarded on-task steps. The body model was trained on 100,000 (myoElbow) or
200,000 (myoFinger) reward-free babble transitions, so the prior consumed 3.3x to 6.7x the
coach's data before the student took a single step. Matching happens only on the
withdrawal step and the scoring step.

That is the single most obvious objection to the whole comparison, and the adversarial
prior-art search found that nobody else has run a matched-budget version either: it is the
one element of the humanoid proposal with no near neighbour in the literature. So it is
worth settling on MyoSuite first, cheaply, before spending GPU time on a humanoid.

The arms
--------
  blank          no aid at all
  coach          constant coach, teacher trained on 30k rewarded on-task steps
  prior_matched  frozen body model built from exactly 30k reward-free babble transitions
  prior_full     frozen body model at the paper's budget (100k on myoElbow)

`prior_matched` uses the checkpoint the paper's own budget sweep already produced, so no
new model has to be trained for it.

What matching does and does not buy
------------------------------------
Matching the raw transition COUNT does not match information content, and the paper should
say so. The coach's 30k steps are rewarded and on-task; the prior's 30k are reward-free and
task-agnostic, and the prior arm additionally receives the environment's analytic reward
evaluated at imagined states. Count matching is the weakest honest form of matching, not
the strongest. It removes one specific objection ("the prior just saw more data") and
leaves the others standing, which should be stated rather than glossed.

Two further asymmetries worth reporting alongside the result:
  - total compute differs, because the coach's 30k steps also had to train a policy,
    while the prior's 30k only had to fit a dynamics model;
  - the prior arm trains on 64 real plus 64 imagined transitions per update while the
    model-free arms train on 128 real, so equal update-to-data is not equal real-data
    gradient signal. That confound is inherited from the TMLR harness and is not fixed here.

Usage
-----
  python run_budget_matched.py --body myoelbow --seeds 12
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

WM_PRIOR_DIR = os.environ.get(
    "WM_PRIOR_DIR", r"C:\Users\maurice\Desktop\robotic_research\wm_prior")
if WM_PRIOR_DIR not in sys.path:
    sys.path.insert(0, WM_PRIOR_DIR)
os.chdir(WM_PRIOR_DIR)

import run_reach  # noqa: E402

OUT_ROOT = os.path.dirname(os.path.abspath(__file__))

BODIES = {
    "myoelbow": {
        "teacher": "teacher_elbow_clean30k.pt",
        "prior_full": "prior_myo.pt",             # 1e5 babble transitions
        "prior_matched": "prior_myo_sweep_30000.pt",  # 3e4, matching the teacher's budget
        "teacher_steps": 30000,
        "prior_full_transitions": 100000,
        "prior_matched_transitions": 30000,
    },
    "myofinger": {
        "teacher": "teacher_finger_clean30k.pt",
        "prior_full": "prior_finger.pt",              # 2e5 babble transitions (TMLR)
        # built by build_prior_finger_matched.py with the build_prior_sizes.py recipe
        "prior_matched": os.path.join(OUT_ROOT, "checkpoints", "prior_finger_matched30k.pt"),
        "teacher_steps": 30000,
        "prior_full_transitions": 200000,
        "prior_matched_transitions": 30000,
    },
}

ARMS = {
    "blank":         dict(cond="blank", prior=None,            coach=False),
    "coach":         dict(cond="coach", prior=None,            coach=True),
    "prior_matched": dict(cond="prior", prior="prior_matched", coach=False),
    "prior_full":    dict(cond="prior", prior="prior_full",    coach=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="myoelbow", choices=sorted(BODIES))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed-list", type=str, default=None,
                    help="comma-separated subset of seeds, to split across processes")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--utd", type=int, default=2)
    ap.add_argument("--ntargets", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seed_list = ([int(x) for x in args.seed_list.split(",")] if args.seed_list
                 else list(range(args.seeds)))

    body = BODIES[args.body]
    arms = [a.strip() for a in args.arms.split(",")]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}")
    for a in arms:
        pk = ARMS[a]["prior"]
        if pk and not os.path.exists(body[pk]):
            raise SystemExit(f"missing checkpoint for arm {a}: {body[pk]}")
    if any(ARMS[a]["coach"] for a in arms) and not os.path.exists(body["teacher"]):
        raise SystemExit(f"missing teacher: {body['teacher']}")

    out_dir = args.out or os.path.join(OUT_ROOT, f"results_budget_{args.body}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"body": args.body, "arms": arms, "seeds": args.seeds,
                   "steps": args.steps, "eval_every": args.eval_every,
                   "hidden": args.hidden, "utd": args.utd, "eval_heldout": True,
                   "teacher_steps": body["teacher_steps"],
                   "prior_full_transitions": body["prior_full_transitions"],
                   "prior_matched_transitions": body["prior_matched_transitions"]}, f, indent=2)

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
            t0 = time.time()
            print(f"[{arm} seed{seed}] starting ({n + 1}/{total})", flush=True)
            curve = run_reach.run(
                condition=cfg["cond"],
                seed=seed,
                total_steps=args.steps,
                hidden=args.hidden,
                utd=args.utd,
                n_train_targets=args.ntargets,
                eval_every=args.eval_every,
                eval_heldout=True,
                prior_path=body[cfg["prior"]] if cfg["prior"] else "prior.pt",
                coach_path=body["teacher"] if cfg["coach"] else None,
                coach0=1.0,
                coach_anneal=10 ** 8,   # constant coach, never withdrawn
                coach_abrupt=False,
            )
            with open(path, "w") as f:
                json.dump(curve, f)
            n += 1
            print(f"[{arm} seed{seed}] done in {time.time() - t0:.0f}s "
                  f"({n}/{total}, {(time.time() - t_start) / 60:.1f} min)", flush=True)

    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
