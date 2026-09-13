"""Attachment sweep in the REPLACING regime (see replace_sac.py for the rationale).

Arms per withdrawal step t_w: abrupt, selfanchor, gateon (Q-gradient on, teacher kept). `constant` (never withdrawn, pure
distillation for the whole run) is shared across t_w and written as constant_seed{s}.json
so analyze_attachment.py can read this directory with --shared pointing at itself; copy
none_seed*.json from results_withdrawal_<body> into the output dir for the context lines.

Usage
-----
  python run_replace_sweep.py --body myoelbow --seeds 12
  python run_replace_sweep.py --body myoelbow --arms constant
  python run_replace_sweep.py --body myoelbow --withdraw-at 4000 --arms abrupt --seed-list 0,1,2
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

import replace_sac  # noqa: E402  (chdirs into wm_prior)
import run_reach  # noqa: E402

OUT_ROOT = os.path.dirname(os.path.abspath(__file__))
BODIES = {
    "myoelbow": {"teacher": "teacher_elbow_clean30k.pt"},
    "myofinger": {"teacher": "teacher_finger_clean30k.pt"},
}
DEFAULT_WITHDRAW = [2000, 4000, 6000, 8000]
ARMS = {
    "constant":   dict(abrupt=False, mode="none", never=True),
    "abrupt":     dict(abrupt=True,  mode="none", never=False),
    "selfanchor": dict(abrupt=False, mode="self", never=False),
    # gateon: at t_w the actor gains its own Q-gradient but KEEPS the real teacher, so
    #   gateon     - constant = gaining own signal only
    #   selfanchor - gateon   = losing the teacher only, own signal on in both
    "gateon":     dict(abrupt=False, mode="none", never=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="myoelbow", choices=sorted(BODIES))
    ap.add_argument("--withdraw-at", type=str,
                    default=",".join(str(x) for x in DEFAULT_WITHDRAW))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed-list", type=str, default=None)
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

    out_dir = args.out or os.path.join(OUT_ROOT, f"results_replace_{args.body}")
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump({"body": args.body, "regime": "replacing (q_weight=0 while teacher attached)",
                       "withdraw_at": DEFAULT_WITHDRAW, "arms": list(ARMS), "seeds": 12,
                       "steps": args.steps, "eval_every": args.eval_every, "hidden": args.hidden,
                       "utd": args.utd, "teacher": body["teacher"], "eval_heldout": True,
                       "note": "constant_seed*.json = pure distillation all run; "
                               "none_seed*.json copied from results_withdrawal_<body>"}, f, indent=2)

    original_sac = run_reach.SAC
    cells = []
    for arm in arms:
        if ARMS[arm]["never"]:
            cells += [(arm, None, s) for s in seed_list]
        else:
            cells += [(arm, t_w, s) for t_w in withdraws for s in seed_list]
    n, t_start = 0, time.time()
    for arm, t_w, seed in cells:
        cfg = ARMS[arm]
        fn = f"{arm}_seed{seed}.json" if t_w is None else f"{arm}_w{t_w}_seed{seed}.json"
        path = os.path.join(out_dir, fn)
        if os.path.exists(path):
            n += 1
            continue
        swap_at = 10 ** 8 if t_w is None else t_w
        run_reach.SAC = replace_sac.make_replace_sac(swap_at, cfg["mode"], args.hidden)
        anneal = t_w if cfg["abrupt"] else 10 ** 8
        tag = f"{arm}{'' if t_w is None else ' w' + str(t_w)} seed{seed}"
        t0 = time.time()
        print(f"[{tag}] starting ({n + 1}/{len(cells)})", flush=True)
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
        print(f"[{tag}] done in {time.time() - t0:.0f}s "
              f"({n}/{len(cells)}, {(time.time() - t_start) / 60:.1f} min)", flush=True)

    run_reach.SAC = original_sac
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
