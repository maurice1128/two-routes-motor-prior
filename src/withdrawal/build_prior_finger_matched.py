"""Build the budget-matched myoFinger prior that B1 needs.

The TMLR study's myoFinger prior (`prior_finger.pt`) was trained on 200,000 reward-free
babble transitions (see wm_prior/build_prior_finger.log). The coach was trained on 30,000
rewarded on-task steps. On myoElbow the paper's own budget sweep left a 30k checkpoint
(`prior_myo_sweep_30000.pt`) that B1 could reuse; on myoFinger no such checkpoint exists,
so this script builds one with exactly the recipe of `build_prior_sizes.py`:

    the first N transitions of the seed-0 OU babble stream, hidden 256, 40 epochs, seed 0

`collect(n, seed=0)` is sequential and deterministic, so `collect(30000)` is the same data
as `collect(200000)[:30000]` -- the matched prior sees a prefix of exactly the pool the
full prior saw, which is what the elbow sweep did too.

The checkpoint is written to this directory (the frozen TMLR tree is not touched) and
its 5-step open-loop fingertip error is printed and saved next to it so the model-quality
axis is on record for the write-up.

Usage:
  python build_prior_finger_matched.py            # 30000 transitions
  python build_prior_finger_matched.py --n 30000 --out checkpoints/prior_finger_matched30k.pt
"""
import argparse
import json
import os
import sys
import time

os.environ["WM_BODY"] = "myofinger"
WM_PRIOR_DIR = os.environ.get(
    "WM_PRIOR_DIR", r"C:\Users\maurice\Desktop\robotic_research\wm_prior")
if WM_PRIOR_DIR not in sys.path:
    sys.path.insert(0, WM_PRIOR_DIR)
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(WM_PRIOR_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(1)
from collect_babble import collect  # noqa: E402
from world_model import train_world_model  # noqa: E402
from build_prior import collect_eval_rollouts  # noqa: E402
from build_prior_sizes import kstep_error  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "checkpoints",
                                                  "prior_finger_matched30k.pt"))
    args = ap.parse_args()
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    t0 = time.time()
    S, A, S2 = collect(args.n, seed=args.seed)
    print(f"qpos coverage: {S[:, :4].min(0).round(2)} -> {S[:, :4].max(0).round(2)}")
    wm = train_world_model(S, A, S2, hidden=args.hidden, epochs=args.epochs,
                           seed=args.seed, device="cpu", verbose=True)
    Ss, As = collect_eval_rollouts(n=1000, T=6)
    tip_m, qerr = kstep_error(wm, Ss, As, k=5)
    torch.save({"state_dict": wm.state_dict(), "hidden": args.hidden,
                "in_mean": wm.in_norm.mean.cpu().numpy(),
                "in_std": wm.in_norm.std.cpu().numpy(),
                "out_mean": wm.out_norm.mean.cpu().numpy(),
                "out_std": wm.out_norm.std.cpu().numpy()}, out)
    info = {"body": "myofinger", "n_babble": args.n, "hidden": args.hidden,
            "epochs": args.epochs, "seed": args.seed,
            "tip_err_mm_k5": tip_m * 1000.0, "qpos_rms_k5": qerr,
            "recipe": "build_prior_sizes.py (prefix of seed-0 OU babble, fixed epochs)",
            "full_prior_reference": "prior_finger.pt = 200k babble, k5 tip err 14.3 mm",
            "wall_s": time.time() - t0}
    with open(out[:-3] + ".json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"N={args.n}  k5 tip_err {tip_m * 1000:.2f} mm  qpos_rms {qerr:.4f}  "
          f"({time.time() - t0:.0f}s)\nsaved {out}")


if __name__ == "__main__":
    main()
