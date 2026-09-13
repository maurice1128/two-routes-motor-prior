"""myoFinger actors at the CONVERGED budget (100k steps) for the project video.

The paper's Table I scores the arms at 100,000 steps; the earlier video actors
(actors_finger/) were 12k policies. run_reach is seed-deterministic, so the run
here IS the paper's run for that seed: the held-out curve it writes is checked
against results_conv_finger_<arm>/<cond>_seed<s>.json checkpoint for checkpoint
before an actor is trusted for the video.

usage:  python train_video100k.py <blank|prior|coach> <seed>
"""
import os, sys, json
os.environ.setdefault("WM_BODY", "myofinger")
import run_reach

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "actors_finger100k"); os.makedirs(OUT, exist_ok=True)
ARMS = {
    "blank": ("blank", {}),
    "prior": ("prior", {"prior_path": "prior_finger.pt"}),
    "coach": ("coach", {"coach_path": "teacher_finger_clean30k.pt", "coach_anneal": 10**8}),
    "randprior": ("prior", {"prior_path": "prior_rand_finger.pt"}),
}
label, sd = sys.argv[1], int(sys.argv[2])
cond, extra = ARMS[label]
dest = os.path.join(OUT, "%s_s%d.pt" % (label, sd))
curve = run_reach.run(cond, seed=sd, total_steps=100000, utd=2, n_train_targets=8,
                      eval_heldout=True, save_actor_path=dest, **extra)
with open(os.path.join(OUT, "%s_seed%d.json" % (label, sd)), "w") as f:
    json.dump(curve, f)
# reproduction check against the paper's run
ref = os.path.join(ROOT, "results_conv_finger_%s" % label, "%s_seed%d.json" % (cond, sd))
if os.path.exists(ref):
    R = {int(r["step"]): r["eval_dist"] for r in json.load(open(ref))}
    bad = sum(1 for r in curve if abs(R.get(int(r["step"]), 1e9) - r["eval_dist"]) > 1e-6)
    print("REPRODUCTION vs paper run: %d/%d checkpoints differ" % (bad, len(curve)), flush=True)
    if bad:
        os.rename(dest, dest + ".NOREPRO")
print("DONE", label, sd, flush=True)
