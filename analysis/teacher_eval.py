import os, numpy as np, torch
from run_reach import load_teacher, evaluate, make_train_targets, fixed_eval_targets
from arm_env import TASK_TARGET_SEED
body = os.environ["WM_BODY"]; tp = os.environ["TEACHER"]
class W: pass
w = W(); w.actor = load_teacher(tp, hidden=128)
train = make_train_targets(8, seed=TASK_TARGET_SEED.get("reach",777))
heldout = fixed_eval_targets(20)
_, dtr = evaluate(w, train, randomize_pose=False)
_, dho = evaluate(w, heldout, randomize_pose=False)
print(f"{body} TEACHER final reach dist:  train {dtr*1000:.1f} mm   held-out {dho*1000:.1f} mm")
