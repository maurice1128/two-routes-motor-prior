# -*- coding: utf-8 -*-
"""Reproduce the two finger referents quoted in Limitations: the clean30k
teacher's held-out reach error and a zero-activation no-op's, under the same
evaluate() as teacher_eval.py (greedy, fixed pose, 20 held-out targets)."""
import os, sys, io
os.environ.setdefault("WM_BODY", "myofinger")
W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
os.chdir(W); sys.path.insert(0, W)
import numpy as np, torch
from run_reach import load_teacher, evaluate, fixed_eval_targets, ACT_DIM

class NoopActor:
    def act(self, o, greedy=True):
        # action -1 in tanh space = zero muscle activation (as random_baseline.py's "noop")
        return torch.full((o.shape[0], ACT_DIM), -1.0)

class Holder: pass
heldout = fixed_eval_targets(20)
t = Holder(); t.actor = load_teacher(os.path.join(W, "teacher_finger_clean30k.pt"), hidden=128)
_, d_teacher = evaluate(t, heldout, randomize_pose=False)
n = Holder(); n.actor = NoopActor()
_, d_noop = evaluate(n, heldout, randomize_pose=False)
line = "myofinger held-out: teacher_clean30k %.1f mm, zero-activation no-op %.1f mm" % (d_teacher * 1000, d_noop * 1000)
print(line)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teacher_noop_finger.txt")
io.open(out, "w", encoding="utf-8").write(line + "\n")
