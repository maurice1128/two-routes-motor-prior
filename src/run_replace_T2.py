# -*- coding: utf-8 -*-
"""Replacing regime on myoFinger with the competent teacher T2.

world_model_zeroshot/b2_pilot/run_replace_sweep.py (read-only) hard-codes the finger teacher as
T1 (teacher_finger_clean30k.pt). This wrapper imports it unchanged and swaps only the finger
teacher path before main() runs, so the recipe is otherwise identical to the elbow runs.
replace_sac chdirs into wm_prior on import, where the T2 file lives.

usage: python run_replace_T2.py --body myofinger --arms abrupt --withdraw-at 4000 --seed-list 0 \
           --steps 100000 --eval-every 1000 --out <abs dir>
"""
import sys
sys.path.insert(0, r"C:\Users\maurice\Desktop\world_model_zeroshot\b2_pilot")
import run_replace_sweep as rrs  # noqa: E402

rrs.BODIES["myofinger"]["teacher"] = "teacher_finger_prior100k.pt"

if __name__ == "__main__":
    rrs.main()
