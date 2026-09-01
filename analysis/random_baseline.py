import os, numpy as np
from run_reach import fixed_eval_targets, make_train_targets
from arm_env import ArmReachEnv, ACT_DIM, TASK_TARGET_SEED
body=os.environ["WM_BODY"]
rng=np.random.default_rng(0)
def eval_policy(targets, mode):
    env=ArmReachEnv(seed=7); ds=[]
    for tg in targets:
        o=env.reset(target=tg, randomize_pose=False)
        for _ in range(100):
            a = rng.uniform(-1,1,ACT_DIM) if mode=="random" else np.full(ACT_DIM,-1.0)  # -1 -> zero activation
            o,r,d,inf=env.step(a)
        ds.append(inf["dist"])
    return np.mean(ds)*1000, np.std(ds)*1000
ho=fixed_eval_targets(20); tr=make_train_targets(8, seed=TASK_TARGET_SEED.get("reach",777))
for mode in ["random","noop"]:
    m1,s1=eval_policy(tr,mode); m2,s2=eval_policy(ho,mode)
    print(f"{body} {mode:7s}: train {m1:.1f}mm (sd{s1:.0f})   held-out {m2:.1f}mm (sd{s2:.0f})")
