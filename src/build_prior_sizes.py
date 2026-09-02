"""Model-quality ablation: train frozen priors from different babble-data budgets, so we
can later relate downstream sample-efficiency to model accuracy. Each prior is saved as
prior_{N}.pt along with its k-step open-loop error (the model-quality axis).

This directly addresses the 'a near-perfect model is trivially powerful' critique: it
characterizes how much cheap task-agnostic data the innate model needs, and shows whether
imperfect models still help.
"""
import numpy as np, torch, json, time
from collect_babble import collect
from world_model import train_world_model
from build_prior import collect_eval_rollouts
from arm_env import fk, JOINT_LO

SIZES = [1000, 3000, 10000, 30000, 100000, 200000]
NQ = len(JOINT_LO)

def kstep_error(wm, Ss, As, k=5):
    Sst = torch.tensor(Ss); Ast = torch.tensor(As)
    wm.eval()
    with torch.no_grad():
        s = Sst[:, 0]
        for t in range(k):
            s = wm(s, Ast[:, t])
        tip_pred = fk(s[:, :NQ].numpy()); tip_true = fk(Ss[:, k, :NQ])
        tip_err = np.linalg.norm(tip_pred - tip_true, axis=1)
        qerr = torch.sqrt(((s[:, :NQ] - Sst[:, k, :NQ]) ** 2).mean()).item()
    return float(tip_err.mean()), qerr

def main(prefix="prior", sizes_json="prior_sizes.json"):
    print("collecting full babble pool (200k) once ...")
    S, A, S2 = collect(200_000, seed=0)
    Ss, As = collect_eval_rollouts(n=1000, T=6)
    log = {}
    for n in SIZES:
        t0 = time.time()
        wm = train_world_model(S[:n], A[:n], S2[:n], hidden=256, epochs=40, seed=0,
                               device="cpu", verbose=False)
        tip_mm, qerr = kstep_error(wm, Ss, As, k=5)
        torch.save({"state_dict": wm.state_dict(), "hidden": 256,
                    "in_mean": wm.in_norm.mean.cpu().numpy(), "in_std": wm.in_norm.std.cpu().numpy(),
                    "out_mean": wm.out_norm.mean.cpu().numpy(), "out_std": wm.out_norm.std.cpu().numpy()},
                   f"{prefix}_{n}.pt")
        log[n] = {"tip_err_mm_k5": tip_mm, "qpos_rms_k5": qerr, "train_s": time.time()-t0}
        print(f"  N={n:7d}  k5 tip_err {tip_mm*1000:6.2f}mm  qpos_rms {qerr:.5f}  ({time.time()-t0:.0f}s)")
    json.dump(log, open(sizes_json, "w"), indent=2)
    print(f"saved priors {prefix}_{{N}}.pt and {sizes_json}")

if __name__ == "__main__":
    import sys
    pre = sys.argv[1] if len(sys.argv) > 1 else "prior"
    sj = sys.argv[2] if len(sys.argv) > 2 else "prior_sizes.json"
    main(pre, sj)
