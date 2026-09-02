"""Go/no-go part (a): build the frozen world-model prior from task-agnostic babbling
and measure its k-step open-loop prediction accuracy. Saves the model for downstream use.
"""
import numpy as np, torch, time, argparse
from collect_babble import collect
from world_model import train_world_model, WorldModel
from arm_env import ArmReachEnv, STATE_DIM, ACT_DIM

def collect_eval_rollouts(n=1000, T=6, seed=999):
    """Contiguous ground-truth rollouts under OU actions for k-step eval."""
    env = ArmReachEnv(seed=seed); rng = np.random.default_rng(seed)
    Ss = np.zeros((n, T + 1, STATE_DIM), np.float32)
    As = np.zeros((n, T, ACT_DIM), np.float32)
    for i in range(n):
        env.reset(randomize_pose=True)
        a = rng.uniform(-1, 1, ACT_DIM)
        Ss[i, 0] = env.get_state()
        for t in range(T):
            a = np.clip(a + 0.15 * (-a) + 0.4 * rng.standard_normal(ACT_DIM), -1, 1)
            env.step(a); As[i, t] = a; Ss[i, t + 1] = env.get_state()
    return Ss, As

def main(n_babble=200_000, hidden=256, epochs=40, seed=0, out="prior.pt"):
    print("== collecting babble (task-agnostic) ==")
    S, A, S2 = collect(n_babble, seed=seed)
    print("qpos coverage:", S[:, :2].min(0).round(2), "->", S[:, :2].max(0).round(2))
    print("== training world model ==")
    t0 = time.time()
    wm = train_world_model(S, A, S2, hidden=hidden, epochs=epochs, seed=seed, device="cpu")
    print("train time %.1fs" % (time.time() - t0))

    print("== k-step open-loop eval (held-out rollouts) ==")
    Ss, As = collect_eval_rollouts(n=1000, T=6)
    Sst = torch.tensor(Ss); Ast = torch.tensor(As)
    wm.eval()
    from arm_env import fk, JOINT_LO
    NQ = len(JOINT_LO)
    with torch.no_grad():
        s = Sst[:, 0]
        # baseline: persistence (predict s_next = s) for reference scale
        for t in range(6):
            s = wm(s, Ast[:, t])
            true = Sst[:, t + 1]
            rms_q = torch.sqrt(((s[:, :NQ] - true[:, :NQ]) ** 2).mean()).item()
            rms_all = torch.sqrt(((s - true) ** 2).mean()).item()
            # persistence error at this horizon for scale
            pers_q = torch.sqrt(((Sst[:, 0, :NQ] - true[:, :NQ]) ** 2).mean()).item()
            print(f"  k={t+1}: qpos RMS {rms_q:.4f} rad (persist {pers_q:.4f})  |  full-state RMS {rms_all:.4f}")

    # tip-position error at k=5 (task-relevant): FK on predicted vs true qpos
    with torch.no_grad():
        s = Sst[:, 0]
        for t in range(5):
            s = wm(s, Ast[:, t])
        tip_pred = fk(s[:, :NQ].numpy()); tip_true = fk(Ss[:, 5, :NQ])
        tip_err = np.linalg.norm(tip_pred - tip_true, axis=1)
    print(f"  k=5 fingertip position error: mean {tip_err.mean()*1000:.1f} mm  median {np.median(tip_err)*1000:.1f} mm")

    torch.save({"state_dict": wm.state_dict(), "hidden": hidden,
                "in_mean": wm.in_norm.mean.cpu().numpy(), "in_std": wm.in_norm.std.cpu().numpy(),
                "out_mean": wm.out_norm.mean.cpu().numpy(), "out_std": wm.out_norm.std.cpu().numpy()}, out)
    print("saved prior ->", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="prior.pt")
    args = ap.parse_args()
    main(n_babble=args.n, epochs=args.epochs, seed=args.seed, out=args.out)
