"""Task-agnostic motor babbling: temporally-correlated random muscle activations,
periodic resets to random poses. Produces (s, a, s_next) transitions with NO task reward.
This is the ONLY data the 'innate' world-model prior is allowed to see.
"""
import numpy as np, argparse, time
from arm_env import ArmReachEnv, STATE_DIM, ACT_DIM

def collect(n_transitions=200_000, seed=0, reset_every=100, ou_theta=0.15, ou_sigma=0.4):
    env = ArmReachEnv(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    S = np.zeros((n_transitions, STATE_DIM), np.float32)
    A = np.zeros((n_transitions, ACT_DIM), np.float32)
    S2 = np.zeros((n_transitions, STATE_DIM), np.float32)
    env.reset(randomize_pose=True)
    a = rng.uniform(-1, 1, ACT_DIM)
    i = 0; t0 = time.time()
    while i < n_transitions:
        if i % reset_every == 0 and i > 0:
            env.reset(randomize_pose=True)
            a = rng.uniform(-1, 1, ACT_DIM)
        # Ornstein-Uhlenbeck correlated exploration in action space
        a = a + ou_theta * (0.0 - a) + ou_sigma * rng.standard_normal(ACT_DIM)
        a = np.clip(a, -1, 1)
        s = env.get_state()
        _, _, _, _ = env.step(a)
        s2 = env.get_state()
        S[i], A[i], S2[i] = s, a, s2
        i += 1
    print("collected %d transitions in %.1fs" % (n_transitions, time.time() - t0))
    return S, A, S2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="babble.npz")
    args = ap.parse_args()
    S, A, S2 = collect(args.n, args.seed)
    np.savez_compressed(args.out, S=S, A=A, S2=S2)
    # quick stats on state coverage
    print("qpos range:", S[:, :2].min(0).round(2), S[:, :2].max(0).round(2))
    print("qvel std:", S[:, 2:4].std(0).round(2))
    print("act range:", S[:, 4:8].min(0).round(2), S[:, 4:8].max(0).round(2))
    print("saved", args.out)
