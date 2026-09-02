"""Re-measure the myoElbow prior's open-loop accuracy, on BOTH distributions.

§3.3 reports "myoElbow ~1.1 deg / 0.019 rad (~19 mrad) joint error" as an
unarchived point value: prior_myo.pt is on disk, but no build or evaluation
transcript for it exists anywhere in the tree. The checkpoint survived, so the
number is recoverable -- this recovers it.

It also measures the thing the paper concedes it never measured. Every fidelity
figure in the paper was taken on the BABBLE distribution, which is not where the
model is queried; the model is queried on states an RL policy visits. That gap
is disclosed as an optimistic bound. Here both are measured, so the bound can be
replaced by the two numbers it stands between.

Protocol follows build_prior.py exactly -- contiguous held-out rollouts, k-step
open-loop, qpos RMS -- with the OU action process build_prior.py uses and the
persistence baseline it prints for scale. The only change is the state
distribution the rollouts start from and the actions that drive them.

Run under .venv_myo (the only environment with myosuite).
"""
import numpy as np, torch, time, json, sys

from myo_env import ArmReachEnv, STATE_DIM, ACT_DIM, _NQ
from world_model import WorldModel, Normalizer

K = 6
N_TRAJ = 1000
EVAL_SEED = 999


def load_prior(path="prior_myo.pt"):
    d = torch.load(path, map_location="cpu", weights_only=False)
    wm = WorldModel(STATE_DIM, ACT_DIM, hidden=d["hidden"])
    wm.load_state_dict(d["state_dict"])
    wm.in_norm = Normalizer(torch.tensor(d["in_mean"]), torch.tensor(d["in_std"]))
    wm.out_norm = Normalizer(torch.tensor(d["out_mean"]), torch.tensor(d["out_std"]))
    wm.eval()
    return wm


def rollouts_babble(n=N_TRAJ, T=K, seed=EVAL_SEED):
    """Held-out contiguous rollouts under the OU babble process, from random poses.
    This is the distribution every fidelity figure in the paper was measured on."""
    env = ArmReachEnv(seed=seed)
    rng = np.random.default_rng(seed)
    S = np.zeros((n, T + 1, STATE_DIM), np.float32)
    A = np.zeros((n, T, ACT_DIM), np.float32)
    for i in range(n):
        env.reset(randomize_pose=True)
        a = rng.uniform(-1, 1, ACT_DIM)
        S[i, 0] = env.get_state()
        for t in range(T):
            a = np.clip(a + 0.15 * (-a) + 0.4 * rng.standard_normal(ACT_DIM), -1, 1)
            env.step(a)
            A[i, t] = a
            S[i, t + 1] = env.get_state()
    return S, A


def rollouts_onpolicy(actor_path, n=N_TRAJ, T=K, seed=EVAL_SEED):
    """Held-out contiguous rollouts driven by a trained prior agent, from the
    task's own start pose and targets. This is where the model is actually
    queried during a run, and no figure in the paper was measured here."""
    from run_reach import fixed_eval_targets, load_teacher
    actor = load_teacher(actor_path)
    targets = fixed_eval_targets(20)
    env = ArmReachEnv(seed=seed)
    rng = np.random.default_rng(seed)
    S = np.zeros((n, T + 1, STATE_DIM), np.float32)
    A = np.zeros((n, T, ACT_DIM), np.float32)
    i = 0
    while i < n:
        tgt = targets[rng.integers(len(targets))]
        obs = env.reset(target=tgt, randomize_pose=False)
        # advance a random number of steps so the sampled states cover the
        # episode, not only its first frames
        for _ in range(int(rng.integers(0, 90))):
            with torch.no_grad():
                a = actor.act(torch.tensor(obs[None], dtype=torch.float32), greedy=True).numpy()[0]
            obs, *_ = env.step(a)
        S[i, 0] = env.get_state()
        ok = True
        for t in range(T):
            with torch.no_grad():
                a = actor.act(torch.tensor(obs[None], dtype=torch.float32), greedy=True).numpy()[0]
            obs, *_ = env.step(a)
            A[i, t] = a
            S[i, t + 1] = env.get_state()
        if ok:
            i += 1
    return S, A


def kstep(wm, S, A, label):
    St = torch.tensor(S); At = torch.tensor(A)
    out = []
    with torch.no_grad():
        s = St[:, 0]
        for t in range(K):
            s = wm(s, At[:, t])
            true = St[:, t + 1]
            rms_q = torch.sqrt(((s[:, :_NQ] - true[:, :_NQ]) ** 2).mean()).item()
            rms_all = torch.sqrt(((s - true) ** 2).mean()).item()
            pers_q = torch.sqrt(((St[:, 0, :_NQ] - true[:, :_NQ]) ** 2).mean()).item()
            out.append({"k": t + 1, "qpos_rms_rad": rms_q, "qpos_rms_mrad": rms_q * 1000,
                        "qpos_rms_deg": np.degrees(rms_q),
                        "persistence_mrad": pers_q * 1000, "full_state_rms": rms_all})
            print("  %-9s k=%d  qpos RMS %8.5f rad = %6.2f mrad = %5.3f deg   "
                  "(persistence %6.2f mrad)  full-state RMS %.4f"
                  % (label, t + 1, rms_q, rms_q * 1000, np.degrees(rms_q),
                     pers_q * 1000, rms_all))
    return out


if __name__ == "__main__":
    print("== myoElbow prior: prior_myo.pt ==")
    wm = load_prior()
    rec = {"checkpoint": "prior_myo.pt", "k": K, "n_traj": N_TRAJ, "eval_seed": EVAL_SEED}

    print("\n== babble distribution (what the paper measured) ==")
    t0 = time.time()
    Sb, Ab = rollouts_babble()
    rec["babble"] = kstep(wm, Sb, Ab, "babble")
    print("  collected in %.1fs" % (time.time() - t0))

    actor = sys.argv[1] if len(sys.argv) > 1 else "actors_video/prior_actor.pt"
    print("\n== on-policy distribution, %s (what the paper never measured) ==" % actor)
    try:
        t0 = time.time()
        So, Ao = rollouts_onpolicy(actor)
        rec["on_policy"] = kstep(wm, So, Ao, "on-policy")
        rec["on_policy_actor"] = actor
        print("  collected in %.1fs" % (time.time() - t0))
    except Exception as e:
        print("  on-policy pass FAILED: %r" % (e,))
        rec["on_policy_error"] = repr(e)

    with open("measure_elbow_prior.json", "w") as f:
        json.dump(rec, f, indent=1)
    print("\nwrote measure_elbow_prior.json")
