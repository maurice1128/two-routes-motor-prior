"""Imagination pretraining: an agent 'born' with a frozen body model practices ENTIRELY
in its head (zero real environment steps), then is deployed. This is the cleanest
instantiation of 'a good world model = talent': the innate model lets the agent acquire
the task off-line, so it needs far fewer REAL interactions than learning from scratch.

The imagined MDP uses:
  - start states sampled synthetically (valid qpos range, small qvel, random muscle act)
  - transitions from the FROZEN babble-trained world model
  - exact reach reward via FK on predicted qpos (reward is a known function, not learned)

Reports real-env performance as a function of imagined-practice budget, and (optionally)
a short real fine-tune to show adaptation.
"""
import numpy as np, torch, time, argparse, json, os
torch.set_num_threads(1)
from arm_env import (ArmReachEnv, OBS_DIM, ACT_DIM, STATE_DIM, fk, reach_reward,
                     task_reward, TASK_TARGET_SEED, JOINT_LO, JOINT_HI)
from sac_dyna import SAC, Replay, obs_from_state, DEV
from run_reach import load_prior, make_train_targets, evaluate

_NQ = len(JOINT_LO)
_NA = STATE_DIM - 2 * _NQ

def sample_start_states(n, rng):
    """Synthetic plausible dynamics states: qpos in range, small qvel, random activation."""
    q = rng.uniform(JOINT_LO, JOINT_HI, size=(n, _NQ))
    qd = rng.normal(0, 0.3, size=(n, _NQ))
    act = rng.uniform(0, 1, size=(n, _NA))
    return np.concatenate([q, qd, act], 1).astype(np.float32)

def imagine_pretrain(seed=0, imagined_steps=40000, horizon=100, batch=128, hidden=128,
                     utd=2, n_train_targets=8, eval_every=2000, prior_path="prior.pt",
                     real_finetune=0, task="reach", eval_heldout=False):
    from run_reach import fixed_eval_targets
    rng = np.random.default_rng(seed)
    model = load_prior(prior_path)
    agent = SAC(hidden=hidden, seed=seed)
    buf = Replay(300_000, OBS_DIM, ACT_DIM)
    train_targets = make_train_targets(n_train_targets, seed=TASK_TARGET_SEED.get(task, 777))
    # held-out generalization test: imagine-train on the 8 goals, score on 20 unseen goals
    eval_targets = fixed_eval_targets(20) if eval_heldout else train_targets
    curve = []
    t0 = time.time()

    # rolling batch of parallel imagined episodes
    B = 64
    s = sample_start_states(B, rng)
    tg = np.array([train_targets[i % len(train_targets)] for i in rng.integers(0, len(train_targets), B)])
    hstep = np.zeros(B, int)

    imagined = 0; warmup = 2000
    while imagined < imagined_steps:
        o = obs_from_state(s, tg)
        with torch.no_grad():
            a = agent.actor.act(torch.tensor(o, device=DEV), greedy=False).cpu().numpy()
            s2 = model(torch.tensor(s, dtype=torch.float32, device=DEV),
                       torch.tensor(a, dtype=torch.float32, device=DEV)).cpu().numpy()
        r = task_reward(s2, tg, np.clip((a + 1.0) * 0.5, 0.0, 1.0), task)  # applied activation, matches real env
        hstep += 1
        o2 = obs_from_state(s2, tg)
        # reach is a pure time-limit task: store done=False so SAC BOOTSTRAPS across the
        # branch cut. This is what lets SHORT imagined rollouts still learn long-horizon
        # reaching (MBPO-style) while limiting how far model error can compound.
        buf.add_batch(o, a, r, o2, np.zeros(len(o), np.float32))
        imagined += B
        # resample fresh start state after `horizon` imagined steps (branch length)
        s = s2
        fin = hstep >= horizon
        if fin.any():
            idx = np.where(fin)[0]
            s[idx] = sample_start_states(len(idx), rng)
            tg[idx] = np.array([train_targets[j] for j in rng.integers(0, len(train_targets), len(idx))])
            hstep[idx] = 0

        # SAC updates: keep update-to-imagined-transition ratio = utd (as in real runs)
        if buf.n >= batch and imagined >= warmup:
            for _ in range(B * utd):
                agent.update(buf.sample(batch))

        if imagined % eval_every < B:
            mret, mdist = evaluate(agent, eval_targets, randomize_pose=False)
            curve.append({"imagined_steps": imagined, "real_steps": 0,
                          "eval_return": mret, "eval_dist": mdist})
            print(f"[imag s{seed}] imagined {imagined:6d} (0 real)  eval_ret {mret:8.2f}  "
                  f"dist {mdist*1000:6.1f}mm  ({time.time()-t0:.0f}s)")

    # optional: short real fine-tune to show adaptation, logged vs REAL steps
    if real_finetune > 0:
        env = ArmReachEnv(seed=seed)
        trng = np.random.default_rng(seed + 555)
        pick = lambda: train_targets[trng.integers(len(train_targets))]
        o = env.reset(target=pick(), randomize_pose=False); rbuf = Replay(100_000, OBS_DIM, ACT_DIM)
        for rs in range(1, real_finetune + 1):
            a = agent.actor.act(torch.tensor(o[None], device=DEV), greedy=False).cpu().numpy()[0]
            o2, r, d, inf = env.step(a)
            r = float(task_reward(o2[:STATE_DIM], env.target, inf["ctrl"], task))
            rbuf.add(o, a, r, o2, float(d)); o = o2
            if d: o = env.reset(target=pick(), randomize_pose=False)
            if rbuf.n >= batch:
                for _ in range(utd):
                    # mix real + imagined
                    nr = batch // 2
                    ro = rbuf.sample(nr); io = buf.sample(batch - nr)
                    agent.update(tuple(torch.cat([ro[i], io[i]], 0) for i in range(5)))
            if rs % eval_every == 0:
                mret, mdist = evaluate(agent, eval_targets, randomize_pose=False)
                curve.append({"imagined_steps": imagined, "real_steps": rs,
                              "eval_return": mret, "eval_dist": mdist})
                print(f"[imag+ft s{seed}] real {rs:6d}  eval_ret {mret:8.2f}  dist {mdist*1000:6.1f}mm")
    return curve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imagined", type=int, default=40000)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--finetune", type=int, default=0)
    ap.add_argument("--ntargets", type=int, default=8)
    ap.add_argument("--task", default="reach", choices=["reach", "reachB", "hold"])
    ap.add_argument("--prior", default="prior.pt")
    ap.add_argument("--heldout", action="store_true", help="score on 20 disjoint held-out targets (generalization test)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    curve = imagine_pretrain(seed=args.seed, imagined_steps=args.imagined, horizon=args.horizon,
                             real_finetune=args.finetune, n_train_targets=args.ntargets,
                             task=args.task, prior_path=args.prior, eval_heldout=args.heldout)
    tag = "" if args.task == "reach" else f"_{args.task}"
    fn = os.path.join(args.out, f"imag{tag}_seed{args.seed}.json")
    with open(fn, "w") as f: json.dump(curve, f)
    print("saved", fn)
