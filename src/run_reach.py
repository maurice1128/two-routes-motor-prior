"""Part (b): sample-efficiency comparison on the reach task.

Conditions (identical SAC, only the model source differs):
  blank    : no model            (model-free SAC, blank slate)
  prior    : frozen pretrained WM (innate body prior)   <- our method
  colearn  : WM trained online    (MBPO-style co-learning)

Logs eval return / final-distance vs real env steps. Multi-seed via --seed.
"""
import numpy as np, torch, time, argparse, json, os
torch.set_num_threads(1)  # keep each run to ~1 core so many can run in parallel cleanly
from arm_env import (ArmReachEnv, OBS_DIM, ACT_DIM, STATE_DIM, fk, reach_reward,
                     task_reward, TASK_TARGET_SEED, JOINT_LO, JOINT_HI)
from world_model import WorldModel, train_world_model, finetune_world_model
from sac_dyna import SAC, Replay, obs_from_state, DEV

def load_prior(path="prior.pt", freeze=True):
    ck = torch.load(path, weights_only=False)
    wm = WorldModel(STATE_DIM, ACT_DIM, ck["hidden"])
    from world_model import Normalizer
    wm.in_norm = Normalizer(ck["in_mean"], ck["in_std"])
    wm.out_norm = Normalizer(ck["out_mean"], ck["out_std"])
    wm.load_state_dict(ck["state_dict"]); wm.eval()
    if freeze:
        for p in wm.parameters(): p.requires_grad_(False)
    return wm

def fixed_eval_targets(n=20, seed=12345):
    rng = np.random.default_rng(seed)
    return [fk(rng.uniform(JOINT_LO, JOINT_HI)) for _ in range(n)]

def load_teacher(path, hidden=128):
    """Load a frozen teacher Actor (a strong pretrained policy) for the coach conditions."""
    from sac_dyna import Actor
    t = Actor(OBS_DIM, ACT_DIM, hidden).to(DEV)
    t.load_state_dict(torch.load(path, weights_only=True)); t.eval()
    return t

def evaluate(agent, targets, max_steps=100, seed=7, randomize_pose=False):
    env = ArmReachEnv(seed=seed)
    rets, dists = [], []
    for tg in targets:
        o = env.reset(target=tg, randomize_pose=randomize_pose); R = 0
        for _ in range(max_steps):
            a = agent.actor.act(torch.tensor(o[None], device=DEV), greedy=True).cpu().numpy()[0]
            o, r, d, inf = env.step(a); R += r
        rets.append(R); dists.append(inf["dist"])
    return float(np.mean(rets)), float(np.mean(dists))

@torch.no_grad()
def dyna_rollout(model, agent, real_buf, n_start=400, k=3, task="reach"):
    """Imagine k-step transitions from sampled real states; exact task reward from state."""
    if real_buf.n < n_start: return None
    idx = np.random.randint(0, real_buf.n, n_start)
    gdim = (OBS_DIM - STATE_DIM) // 2               # goal/target dim (2 for arm, 3 for finger)
    state = real_buf.o[idx, :STATE_DIM].copy()      # dynamics state
    target = real_buf.o[idx, STATE_DIM+gdim:STATE_DIM+2*gdim].copy()  # target dims in obs
    out = []
    s = state
    for _ in range(k):
        o = obs_from_state(s, target)
        a = agent.actor.act(torch.tensor(o, device=DEV), greedy=False).cpu().numpy()
        s2 = model(torch.tensor(s, dtype=torch.float32, device=DEV),
                   torch.tensor(a, dtype=torch.float32, device=DEV)).cpu().numpy()
        r = task_reward(s2, target, np.clip((a + 1.0) * 0.5, 0.0, 1.0), task)  # applied activation, matches real env
        o2 = obs_from_state(s2, target)
        out.append((o, a, r, o2, np.zeros(len(o), np.float32)))
        s = s2
    return out

def make_train_targets(n, seed=777):
    """Fixed set of n reachable targets used for training resets (n=0 -> fresh random each ep)."""
    rng = np.random.default_rng(seed)
    return [fk(rng.uniform(JOINT_LO, JOINT_HI)) for _ in range(max(n, 1))]

def run(condition, seed=0, total_steps=40000, start_steps=1000, batch=128, hidden=128,
        real_ratio=0.5, dyna_every=100, k=3, dyna_starts=200, utd=2, n_train_targets=8,
        fixed_start=True, model_retrain_every=2000, eval_every=2000, prior_path="prior.pt",
        task="reach", eval_heldout=False, coach_path=None, coach0=1.0, coach_anneal=8000, coach_abrupt=False, prior_off=10**9, prior_purge=False,
        save_actor_path=None):
    assert condition in ("blank", "prior", "colearn", "warm", "coach", "priorcoach")
    uses_prior = condition in ("prior", "priorcoach")
    uses_model = condition in ("prior", "colearn", "warm", "priorcoach")
    uses_coach = condition in ("coach", "priorcoach")
    env = ArmReachEnv(seed=seed)
    agent = SAC(hidden=hidden, seed=seed)
    if uses_coach:
        agent.set_teacher(load_teacher(coach_path, hidden), coef=coach0)
    real = Replay(200_000, OBS_DIM, ACT_DIM)
    model_buf = Replay(400_000, OBS_DIM, ACT_DIM) if uses_model else None
    train_targets = make_train_targets(n_train_targets, seed=TASK_TARGET_SEED.get(task, 777))
    # eval_heldout=True -> score on a DISJOINT held-out target set (seed 12345 != train seed),
    # the honest generalization test (train on 8 goals, eval on 20 unseen goals).
    eval_targets = fixed_eval_targets(20) if eval_heldout else (
        train_targets if n_train_targets > 0 else fixed_eval_targets())
    rp = (not fixed_start)  # randomize_pose
    trng = np.random.default_rng(seed + 555)
    pick = lambda: train_targets[trng.integers(len(train_targets))] if n_train_targets > 0 else None

    model = None
    if uses_prior:
        model = load_prior(prior_path, freeze=True)
    elif condition == "warm":
        model = load_prior(prior_path, freeze=False)  # babble-pretrained, keeps co-adapting

    o = env.reset(target=pick(), randomize_pose=rp); ep_ret = 0; curve = []
    t0 = time.time()
    for step in range(1, total_steps + 1):
        if uses_coach:
            if coach_abrupt:   # matched-withdrawal control: hold full weight, then cut to 0
                agent.coach_coef = coach0 if step < coach_anneal else 0.0
            else:              # kickstarting: linear taper from step 0 to coach_anneal
                agent.coach_coef = coach0 * max(0.0, 1.0 - step / coach_anneal)
        if step < start_steps:
            a = env.rng.uniform(-1, 1, ACT_DIM)
        else:
            a = agent.actor.act(torch.tensor(o[None], device=DEV), greedy=False).cpu().numpy()[0]
        o2, r, d, inf = env.step(a)
        r = float(task_reward(o2[:STATE_DIM], env.target, inf["ctrl"], task))  # task-specific
        ep_ret += r
        real.add(o, a, r, o2, float(d)); o = o2
        if d: o = env.reset(target=pick(), randomize_pose=rp); ep_ret = 0

        # build/refresh co-learned model (from scratch), or fine-tune warm-started model
        if step >= start_steps and (step - start_steps) % model_retrain_every == 0:
            S = real.o[:real.n, :STATE_DIM]; A = real.a[:real.n]; S2 = real.o2[:real.n, :STATE_DIM]
            if condition == "colearn":
                model = train_world_model(S, A, S2, hidden=256, epochs=8, bs=512,
                                          device=DEV, verbose=False, seed=seed)
                for p in model.parameters(): p.requires_grad_(False)
            elif condition == "warm":
                finetune_world_model(model, S, A, S2, epochs=8, bs=512, device=DEV, seed=seed)

        # Dyna imagination
        if step == prior_off and model_buf is not None and prior_purge:
            model_buf.n = 0  # strict withdrawal: also purge previously imagined transitions
        if model is not None and step >= start_steps and step < prior_off and step % dyna_every == 0:
            roll = dyna_rollout(model, agent, real, n_start=dyna_starts, k=k, task=task)
            if roll:
                for (oo, aa, rr, oo2, dd) in roll:
                    model_buf.add_batch(oo, aa, rr, oo2, dd)

        # SAC updates: utd gradient steps per env step (identical count across conditions)
        if step >= start_steps:
            for _ in range(utd):
                if condition == "blank" or model_buf is None or model_buf.n < batch:
                    agent.update(real.sample(batch))
                else:
                    nr = int(batch * real_ratio)
                    ro = real.sample(nr); mo = model_buf.sample(batch - nr)
                    mix = tuple(torch.cat([ro[i], mo[i]], 0) for i in range(5))
                    agent.update(mix)

        if step % eval_every == 0:
            mret, mdist = evaluate(agent, eval_targets, randomize_pose=rp)
            curve.append({"step": step, "eval_return": mret, "eval_dist": mdist, "alpha": agent.alpha})
            print(f"[{condition} s{seed}] step {step:6d}  eval_ret {mret:8.2f}  dist {mdist*1000:6.1f}mm  "
                  f"alpha {agent.alpha:.3f}  ({time.time()-t0:.0f}s)")
    if save_actor_path is not None:
        torch.save(agent.actor.state_dict(), save_actor_path)
        print("saved teacher actor ->", save_actor_path)
    return curve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["blank", "prior", "colearn", "warm", "coach", "priorcoach"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--utd", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ntargets", type=int, default=8)
    ap.add_argument("--task", default="reach", choices=["reach", "reachB", "hold"])
    ap.add_argument("--prior", default="prior.pt")
    ap.add_argument("--coach", default=None, help="teacher actor .pt for coach/priorcoach conditions")
    ap.add_argument("--heldout", action="store_true", help="evaluate on 20 disjoint held-out targets (generalization test)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--coach_abrupt", action="store_true", help="hold coach weight at coach0 until --coach_anneal then cut to 0 (matched withdrawal, no taper)")
    ap.add_argument("--prior_purge", action="store_true", help="at --prior_off, ALSO purge the imagined-transition buffer (strict withdrawal)")
    ap.add_argument("--prior_off", type=int, default=10**9,
                    help="stop Dyna use of the frozen prior after N steps (symmetric withdrawal control)")
    ap.add_argument("--coach_anneal", type=int, default=8000,
                    help="LINEAR TAPER HORIZON: weight = coach0*max(0,1-step/coach_anneal), recomputed every step, so the "
                         "weight starts fading at step 0 and reaches 0 here. This is a fade, NOT a removal at this step. "
                         "For a genuinely CONSTANT coach set this far above --steps (e.g. 1e8/1e9); setting it merely "
                         ">= --steps is NOT constant (at --coach_anneal 12000 with --steps 12000 the weight is still 0.5 "
                         "at 6k). For a step-function removal at step N use --coach_abrupt --coach_anneal N.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    curve = run(args.cond, seed=args.seed, total_steps=args.steps, utd=args.utd,
                hidden=args.hidden, batch=args.batch, n_train_targets=args.ntargets,
                task=args.task, prior_path=args.prior, eval_heldout=args.heldout, coach_path=args.coach,
                coach_anneal=args.coach_anneal, coach_abrupt=args.coach_abrupt, prior_off=args.prior_off, prior_purge=args.prior_purge)
    tag = "" if args.task == "reach" else f"_{args.task}"
    fn = os.path.join(args.out, f"{args.cond}{tag}_seed{args.seed}.json")
    with open(fn, "w") as f: json.dump(curve, f)
    print("saved", fn)
