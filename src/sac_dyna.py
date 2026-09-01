"""Compact SAC with an optional Dyna (model-based imagination) hook.

The three experimental conditions share this identical SAC; the ONLY difference is the
model source passed in:
  A 'blank'   : model=None                         -> pure model-free SAC (no prior)
  B 'prior'   : model=frozen pretrained WorldModel  -> Dyna with innate body prior
  C 'colearn' : model=WorldModel trained online     -> MBPO-style co-learning

Synthetic-transition rewards are computed with the exact reach reward (FK on predicted
qpos), which is legitimate because reward is a known function of state, not learned.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from arm_env import OBS_DIM, ACT_DIM, STATE_DIM, fk, reach_reward, JOINT_LO
from world_model import WorldModel
_NQ = len(JOINT_LO)

DEV = "cpu"

def mlp(sizes, act=nn.SiLU, out_act=nn.Identity, layernorm=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if layernorm: layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(act())
        else:
            layers.append(out_act())
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.body = mlp([obs_dim, hidden, hidden])
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
    def forward(self, o):
        h = self.body(o); mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), -5, 2)
        return mu, log_std
    def sample(self, o):
        mu, log_std = self(o); std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample(); a = torch.tanh(u)
        logp = dist.log_prob(u).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp
    @torch.no_grad()
    def act(self, o, greedy=False):
        mu, log_std = self(o)
        a = torch.tanh(mu) if greedy else torch.tanh(torch.distributions.Normal(mu, log_std.exp()).sample())
        return a

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, hidden, hidden, 1], layernorm=True)
        self.q2 = mlp([obs_dim + act_dim, hidden, hidden, 1], layernorm=True)
    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

class Replay:
    def __init__(self, cap, obs_dim, act_dim):
        self.o = np.zeros((cap, obs_dim), np.float32); self.a = np.zeros((cap, act_dim), np.float32)
        self.r = np.zeros(cap, np.float32); self.o2 = np.zeros((cap, obs_dim), np.float32)
        self.d = np.zeros(cap, np.float32); self.cap = cap; self.n = 0; self.i = 0
    def add(self, o, a, r, o2, d):
        self.o[self.i] = o; self.a[self.i] = a; self.r[self.i] = r; self.o2[self.i] = o2; self.d[self.i] = d
        self.i = (self.i + 1) % self.cap; self.n = min(self.n + 1, self.cap)
    def add_batch(self, o, a, r, o2, d):
        for k in range(len(o)): self.add(o[k], a[k], r[k], o2[k], d[k])
    def sample(self, bs):
        idx = np.random.randint(0, self.n, bs)
        t = lambda x: torch.tensor(x[idx], device=DEV)
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d)

def obs_from_state(state, target):
    """Rebuild policy obs [state(8), tip(2), target(2)] from dynamics state + target."""
    tip = fk(state[..., :_NQ])
    return np.concatenate([state, tip, target], axis=-1).astype(np.float32)

class SAC:
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256, gamma=0.99, tau=0.005,
                 lr=3e-4, target_entropy=None, seed=0):
        torch.manual_seed(seed); np.random.seed(seed)
        self.actor = Actor(obs_dim, act_dim, hidden).to(DEV)
        self.critic = Critic(obs_dim, act_dim, hidden).to(DEV)
        self.ctarg = Critic(obs_dim, act_dim, hidden).to(DEV)
        self.ctarg.load_state_dict(self.critic.state_dict())
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha = torch.tensor(np.log(0.3), requires_grad=True, device=DEV)
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = target_entropy if target_entropy is not None else -0.5 * act_dim
        self.gamma = gamma; self.tau = tau
        self.teacher = None; self.coach_coef = 0.0  # 'coach': frozen teacher actor + distillation aux loss

    def set_teacher(self, teacher_actor, coef=1.0):
        """teacher_actor: a frozen Actor; coach_coef weights the behavior-cloning aux loss."""
        self.teacher = teacher_actor
        for p in self.teacher.parameters(): p.requires_grad_(False)
        self.teacher.eval(); self.coach_coef = coef

    @property
    def alpha(self): return self.log_alpha.exp().item()

    def update(self, batch):
        o, a, r, o2, d = batch
        with torch.no_grad():
            a2, logp2 = self.actor.sample(o2)
            q1t, q2t = self.ctarg(o2, a2)
            qt = torch.min(q1t, q2t) - self.log_alpha.exp() * logp2
            y = r + self.gamma * (1 - d) * qt
        q1, q2 = self.critic(o, a)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad(); q_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0); self.q_opt.step()

        ap, logp = self.actor.sample(o)
        q1p, q2p = self.critic(o, ap)
        pi_loss = (self.log_alpha.exp().detach() * logp - torch.min(q1p, q2p)).mean()
        if self.teacher is not None and self.coach_coef > 0:
            # coach: pull the student's greedy action toward the teacher's on visited states
            with torch.no_grad():
                tmu, _ = self.teacher(o); ta = torch.tanh(tmu)
            smu, _ = self.actor(o); sa = torch.tanh(smu)
            pi_loss = pi_loss + self.coach_coef * ((sa - ta) ** 2).mean()
        self.pi_opt.zero_grad(); pi_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0); self.pi_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.a_opt.zero_grad(); alpha_loss.backward(); self.a_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.ctarg.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        return q_loss.item(), pi_loss.item()
