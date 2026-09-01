"""Forward dynamics world model: (s, a) -> delta_s, with input/output normalization.
Trains on babble data. Provides k-step rollout for evaluation and for Dyna imagination.
"""
import numpy as np, torch, torch.nn as nn

class Normalizer:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(np.clip(std, 1e-4, None), dtype=torch.float32)
    def norm(self, x): return (x - self.mean) / self.std
    def denorm(self, x): return x * self.std + self.mean
    def to(self, dev):
        self.mean = self.mean.to(dev); self.std = self.std.to(dev); return self

class WorldModel(nn.Module):
    def __init__(self, state_dim=8, act_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
        self.in_norm = None   # over [s,a]
        self.out_norm = None  # over delta_s
        self.state_dim = state_dim; self.act_dim = act_dim

    def fit_normalizers(self, S, A, S2):
        X = np.concatenate([S, A], 1); D = S2 - S
        self.in_norm = Normalizer(X.mean(0), X.std(0))
        self.out_norm = Normalizer(D.mean(0), D.std(0))

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        xn = self.in_norm.norm(x)
        dn = self.net(xn)
        return s + self.out_norm.denorm(dn)  # predicted next state

    def to(self, dev):
        super().to(dev)
        if self.in_norm: self.in_norm.to(dev); self.out_norm.to(dev)
        return self


def train_world_model(S, A, S2, hidden=256, epochs=30, bs=1024, lr=1e-3,
                      device="cpu", val_frac=0.1, verbose=True, seed=0):
    torch.manual_seed(seed)
    wm = WorldModel(S.shape[1], A.shape[1], hidden)
    wm.fit_normalizers(S, A, S2)
    wm.to(device)
    n = len(S); idx = np.random.default_rng(seed).permutation(n)
    nval = int(n * val_frac); vi, ti = idx[:nval], idx[nval:]
    St = torch.tensor(S, device=device); At = torch.tensor(A, device=device); S2t = torch.tensor(S2, device=device)
    opt = torch.optim.Adam(wm.parameters(), lr=lr)
    dn_target_v = wm.out_norm.norm(S2t[vi] - St[vi])
    for ep in range(epochs):
        wm.train(); perm = ti[np.random.default_rng(seed + ep).permutation(len(ti))]
        for b in range(0, len(perm), bs):
            j = perm[b:b + bs]
            pred = wm(St[j], At[j])
            dn_pred = wm.out_norm.norm(pred - St[j])
            dn_tgt = wm.out_norm.norm(S2t[j] - St[j])
            loss = ((dn_pred - dn_tgt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            wm.eval()
            with torch.no_grad():
                pv = wm(St[vi], At[vi])
                vloss = ((wm.out_norm.norm(pv - St[vi]) - dn_target_v) ** 2).mean().item()
            print(f"  ep{ep:3d} val_norm_mse {vloss:.4f}")
    return wm


def finetune_world_model(wm, S, A, S2, epochs=8, bs=512, lr=3e-4, device="cpu", seed=0):
    """Continue training an EXISTING world model on new (real) data, keeping its normalizers.
    Used for the 'warm-start' condition: a babble-pretrained model that keeps co-adapting
    on-task -- the middle point between frozen-prior and from-scratch co-learning."""
    wm.to(device); wm.train()
    St = torch.tensor(S, device=device); At = torch.tensor(A, device=device); S2t = torch.tensor(S2, device=device)
    opt = torch.optim.Adam(wm.parameters(), lr=lr)
    n = len(S)
    for ep in range(epochs):
        perm = np.random.default_rng(seed + ep).permutation(n)
        for b in range(0, n, bs):
            j = perm[b:b + bs]
            pred = wm(St[j], At[j])
            dn_pred = wm.out_norm.norm(pred - St[j])
            dn_tgt = wm.out_norm.norm(S2t[j] - St[j])
            loss = ((dn_pred - dn_tgt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    wm.eval()
    return wm


@torch.no_grad()
def eval_kstep(wm, S, A, S2_seq=None, k=5, device="cpu", n_traj=2000, horizon_data=None):
    """Open-loop k-step prediction error using consecutive transitions from a held-out
    contiguous rollout. horizon_data: (S_seq [N,T,8], A_seq [N,T,4]) ground-truth rollouts."""
    wm.eval()
    Ss, As = horizon_data
    Ss = torch.tensor(Ss[:n_traj], device=device); As = torch.tensor(As[:n_traj], device=device)
    s = Ss[:, 0]
    errs = []
    for t in range(k):
        s = wm(s, As[:, t])
        true = Ss[:, t + 1]
        # report per-dim RMS on qpos (first 2) which drives the task
        err_q = torch.sqrt(((s[:, :2] - true[:, :2]) ** 2).mean()).item()
        err_all = torch.sqrt(((s - true) ** 2).mean()).item()
        errs.append((t + 1, err_q, err_all))
    return errs
