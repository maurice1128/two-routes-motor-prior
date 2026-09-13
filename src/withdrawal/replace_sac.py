"""A coach that REPLACES the student's own error signal instead of adding to it.

Why
---
The attachment sweep (research/12, §1) found that the genuine-guidance component of the
withdrawal cost DECAYS with attachment duration on myoElbow and is ~0 throughout on
myoFinger, the opposite of the guidance hypothesis. The explanation offered there is
mechanistic: in the TMLR harness the distillation loss is ADDED to the SAC actor loss, so
the student keeps learning from its own reward gradient the whole time and there is
nothing for dependence to accumulate on. In the human guidance experiments (and in
RSL-RL's `Distillation`), the aid REPLACES the learner's own error signal.

That explanation makes a prediction that this file lets us test: if the coach replaces
the actor's reward gradient while attached, the guidance component should no longer
decay with attachment (the actor never practices learning from its own signal), so the
2x2 {additive, replacing} x {abrupt, selfanchor} should separate along the attachment
axis in a way the additive regime alone cannot.

What it does
------------
Subclass of the frozen `sac_dyna.SAC` that overrides `update` with one change: the SAC
actor term (entropy-regularised Q maximisation) is multiplied by `q_weight`, which is 0
while the EXTERNAL teacher is attached and 1 from the withdrawal step on. Everything else
(critic training on rewards, alpha update, target update) is untouched, so the student
still SEES rewards; it just does not use them to move its actor until withdrawal.

Arms, with `w` = withdrawal step and `coach_coef` driven by run_reach as before:
  constant    q_weight = 0 for the whole run, real teacher throughout (pure distillation)
  abrupt      at w: coach_coef -> 0 (run_reach), q_weight -> 1      => pure SAC afterwards
  selfanchor  at w: teacher -> frozen self snapshot, coach_coef stays, q_weight -> 1
              => SAC + BC-to-self afterwards (the loss term survives, as in the additive
                 selfanchor; the student additionally gains its own signal, which in the
                 replacing regime is inseparable from losing the teacher)

Contrasts keep their meaning:
  abrupt - selfanchor  = losing the loss term only            (same confound as before)
  selfanchor - constant = losing the teacher (and, in this regime, gaining own signal)

The env-step counter is the same trick as anchor_sac.py: run_reach writes `coach_coef`
exactly once per env step.
"""
import copy
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

WM_PRIOR_DIR = os.environ.get(
    "WM_PRIOR_DIR", r"C:\Users\maurice\Desktop\robotic_research\wm_prior")
if WM_PRIOR_DIR not in sys.path:
    sys.path.insert(0, WM_PRIOR_DIR)
if os.path.abspath(os.getcwd()) != os.path.abspath(WM_PRIOR_DIR):
    os.chdir(WM_PRIOR_DIR)

from sac_dyna import SAC, Actor, DEV, OBS_DIM, ACT_DIM  # noqa: E402


def make_replace_sac(swap_at, mode="none", hidden=128, verbose=True):
    """SAC whose actor ignores its own Q-gradient while the external teacher is attached.

    mode="none"    never swap the teacher (constant / abrupt arms; abrupt's coach_coef is
                   cut by run_reach, and q_weight turns on at swap_at either way)
    mode="self"    at swap_at, replace the teacher by a frozen copy of the student's actor
    """

    class ReplaceSAC(SAC):
        def __init__(self, *a, **kw):
            self._env_step = 0
            self._swapped = False
            self._coach_coef = 0.0
            self.q_weight = 0.0
            super().__init__(*a, **kw)

        @property
        def coach_coef(self):
            return self._coach_coef

        @coach_coef.setter
        def coach_coef(self, value):
            self._coach_coef = value
            self._env_step += 1
            if not self._swapped and self._env_step >= swap_at:
                self._swapped = True
                self.q_weight = 1.0
                if mode == "self":
                    anchor = copy.deepcopy(self.actor)
                    for p in anchor.parameters():
                        p.requires_grad_(False)
                    anchor.eval()
                    self.teacher = anchor
                if verbose:
                    print(f"    [replace] env-step {self._env_step}: q_weight 0 -> 1, "
                          f"teacher {'-> self snapshot' if mode == 'self' else 'unchanged'}, "
                          f"coach_coef {self._coach_coef:.3f}", flush=True)

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
            sac_term = (self.log_alpha.exp().detach() * logp - torch.min(q1p, q2p)).mean()
            pi_loss = self.q_weight * sac_term          # <- the only change vs sac_dyna.SAC
            if self.teacher is not None and self.coach_coef > 0:
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

    ReplaceSAC.__name__ = f"ReplaceSAC_{mode}_{swap_at}"
    return ReplaceSAC
