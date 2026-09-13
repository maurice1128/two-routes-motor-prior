"""The separating control the TMLR paper names but did not run.

The problem
-----------
In the TMLR study the coach is withdrawn by setting its distillation weight to zero at
env-step 8000. That single act does two things at once:

  1. it removes the TEACHER'S INFORMATION from the objective, and
  2. it deletes an auxiliary loss term of weight about 1.0 in one step.

The paper's own Limitation 4 says so, says the confound pushes the prior's removal cost
toward zero, and says the separating control was not run. Any humanoid follow-up that
withdraws a distillation loss inherits the same confound, so the control has to exist
before the headline retention claim can mean anything.

The control
-----------
At the withdrawal step, instead of deleting the loss term, swap the teacher for a FROZEN
SNAPSHOT OF THE STUDENT'S OWN ACTOR taken at that moment. The loss term survives at the
same weight and the same functional form, but it now carries no external information: it
is an anchor to where the student already was. Then

    abrupt      - constant   = losing the teacher AND the loss term  (what the paper measured)
    selfanchor  - constant   = losing the teacher only
    abrupt      - selfanchor = losing the loss term only

and the third quantity is the confound, measured rather than assumed.

A second arm, `randanchor`, swaps in a randomly initialised actor instead. That is the
swap-at-8k version of the paper's `randcoach`, which the paper lists as a next step and
which bounds how much of any effect is "being pulled toward something arbitrary" rather
than "being pulled toward nothing".

How the step counter works
--------------------------
`run_reach.run` assigns `agent.coach_coef` exactly once per environment step whenever the
condition uses a coach. Making `coach_coef` a property therefore gives an exact env-step
counter without touching the frozen TMLR code, and the swap fires on the right step even
if the update-to-data ratio or the warmup length changes. Counting optimizer updates
instead would silently drift if either did.
"""
import copy
import os
import sys

import torch

WM_PRIOR_DIR = os.environ.get(
    "WM_PRIOR_DIR", r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
)
if WM_PRIOR_DIR not in sys.path:
    sys.path.insert(0, WM_PRIOR_DIR)
if os.path.abspath(os.getcwd()) != os.path.abspath(WM_PRIOR_DIR):
    os.chdir(WM_PRIOR_DIR)

from sac_dyna import SAC, Actor, DEV, OBS_DIM, ACT_DIM  # noqa: E402


def make_anchor_sac(swap_at, mode="self", hidden=128, verbose=True):
    """Return a SAC subclass that swaps its teacher at env-step `swap_at`.

    mode="self"   freeze a copy of the student's own actor at the swap step
    mode="random" freeze a freshly initialised actor
    mode="none"   never swap (identical to plain SAC; useful for a null check)
    """

    class AnchorSAC(SAC):
        def __init__(self, *a, **kw):
            self._env_step = 0
            self._swapped = False
            self._coach_coef = 0.0
            super().__init__(*a, **kw)

        # `run_reach.run` writes this once per env step, which is what we count.
        @property
        def coach_coef(self):
            return self._coach_coef

        @coach_coef.setter
        def coach_coef(self, value):
            self._coach_coef = value
            self._env_step += 1
            if not self._swapped and self._env_step >= swap_at and mode != "none":
                self._do_swap()

        def _do_swap(self):
            self._swapped = True
            if mode == "self":
                anchor = copy.deepcopy(self.actor)
            elif mode == "random":
                anchor = Actor(OBS_DIM, ACT_DIM, hidden).to(DEV)
            else:
                raise ValueError(f"unknown anchor mode {mode!r}")
            for p in anchor.parameters():
                p.requires_grad_(False)
            anchor.eval()
            self.teacher = anchor
            if verbose:
                print(f"    [anchor] env-step {self._env_step}: teacher -> {mode} snapshot, "
                      f"coach_coef stays {self._coach_coef:.4f}", flush=True)

    AnchorSAC.__name__ = f"AnchorSAC_{mode}_{swap_at}"
    return AnchorSAC


def install(run_reach_module, swap_at, mode, hidden=128, verbose=True):
    """Point `run_reach`'s SAC name at the anchor subclass. Returns the original class."""
    original = run_reach_module.SAC
    run_reach_module.SAC = make_anchor_sac(swap_at, mode, hidden, verbose)
    return original
