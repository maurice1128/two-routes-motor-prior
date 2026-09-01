# Two Routes to a Motor Prior

Code for *Two Routes to a Motor Prior: Guidance Versus a Frozen Body Model* — a controlled dissection of two ways to hand an RL agent a motor prior on MyoSuite musculoskeletal bodies, under one SAC backbone, with n=12 matched seeds throughout.

**Project page:** https://maurice1128.github.io/projects/motor-prior.html
**Paper:** [`two_routes_motor_prior.pdf`](https://maurice1128.github.io/assets/papers/two_routes_motor_prior.pdf) — manuscript as submitted, still under revision, not posted to any preprint server.

---

## The two routes

| | what it is | where it comes from |
|---|---|---|
| **Forward-model route** | a frozen body forward model `(s,a) → Δs`, used for short-horizon Dyna rollouts | learned once from task-agnostic, reward-free motor babbling (`imagine_pretrain.py`), then frozen |
| **Guidance route** | a distilled coach, a frozen teacher policy acting through an auxiliary distillation loss | a `blank` agent trained for 30,000 rewarded on-task steps (`train_teacher2.py`) |

Each route gets the adversarial control the other has: a random-initialized frozen model (`randprior`) and a random-initialized frozen teacher in the identical distillation loss (`randcoach`).

**One scope condition that travels with every reading of the forward-model arm:** its Dyna rollouts evaluate the environment's analytic reward at the imagined states. That arm holds a forward model *and* a known reward function, which no model-free arm has.

---

## Layout

```
src/                     the study proper
  run_reach.py           entry point; one function runs one condition at one seed
  sac_dyna.py            SAC with the short-horizon Dyna loop
  world_model.py         the forward model and its training
  imagine_pretrain.py    task-agnostic OU-noise babbling; writes the frozen prior
  train_teacher2.py      trains a teacher from blank for 30k rewarded steps
  myo_env.py             myoElbow wrapper
  myofinger_env.py       myoFinger wrapper
  myohand_env.py         myoHand wrapper
  arm_env.py             the toy planar arm used in pre-study pilots

launchers/               the invocations behind each reported condition
analysis/                recomputation, statistics and figures
RECOMPUTED.md            the recomputation register the paper cites
```

## Running one condition

```python
import run_reach
run_reach.run("prior", seed=0, total_steps=12000, utd=2,
              n_train_targets=8, eval_heldout=True)
```

Conditions are `blank`, `prior`, `coach`, `priorcoach`, `randprior`, and the withdrawal variants. Defaults are the single set in `run_reach.py`: hidden 128, batch 128, update-to-data 2, `real_ratio` 0.5, `dyna_every` 100, `dyna_starts` 200, k = 3, 8 train targets and 20 disjoint held-out targets on fixed seeds.

Each run writes one JSON per seed: `{"step", "eval_return", "eval_dist", "alpha"}` at six checkpoints (2k, 4k, 6k, 8k, 10k, 12k).

## Reproducing the statistics

`analysis/recompute_all.py` rebuilds the contrast register from the per-seed JSONs. The primary estimator is the paired *t* 95% CI, `mean(d) ± t₀.₉₇₅,ₙ₋₁ · sd(d)/√n`; a BCa bootstrap and an exact Wilcoxon accompany it as sensitivity checks and license no claim the primary estimator does not. `RECOMPUTED.md` is the output.

`analysis/make_figs.py` verifies every plotted contrast against that register before writing an image, with four train-target `coach(anneal) − blank` cells the exception — they are not in the register and are flagged as such.

---

## What is not here

Stated plainly, because the paper states it:

- **The run data.** 55 MB of per-seed JSONs across 55 condition directories, plus checkpoints. Not committed. `RECOMPUTED.md` carries every contrast computed from them.
- **Seven analysed conditions have no archived launcher.** `results_noanneal_*`, `results_abruptcoach_*`, `results_cleancoach_*`, `results_cleancoach_train_*`, `results_pcconst_*`, `results_prioroff_*`, `results_priorpurge_*` — fourteen directories in total. Their invocations are reconstructible from the run log and checkpoint structure but not from a committed script. This is disclosed in the paper's limitations.
- **The elbow prior's build transcript.** `prior_myo.pt` is loaded by every elbow `prior`, `randprior` and `priorcoach` run, but no log of its babbling or its open-loop evaluation exists. Both the ≈19 mrad fidelity figure and the elbow babble budget are unarchived point values.
- **`randcoach` is absent from `RECOMPUTED.md`.** That arm was added last; its contrasts are computed by `analysis/analyze_randcoach.py` under the primary estimator only, with no BCa or Wilcoxon check.

## Requirements

Python 3.10, PyTorch, [MyoSuite](https://github.com/MyoHub/myosuite) and MuJoCo. The muscle environments will not import without MyoSuite; the analysis scripts run without it.

## Citation

Not yet published. Please link the project page rather than citing a version number.
