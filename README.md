# What a Motor Prior Is Worth

Code, launchers, analysis and audit scripts for

> **What a Motor Prior Is Worth: A Frozen Body Model and a Teacher Compared to Convergence and Through Withdrawal.**
> Mu-Hua Wang, 2026. Manuscript prepared for *IEEE Transactions on Cognitive and Developmental Systems*; not posted to any preprint server.

**Project page:** https://maurice1128.github.io/projects/motor-prior.html
**Paper:** [`paper/what_a_motor_prior_is_worth.pdf`](paper/what_a_motor_prior_is_worth.pdf) (LaTeX source and figures alongside it).

This paper merges and supersedes two earlier drafts by the same author: *Two Routes to a Motor Prior* (the first version of this repository, which scored every arm at a 12,000-step endpoint) and a separate study of teacher withdrawal. Everything reported now is run to a converged budget of 100,000 steps or carried over with its raw per-seed data re-audited.

---

## The claim, in three parts

Two priors for one soft actor-critic learner on two MyoSuite bodies (myoElbow: 1 joint, six muscles; myoFinger: 4 joints, five muscles), twelve matched seeds, paired-*t* 95 % intervals, every number held-out.

| | what it is | where it comes from |
|---|---|---|
| **Body model** | a frozen forward model `(s,a) → s'` used for short-horizon Dyna rollouts; the arm is handed the environment's analytic reward at imagined states | learned once from reward-free Ornstein–Uhlenbeck babbling (`src/build_prior.py`), then frozen |
| **Teacher** | a frozen actor attached through a distillation term on the student's actor loss at constant weight | a `blank` agent trained for 30k rewarded steps (`src/train_teacher2.py`) |

1. **The body model helps where the body is hard.** Its advantage over model-free learning is a transient of the first 12k on the elbow (−4.09 [−11.11, +2.94] mrad at 100k) and stable on the finger (−40.29 [−57.95, −22.64] mm, significant at every checkpoint from 14k).
2. **Guidance is only as good as its teacher.** On the elbow the coached student keeps a small lead (−8.06 [−15.83, −0.28]); on the finger, whose teacher is within 3.2 mm of doing nothing, it ends +28.33 [+16.11, +40.56] worse than no aid.
3. **Withdrawing either prior costs no dependence.** With the learner's real-data signal held fixed, losing the body model costs +22.11 [+4.94, +39.28] mrad at 12k and +5.05 [−0.98, +11.08] at 100k. Replacing the teacher by a zero-information snapshot of the student shows the immediate drop is the deletion of a loss term, that no component grows with attachment across seven durations and two coaching regimes, and that what persists at 100k is the teacher's information (+8.16 [+2.17, +14.14]), not the deleted term (+2.13 [−3.64, +7.90]).

---

## Layout

```
src/                         the study proper (WM_BODY=myoelbow|myofinger selects the body)
  run_reach.py               entry point; one function runs one condition at one seed
  sac_dyna.py                SAC with the short-horizon Dyna loop
  world_model.py             the forward model and its training
  collect_babble.py          task-agnostic Ornstein-Uhlenbeck babbling
  build_prior.py             builds one frozen prior and measures its open-loop error
  train_teacher2.py          trains a teacher from blank for 30k rewarded steps
  run_prior_matched.py       body-model withdrawal: keep / soft / purge / matched (matched-replay) arms
  run_coach_anchor.py        teacher withdrawal: constant / abrupt / selfanchor, with --swap-offset
  train_video100k.py         re-runs a seed at 100k with the actor saved, for the videos
  withdrawal/                the teacher-withdrawal study as first run (anchor_sac, replace_sac,
                             attachment sweep, replacing regime, budget matching)
  myo_env.py, myofinger_env.py, myohand_env.py, arm_env.py   body wrappers

launchers/                   the invocations behind each reported table
  run_converged.ps1          six core arms x two bodies x 12 seeds to 100k        (Table I)
  run_matched_withdrawal.ps1, run_soft_arm.ps1   body-model withdrawal at 12k     (Table III)
  run_withdraw100k.ps1       both withdrawal decompositions to 100k               (Table VI)
  run_selfanchor_off2.ps1    selfanchor with the swap on loop step t_w-2 (the 12k table's arm)
  run_video100k.ps1          actors for the project videos

analysis/
  analyze_tcds.py            convergence + matched-withdrawal statistics
  make_figs_converged.py     Fig. 1 and Fig. 2 from the per-seed JSON
  withdrawal/                analysis and plots of the carried 12k experiments (Tables II, IV, V, VII)
  audits/                    seven scripts that recompute every table cell, significance mark,
                             prose number and Method hyperparameter from raw JSON and logs
  teacher_eval.py            scores a teacher checkpoint on train and held-out targets
  measure_elbow_prior.py     the elbow prior's open-loop error on both distributions

paper/                       paper.tex, refs.bib, figs/, and the PDF
records/                     HANDOFF.md (the verified-numbers register the project page is built
                             from) and the outputs of the two elbow-prior reproductions below
RECOMPUTED.md                the recomputation register of the first draft (kept for the record)
```

## Running one condition

```python
import os; os.environ["WM_BODY"] = "myofinger"
import run_reach
run_reach.run("prior", seed=0, total_steps=100000, utd=2,
              n_train_targets=8, eval_heldout=True, prior_path="prior_finger.pt")
```

Conditions are `blank`, `prior`, `coach`, `priorcoach`; `randprior` and `randcoach` are `prior` and `coach` with a random-initialised checkpoint. Defaults are the single set in `run_reach.py`: hidden 128, batch 128, update-to-data 2, `real_ratio` 0.5 (64 real + 64 imagined per update), 1,000 warm-up steps, 8 train targets and 20 disjoint held-out targets on fixed seeds. The constant-weight coach is `coach_anneal=10**8`. Each run writes one JSON per seed with `{"step", "eval_return", "eval_dist", "alpha"}` every 2,000 steps.

Runs are seed-deterministic: the 100k grid reproduces the earlier 12k study's arm means at its 12k checkpoint seed for seed (144/144), which is how the two budgets are compared on the same footing.

## Reproducing the statistics and the audits

`analysis/analyze_tcds.py` recomputes the convergence and body-model-withdrawal contrasts; `analysis/withdrawal/analyze_*.py` the teacher-withdrawal decompositions, attachment slopes (per-seed OLS against t_w, then a paired-*t* interval across seeds) and budget matching.

`analysis/audits/` is the check the paper was released under: every cell of the seven tables, every bold or starred mark, every interval and rounded figure in the prose, the recovery counts, the finger teacher's 169.1 mm against the 172.3 mm zero-activation no-op, and the Method's hyperparameters against `run_reach.py`, recomputed from the per-seed JSON and build logs. 475 items, 0 mismatches at release. The scripts carry the absolute paths of the machine they ran on at the top of each file; change `W` (this study's results) and `Z` (the carried study's) to run them elsewhere.

`records/HANDOFF.md` §4 lists every number the project page is allowed to show.

---

## Reproducing the elbow prior

The paper reports that no build or open-loop-evaluation transcript survives for
`prior_myo.pt`, the frozen model every myoElbow `prior`, `randprior` and
`priorcoach` run loads. The checkpoint survived, so both the figure and the
budget are recoverable from the artefact. Both reproductions are here.

**Its open-loop accuracy**, on the babble distribution the paper measured and on
the on-policy distribution it did not:

```bash
WM_BODY=myoelbow python analysis/measure_elbow_prior.py
```

Five-step qpos RMS is 19.34 mrad on babble, reproducing the ~19 mrad the paper
carries, and 24.86 mrad on-policy. At the three-step horizon the Dyna loop uses,
the two are 10.94 and 10.37, within 5% of each other. The on-policy rollouts are
driven by a single-seed agent, so that column is n = 1 in the policy.

**Its babble budget**, which the paper had called unquantified:

```bash
WM_BODY=myoelbow python src/build_prior_sizes.py prior_myo_sweep prior_myo_sizes.json
```

The 10^5 model comes out **bit-identical** to `prior_myo.pt` — equal element by
element across all 71,688 parameters, with the four stored normalizer statistics
matching — so the elbow budget is 10^5, against 2×10^5 on myoFinger and 3×10^5
on myoHand. Neighbouring budgets are not close: 3×10^4 gives 0.02518 rad at five
steps and 2×10^5 gives 0.00766, against the checkpoint's 0.01934.

A reproduction is not a transcript. It shows that a committed script at its own
defaults yields the artefact today; it does not record what was run on the day,
and it fixes seed 0, hidden 256 and 40 epochs because the script does.

Outputs of both are in `records/`.

## What is not here

Stated plainly, because the paper states it:

- **The run data.** The per-seed JSONs (1,044 runs behind the merged paper's tables) and checkpoints are not committed; `records/HANDOFF.md` §3 says where they live. `RECOMPUTED.md` carries every contrast computed from them.
- **Seven conditions of the first draft have no archived launcher.** Every table of the merged paper has a committed launcher (the 12k columns of Table I are the 12k checkpoints of the 100k runs). `results_noanneal_*`, `results_abruptcoach_*`, `results_cleancoach_*`, `results_cleancoach_train_*`, `results_pcconst_*`, `results_prioroff_*`, `results_priorpurge_*` — fourteen directories in total. Their invocations are reconstructible from the run log and checkpoint structure but not from a committed script. This is disclosed in the paper's limitations.
- **The elbow prior's build transcript.** `prior_myo.pt` is loaded by every elbow `prior`, `randprior` and `priorcoach` run, but no log of its babbling or its open-loop evaluation exists. Both the ≈19 mrad fidelity figure and the elbow babble budget are unarchived point values.
- **`randcoach` is absent from `RECOMPUTED.md`.** That arm was added last; its contrasts are computed by `analysis/analyze_randcoach.py` under the primary estimator only, with no BCa or Wilcoxon check.

## Requirements

Python 3.11, PyTorch, MuJoCo, MyoSuite, NumPy; matplotlib for the figures; imageio and Pillow for the videos. The study was run on CPU, one torch thread per run.
