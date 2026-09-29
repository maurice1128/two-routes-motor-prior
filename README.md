# What a Motor Prior Is Worth

Code, launchers, analysis and audit scripts for

> **What a Motor Prior Is Worth: A Body Model and a Teacher Compared to Convergence and Through Withdrawal.**
> Mu-Hua Wang, 2026. Manuscript prepared for *IEEE Transactions on Cognitive and Developmental Systems*; not posted to any preprint server.

**Project page:** https://maurice1128.github.io/projects/motor-prior.html
**Paper:** [`paper/what_a_motor_prior_is_worth.pdf`](paper/what_a_motor_prior_is_worth.pdf) (LaTeX source, generated tables and figures alongside it).

This paper merges and supersedes two earlier drafts by the same author: *Two Routes to a Motor Prior* (the first version of this repository, which scored every condition at a 12,000-step endpoint) and a separate study of teacher withdrawal. Everything reported is run to 100,000 steps with a per-condition plateau check.

---

## The claim

One soft actor-critic learner on six muscle-driven bodies — myoElbow (1 joint, six muscles, mrad), myoFinger (4 joints, five muscles, mm) and four planar muscle arms of one to four joints (mm) — given one of two frozen priors. At least twelve matched seeds per condition (24 on the 3- and 4-joint arms), paired-*t* 95 % intervals, every number on held-out targets.

| | what it is | where it comes from |
|---|---|---|
| **Body model** | a frozen forward model `(s,a) → s'` used for short-horizon Dyna rollouts; the condition is handed the environment's analytic reward at imagined states | learned once from reward-free Ornstein–Uhlenbeck babbling (`src/build_prior.py`), then frozen |
| **Teacher** | a frozen actor attached through a distillation term `c·‖tanh μ_s − tanh μ_T‖²`, added to the student's own gradient or replacing it until withdrawal | T1: a model-free actor after 30k rewarded steps (`src/train_teacher2.py`); T2 (finger): the actor of a 100k body-model run |

**Main result — the withdrawal cost decomposes, and what remains at convergence is not dependence.**
Replacing a withdrawn teacher by a frozen snapshot of the student (`selfanchor`: the loss term stays, no new information) separates the two things withdrawal removes. The teacher's share of the immediate drop *falls* with attachment on both bodies (−6.23 [−7.79, −4.68] mrad and −4.65 [−7.05, −2.25] mm per 1,000 attached steps); the deleted term's share does not. Once training has converged, a learner that lost its teacher is never detectably worse than one never guided — 22 cells over two bodies, seven attachment durations and two teaching regimes — and with a competent teacher it is ahead, by more the longer it was guided (−8.44 [−9.89, −6.99] mm per 1,000 steps). A withdrawn body model can leave the learner short of one that kept it (finger +15.31 [+5.70, +24.91] mm) but never behind one that never had it (finger −24.99 [−44.86, −5.11]).

**Second result — a prior is worth its content.** The babble model beats model-free learning on four of six bodies and a random model on all six; the random model's harm grows with joint count (+0.41 [+0.18, +0.63] of model-free error per joint), the trained model's benefit does not. A teacher is worth its competence: on the finger a near-no-op teacher leaves the student +28.33 [+16.11, +40.56] mm behind no aid, a competent one puts it ahead of every condition without that teacher.

---

## Layout

```
src/                         the study proper (WM_BODY=myoelbow|myofinger|arm1..arm4 selects the body)
  run_reach.py               entry point; one function runs one condition at one seed
  sac_dyna.py                SAC with the short-horizon Dyna loop
  world_model.py, collect_babble.py, build_prior.py, build_prior_sizes.py   the frozen body model
  build_randprior.py         the random-initialised control model
  train_teacher2.py          trains a teacher from blank for 30k rewarded steps
  run_blank64.py             model-free SAC with 64 distinct real transitions per update (= batch 64)
  run_prior_matched.py       body-model withdrawal: keep / soft / purge / matched
  run_withdrawal100k.py      teacher withdrawal to 100k: constant / abrupt / selfanchor / randanchor / none
  run_replace_T2.py          the replacing regime with teacher T2 (wraps withdrawal/run_replace_sweep.py)
  withdrawal/                anchor_sac, replace_sac, attachment sweep, replacing regime, budget matching
  arm_env.py + arm{1..4}_muscle.xml, myo_env.py, myofinger_env.py, myohand_env.py   bodies

launchers/                   the invocations behind each table (PowerShell, resumable, multi-pass)
  run_converged.ps1          core MyoSuite grid to 100k (T2, blank64: run_v3_batch.ps1) (Table I)
  run_arms100k.ps1, run_arms100k_more.ps1, run_arms100k_prior2.ps1   joint-count series  (Table II)
  run_withdraw100k.ps1, run_sweep100k.ps1, run_v3_batch.ps1   teacher withdrawal       (Tables III, IV)
  run_v3c_gaps.ps1           finger replacing regime and finger body-model withdrawal   (Tables V, VI)

analysis/
  v3/make_tables_v3.py, v3/make_table_replace.py   Tables II, IV, V generated from the per-seed JSON
  v3/make_figs_v3.py         the three figures
  v3/*_stats.py, v3/v3*.py   the statistics behind every table and prose number
  v3/count_runs_v3.py        run count and CPU time
  audits/audit_v3b.py        recomputes every table cell, star, interval and worded claim of the paper
  audits/v2/                 the audits of the previous version, kept for the record
  videos/                    the explainer and comparison videos on the project page

paper/                       paper.tex, refs.bib, tables/ (generated), figs/, and the PDF
records/                     HANDOFF.md (the verified-numbers register) and prior-build logs
RECOMPUTED.md                the recomputation register of the first draft (kept for the record)
```

## Running one condition

```python
import os; os.environ["WM_BODY"] = "myofinger"
import run_reach
run_reach.run("prior", seed=0, total_steps=100000, utd=2,
              n_train_targets=8, eval_heldout=True, prior_path="prior_finger.pt")
```

Conditions are `blank`, `prior`, `coach`, `priorcoach`; `randprior` and `randcoach` are `prior` and `coach` with a random-initialised checkpoint. Defaults are the single set in `run_reach.py`: hidden 128, batch 128, update-to-data 2, 64 real + 64 imagined per Dyna update, 1,000 warm-up steps, 8 train targets and 20 disjoint held-out targets on fixed seeds. The constant-weight coach is `coach_anneal=10**8`. Each run writes one JSON per seed with `{"step", "eval_return", "eval_dist", "alpha"}`.

Runs are seed-deterministic on CPU. Every run of the earlier 12k study with a counterpart here reproduces it to the digit at the checkpoints they share.

## Reproducing the statistics

`analysis/audits/audit_v3b.py` is the check the paper was released under: every cell of the six tables, every star or bold mark, every interval in the prose, every worded claim (each phrase must appear in the paper *and* be true of the data) and the plateau count, recomputed from the per-seed JSON. **1,245 items, 0 mismatches** at release. Scripts carry the absolute paths of the machine they ran on at the top of each file; change `W` (this study's results) and `Z` (the carried 12k study's) to run them elsewhere.

`records/HANDOFF.md` §4 lists every number the project page may show; §5 lists wordings that are wrong and must not be used.

## Notes on the body models

- The arm1 and arm4 priors used in the paper are `prior_arm{1,4}_200k.pt`, rebuilt with `build_prior.py --n 200000 --seed 0` (logs in `records/`). The checkpoints first used for those two arms could not be reproduced from any recorded budget or seed, so their runs were replaced; the arm2 and arm3 priors reproduce bit for bit from the same recipe.
- The myoElbow prior `prior_myo.pt` is reproduced bit for bit by `build_prior_sizes.py` at 10^5 transitions (open-loop five-step qpos RMS 19.34 mrad on babble).

## What is not here

- **The run data.** The per-seed JSONs (1,500 runs, about 1,500 single-thread CPU-hours) and checkpoints are not committed; `records/HANDOFF.md` §3 says where they live.
- **Seven conditions of the first draft have no archived launcher** (`results_noanneal_*`, `results_abruptcoach_*`, `results_cleancoach_*`, `results_cleancoach_train_*`, `results_pcconst_*`, `results_prioroff_*`, `results_priorpurge_*`). None of them enters the current paper.

## Requirements

Python 3.11, PyTorch, MuJoCo, MyoSuite, NumPy; matplotlib for the figures; imageio and Pillow for the videos; python-docx for the Word version. The study was run on CPU, one torch thread per run.
