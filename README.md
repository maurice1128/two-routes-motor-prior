# Learned Body Models and Withdrawn Teachers

Code, launchers, analysis and audit scripts for

> **Learned Body Models and Withdrawn Teachers in Muscle-Driven Motor Learning.**
> Manuscript in preparation for *IEEE Transactions on Cognitive and Developmental Systems*; not posted to any preprint server.
> (Earlier titles: *Given and Taken Away*; *What a Motor Prior Is Worth*; *Two Routes to a Motor Prior*.)

**Project page:** https://maurice1128.github.io/projects/motor-prior.html
**Paper:** [`paper/learned_body_models_withdrawn_teachers.pdf`](paper/learned_body_models_withdrawn_teachers.pdf) (LaTeX source, generated tables and figures alongside it).

---

## The study

One soft actor-critic learner on six muscle-driven bodies -- myoElbow (1 joint, six muscles, mrad), myoFinger (4 joints, five muscles, mm) and four planar muscle arms of one to four joints (mm). Twelve matched seeds per condition (24 on the 3- and 4-joint arms), paired-*t* 95 % intervals, held-out targets. Every claim is checked under two endpoints (mean of the evaluations at 90k-100k, and the single 100k evaluation) and only claims that hold under both are made (`analysis/v4/robust_v4.py`, `analysis/v4/make_slim.py`).

* **Teacher.** A frozen actor joined through an imitation term is withdrawn at 2k-8k steps. On myoFinger, with a competent teacher, the withdrawn learner ends ahead of one never guided and keeps more than half of the teacher's benefit (about a quarter when withdrawn at 2k, about all of it from 7k). A *self-anchor* (the teacher replaced by a frozen copy of the student) shows that after 4k or more steps of attachment most of the drop at withdrawal is the removed imitation term rather than lost knowledge.
* **Body model.** A frozen forward model learned once from reward-free babbling, used for short Dyna rollouts, is compared with a random model in the identical machinery, whose predictions barely depend on the action. That random model becomes increasingly harmful as joints are added (planar arms of one to four joints); the trained model avoids the harm. Against model-free learning with the same real data per update, the trained model's own advantage does not grow with joints.

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
  run_long200k.ps1           myoElbow teacher withdrawal extended to 200k (none / constant / abrupt x 7 t_w)
  run_bm_arm4.ps1            body-model withdrawal on the four-joint arm (purge / matched, 24 seeds)

analysis/
  v4/common.py               data access and statistics (late-mean endpoint) used by every v4 script
  v4/make_tables_v4.py, v4/extra_v4.py, v4/long200k.py, v4/arm4_bm.py   every table and quoted number -> v4/numbers.json
  v4/make_slim.py, v4/slim2_stats.py, v4/robust_v4.py   the paper's tables, primary family (Holm) and two-endpoint screen
  v4/make_figs_slim.py       the three figures of the paper; v4/count_runs_slim.py run count (1,056 runs)
  v4/make_figs_v4.py, v4/plateau_v4.py, v4/count_runs_v4.py   the full v4 draft before subtraction
  v4/audits/audit_v4.py      checks every decimal in the prose and every table cell against numbers.json and raw JSON
  v3/make_tables_v3.py, v3/make_table_replace.py   the previous version's tables (kept for the record)
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

`analysis/v4/audits/audit_v4.py` is the check of the current version (prose decimals and table cells against `numbers.json`, recomputed from the per-seed JSON; 0 mismatches). `analysis/audits/audit_v3b.py` is the check the previous version was released under: every cell of the six tables, every star or bold mark, every interval in the prose, every worded claim (each phrase must appear in the paper *and* be true of the data) and the plateau count, recomputed from the per-seed JSON. **1,245 items, 0 mismatches** at release. Scripts carry the absolute paths of the machine they ran on at the top of each file; change `W` (this study's results) and `Z` (the carried 12k study's) to run them elsewhere.

`records/HANDOFF.md` §4 lists every number the project page may show; §5 lists wordings that are wrong and must not be used.

## Notes on the body models

- The arm1 and arm4 priors used in the paper are `prior_arm{1,4}_200k.pt`, rebuilt with `build_prior.py --n 200000 --seed 0` (logs in `records/`). The checkpoints first used for those two arms could not be reproduced from any recorded budget or seed, so their runs were replaced; the arm2 and arm3 priors reproduce bit for bit from the same recipe.
- The myoElbow prior `prior_myo.pt` is reproduced bit for bit by `build_prior_sizes.py` at 10^5 transitions (open-loop five-step qpos RMS 19.34 mrad on babble).

## What is not here

- **The run data.** The per-seed JSONs (1,500 runs, about 1,500 single-thread CPU-hours) and checkpoints are not committed; `records/HANDOFF.md` §3 says where they live.
- **Seven conditions of the first draft have no archived launcher** (`results_noanneal_*`, `results_abruptcoach_*`, `results_cleancoach_*`, `results_cleancoach_train_*`, `results_pcconst_*`, `results_prioroff_*`, `results_priorpurge_*`). None of them enters the current paper.

## Requirements

Python 3.11, PyTorch, MuJoCo, MyoSuite, NumPy; matplotlib for the figures; imageio and Pillow for the videos; python-docx for the Word version. The study was run on CPU, one torch thread per run.
