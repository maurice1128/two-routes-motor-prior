# HANDOFF — motor-prior project (TCDS paper, v3)

Live source of truth for the project page (`maurice1128.github.io/projects/motor-prior.html`),
the professor-facing artifact page, and the code repository README. Every number a page
shows must appear in §4 below; §4 is copied from the audited paper. The previous register
(v2, 2026-09-13) is kept as `HANDOFF_v2_2026-09-13.md` and is **superseded**.

## 1. What the paper is now (2026-09-25)

- **Title:** *What a Motor Prior Is Worth: A Body Model and a Teacher Compared to Convergence
  and Through Withdrawal.* Sole author Mu-Hua Wang (anonymous in the submission). Target IEEE TCDS.
- **Files:** source `wm_prior/tcds_v3/paper.tex` (+ `refs.bib`, `figs/`, `tables/`);
  PDF and Word in `Desktop/PAPER_FIX/world_model/world_model_paper_TCDS_v3_2026-09-25.{pdf,docx}`.
  10 pages (TCDS free limit), abstract 249 words (limit 250), 6 tables, 3 figures, 75 references.
- **One thesis:** the cost of withdrawing a motor prior decomposes, and what remains at
  convergence is not dependence. On either route (teacher, body model), either MyoSuite body and
  either teaching regime, a learner that had a prior and lost it is **never detectably worse than
  one that never had it**; what can remain is a shortfall against *keeping* the prior.
- **Second result:** what a prior is worth is bound to its content — the babble model beats
  model-free learning on 4 of 6 bodies and a random model on all 6 (whose harm grows with joint
  count); a teacher is worth its competence.
- **Replaces** the TMLR draft *Two Routes to a Motor Prior* (desk-rejected 2026-09-09) and the
  separate teacher-withdrawal draft (`world_model_zeroshot`, read-only, never submit separately).

## 2. Setup

- Six bodies. myoElbowPose1D6M (1 DoF, six muscles, mrad); myoFingerReachRandom-v0 (4 joints,
  5 muscles, mm); planar muscle arms with 1/2/3/4 joints and 2/4/8/10 muscles (mm).
- One SAC backbone (hidden 128, batch 128, UTD 2). Reach, 100-step episodes, 8 training targets,
  20 disjoint held-out targets (every number is held-out). 100,000 steps, plateau checked per
  condition (paired change 100k−80k): 5 of 97 conditions still moving.
- Body model: frozen forward model from reward-free OU babbling, short Dyna (k=3), 64 real +
  64 imagined per update, analytic reward at imagined states. Controls: `randprior` (random frozen
  model), `blank64` (= SAC at batch 64, the real-data content of a Dyna update).
- Teacher: distillation term `c·||tanh μ_s − tanh μ_T||²`, c = 1. T1 = 30k-step model-free actor
  (elbow 27.2 mrad; finger 169.1 mm ≈ no-op 172.3); T2 (finger) = actor of a 100k body-model run,
  seed 1 (35.6 mm). Additive regime (q=1) and replacing regime (q=0 until withdrawal).
- Withdrawal arms: none / constant / abrupt / selfanchor (frozen student snapshot, keeps the
  term, no new information) / randanchor / gateon (replacing only). Body model: keep / soft /
  purge / matched (matched holds real data per update at 64).
- n = 12 matched seeds (24 on the 3- and 4-joint arms); paired-t 95% intervals; `*` = excludes 0.
- **1,500 runs, ~1,500 single-thread CPU-hours.** Runs are deterministic on CPU; every run of the
  earlier 12k study with a counterpart here is reproduced to the digit.

## 3. Where the raw data are (wm_prior/)

- Core grid: `results_conv_{elbow,finger}_{blank,blank64,prior,randprior,coach,randcoach,priorcoach}`,
  `results_conv_finger_{coachT2,priorcoachT2}`.
- Joint series: `results_arms100k_arm{1..4}_{blank,blank64,randprior}`, prior = `_prior` (arm2, arm3)
  and `_prior2` (arm1, arm4: priors rebuilt by the recorded recipe, `prior_arm{1,4}_200k.pt`;
  the old `prior_arm1.pt`/`prior_arm4.pt` could not be reproduced and are not used).
- Teacher withdrawal: `results_coach100k_elbow`, `results_coach100k_elbow_off2`,
  `results_coach100k_finger_T2`, `results_sweep100k_{elbow,finger_T2}`; 1k-grid elbow dips from
  `world_model_zeroshot/b2_pilot/results_{attachment,withdrawal}_myoelbow` (reproduced, read-only).
- Replacing regime: `results_replace100k_{myoelbow,myofinger}`.
- Body-model withdrawal: `results_matched100k_{elbow,finger}`.

## 4. Verified numbers (from the audited paper; 100k unless stated)

### 4.1 Withdrawing the teacher (Tables III–V)
- At t_w = 8k the immediate dip is the deleted term: elbow +40.22 of +45.17 mrad; finger (T2)
  +46.22 of +49.20 mm. At t_w = 2k the dip is mostly the teacher's, on both bodies.
- Teacher share of the dip falls with attachment: slope −6.23* mrad and −4.65* mm per 1k steps.
  Loss-term share does not fall; on the finger it grows (+7.07*), carrying the growth of
  abrupt−none at the dip (+9.34*).
- At 100k, additive regime: elbow abrupt−none not detected at any t_w (slope +0.50 n.s.);
  total +7.6 to +17.0 mrad at every t_w (teacher component 3 to 9 mrad, 4 of 7 detected; loss
  term never). Finger: withdrawn learner better than never-guided at 6 of 7 t_w, slope −8.44*;
  teacher component +53.28* at 2k → +0.18 at 8k.
- Replacing regime: elbow collapse = switching on the student's gradient (gateon ≈ selfanchor,
  within 5.0 mrad; selfanchor−constant post-4k slope +5.77*). Finger collapse = deleted term
  (abrupt 59–93 mm above constant 4k steps later; keeping the teacher only 3–7 mm; no growth,
  slope −2.19 n.s.). At 100k plain SAC after withdrawal: elbow level with never-guided (+1.0 to
  +3.2, none detected); finger 38–59 mm ahead (slope −3.20*). Pure distillation vs model-free:
  elbow −3.02 n.s., finger −58.32*.
- Across 22 withdrawal cells (7+7 additive, 4+4 replacing): the withdrawn learner is never
  detectably worse than one never guided.

### 4.2 Withdrawing the body model (Table VI, at 8k)
- Elbow: purge−keep +7.17 n.s. at 12k; matched−keep +22.11* at 12k, +5.05 n.s. at 100k;
  withdrawn arms vs never-modelled at 100k +0.97 to +3.32, none detected.
- Finger: matched−keep +9.97 n.s. at 12k, **+15.31* at 100k** (purge +20.15*): the imagination
  forgone after withdrawal. Withdrawn arms vs never-modelled −20 to −36 mm*; matched keeps 62%
  of the model's advantage.

### 4.3 What the priors are worth (Tables I, II)
- Elbow prior−blank −20.72* at 12k, −4.09 n.s. at 100k. Finger −27.10 n.s. at 12k, −40.29* at 100k.
- Arms prior−blank at 100k: 1j −8.87*, 2j −7.85*, 3j −23.76 n.s., 4j −100.18* (−49%).
  Body model beats randprior on all six bodies. randprior harm grows with joints (slope +0.41 of
  blank per joint*, Wilcoxon p = 0.003); the model's own benefit does not (slope −0.10 n.s.).
- blank64 (= batch 64) helps the finger (−14.59*), a known small-batch effect.
- Teacher: elbow coach−blank −8.06* (Wilcoxon p = 0.052). Finger T1 coach +28.33* worse than
  blank; T2 coach 34.50 mm, −60.17* vs blank, −19.87* vs prior, level with T2+prior.

## 5. Phrasings that are wrong — do not use

- "Withdrawing either prior costs no dependence" (v1/v2 wording) → use "never worse than never
  having had it; what can remain is a shortfall against keeping it".
- "The body model helps where the body is hard" / "harder body, more benefit" — not supported
  (slope −0.10 n.s.); only the random model's *harm* grows with joints.
- "The body decides between the routes" — refuted; the teacher's quality decides.
- "Only two bodies" — six (two MyoSuite + four planar arms).
- "Keeping the teacher does not relieve the collapse" — elbow only; on the finger it does.
- "The human literature cannot give an informationless stand-in" — false (erroneous KR, error
  clamps, random feedback exist); say "such controls are rare".
- "No work runs a control for the loss term" — too broad; say "no study splits the withdrawal
  cost by retargeting the term to the learner's own frozen snapshot".
- "Dissipated by convergence" for the body model — on the finger the shortfall vs keep grows.
- Any "not significant" read as "equal" — it is "not detected at this power".

## 6. How to verify

- `tcds_v3/audits/audit_v3b.py` recomputes every table cell, every interval in the prose and
  every worded claim from raw JSON: **1,245 items, 0 mismatches** (2026-09-25).
- Tables II, IV, V are generated from JSON: `tcds_v3/make_tables_v3.py`, `make_table_replace.py`.
  Figures: `tcds_v3/make_figs_v3.py`. Run counts: `tcds_v3/count_runs_v3.py`.
- Word: `tcds_v3/tex2docx_v3.py` (run with `.venv_mm`), check with `check_docx.py`.
- Literature checks: `Desktop/deep research/motor_prior_tcds/results/` (R1–R6 + SUMMARY).
