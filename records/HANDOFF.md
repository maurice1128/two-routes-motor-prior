# HANDOFF — motor-prior project (merged TCDS paper)

Live source of truth for the project page (`maurice1128.github.io/projects/motor-prior.html`),
the professor-facing artifact page, and the code repository README. Every number a page
shows must appear in §4 below; §4 is copied from the audited paper and nothing here is
recomputed by hand.

## 1. What the paper is now (2026-09-13)

- **Title:** *What a Motor Prior Is Worth: A Frozen Body Model and a Teacher Compared to
  Convergence and Through Withdrawal.* Sole author Mu-Hua Wang. Target venue IEEE TCDS.
- **It replaces two earlier drafts:** the TMLR submission *Two Routes to a Motor Prior*
  (desk-rejected 2026-09-09, scored everything at a 12k endpoint) and the separate
  teacher-withdrawal draft (`world_model_zeroshot`). Both are merged into one paper; the
  zeroshot draft must not be submitted on its own.
- **Source:** `wm_prior/tcds_merged/paper.tex` (+ `refs.bib`, `figs/`). PDF is 9 pages,
  3 figures, 7 tables, ~4,780 body words.
- **Thesis, three parts:** (1) the body model helps where the body is hard; (2) guidance is
  only as good as its teacher; (3) withdrawing either prior costs no dependence.

## 2. Bodies, arms, budget

- myoElbowPose1D6M: 1 DoF, **six** muscles, error in mrad. myoFingerReachRandom-v0: 4 joints,
  5 muscles, error in mm. Never compared in magnitude.
- SAC hidden 128, batch 128, UTD 2, 1,000 warm-up, lr 3e-4. Reach, 100-step episodes, 8 train
  targets, 20 disjoint held-out targets (all numbers are held-out).
- Body model: frozen forward model from reward-free OU babbling (1e5 transitions elbow,
  2e5 finger), short-horizon Dyna, 64 real + 64 imagined per update; the arm is handed the
  analytic reward at imagined states. Teacher: `teacher_{elbow,finger}_clean30k.pt`, 30k
  rewarded steps, distillation term at constant weight c = 1 (`--coach_anneal 100000000`).
- Controls: randprior, randcoach, priorcoach. Withdrawal of the body model at 8k:
  keep / soft / purge / matched. Withdrawal of the teacher at t_w: none / constant / abrupt /
  selfanchor / randanchor (+ gateon in the replacing regime).
- **Budget: 100,000 steps** for the core grid and both 100k decompositions; the attachment
  sweep, replacing regime and budget matching are 12k experiments carried from the second
  draft. n = 12 matched seeds everywhere; paired-t 95% CI, df = 11.

## 3. Where the raw data are

- Core grid 100k: `wm_prior/results_conv_{elbow,finger}_{blank,prior,randprior,coach,randcoach,priorcoach}/`.
- Body-model withdrawal: `results_matched_{elbow,finger}` (12k), `results_matched100k_elbow` (100k).
- Teacher withdrawal 100k: `results_coach100k_elbow` (constant, abrupt), `results_coach100k_elbow_off2`
  (selfanchor, swap at loop step t_w−2 = the arm of the 12k table).
- Carried 12k experiments (read-only): `world_model_zeroshot/b2_pilot/results_{withdrawal,attachment,replace}_*`,
  `b1_pilot/results_budget_*`.
- Video actors at 100k: `wm_prior/actors_finger100k/` (train_video100k.py; each actor's held-out
  curve is checked against the paper's run checkpoint for checkpoint before use).

## 4. Verified numbers (the only ones a page may show)

All from the audited paper (seven audit scripts, 0 mismatches, 2026-09-13). Format:
mean [95% CI]; * = interval excludes zero. Negative favours the first-named arm.

### 4.1 Core grid at 100k (Table I)

| arm | elbow 12k | elbow 100k | finger 12k | finger 100k |
|---|---|---|---|---|
| blank (model-free) | 70.73 | 29.87 | 117.60 | 94.66 |
| prior (body model) | 50.02 | 25.78 | 90.50 | 54.37 |
| randprior | 106.03 | 46.08 | 129.98 | 103.08 |
| coach | 39.13 | 21.81 | 122.77 | 123.00 |
| randcoach | 93.26 | 39.50 | 142.02 | 109.14 |
| priorcoach | 38.49 | 24.63 | 116.13 | 114.62 |

| contrast | elbow 12k | elbow 100k | finger 12k | finger 100k |
|---|---|---|---|---|
| coach − prior | −10.89 | −3.97 [−8.47, +0.53] | +32.27 | **+68.63 [+59.19, +78.07]*** |
| prior − blank | −20.72 | −4.09 [−11.11, +2.94] | −27.10 | **−40.29 [−57.95, −22.64]*** |
| coach − blank | −31.61 | **−8.06 [−15.83, −0.28]*** | +5.17 | **+28.33 [+16.11, +40.56]*** |
| priorcoach − blank | −32.25 | −5.24 [−12.93, +2.45] | −1.47 | **+19.96 [+5.74, +34.17]*** |
| priorcoach − coach | −0.64 | +2.81 [−1.02, +6.65] | −6.64 | **−8.38 [−15.57, −1.19]*** |
| prior − randprior | −56.01 | **−20.29 [−32.00, −8.59]*** | −39.48 | **−48.71 [−62.32, −35.09]*** |
| coach − randcoach | −54.14 | **−17.69 [−33.70, −1.68]*** | −19.25 | **+13.86 [+7.62, +20.09]*** |

Random controls vs blank at 100k: elbow randprior +16.21 [+6.82, +25.59]*, randcoach +9.63 [−8.07, +27.34];
finger randprior +8.41 [−14.17, +30.99], randcoach +14.48 [+0.74, +28.21]*.
12k intervals quoted in prose: elbow prior−blank −20.72 [−35.8, −5.6]; coach−blank −31.61 [−48.6, −14.6];
coach−prior −10.89 [−19.3, −2.5]; finger prior−blank −27.10 [−66.3, +12.1]; coach−prior +32.27 [+16.5, +48.0].
Checkpoint counts (of 50, every 2k): elbow prior−blank significant at 7 (8k, 10k, 12k + four isolated);
elbow coach−blank 43; elbow coach−prior 38 (last 96k); finger prior−blank 45 (every checkpoint from 14k);
finger coach−prior > 0 at every checkpoint from 12k; finger coach−blank > 0 at 27 of 31 from 40k.
Plateau (last 20k, elbow): blank +2.7 (SD 11.3), prior −2.9 (9.4), coach −1.7 (3.5), randprior −25.0 (29.6).
Finger coach 122.77 at 12k, 123.00 at 100k. Finger teacher 169.1 mm vs zero-activation no-op 172.3 mm
(3.2 mm; zero activation = action −1).

### 4.2 Body-model withdrawal, elbow (Tables III, VI)

| contrast | 12k | 100k |
|---|---|---|
| purge − keep | +7.17 [−10.95, +25.29] | +5.34 [−3.14, +13.81] |
| soft − keep | +2.09 [−2.85, +7.04] | **+7.40 [+2.72, +12.08]*** |
| matched − keep | **+22.11 [+4.94, +39.28]*** | +5.05 [−0.98, +11.08] |
| soft − matched | **−20.02 [−37.34, −2.70]*** | +2.35 [−4.42, +9.11] |
| purge − matched | −14.94 [−44.62, +14.73] | +0.28 [−8.15, +8.72] |

Means 12k: keep 50.02, soft 52.11, purge 57.19, matched 72.13. Means 100k: keep 25.78, soft 33.18,
purge 31.12, matched 30.84. Change across the cut (8k→10k), matched − keep: +21.47 [+4.54, +38.40]*.
Finger 12k: purge − keep +14.19 [−0.95, +29.33]; matched − keep +9.97 [−1.50, +21.45]; soft − keep
−2.39 [−13.74, +8.97]; soft − matched −12.36 [−24.84, +0.12]; purge − matched +4.22 [−12.76, +21.19].

### 4.3 Teacher withdrawal at t_w = 8000 (Table IV) and at 100k (Table VI)

| contrast | elbow dip | elbow 12k | finger dip | finger 12k | elbow 100k |
|---|---|---|---|---|---|
| abrupt − constant (total) | +45.17* | +16.00* [+6.49, +25.52] | +13.86* | +46.37* [+22.88, +69.87] | **+10.29 [+3.96, +16.62]*** |
| selfanchor − constant (teacher) | +4.95 | +9.42* [+3.13, +15.70] | −0.27 | −11.29* [−20.65, −1.93] | **+8.16 [+2.17, +14.14]*** |
| abrupt − selfanchor (loss term) | +40.22* | +6.59 | +14.13* | +57.66* | +2.13 [−3.64, +7.90] |
| randanchor − selfanchor | +47.44* | +20.66* | +20.13* | +17.20* | — |
| constant − none | — | −31.61* | — | +5.17 [−26.49, +36.82] | — |

Loss term = 89% of the elbow dip, all of the finger dip. Seeds recovering pre-withdrawal error (of 12):
elbow none 11, constant 7, abrupt 6, selfanchor 8, randanchor 5; finger 12, 6, 6, 12, 3.
Finger abrupt: 131 mm at t_w+1000, 169 at 12k. Teacher means 100k: constant 21.81, abrupt 32.10, selfanchor 29.97.

### 4.4 Attachment sweep (Table V, 2k–8k, slopes per 1,000 attached steps)

Elbow dip: teacher slope **−6.23 [−7.79, −4.68]***, loss term +4.76 [−0.13, +9.65], total −1.47 [−5.79, +2.84].
Elbow post-4k: teacher −2.88 [−5.86, +0.10], loss term −3.11 [−8.69, +2.46], total **−6.00 [−10.60, −1.39]***.
Elbow endpoint: teacher −2.26 [−4.84, +0.31], loss term +0.77 [−2.82, +4.36].
Finger dip: teacher −0.35 [−1.46, +0.75], loss term **+2.09 [+0.46, +3.72]***, total **+1.74 [+0.25, +3.22]***.
Finger post-4k: teacher **−2.87 [−4.27, −1.47]***, loss term **+5.74 [+1.49, +9.98]***, total +2.87 [−1.74, +7.47].
Finger endpoint: teacher −1.21 [−2.52, +0.10], loss term **+9.67 [+5.54, +13.80]***.
constant − none at each t_w (elbow): −86, −90, −66, −55, −24, −25, −44.

### 4.5 Replacing regime, elbow (Table VII)

selfanchor − constant: dips 58.7*, 104.7*, 109.4*, 67.2*; slope post-4k **+5.77 [+1.30, +10.25]***, endpoint **+8.66 [+2.99, +14.33]***.
gateon − constant: 59.1*, 105.4*, 110.2*, 64.1*; +5.90 [+0.26, +11.54]*, +8.82 [+2.58, +15.06]*.
selfanchor − gateon: −0.4, −0.8, −0.8, +3.0; −0.13 [−3.44, +3.19], −0.15 [−5.25, +4.95] (narrowest interval [−2.5, +1.7]).
abrupt − selfanchor: 54.4*, 86.5*, 58.6*, 76.4*; +4.32 [−4.60, +13.24], +4.06 [−1.82, +9.95].
abrupt − constant: 113.1*, 191.2*, 168.0*, 143.6*; +10.10 [+0.52, +19.68]*, +12.72 [+6.47, +18.98]*.
Endpoints: constant 27.86 (additive constant 39.13); abrupt 107.7, 133.0, 168.2, 180.8 (0 of 12 recover at t_w 4000, 6000);
selfanchor 45.9, 59.8, 110.8, 86.7; a student that never had a teacher 70.73.

### 4.6 Budget matching, 12k (Table II)

Body model saw 3.3× (elbow) / 6.7× (finger) the teacher's transitions. prior_30k − prior_full:
+1.55 [−0.87, +3.97] mrad, +2.75 [−12.77, +18.27] mm. prior_full − coach: +10.89 [+2.52, +19.26]* mrad,
−32.27 [−48.05, −16.49]* mm. prior_30k − coach: +12.44 [+3.27, +21.61]*, −29.52 [−43.17, −15.87]*.
Means (SD): none 70.73 (22.77) / 117.60 (43.03); coach 39.13 (7.77) / 122.77 (12.18); prior 30k 51.57 (8.73) /
93.25 (22.02); prior full 50.02 (8.18) / 90.50 (25.57). Finger model 5-step error 14.3 → 25.1 mm.

### 4.7 Counts (for the "what was run" cards)

Per-seed training runs behind the paper's tables: own 324 (core grid 144; body-model withdrawal
12k 96; body-model withdrawal 100k 48; teacher withdrawal 100k 36 = constant, abrupt, selfanchor off2)
+ carried from the second draft 720 (teacher withdrawal 12k 120; attachment sweep 336; replacing
regime 168; budget matching 96) = **1,044**. A further 12 (selfanchor at offset 0) are a robustness
extra not in any table. Seed-for-seed reproductions of the earlier 12k runs at the 12k checkpoint:
144/144 (grid), 48/48 (withdrawal arms), 12/12 (selfanchor). Audit items recomputed from raw JSON
or logs with 0 mismatches: 70 + 82 + 18 + 158 + 21 + 31 + 95 = **475**. Rewards seen by the body
model: 0.

## 5. Phrasings earlier audits rejected (do not reintroduce)

- "both random controls are worse than model-free" — false at 100k (see §4.1 random controls).
- "the elbow benefit reverses at 30k/50k" — an artefact of a wrong teacher checkpoint; discarded.
- "by 40k" for the finger coach — it is "27 of 31 checkpoints from 40k".
- "the only window" for elbow prior−blank — it is "the only run of consecutive significant checkpoints (8k–12k)"; four isolated later ones exist.
- "elbow has three muscles" — six.
- "within ±3 mrad" for selfanchor − gateon — 3.04; say "within 3.1 mrad".
- Any "not significant" read as "equal" — it is "not detected at this power".
- "no public code repository" — one exists: github.com/maurice1128/two-routes-motor-prior.

## 6. How to verify

Scratchpad audit scripts (session ce654e03): merged_audit_tcds.py (Tables I/III), audit_prose.py,
audit_merged_carried.py (Tables IV/V/VII/II), audit_w100k.py (Table VI), audit_prose_v2.py,
audit_marks.py (bold/stars + prose CIs + hyperparameters vs run_reach.py), audit_stars_sweep_replace.py.
Figures: `tcds_merged/make_figs.py` (converged_curves.png, withdrawal100k_curves.png) from the same JSON.
