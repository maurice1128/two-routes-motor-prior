# HANDOFF — motor-prior project (TCDS paper, v4 final, 2026-10-02)

Live source of truth for the project page (`maurice1128.github.io/projects/motor-prior.html`),
the professor-facing artifact page and the code repository README. **Every number a page shows
must appear in §2 below**; §2 is copied from the audited paper (`wm_prior/tcds_v4/paper.tex`,
numbers in `tcds_v4/numbers.json`, checked by `tcds_v4/audits/audit_v4.py`, 0 mismatches).
The previous register (v3, 2026-09-25) is kept as `HANDOFF_v3_2026-09-25.md`; most of its claims
are withdrawn (§3).

## 1. The paper

**Withdrawn Teachers and Learned Body Models in Muscle-Driven Motor Learning** — 7 pages, IEEE TCDS
format, double-anonymous. PDF: `PAPER_FIX/world_model/world_model_paper_TCDS_v4final_2026-10-01.pdf`
(the Word file next to it is a reading copy only). Code: github.com/maurice1128/two-routes-motor-prior.

One soft actor-critic learner, six muscle-driven bodies (myoElbow, myoFinger, planar arms of 1–4
joints), 12 matched seeds per condition (24 on the 3- and 4-joint arms), 100k steps (elbow
withdrawal also 200k). Final performance is reported under two endpoints (mean of 90–100k, and the
single 100k evaluation); claims are made only when they hold under both.

## 2. Numbers a page may show

**Teacher withdrawal (myoFinger, competent teacher T2, withdrawal at 2k–8k steps)**
- Withdrawn minus never guided, pooled: −43.25 [−54.72, −31.79] mm (single evaluation −40.63 [−53.05, −28.22]).
- Teacher's worth (kept minus never guided): −61.71 [−74.98, −48.44] mm.
- Share of the teacher's benefit kept, per withdrawal time 2k…8k: 24, 36, 59, 78, 91, 101, 101 % (single evaluation 22 … 99 %).
- Averaged over withdrawal times: more than half (one-sided p = 0.0008; single evaluation 0.0018); at one-sided 95 % confidence at least 62 % (60 %).
- Slope of the advantage over attachment time: −8.70 [−10.07, −7.33] mm per 1,000 attached steps.
- Elbow (teacher worth −8.65 mrad): not detectably different from never guided; at 190–200k: +0.04 [−2.11, +2.19] mrad.

**The drop after withdrawal (self-anchor decomposition)**
- myoFinger, withdrawal at 4,000 steps or later: the imitation-term component exceeds the teacher component by 17.87 to 46.48 mm, each significant after Holm correction across withdrawal times; the imitation term carries 65–94 % of the drop (point estimates).

**Body model (trained vs random model in identical machinery)**
- Random minus trained, planar arms of 1–4 joints: 4.92, 28.42, 132.89, 357.20 mm (round: 5, 28, 133, 357).
- Between bodies: 4 vs 1 joint −352.27 mm, 3 vs 1 joint −127.96 mm, p < 0.001 under both endpoints.
- The random model's predictions depend on the action 2–4 % as much as the trained model's.
- myoFinger: trained minus random −50.11 [−64.54, −35.68] mm; trained minus model-free −46.19 [−58.70, −33.69] mm.
- myoElbow: trained minus random −25.78 [−34.71, −16.84] mrad.
- Teacher competence, myoFinger: competent teacher's student 37.46 mm vs near-motionless teacher's 120.53 mm.

**Design and checks**
- 1,056 runs, about 1,100 single-thread CPU-hours.
- Ten primary tests, Holm-corrected, largest corrected p = 0.0018.

## 3. Withdrawn wording — must not appear anywhere

- "never behind" / "no dependence" as equivalence → say "not detectably worse".
- "the body model's benefit grows with joints (10 % → 47 %)" → mixed with a batch-size effect.
- "what the model has learned matters more the more joints there are" → say "a random model becomes increasingly harmful as joints are added; the trained model avoids that harm".
- "keeps 70 %", "at least 60 %", "under half / a quarter of the teacher's worth" → say "more than half on average; from about a quarter (2k) to about all (7k+)".
- "early withdrawal = lost knowledge" → not significant after correction.
- "the drop is the change of objective, not forgetting" → only "on the finger, after ≥ 4k steps, the drop is mostly the removed imitation term".
- Body-model withdrawal, the replacing regime → removed from the paper.
- Titles "What a Motor Prior Is Worth", "Given and Taken Away", "Learned Body Models and Withdrawn Teachers".

## 4. Scope that travels with every number

Single-task reach in simulation on deterministic, fully observed bodies; imagined reward uses exact
forward kinematics. One teacher per body (finger teacher = best of twelve runs); one trained and one
random model per body. Joint count, muscle count and biarticular muscles change together along the
arm series; the one-joint arm cannot reach half of its held-out targets. Guided-phase transitions stay
in the replay buffer.
