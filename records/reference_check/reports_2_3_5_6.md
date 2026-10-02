# Reference check — batches 2, 3, 5, 6 (saved from agent output, 2026-10-03)

## Batch 2
[2] berg2023 — META OK — CLAIM OK (SAC/PPO, model-free; slow) — arxiv.org/pdf/2307.03716
[10] schmitt2018 — META OK (arXiv) — CLAIM: sentence "pipelines in robotics" MISMATCH minor (Kickstarting = DMLab-30, not robotics); annealing OK
[18] mckay2022 — META OK — CLAIM OK on OSF preprint (no significant effect of reduced frequency); published version paywalled → library
[26] miki2022 — META OK — CLAIM OK (privileged teacher, student rollouts)
[34] uchendu2023 — META OK — CLAIM MISMATCH partial: JSRL attributes degradation to a poor/untrained critic, not removal of a constraint; prior policy not cloned
[42] ha2018 — META OK — CLAIM OK (random-policy rollouts, model trained once)
[50] yu2020 — META OK (optional pp. 14129–14142) — CLAIM OK
[58] haarnoja2018 — META OK — CLAIM OK only together with haarnoja2018b (auto-temperature is in 2018b)

## Batch 3
[3] chiappa2023 — META OK — CLAIM OK
[11] agarwal2022 — META OK, add vol. 35, pp. 28955–28971 — CLAIM: "in robotics" MISMATCH minor (Atari/DMC/balloon); collapse at weaning reported; authors hypothesise objective inconsistency as the cause
[19] sullivan2008 — META OK — CLAIM: abstract supports (reduced feedback → worse retention); paywalled → library; safer "retained the skill less well"
[27] kumar2021 — META OK — CLAIM OK (nuance: distils latent extrinsics)
[35] zhang2023 — META OK — CLAIM MISMATCH: "sometimes" not "often"; offline IQL policy, not cloned; cause = distribution shift / cold-start critic, not constraint removal
[43] hafner2020 — META OK — CLAIM OK
[51] yu2021 — META OK — CLAIM OK
[59] haarnoja2018b — META OK — CLAIM OK

## Batch 5
[5] wolpert1995 — META OK — CLAIM OK
[13] voelcker2022 — META OK — CLAIM OK; "more so in higher dimension" only partly supported (task-irrelevant distractor dimensions, no model-free baseline there)
[21] buekers1992 — META OK — CLAIM abstract-consistent (erroneous KR followed); paywalled → library
[29] sarnthein2023 — META OK — CLAIM minor: "Supervised distillation" wrong word (label term off) → "Knowledge distillation"
[37] asada2009 — META OK — CLAIM OK
[45] hafner2025 — META OK — CLAIM OK
[53] kim2025 — META OK — CLAIM OK
[61] claude2026 — META OK; suggest URL https://www.anthropic.com/claude/opus and an access date (IEEE online style); name other model versions if used

## Batch 6
[6] kawato1999 — META OK — CLAIM OK
[14] schmidt1989 — META OK — CLAIM abstract-consistent; paywalled → library
[22] kitago2013 — META OK — CLAIM OK (error clamp keeps cursor, removes information)
[30] li2023proto — META OK (arXiv) — CLAIM OK (KL to previous-iteration policy stabilises fine-tuning)
[38] cangelosi2015 — META OK (book) — CLAIM: full text not accessible → library (ch. 5)
[46] hansen2024 — META OK — CLAIM OK
[54] berdica2024 — META OK — CLAIM MISMATCH minor: policies trained/tested in the learned model, not run on the real soft robot ("drive" overstates)

## Batch 8
[8] salmoni1984 — META OK — CLAIM: paywalled; secondary source (Winstein & Schmidt 1990 full text) supports both sentences → library
[16] winstein1990 — META OK — CLAIM OK (Exp. 2: faded 50% schedule enhanced delayed no-KR retention)
[24] chen2019lbc — META OK; add PMLR vol. 100, pp. 66–75 — CLAIM OK (DAgger on student rollouts)
[32] zhao2022 — META OK — CLAIM MISMATCH minor: BC term pulls toward logged data / behaviour policy, not the learner's own earlier policy
[40] bongard2006 — META OK — CLAIM OK
[48] seo2022 — META OK — CLAIM OK
[56] zheng2025 — META OK (arXiv) — CLAIM OK for planning half (model-only learning covered by berdica2024 / wu2024modex)

## Batch 4 (all full text read)
[4] schumacher2023 — META OK (ICLR 2023) — CLAIM OK
[12] janner2019 — META OK — CLAIM OK; "higher dimension" only indirect (rests on voelcker2022)
[20] soderstrom2015 — META OK (add DOI 10.1177/1745691615569000) — CLAIM OK
[28] yuan2020 — META OK (add pp. 3902–3910, DOI 10.1109/CVPR42600.2020.00396) — CLAIM OK (student itself; poorly trained teacher; random network rests on sarnthein2023)
[36] lungarella2003 — META OK — CLAIM OK
[44] hafner2021 — META OK — CLAIM OK
[52] feng2023 — META OK — CLAIM OK
[60] sutton1991 — META OK — CLAIM OK; "Dyna-style rollouts of three steps" more exact

## Batch 1
[1] bernstein1967 — META OK — CLAIM OK (full text: redundant DOF, multiplicity of muscle action)
[9] schmidt1992 — META ERROR: pages 207–218 — CLAIM: (b),(c) OK on pp. 207–208; (a) "dependent" → library (pp. 209+)
[17] marschall2007 — META ERROR: pages 74–85 — CLAIM caveat: results show no effect for isolated parameter learning → "limited to particular laboratory tasks and extensive practice"
[25] lee2020 — META OK — CLAIM OK
[33] nair2020 — META OK (arXiv; not published) — CLAIM OK
[41] rolf2010 — META OK — CLAIM: abstract-consistent (inverse kinematics via goal-directed exploration) → library
[49] kidambi2020 — META OK (optional pp. 21810–21823) — CLAIM OK
[57] caggiano2022 — META OK — CLAIM OK
