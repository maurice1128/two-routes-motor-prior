"""The explainer for the v3 TCDS paper (2026-09-25).

Same construction and footage as make_explainer100k.py (every policy on screen is a
100,000-step policy from actors_finger100k/, the paper's own run for that seed,
re-executed with the network saved). The cards are rewritten to the v3 claims, and
the withdrawal part shows the paper's own figures from tcds_v3/figs. Every number on a
card is a value of the audited v3 paper (tcds_v3/audits/audit_v3b.py).

Retracted v2 wording that must not reappear: "the body model helps where the body is
hard", "withdrawing either costs no dependence", "two bodies", "guidance vs the body
model" with the near-no-op teacher T1.
"""
import os, warnings
os.environ.setdefault("WM_BODY", "myofinger")
warnings.filterwarnings("ignore")

import numpy as np, mujoco, imageio.v2 as imageio
from PIL import Image
import arm_env, run_reach
from make_explainer import Stage, card, compose, W, PH, FPS

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "videos"); os.makedirs(OUT, exist_ok=True)
ACT = os.path.join(ROOT, "actors_finger100k")
FIGS = os.path.join(ROOT, "tcds_v3", "figs")


def image_card(path, foot, num, hold=7.0):
    """A figure from the paper, fitted into the panel area, with the footer."""
    im = Image.open(path).convert("RGB")
    s = min(W / im.width, PH / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)))
    canvas = Image.new("RGB", (W, PH), (255, 255, 255))
    canvas.paste(im, ((W - im.width) // 2, (PH - im.height) // 2))
    return [compose([np.asarray(canvas)], foot, num)] * int(FPS * hold)


def main():
    st = Stage()
    T = run_reach.fixed_eval_targets(20)
    A = lambda name, s=0: os.path.join(ACT, "%s_s%d.pt" % (name, s))
    f = []

    # -- 1. the question ------------------------------------------------
    f += card(["Guidance breeds dependence - or does it?"],
              ["Give a learner a head start, then take it away.",
               "Is it worse off than a learner that never had it?"])

    # -- 2. where a body model comes from --------------------------------
    f += card(["One head start: a model of its own body."],
              ["Before any task, the finger moves at random and records what its body does."])
    rng = np.random.default_rng(3)
    st.env.reset(randomize_pose=True)
    a = np.zeros(arm_env.ACT_DIM)
    st.cam.distance = float(st.env.m.stat.extent) * 0.95
    look = None
    for t in range(150):
        if t % 12 == 0:
            a = rng.uniform(-1, 1, arm_env.ACT_DIM)
        st.env.step(a)
        mujoco.mj_forward(st.env.m, st.env.d)
        tip_p = st.env.d.site_xpos[st.tip]
        p = tip_p + 0.30 * (st.env.d.xpos[1:].mean(axis=0) - tip_p)
        look = p.copy() if look is None else 0.90 * look + 0.10 * p
        st.cam.lookat[:] = look
        st.wide.update_scene(st.env.d, st.cam)
        f.append(compose([st.wide.render()],
                         "Motor babbling  -  no target, no reward, no task",
                         "200,000 transitions. The only data the body model ever sees. Rewards seen: 0."))
    st.cam.distance = float(st.env.m.stat.extent) * 0.58

    # -- 3. frozen, and used to imagine ----------------------------------
    f += card(["That model is then frozen."],
              ["It is never updated again. It is used to imagine short",
               "rollouts while the policy learns the task for real."])
    f += st.duo(A("prior"), A("blank"), "frozen body model", "model-free SAC",
                "Body model vs nothing: same held-out target, frame for frame",
                "myoFinger, n=12 at 100k:  -40.29 mm [-57.95, -22.64]", T[:2])
    f += card(["Is it what the model knows,", "or just the extra machinery?"],
              ["The control: an untrained model, randomly initialised,",
               "in exactly the same imagination loop, to the same budget."])
    f += st.duo(A("prior"), A("randprior"), "babble-trained model", "RANDOM model, same loop",
                "Identical machinery. Only the model's content differs.",
                "n=12 at 100k:  -48.71 mm [-62.32, -35.09]  in favour of the trained model", T[:2])
    f += image_card(os.path.join(FIGS, "joint_series.png"),
                    "Four planar arms of 1-4 joints: the trained model beats a random one on every arm.",
                    "A random model's harm grows with joint count (+0.41 of model-free error per joint); the model's own benefit does not.")

    # -- 4. the other head start ------------------------------------------
    f += card(["The other head start: a teacher."],
              ["It pulls the student toward its own actions",
               "through a distillation term in the loss."])
    f += st.solo(A("coach"), "student of a near-no-op teacher",
                 "A teacher is worth its competence",
                 "This teacher scores 169.1 mm; doing nothing scores 172.3. Its student ends +28.33 mm behind no aid.", T[:1])
    f += card(["With a competent teacher on the same finger"],
              ["the student ends at 34.50 mm, ahead of every condition without that teacher.",
               "The same loss, seeds and body: only the teacher differs."])

    # -- 5. withdrawal ------------------------------------------------------
    f += card(["Now take the teacher away."],
              ["After 2,000 to 8,000 guided steps, on both MyoSuite bodies.",
               "Control: keep the loss term, but point it at a frozen copy of the student."])
    f += image_card(os.path.join(FIGS, "sweep100k_components.png"),
                    "Right after withdrawal (left) the teacher's share (blue) falls with attachment; the deleted term's (red) does not.",
                    "At 100k (right): the withdrawn learner is never behind one never guided, and on the finger it is ahead.")
    f += card(["Not dependence."],
              ["On both bodies, at every attachment duration, whether the teacher",
               "added to or replaced the learner's own gradient: once training has",
               "converged, a learner that lost its teacher never ends behind one that never had it."],
              hold=5.0)
    f += card(["The body model behaves alike."],
              ["Withdrawn at 8,000 steps, it leaves the finger learner 15 mm short of",
               "one that kept it - and 25 mm ahead of one that never had it."],
              hold=4.5)

    # -- 6. where it stops ---------------------------------------------------
    f += card(["Where this stops."],
              ["Single-task reach, in simulation, on six muscle-driven bodies.",
               "A harder body makes a wrong model cost more, not a right one worth more."],
              hold=4.5)

    # -- 7. the claim ----------------------------------------------------------
    f += card(["What remains after withdrawal is a shortfall,",
               "not dependence.",
               "A prior is worth what it contains."],
              ["What a Motor Prior Is Worth  -  2026"], hold=6.0)

    dest = os.path.join(OUT, "explainer_v3.mp4")
    imageio.mimwrite(dest, f, fps=FPS, quality=7, macro_block_size=1)
    print("wrote %s  (%d frames, %.0f s, %.2f MB)"
          % (dest, len(f), len(f) / FPS, os.path.getsize(dest) / 1e6))


if __name__ == "__main__":
    main()
