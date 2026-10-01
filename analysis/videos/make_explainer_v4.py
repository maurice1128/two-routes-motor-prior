"""The explainer for the v4 TCDS paper (2026-10-01): "Given and Taken Away".

Same construction and footage as make_explainer_v3.py (every policy on screen is a 100,000-step
policy from actors_finger100k/, the paper's own run for that seed, re-executed with the network
saved). Cards carry the v4 story and v4 numbers (late mean, tcds_v4/numbers.json); figures come
from tcds_v4/figs.

v4 story: no dependence on a teacher; the body model matters more the harder the body.
Wording that must not reappear: "never behind" (stated as equivalence), "four of six bodies",
"the model's own benefit does not grow", "behaves alike" for the body model.
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
FIGS = os.path.join(ROOT, "tcds_v4", "figs")


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
    f += card(["Give a learner help, then take it away."],
              ["Is it worse off than a learner that never had help?",
               "And which help is worth more: a body model, or a teacher?"])

    # -- 2. where a body model comes from --------------------------------
    f += card(["Help one: a model of its own body."],
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
              ["It is used to imagine short rollouts",
               "while the policy learns the task for real."])
    f += st.duo(A("prior"), A("blank"), "frozen body model", "model-free SAC",
                "Body model vs nothing: same held-out target, frame for frame",
                "myoFinger, n=12, mean over 90-100k:  -46.19 mm [-58.70, -33.69]", T[:2])
    f += card(["Is it what the model knows,", "or just the extra machinery?"],
              ["The control: a random model, in exactly the same",
               "imagination loop, to the same budget."])
    f += st.duo(A("prior"), A("randprior"), "babble-trained model", "RANDOM model, same loop",
                "Identical machinery. Only the model's content differs.",
                "n=12:  -50.11 mm [-64.54, -35.68]  in favour of the trained model", T[:2])
    f += image_card(os.path.join(FIGS, "teaser_slim.png"),
                    "Right: a random model that ignores the action gets more harmful as joints are added.",
                    "Random minus trained model, arms of 1-4 joints: 5, 28, 133, 357 mm. The trained model avoids it.")

    # -- 4. the other help -------------------------------------------------
    f += card(["Help two: a teacher."],
              ["It pulls the student toward its own actions",
               "through an imitation term in the loss."])
    f += st.duo(A("coachT2"), A("coach"), "competent teacher", "near-motionless teacher",
                "A teacher is worth its competence. Same body, seeds and loss.",
                "n=12:  37.46 mm against 120.53 mm. The motionless teacher's student is worse than no help.", T[:2])

    # -- 5. withdrawal ------------------------------------------------------
    f += card(["Now take the teacher away."],
              ["After 2,000 to 8,000 guided steps, on both MyoSuite bodies."])
    f += image_card(os.path.join(FIGS, "withdrawal_curves_v4.png"),
                    "Withdrawn (dashed): a sharp drop, then recovery. Never guided: grey. Teacher kept: red.",
                    "Finger: 43 mm ahead of never guided; more than half of the teacher's benefit kept.")
    f += card(["No lasting deficit."],
              ["On neither body does the withdrawn learner end detectably",
               "behind a learner that was never guided."],
              hold=4.5)
    f += card(["The drop after withdrawal"],
              ["Control: keep the imitation term, but point it at a frozen copy of the student.",
               "On the finger, after longer attachment, most of the drop is the removed term."],
              hold=5.0)
    f += image_card(os.path.join(FIGS, "dip_slim.png"),
                    "Blue: lost teacher knowledge.  Red: the deleted imitation term.",
                    "On the finger, after 4,000 or more guided steps, the drop is mostly red.")

    # -- 7. where it stops ---------------------------------------------------
    f += card(["Where this stops."],
              ["Single-task reach, in simulation, on six muscle-driven bodies.",
               "The competent teacher is the best of twelve candidate runs."],
              hold=4.5)

    # -- 8. the claim ----------------------------------------------------------
    f += card(["A competent teacher can be taken away.",
               "A body model is worth its content."],
              ["Withdrawn Teachers and Learned Body Models  -  2026"], hold=6.0)

    dest = os.path.join(OUT, "explainer_v4.mp4")
    imageio.mimwrite(dest, f, fps=FPS, quality=7, macro_block_size=1)
    print("wrote %s  (%d frames, %.0f s, %.2f MB)"
          % (dest, len(f), len(f) / FPS, os.path.getsize(dest) / 1e6))


if __name__ == "__main__":
    main()

