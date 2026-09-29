"""The long explainer: the whole study, in the order a stranger needs it.

The short version answered only "what is a frozen babble-learned body model".
This one carries the argument: two routes, where they end up, what happens when
each is taken away, whether it is the model's content or its machinery, and
where the whole thing stops working.

Every number on a card comes from the study (n=12 matched seeds, held-out
targets unless the card says otherwise).  The footage is myoFinger seed 0 at
the 12,000-step endpoint -- it shows what an arm does, never what the result
is, because a single seed can run opposite to a twelve-seed mean.
"""
import os, warnings
os.environ.setdefault("WM_BODY", "myofinger")
warnings.filterwarnings("ignore")

import numpy as np, torch, mujoco, imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import arm_env, run_reach, viz
from sac_dyna import Actor

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "videos"); os.makedirs(OUT, exist_ok=True)
W, PH, FH, FPS = 900, 430, 70, 25
H = PH + FH
INK, DIM, BG, PANEL = (238, 236, 230), (146, 154, 148), (15, 18, 19), (12, 15, 16)
FONTS = "C:/Windows/Fonts/"


def _font(name, size):
    """PIL's default bitmap face is illegible at 900px; load a real one."""
    try:
        return ImageFont.truetype(FONTS + name, size)
    except OSError:
        return ImageFont.load_default()


F_CARD = _font("seguisb.ttf", 31)
F_SUB = _font("segoeui.ttf", 21)
F_FOOT = _font("segoeui.ttf", 18)


def card(lines, sub=(), hold=3.6):
    im = Image.new("RGB", (W, H), BG); g = ImageDraw.Draw(im)
    block = 42 * len(lines) + (18 + 31 * len(sub) if sub else 0)
    y = (H - block) // 2
    for ln in lines:
        g.text(((W - g.textlength(ln, font=F_CARD)) / 2, y), ln, fill=INK, font=F_CARD)
        y += 42
    if sub:
        y += 18
        for s in sub:
            g.text(((W - g.textlength(s, font=F_SUB)) / 2, y), s, fill=DIM, font=F_SUB)
            y += 31
    return [np.asarray(im)] * int(FPS * hold)


def compose(panels, foot, num=None):
    """Panels side by side, with a persistent explanatory footer beneath."""
    strip = np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0]
    im = Image.new("RGB", (W, H), PANEL)
    im.paste(Image.fromarray(strip), (0, 0))
    g = ImageDraw.Draw(im)
    if len(panels) > 1:      # otherwise two panels read as one continuous scene
        g.rectangle([W // 2 - 1, 0, W // 2, PH], fill=(58, 62, 66))
    g.rectangle([0, PH, W, H], fill=PANEL)
    g.text((18, PH + 14), foot, fill=(196, 198, 202), font=F_FOOT)
    if num:
        g.text((18, PH + 40), num, fill=(126, 132, 136), font=F_FOOT)
    return np.asarray(im)


def load(path):
    a = Actor(arm_env.OBS_DIM, arm_env.ACT_DIM, hidden=128)
    a.load_state_dict(torch.load(path, map_location="cpu")); a.eval(); return a


class Stage:
    def __init__(self):
        self.env = arm_env.ArmReachEnv(seed=7)
        self.env.reset(randomize_pose=False)
        for nm in ("IFtip_target", "wrist_target"):   # hide the model's own decoy marker
            i = mujoco.mj_name2id(self.env.m, mujoco.mjtObj.mjOBJ_SITE, nm)
            if i >= 0:
                self.env.m.site_rgba[i] = [0, 0, 0, 0]
        # MuJoCo's offscreen framebuffer defaults to 640x480; widen it before
        # constructing any renderer or a 900px render is refused.
        self.env.m.vis.global_.offwidth = max(self.env.m.vis.global_.offwidth, W)
        self.env.m.vis.global_.offheight = max(self.env.m.vis.global_.offheight, PH)
        self.wide = mujoco.Renderer(self.env.m, height=PH, width=W)
        self.half = mujoco.Renderer(self.env.m, height=PH, width=W // 2)
        self.tip = mujoco.mj_name2id(self.env.m, mujoco.mjtObj.mjOBJ_SITE, "IFtip")
        self.cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = float(self.env.m.stat.extent) * 0.58
        self.cam.azimuth, self.cam.elevation = 140.0, -25.0

    def aim(self, target):
        mujoco.mj_forward(self.env.m, self.env.d)
        p = self.env.d.site_xpos[self.tip]; t = np.asarray(target, float)
        if t.shape[0] < p.shape[0]:
            t = np.concatenate([t, p[t.shape[0]:]])
        self.cam.lookat[:] = (p + t) / 2.0

    @torch.no_grad()
    def rollout(self, actor, target, rend, steps=100):
        o = self.env.reset(target=target, randomize_pose=False)
        frames, dists, trail, d0 = [], [], [], None
        for _ in range(steps):
            a = actor.act(torch.tensor(o[None], dtype=torch.float32),
                          greedy=True).numpy()[0]
            o, r, dn, inf = self.env.step(a)
            tip = self.env.d.site_xpos[self.tip].copy()
            trail.append(tip); trail[:] = trail[-26:]
            rend.update_scene(self.env.d, self.cam)
            d = viz.decorate(rend.scene, target, tip, trail)
            if d0 is None:
                d0 = d
            frames.append(rend.render()); dists.append(d)
        return frames, dists, d0

    def solo(self, path, label, foot, num, targets):
        out = []
        act = load(path)
        for tg in targets:
            self.env.reset(target=tg, randomize_pose=False); self.aim(tg)
            fr, ds, d0 = self.rollout(act, tg, self.wide)
            for i in range(len(fr)):
                out.append(compose([viz.bar(fr[i], ds[i], d0, label)], foot, num))
        return out

    def duo(self, pl, pr, ll, lr, foot, num, targets):
        out = []
        al, ar = load(pl), load(pr)
        for tg in targets:
            self.env.reset(target=tg, randomize_pose=False); self.aim(tg)
            fl, dl, l0 = self.rollout(al, tg, self.half)
            fr, dr, r0 = self.rollout(ar, tg, self.half)
            for i in range(min(len(fl), len(fr))):
                out.append(compose([viz.bar(fl[i], dl[i], l0, ll),
                                    viz.bar(fr[i], dr[i], r0, lr)], foot, num))
        return out


def main():
    st = Stage()
    T = run_reach.fixed_eval_targets(20)
    FING = os.path.join(ROOT, "actors_finger")
    WD = os.path.join(ROOT, "actors_withdraw")
    f = []

    # -- 1. the question ------------------------------------------------
    f += card(["Two ways to give a learner a head start."],
              ["A model of its own body. Or a coach that already solved the task.",
               "This is about how differently the two behave."])

    # -- 2. where a body model comes from --------------------------------
    f += card(["Before the arm is given any task,"],
              ["it moves at random and records what its body does."])
    rng = np.random.default_rng(3)
    st.env.reset(randomize_pose=True)
    a = np.zeros(arm_env.ACT_DIM)
    # There is no target to frame against here, so follow the fingertip itself --
    # smoothed, because babbling throws it around and a hard follow reads as jitter.
    st.cam.distance = float(st.env.m.stat.extent) * 0.95
    look = None
    for t in range(180):
        if t % 12 == 0:
            a = rng.uniform(-1, 1, arm_env.ACT_DIM)
        st.env.step(a)
        mujoco.mj_forward(st.env.m, st.env.d)
        # aiming at the fingertip alone leaves the finger hanging off the top of
        # the frame, since it extends upward from there; lean the aim point a
        # third of the way toward the model's centroid and it sits centred
        tip_p = st.env.d.site_xpos[st.tip]
        p = tip_p + 0.30 * (st.env.d.xpos[1:].mean(axis=0) - tip_p)
        look = p.copy() if look is None else 0.90 * look + 0.10 * p
        st.cam.lookat[:] = look
        st.wide.update_scene(st.env.d, st.cam)
        f.append(compose([st.wide.render()],
                         "Motor babbling  -  no target, no reward, no task",
                         "200,000 transitions. The only data the body model ever sees."))

    st.cam.distance = float(st.env.m.stat.extent) * 0.58   # restore the study framing

    # -- 3. frozen, and used to imagine ----------------------------------
    f += card(["That model is then frozen."],
              ["It is never updated again. It is used to imagine short",
               "rollouts while the policy learns the task for real."])
    f += st.solo(os.path.join(FING, "prior_s0.pt"), "frozen body model",
                 "The same body, reaching a target it was never trained on",
                 "myoFinger   seed 0   12,000 training steps", T[:1])

    # -- 4. the other route ----------------------------------------------
    f += card(["The other route skips the body entirely."],
              ["A teacher that already solved the task pulls the student",
               "toward its own actions. No model of anything."])
    f += st.solo(os.path.join(FING, "coach_s0.pt"), "distilled coach",
                 "Same task, same budget, guidance instead of a body model",
                 "myoFinger   seed 0   12,000 training steps", T[:1])

    # -- 5. where they end up ---------------------------------------------
    f += card(["Guidance is faster at first.", "It does not stay ahead."],
              ["The coach leads early on both bodies. On myoFinger the",
               "order has reversed by the end of the budget."])
    f += st.duo(os.path.join(FING, "coach_s0.pt"), os.path.join(FING, "prior_s0.pt"),
                "distilled coach", "frozen body model",
                "Both at 12,000 steps, the same held-out target, frame for frame",
                "Study endpoint, n=12:   coach 122.8 mm      body model 90.5 mm", T[:2])

    # -- 6. what happens when the aid is taken away -----------------------
    f += card(["Is the head start owned, or rented?"],
              ["Each aid was cut at 8,000 steps of a 12,000-step budget,",
               "and the two routes answer differently."])
    f += st.duo(os.path.join(WD, "coach_kept_s0.pt"), os.path.join(WD, "coach_cut_s0.pt"),
                "coach kept", "coach cut at 8k",
                "Cutting the coach costs more than the coach was worth",
                "n=12 paired:  +46.4 mm  [+22.9, +69.9]", T[:1])
    f += st.duo(os.path.join(WD, "prior_kept_s0.pt"), os.path.join(WD, "prior_cut_s0.pt"),
                "body model kept", "body model cut at 8k",
                "Cutting the body model, we cannot show a cost at all",
                "n=12 paired:  +14.2 mm  [-0.9, +29.3]  -  the interval contains zero", T[:1])

    # -- 7. content or machinery ------------------------------------------
    f += card(["Is it what the model knows,", "or just the extra machinery?"],
              ["The control: an untrained model, randomly initialised,",
               "in exactly the same imagination loop."])
    f += st.duo(os.path.join(FING, "prior_s0.pt"), os.path.join(FING, "randprior_s0.pt"),
                "babble-trained model", "RANDOM model, same loop",
                "Identical machinery. Only the model's content differs.",
                "n=12:  -40.8 mm in favour of the trained model  (train targets)", T[:2])
    f += card(["A wrong model is not a cheap model.", "It is worse than no model."],
              ["On myoElbow the random-model arm is significantly worse",
               "than plain model-free RL: +23.8 mrad [+1.7, +45.8]."])

    # -- 8. where it stops -------------------------------------------------
    f += card(["Where this stops."],
              ["Two bodies are analysed: myoElbow, 1 joint, and myoFinger, 4.",
               "On myoHand - 24 joints, 39 muscles - nothing improves in 60,000 steps.",
               "This study trains for 12,000. Such bodies can need hundreds of millions."],
              hold=5.0)

    # -- 9. the claim ------------------------------------------------------
    f += card(["Guidance and a body model", "are not interchangeable."],
              ["Guidance buys early speed, and has to stay attached to keep paying.",
               "The body model's advantage arrives late - and only if it is right."],
              hold=5.0)

    dest = os.path.join(OUT, "explainer_babble.mp4")
    imageio.mimwrite(dest, f, fps=FPS, quality=7, macro_block_size=1)
    print("wrote %s  (%d frames, %.0f s, %.2f MB)"
          % (dest, len(f), len(f) / FPS, os.path.getsize(dest) / 1e6))


if __name__ == "__main__":
    main()
