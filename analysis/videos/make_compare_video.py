"""Side-by-side rollouts on myoFinger: two arms, same target, same timestep.

Designed so the result is visible without reading anything.  A red sphere marks
the target and turns green when the fingertip arrives; a blue trail shows the
path taken; a bar under each panel empties as the fingertip closes on the
target.  Whoever's bar empties first, and whose sphere turns green, is the one
that got there.

Four seeds are played in sequence rather than one, because every claim in the
paper is a 12-seed mean and a single seed can run opposite to it.  The banner
carries the study's own 12-seed endpoint mean so the sample on screen is never
mistaken for the result.

usage:  python make_compare_video.py coach prior
"""
import os, sys, warnings
os.environ.setdefault("WM_BODY", "myofinger")
warnings.filterwarnings("ignore")

import numpy as np, torch, mujoco, imageio.v2 as imageio
from PIL import Image, ImageDraw
import arm_env, run_reach, viz

BODY = os.environ.get("WM_BODY", "myofinger")
TIP_SITE = "IFtip" if BODY == "myofinger" else "tip"
# WM_ACTORS overrides the actor directory so a late-added arm (randcoach) can be
# played without a second copy of this script.
ACT_DIR = os.environ.get("WM_ACTORS",
                         "actors_finger" if BODY == "myofinger" else "actors_video")
from sac_dyna import Actor

ROOT = os.path.dirname(os.path.abspath(__file__))
ACT = None  # set in main()
OUT = os.path.join(ROOT, "videos"); os.makedirs(OUT, exist_ok=True)
W, H, FPS, TRAIL = 460, 400, 30, 26
STUDY = {"blank": "117.6 mm", "prior": "90.5 mm", "coach": "122.8 mm",
         "randprior": "train targets only", "randcoach": "142.0 mm"}
if ACT_DIR.endswith("100k"):
    # converged budget: Table I of the merged paper, held-out, n=12, 100,000 steps
    STUDY = {"blank": "94.7 mm", "prior": "54.4 mm", "coach": "123.0 mm",
             "randprior": "103.1 mm", "randcoach": "109.1 mm"}
    BUDGET_TAG = "100k"
else:
    BUDGET_TAG = "12k"
NAME = {"blank": "model-free SAC", "prior": "frozen body model",
        "coach": "distilled coach", "randprior": "RANDOM body model",
        "randcoach": "RANDOM teacher, same loss"}


def load(p):
    a = Actor(arm_env.OBS_DIM, arm_env.ACT_DIM, hidden=128)
    a.load_state_dict(torch.load(p, map_location="cpu")); a.eval(); return a


@torch.no_grad()
def run_one(actor, target, env, rend, cam, steps=100):
    o = env.reset(target=target, randomize_pose=False)
    tipid = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
    frames, dists, trail = [], [], []
    d0 = None
    for _ in range(steps):
        a = actor.act(torch.tensor(o[None], dtype=torch.float32), greedy=True).numpy()[0]
        o, r, dn, inf = env.step(a)
        tip = env.d.site_xpos[tipid].copy()
        trail.append(tip); trail[:] = trail[-TRAIL:]
        rend.update_scene(env.d, cam)
        d = viz.decorate(rend.scene, target, tip, trail)
        if d0 is None: d0 = d
        frames.append(rend.render()); dists.append(d)
    return frames, dists, d0


def main(left, right):
    global ACT
    ACT = os.path.join(ROOT, ACT_DIR)
    def pat(a, sd):
        return "%s_actor.pt" % a if ACT_DIR == "actors_video" else "%s_s%d.pt" % (a, sd)
    cand = [0] if ACT_DIR == "actors_video" else [0, 1, 2, 3]
    seeds = [s for s in cand
             if os.path.exists(os.path.join(ACT, pat(left, s)))
             and os.path.exists(os.path.join(ACT, pat(right, s)))]
    if not seeds:
        print("no matching seeds yet for %s / %s" % (left, right)); return 1
    targets = run_reach.fixed_eval_targets(20)[:2]
    env = arm_env.ArmReachEnv(seed=7); env.reset(randomize_pose=False)
    # the model ships its own IFtip_target marker at a fixed pose; hide it so the
    # only sphere on screen is the episode's actual target
    for nm in ("IFtip_target", "wrist_target"):
        i = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_SITE, nm)
        if i >= 0:
            env.m.site_rgba[i] = [0, 0, 0, 0]
    rend = mujoco.Renderer(env.m, height=H, width=W)
    tipid = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.distance = float(env.m.stat.extent) * 0.58
    cam.azimuth, cam.elevation = 140.0, -25.0

    def aim(target):
        """Frame the fingertip and this episode's target together."""
        mujoco.mj_forward(env.m, env.d)
        p = env.d.site_xpos[tipid]
        t = np.asarray(target, float)
        if t.shape[0] < p.shape[0]:
            t = np.concatenate([t, p[t.shape[0]:]])
        cam.lookat[:] = (p + t) / 2.0

    out = []
    for sd in seeds:
        al = load(os.path.join(ACT, pat(left, sd)))
        ar = load(os.path.join(ACT, pat(right, sd)))
        for tg in targets:
            env.reset(target=tg, randomize_pose=False); aim(tg)
            fl, dl, l0 = run_one(al, tg, env, rend, cam)
            fr, dr, r0 = run_one(ar, tg, env, rend, cam)
            for i in range(min(len(fl), len(fr))):
                a = viz.bar(fl[i], dl[i], l0, NAME[left], "seed %d" % sd)
                b = viz.bar(fr[i], dr[i], r0, NAME[right], "seed %d" % sd)
                quad = np.concatenate([a, b], axis=1)
                pad = np.zeros((34, quad.shape[1], 3), dtype=quad.dtype)
                im = Image.fromarray(np.concatenate([quad, pad], axis=0))
                g = ImageDraw.Draw(im)
                g.rectangle([0, im.height - 34, im.width, im.height], fill=(18, 18, 22))
                g.text((12, im.height - 25),
                       "%s  |  study endpoint mean at %s" % ("myoFinger" if BODY=="myofinger" else "myoElbow", BUDGET_TAG) + ", n=12 held-out:  %s %s   vs   %s %s"
                       % (NAME[left], STUDY[left], NAME[right], STUDY[right]),
                       fill=(180, 180, 190), font=viz.F_SUB)
                out.append(np.asarray(im))
    dest = os.path.join(OUT, "compare_%s_vs_%s_%s.mp4" % (left, right, BUDGET_TAG))
    imageio.mimwrite(dest, out, fps=FPS, quality=8, macro_block_size=1)
    print("wrote %s  (%d seeds, %d frames)" % (dest, len(seeds), len(out)))
    return 0


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "coach"
    b = sys.argv[2] if len(sys.argv) > 2 else "prior"
    sys.exit(main(a, b))
