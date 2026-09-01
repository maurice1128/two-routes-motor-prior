"""MyoSuite myoElbow adapter exposing the SAME interface as arm_env, so the whole
world-model-prior pipeline runs unchanged on a RECOGNIZED muscle benchmark.

Body: myoElbowPose1D6MRandom-v0 — 1-DoF elbow, 6 Hill muscles (MyoSuite/L4DC).
We use the MyoSuite env purely for its muscle DYNAMICS (mj step); we compute our OWN
pose reward so it is identical for real and imagined transitions (required for imagination).

Dynamics state: s = [qpos(1), qvel(1), act(6)] -> 8   (same size as arm2)
Goal: the 1-D target elbow angle, embedded as 2-D [angle, 0] so downstream code that
assumes a 2-D fingertip/target works verbatim.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import mujoco
from myosuite.utils import gym as _gym

ENV_ID = "myoElbowPose1D6MRandom-v0"
JOINT_LO = np.array([0.0])
JOINT_HI = np.array([2.269])
STATE_DIM = 8      # qpos1 + qvel1 + act6
ACT_DIM = 6
OBS_DIM = STATE_DIM + 4   # + goal-embed(2) + target(2)
_NQ = 1
_NA = 6

def fk(qpos):
    """Goal-space 'position' of the elbow: the joint angle, embedded in 2-D as [angle, 0]."""
    q = np.asarray(qpos)
    ang = q[..., 0]
    return np.stack([ang, np.zeros_like(ang)], axis=-1)

def reach_reward(qpos_next, target, ctrl=None):
    ang = fk(qpos_next)
    dist = np.linalg.norm(ang - target, axis=-1)   # = |angle - target_angle|
    r = -dist + 0.1 * (dist < 0.10)
    if ctrl is not None:
        r = r - 0.001 * np.sum(np.asarray(ctrl) ** 2, axis=-1)
    return r

def task_reward(state_next, target, ctrl=None, task="reach"):
    state_next = np.asarray(state_next)
    qpos = state_next[..., :_NQ]; qvel = state_next[..., _NQ:2 * _NQ]
    r = reach_reward(qpos, target, ctrl)
    if task == "hold":
        r = r - 0.05 * np.linalg.norm(qvel, axis=-1)
    return r

TASK_TARGET_SEED = {"reach": 777, "reachB": 999, "hold": 777}

class ArmReachEnv:
    """Same API as arm_env.ArmReachEnv. Loads the validated MyoSuite elbow muscle model,
    then drives it with RAW MuJoCo (mj_step) — bypassing the gym wrapper's reset/step
    semantics for clean, consistent dynamics (identical treatment to the custom arms)."""
    def __init__(self, frame_skip=10, max_steps=100, seed=0):
        env = _gym.make(ENV_ID)
        self.m = env.unwrapped.mj_model            # keep the validated muscle model
        self.d = mujoco.MjData(self.m)
        env.close()
        self.frame_skip = frame_skip; self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.target = np.array([1.0, 0.0]); self.t = 0

    def get_state(self):
        return np.concatenate([self.d.qpos[:_NQ].copy(), self.d.qvel[:_NQ].copy(),
                               self.d.act[:_NA].copy()]).astype(np.float64)

    def set_state(self, s):
        self.d.qpos[:_NQ] = s[:_NQ]; self.d.qvel[:_NQ] = s[_NQ:2 * _NQ]
        self.d.act[:_NA] = s[2 * _NQ:2 * _NQ + _NA]
        mujoco.mj_forward(self.m, self.d)

    def _obs(self):
        st = self.get_state()
        return np.concatenate([st, fk(st[:_NQ]), self.target]).astype(np.float32)

    def sample_target(self):
        return fk(self.rng.uniform(JOINT_LO, JOINT_HI))

    def reset(self, target=None, randomize_pose=True):
        mujoco.mj_resetData(self.m, self.d)
        if randomize_pose:
            self.d.qpos[:_NQ] = self.rng.uniform(JOINT_LO, JOINT_HI)
        self.d.qvel[:_NQ] = 0.0; self.d.act[:_NA] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.target = self.sample_target() if target is None else np.asarray(target, float)
        self.t = 0
        return self._obs()

    def step(self, action):
        ctrl = np.clip((np.asarray(action) + 1.0) * 0.5, 0.0, 1.0)  # -> muscle activation [0,1]
        self.d.ctrl[:_NA] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.m, self.d)
        self.t += 1
        obs = self._obs()
        r = float(reach_reward(self.d.qpos[:_NQ], self.target, ctrl))
        done = self.t >= self.max_steps
        dist = float(abs(self.d.qpos[0] - self.target[0]))
        return obs, r, done, {"dist": dist, "ctrl": ctrl}


if __name__ == "__main__":
    e = ArmReachEnv(seed=0)
    o = e.reset()
    print("BODY=myoelbow STATE_DIM", STATE_DIM, "ACT_DIM", ACT_DIM, "OBS_DIM", OBS_DIM, "obs", o.shape)
    tot = 0
    for _ in range(100):
        o, r, d, inf = e.step(e.rng.uniform(-1, 1, ACT_DIM)); tot += r
    print("random return %.2f final dist %.3f rad  target %.2f" % (tot, inf["dist"], e.target[0]))
    s = e.get_state(); e.set_state(s); print("state dim", s.shape, "set_state ok")
