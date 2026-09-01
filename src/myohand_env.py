"""MyoSuite myoHand adapter — same public interface as arm_env / myofinger_env, so the whole
world-model-prior pipeline runs unchanged on the HARDEST recognized body: a full hand.

Body: myoHandReachRandom-v0 — 23 DoF, 39 Hill muscles.
We use the MyoSuite env purely for its muscle DYNAMICS (mj_step); we compute our OWN
reach reward so it is identical for real and imagined transitions (required for imagination).

Design (single-fingertip reach — mirrors myofinger so ONLY body complexity changes):
Dynamics state: s = [qpos(23), qvel(23), act(39)] -> 85
Goal: the 3-D world position of site 'IFtip' (index fingertip). fk() requires forward
kinematics (mj_forward), so we keep a persistent MjModel+MjData and make fk a deterministic
function of qpos ONLY. Drive ALL 39 muscles. GOAL_DIM stays 3 (identical to finger).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import mujoco
from myosuite.utils import gym as _gym

ENV_ID = "myoHandReachRandom-v0"
STATE_DIM = 85     # qpos23 + qvel23 + act39
ACT_DIM = 39
GOAL_DIM = 3
OBS_DIM = STATE_DIM + 2 * GOAL_DIM   # state + fk(3) + target(3) = 91
_NQ = 23
_NA = 39

def _load_model():
    env = _gym.make(ENV_ID)
    m = env.unwrapped.mj_model
    env.close()
    return m

# Persistent model/data used ONLY for forward kinematics (fk). Deterministic in qpos.
_FK_M = _load_model()
_FK_D = mujoco.MjData(_FK_M)
_IFTIP = mujoco.mj_name2id(_FK_M, mujoco.mjtObj.mjOBJ_SITE, "IFtip")
# Joint ranges from the validated model (all 23 joints are 1-DoF -> nq==nv==23).
JOINT_LO = _FK_M.jnt_range[:_NQ, 0].copy()
JOINT_HI = _FK_M.jnt_range[:_NQ, 1].copy()

def fk(qpos):
    """3-D fingertip (IFtip) world position; deterministic function of qpos only.
    Supports single (nq,) or batched (...,nq) input, returning (...,3)."""
    q = np.asarray(qpos, dtype=np.float64)
    Q = q.reshape(-1, _NQ)
    out = np.empty((Q.shape[0], 3), dtype=np.float64)
    for i in range(Q.shape[0]):
        _FK_D.qpos[:_NQ] = Q[i]
        mujoco.mj_forward(_FK_M, _FK_D)
        out[i] = _FK_D.site_xpos[_IFTIP]
    return out.reshape(*q.shape[:-1], 3)

def reach_reward(qpos_next, target, ctrl=None):
    tip = fk(qpos_next)
    dist = np.linalg.norm(tip - target, axis=-1)
    r = -dist + 0.1 * (dist < 0.01)
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
    """Same API as arm_env.ArmReachEnv. Loads the validated MyoSuite hand muscle model,
    then drives it with RAW MuJoCo (mj_step) for clean, consistent dynamics."""
    def __init__(self, frame_skip=10, max_steps=100, seed=0):
        env = _gym.make(ENV_ID)
        self.m = env.unwrapped.mj_model
        self.d = mujoco.MjData(self.m)
        env.close()
        self.tip_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "IFtip")
        self.frame_skip = frame_skip; self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.target = fk(0.5 * (JOINT_LO + JOINT_HI)); self.t = 0

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
        tip = self.d.site_xpos[self.tip_id]
        dist = float(np.linalg.norm(tip - self.target))
        return obs, r, done, {"dist": dist, "ctrl": ctrl}


if __name__ == "__main__":
    e = ArmReachEnv(seed=0)
    o = e.reset()
    print("BODY=myohand STATE_DIM", STATE_DIM, "ACT_DIM", ACT_DIM, "OBS_DIM", OBS_DIM, "obs", o.shape)
    tot = 0
    for _ in range(100):
        o, r, d, inf = e.step(e.rng.uniform(-1, 1, ACT_DIM)); tot += r
    print("random return %.2f final dist %.4f m" % (tot, inf["dist"]))
    s = e.get_state(); e.set_state(s); print("state dim", s.shape, "set_state ok")
    q = e.rng.uniform(JOINT_LO, JOINT_HI)
    a = fk(q); b = fk(q)
    print("fk deterministic:", np.allclose(a, b), "fk=", np.round(a, 4))
