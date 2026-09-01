"""Muscle-body reach/pose environment for the world-model-as-prior study.

Body is selected by WM_BODY env var:
  'arm2'     : custom 2-link, 4 muscles    -> state 8,  act 4  (easy toy)
  'arm3'     : custom 3-link, 8 muscles    -> state 14, act 8  (harder: biarticular)
  'myoelbow' : MyoSuite myoElbow, 6 muscles -> state 8, act 6  (recognized benchmark; see myo_env.py)

Dynamics state (what the world model predicts): s = [qpos(nq), qvel(nq), act(na)]
Policy observation:                             o = [state, goal(2), target(2)]
The reward is a deterministic function of state, so identical reward can be computed for
real AND imagined states (required for imagination-based training).
"""
import os

BODY = os.environ.get("WM_BODY", "arm3")

if BODY.startswith("myo"):
    # Delegate to the MyoSuite adapter (identical public interface).
    if BODY == "myofinger":
        from myofinger_env import (ArmReachEnv, fk, reach_reward, task_reward, STATE_DIM, ACT_DIM,
                                   OBS_DIM, JOINT_LO, JOINT_HI, TASK_TARGET_SEED)
    elif BODY == "myohand":
        from myohand_env import (ArmReachEnv, fk, reach_reward, task_reward, STATE_DIM, ACT_DIM,
                                 OBS_DIM, JOINT_LO, JOINT_HI, TASK_TARGET_SEED)
    else:
        from myo_env import (ArmReachEnv, fk, reach_reward, task_reward, STATE_DIM, ACT_DIM,
                             OBS_DIM, JOINT_LO, JOINT_HI, TASK_TARGET_SEED)
else:
    import numpy as np
    import mujoco

    _CFG = {
        "arm1": {"xml": "arm1_muscle.xml", "L": [0.25],
                 "lo": [-2.0], "hi": [2.0]},
        "arm2": {"xml": "arm2_muscle.xml", "L": [0.25, 0.22],
                 "lo": [-1.0, -1.14], "hi": [1.04, 0.20]},
        "arm3": {"xml": "arm3_muscle.xml", "L": [0.25, 0.22, 0.16],
                 "lo": [-1.0, -1.2, -1.0], "hi": [1.0, 0.2, 1.0]},
        "arm4": {"xml": "arm4_muscle.xml", "L": [0.22, 0.20, 0.15, 0.12],
                 "lo": [-1.0, -1.2, -1.0, -1.0], "hi": [1.0, 0.2, 1.0, 1.0]},
    }[BODY]

    XML = _CFG["xml"]
    _L = np.array(_CFG["L"])
    JOINT_LO = np.array(_CFG["lo"])
    JOINT_HI = np.array(_CFG["hi"])
    _NQ = len(_L)

    _m0 = mujoco.MjModel.from_xml_path(XML)
    ACT_DIM = _m0.nu
    STATE_DIM = 2 * _NQ + _m0.na
    OBS_DIM = STATE_DIM + 4
    _NA = _m0.na

    def fk(qpos):
        q = np.asarray(qpos)
        th = np.cumsum(q[..., :_NQ], axis=-1)
        x = np.sum(np.stack([_L[i] * np.cos(th[..., i]) for i in range(_NQ)], -1), -1)
        y = np.sum(np.stack([_L[i] * np.sin(th[..., i]) for i in range(_NQ)], -1), -1)
        return np.stack([x, y], axis=-1)

    def reach_reward(qpos_next, target, ctrl=None):
        tip = fk(qpos_next)
        dist = np.linalg.norm(tip - target, axis=-1)
        r = -dist + 0.1 * (dist < 0.05)
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
        def __init__(self, frame_skip=10, max_steps=100, seed=0):
            self.model = mujoco.MjModel.from_xml_path(XML)
            self.data = mujoco.MjData(self.model)
            self.nq = self.model.nq; self.na = self.model.na
            self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
            self.frame_skip = frame_skip; self.max_steps = max_steps
            self.rng = np.random.default_rng(seed)
            self.target = np.array([0.3, 0.2]); self.t = 0

        def get_state(self):
            d = self.data
            act = d.act.copy() if self.model.na > 0 else np.zeros(0)
            return np.concatenate([d.qpos.copy(), d.qvel.copy(), act]).astype(np.float64)

        def set_state(self, s):
            d = self.data
            d.qpos[:] = s[0:self.nq]; d.qvel[:] = s[self.nq:2 * self.nq]
            if self.model.na > 0:
                d.act[:] = s[2 * self.nq:2 * self.nq + self.na]
            mujoco.mj_forward(self.model, d)

        def _obs(self):
            tip = self.data.site_xpos[self.tip_id][:2].copy()
            return np.concatenate([self.get_state(), tip, self.target]).astype(np.float32)

        def sample_target(self):
            return fk(self.rng.uniform(JOINT_LO, JOINT_HI))

        def reset(self, target=None, randomize_pose=True):
            mujoco.mj_resetData(self.model, self.data)
            if randomize_pose:
                self.data.qpos[:] = self.rng.uniform(JOINT_LO * 0.5, JOINT_HI * 0.5)
            self.target = self.sample_target() if target is None else np.asarray(target, float)
            self.data.mocap_pos[0] = [self.target[0], self.target[1], 0.0]
            mujoco.mj_forward(self.model, self.data)
            self.t = 0
            return self._obs()

        def step(self, action):
            ctrl = np.clip((np.asarray(action) + 1.0) * 0.5, 0.0, 1.0)
            self.data.ctrl[:] = ctrl
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
            self.t += 1
            obs = self._obs()
            r = float(reach_reward(self.data.qpos, self.target, ctrl))
            done = self.t >= self.max_steps
            tip = self.data.site_xpos[self.tip_id][:2]
            info = {"dist": float(np.linalg.norm(tip - self.target)), "ctrl": ctrl}
            return obs, r, done, info


if __name__ == "__main__":
    print(f"BODY={BODY}  STATE_DIM={STATE_DIM}  ACT_DIM={ACT_DIM}  OBS_DIM={OBS_DIM}")
    env = ArmReachEnv(seed=0)
    o = env.reset()
    tot = 0
    for _ in range(100):
        o, r, dn, inf = env.step(env.rng.uniform(-1, 1, ACT_DIM)); tot += r
    print("random return %.2f final dist %.3f" % (tot, inf["dist"]))
