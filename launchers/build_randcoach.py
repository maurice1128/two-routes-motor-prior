"""The adversarial control for the GUIDANCE route, mirroring build_randprior.py.

`randprior` asks whether the frozen model's benefit comes from what it learned or
from the Dyna machinery around it, by swapping in an untrained model of identical
architecture.  Nothing has ever asked the same question of the coach: on myoFinger
the teacher sits essentially at the no-op referent (169.1 mm against 172.3), so
"distil toward a fixed non-flailing action target" predicts the observed early gain
independently of anything the teacher encodes about the task.  This builds that
control -- a teacher with the SAME architecture and RANDOM weights.

Same construction as build_randprior.py: read the trained artefact to recover the
shape, then emit an untrained network of exactly that shape.  A teacher checkpoint
is a bare Actor state_dict (no normalizer stats to carry across, unlike a prior).
"""
import torch
import arm_env
from sac_dyna import Actor


def build(src, dst, seed=0):
    ck = torch.load(src, map_location="cpu", weights_only=True)
    hidden = ck["body.0.weight"].shape[0]
    obs_dim = ck["body.0.weight"].shape[1]
    act_dim = ck["mu.weight"].shape[0]
    torch.manual_seed(seed)
    t = Actor(obs_dim, act_dim, hidden)          # fresh random-init weights
    sd = t.state_dict()
    assert set(sd) == set(ck), "architecture mismatch: %s vs %s" % (
        sorted(set(sd) ^ set(ck)), src)
    for k in sd:
        assert sd[k].shape == ck[k].shape, "shape mismatch at %s" % k
    torch.save(sd, dst)
    print("built %s: obs=%d act=%d hidden=%d (random weights, teacher architecture)"
          % (dst, obs_dim, act_dim, hidden))


if __name__ == "__main__":
    build("teacher_elbow_clean30k.pt", "teacher_rand_elbow.pt", seed=0)
    build("teacher_finger_clean30k.pt", "teacher_rand_finger.pt", seed=0)
