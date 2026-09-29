"""EXPERIMENT A control: build a RANDOM-INIT prior with the SAME architecture and the
SAME normalizer stats as a trained prior, but UNTRAINED network weights. Isolates whether
the prior benefit is LEARNED body dynamics vs generic frozen-model Dyna regularization.
"""
import sys, torch, numpy as np
from world_model import WorldModel

def build(src, dst, seed=0):
    ck = torch.load(src, weights_only=False)
    hidden = ck["hidden"]
    in_dim = ck["state_dict"]["net.0.weight"].shape[1]
    state_dim = ck["state_dict"]["net.4.weight"].shape[0]
    act_dim = in_dim - state_dim
    torch.manual_seed(seed)
    wm = WorldModel(state_dim, act_dim, hidden)   # fresh random-init weights
    # keep EXACTLY the trained prior's normalizer stats (only network weights differ)
    out = {"state_dict": wm.state_dict(), "hidden": hidden,
           "in_mean": ck["in_mean"], "in_std": ck["in_std"],
           "out_mean": ck["out_mean"], "out_std": ck["out_std"]}
    torch.save(out, dst)
    print(f"built {dst}: state_dim={state_dim} act_dim={act_dim} hidden={hidden} (random weights, trained-prior norms)")

if __name__ == "__main__":
    build("prior_myo.pt", "prior_rand_elbow.pt", seed=0)
    build("prior_finger.pt", "prior_rand_finger.pt", seed=0)
