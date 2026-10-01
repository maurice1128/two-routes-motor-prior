# -*- coding: utf-8 -*-
"""Shared data access and statistics for the v4 paper.

Endpoint (council 2026-09-29, option A): every run is scored by the mean of its six held-out
evaluations at 90k, 92k, ..., 100k ("late mean"), not by the single 100k evaluation, because single
evaluations of the same run fluctuate by several mrad/mm between neighbouring checkpoints.
Early endpoints (12k) and immediate dips stay single-checkpoint, as defined in the paper."""
import json, os, glob, math
import numpy as np

W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
Z = r"C:\Users\maurice\Desktop\world_model_zeroshot\b2_pilot"
LATE = list(range(90000, 100001, 2000))
EARLY_WINDOW = list(range(70000, 80001, 2000))          # for the plateau check
TWS = [2000, 3000, 4000, 5000, 6000, 7000, 8000]
TW4 = [2000, 4000, 6000, 8000]
S12 = list(range(12))
T2 = {11: 2.201, 23: 2.069}      # two-sided 95%, by df
T1 = {11: 1.796, 23: 1.714}      # one-sided 95%, by df


def load(d, c, root=W):
    r = {}
    for f in glob.glob(os.path.join(root, d, "%s_seed*.json" % c)):
        s = int(f.split("_seed")[1].split(".")[0])
        r[s] = {int(x["step"]): x["eval_dist"] * 1000 for x in json.load(open(f))}
    return r


def end(run, s):
    return float(np.mean([run[s][k] for k in LATE]))


def win(run, s, steps):
    return float(np.mean([run[s][k] for k in steps]))


def ci(v):
    v = np.asarray(v, float); n = len(v); m = float(v.mean()); se = float(v.std(ddof=1)) / math.sqrt(n)
    return m, m - T2[n - 1] * se, m + T2[n - 1] * se


def upper1(v):
    """one-sided 95% upper bound of the mean"""
    v = np.asarray(v, float); n = len(v); return float(v.mean() + T1[n - 1] * v.std(ddof=1) / math.sqrt(n))


def sig(c):
    return c[1] > 0 or c[2] < 0


def slope(per_seed_series, xs):
    return [float(np.polyfit(xs, y, 1)[0]) for y in per_seed_series]


# ------------------------------------------------------------------ the data sets used in the paper
def core():
    E = {k: load("results_conv_elbow_%s" % k, c) for k, c in (("blank", "blank"), ("blank64", "blank64"), ("prior", "prior"), ("randprior", "prior"), ("coach", "coach"), ("randcoach", "coach"), ("priorcoach", "priorcoach"))}
    F = {k: load("results_conv_finger_%s" % k, c) for k, c in (("blank", "blank"), ("blank64", "blank64"), ("prior", "prior"), ("randprior", "prior"), ("coach", "coach"), ("randcoach", "coach"), ("priorcoach", "priorcoach"), ("coachT2", "coach"), ("priorcoachT2", "priorcoach"))}
    return E, F


PRIOR_DIR = {"arm1": "prior2", "arm2": "prior", "arm3": "prior", "arm4": "prior2"}


def arms():
    out = {}
    for b in ("arm1", "arm2", "arm3", "arm4"):
        A = {"blank": load("results_arms100k_%s_blank" % b, "blank"), "blank64": load("results_arms100k_%s_blank64" % b, "blank64"),
             "prior": load("results_arms100k_%s_%s" % (b, PRIOR_DIR[b]), "prior"), "randprior": load("results_arms100k_%s_randprior" % b, "prior")}
        seeds = sorted(set.intersection(*[set(v) for v in A.values()]))
        out[b] = (A, seeds)
    return out


def sweep(body):
    """additive regime, both bodies: (constant, none, {tw: (abrupt, selfanchor)}, dip-grid version)"""
    if body == "elbow":
        con = load("results_coach100k_elbow", "constant"); none = load("results_conv_elbow_blank", "blank")
        arms_ = {tw: (load("results_sweep100k_elbow/w%d" % tw, "abrupt"), load("results_sweep100k_elbow/w%d" % tw, "selfanchor")) for tw in TWS[:-1]}
        arms_[8000] = (load("results_coach100k_elbow", "abrupt"), load("results_coach100k_elbow_off2", "selfanchor"))
        dip = {tw: (load("results_attachment_myoelbow", "abrupt_w%d" % tw, Z), load("results_attachment_myoelbow", "selfanchor_w%d" % tw, Z)) for tw in TWS}
        dcon = load("results_withdrawal_myoelbow", "constant", Z); dnone = load("results_withdrawal_myoelbow", "none", Z)
        ra = load("results_coach100k_elbow", "randanchor"); dra = load("results_withdrawal_myoelbow", "randanchor", Z)
    else:
        con = load("results_sweep100k_finger_T2", "constant"); none = load("results_sweep100k_finger_T2", "none")
        arms_ = {tw: (load("results_sweep100k_finger_T2/w%d" % tw, "abrupt"), load("results_sweep100k_finger_T2/w%d" % tw, "selfanchor")) for tw in TWS[:-1]}
        arms_[8000] = (load("results_coach100k_finger_T2", "abrupt"), load("results_coach100k_finger_T2", "selfanchor"))
        dip, dcon, dnone = arms_, con, none
        ra = load("results_coach100k_finger_T2", "randanchor"); dra = ra
    return dict(con=con, none=none, arms=arms_, dip=dip, dcon=dcon, dnone=dnone, ra=ra, dra=dra)


def replace(body):
    rdir = "results_replace100k_my%s" % ("oelbow" if body == "elbow" else "ofinger")
    con = load(rdir, "constant")
    none = load("results_conv_elbow_blank", "blank") if body == "elbow" else load("results_sweep100k_finger_T2", "none")
    arm = {a: {tw: load(rdir, "%s_w%d" % (a, tw)) for tw in TW4} for a in ("abrupt", "selfanchor", "gateon")}
    return con, none, arm


def bodymodel(body):
    E, F = core()
    B = E if body == "elbow" else F
    M = {a: load("results_matched100k_%s" % body, a) for a in ("soft", "purge", "matched")}
    M["keep"] = B["prior"]; M["blank"] = B["blank"]
    return M
