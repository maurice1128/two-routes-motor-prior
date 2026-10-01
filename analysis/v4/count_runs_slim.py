# -*- coding: utf-8 -*-
"""Counts the per-seed training runs behind the stage-B tables and prose, and sums their
wall-clock from the run logs (last '(Ns)' or 'done in Ns' in each log). Fills NRUNS/NHOURS."""
import os, re, glob
W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
ZS = r"C:\Users\maurice\Desktop\world_model_zeroshot"
PRIOR_DIR = {"arm1": "prior", "arm2": "prior", "arm3": "prior", "arm4": "prior"}
for line in open(os.path.join(W, "tcds_v4", "common.py"), encoding="utf-8"):
    if line.startswith("PRIOR_DIR"):
        PRIOR_DIR = eval(line.split("=", 1)[1].split("#")[0])

groups = {}
def add(g, d, c, root=W, n=None):
    fs = sorted(glob.glob(os.path.join(root, d, "%s_seed*.json" % c)))
    if n is not None: fs = [f for f in fs if int(re.search(r"_seed(\d+)", f).group(1)) < n]
    groups.setdefault(g, []).extend(fs)

# Table I
for k, c in (("blank", "blank"), ("blank64", "blank64"), ("prior", "prior"), ("randprior", "prior")):
    add("I core elbow", "results_conv_elbow_%s" % k, c)
for k, c in (("blank", "blank"), ("blank64", "blank64"), ("prior", "prior"), ("randprior", "prior")):
    add("I core finger", "results_conv_finger_%s" % k, c)
# Table II
for b in ("arm1", "arm2", "arm3", "arm4"):
    add("II arms", "results_arms100k_%s_blank" % b, "blank"); add("II arms", "results_arms100k_%s_blank64" % b, "blank64")
    add("II arms", "results_arms100k_%s_%s" % (b, PRIOR_DIR[b]), "prior"); add("II arms", "results_arms100k_%s_randprior" % b, "prior")
# Tables III/IV elbow (100k runs) + 12k-study runs that supply the 1k-grid dips
for tw in (2000, 3000, 4000, 5000, 6000, 7000):
    add("III/IV elbow sweep", "results_sweep100k_elbow/w%d" % tw, "abrupt"); add("III/IV elbow sweep", "results_sweep100k_elbow/w%d" % tw, "selfanchor")
add("III/IV elbow 8k", "results_coach100k_elbow", "abrupt"); add("III/IV elbow 8k", "results_coach100k_elbow_off2", "selfanchor"); add("III/IV elbow 8k", "results_coach100k_elbow", "constant")
for tw in (2000, 3000, 4000, 5000, 6000, 7000, 8000):
    add("III/IV elbow 12k dips", "b2_pilot/results_attachment_myoelbow", "abrupt_w%d" % tw, ZS); add("III/IV elbow 12k dips", "b2_pilot/results_attachment_myoelbow", "selfanchor_w%d" % tw, ZS)
for c in ("constant", "none"):
    add("III/IV elbow 12k dips", "b2_pilot/results_withdrawal_myoelbow", c, ZS)
# Tables III/IV finger T2
for tw in (2000, 3000, 4000, 5000, 6000, 7000):
    add("III/IV finger sweep", "results_sweep100k_finger_T2/w%d" % tw, "abrupt"); add("III/IV finger sweep", "results_sweep100k_finger_T2/w%d" % tw, "selfanchor")
add("III/IV finger sweep", "results_sweep100k_finger_T2", "constant"); add("III/IV finger sweep", "results_sweep100k_finger_T2", "none")
for c in ("abrupt", "selfanchor"): add("III/IV finger 8k", "results_coach100k_finger_T2", c)
# Table V
# Table VI + finger 12k withdrawal sentence + budget sentence
for c in ("none", "constant"): add("elbow 200k", "results_long200k_elbow", c)
for tw in (2000, 3000, 4000, 5000, 6000, 7000, 8000): add("elbow 200k", "results_long200k_elbow/w%d" % tw, "abrupt")

# carried 12k-study runs log their time in batched logs: "[<arm> [w<tw> ]seed<N>] done in <s>s"
CARRIED = {}
for pat in ("b2_pilot/logs_sweep/*.log", "b2_pilot/b2_*.log", "b1_pilot/b1_*.log", "b1_pilot/logs_finger/*.log"):
    for lg in sorted(glob.glob(os.path.join(ZS, pat))):
        body = "finger" if "finger" in lg else "elbow"
        for m in re.finditer(r"\[(\w+)(?: w(\d+))? seed(\d+)\] done in (\d+(?:\.\d+)?)s", open(lg, errors="ignore").read()):
            arm, tw, sd, t = m.group(1), m.group(2), int(m.group(3)), float(m.group(4))
            CARRIED[(body, arm, tw, sd)] = t          # later (retry) logs overwrite earlier ones


def carried_time(f):
    s = int(re.search(r"_seed(\d+)\.json$", f).group(1)); base = os.path.basename(f)
    body = "finger" if "finger" in f else "elbow"
    m = re.match(r"(\w+?)_w(\d+)_seed", base)
    arm, tw = (m.group(1), m.group(2)) if m else (base.split("_seed")[0], None)
    return CARRIED.get((body, arm, tw, s))


total = 0; hours = 0.0; nolog = []
for g, fs in groups.items():
    gh = 0.0
    for f in fs:
        total += 1
        cands = [re.sub(r"_seed(\d+)\.json$", r"_s\1.log", f), re.sub(r"_seed(\d+)\.json$", r"_seed\1.log", f)]
        t = None
        for lg in cands:
            if os.path.exists(lg):
                txt = open(lg, errors="ignore").read()
                m = re.findall(r"\((\d+)s\)", txt) or re.findall(r"done in (\d+(?:\.\d+)?)s", txt)
                if m: t = float(m[-1]); break
        if t is None and f.startswith(ZS): t = carried_time(f)
        if t is None: nolog.append(f)
        else: gh += t / 3600
    hours += gh
    print("%-32s runs %4d  hours %7.1f" % (g, len(fs), gh))
print("TOTAL runs %d, CPU-hours %.0f, runs without a timed log: %d" % (total, hours, len(nolog)))
for f in nolog[:10]: print("   no log:", f)
