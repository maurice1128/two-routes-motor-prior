# -*- coding: utf-8 -*-
"""Stage-B figures, drawn from the same per-seed JSON the tables are audited against.
Mean across seeds, band = +-1 SE (curves) or paired-t 95% interval (component plots)."""
import json, os, math, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8})

W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
OUT = os.path.join(W, "tcds_v3", "figs"); os.makedirs(OUT, exist_ok=True)
CK = list(range(2000, 100001, 2000))
TQ = {12: 2.201, 24: 2.069}
PRIOR_DIR = {"arm1": "prior2", "arm2": "prior", "arm3": "prior", "arm4": "prior2"}   # arm1/arm4: priors rebuilt by the recorded recipe (prior_arm{1,4}_200k.pt)


def load(d, c, seeds=range(12)):
    out = []
    for s in seeds:
        r = {int(x["step"]): x["eval_dist"] * 1000 for x in json.load(open(os.path.join(W, d, "%s_seed%d.json" % (c, s))))}
        out.append([r[k] for k in CK])
    return np.array(out)


def loadd(d, c):
    r = {}
    for f in glob.glob(os.path.join(W, d, "%s_seed*.json" % c)):
        s = int(f.split("_seed")[1].split(".")[0]); r[s] = {int(x["step"]): x["eval_dist"] * 1000 for x in json.load(open(f))}
    return r


def band(ax, y, label, color, ls="-"):
    m = y.mean(0); se = y.std(0, ddof=1) / math.sqrt(y.shape[0])
    ax.plot(CK, m, ls, color=color, lw=1.2, label=label)
    ax.fill_between(CK, m - se, m + se, color=color, alpha=0.15, lw=0)


def ci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n); t = TQ[n]
    return m, t * se


# ---- Fig 1: converged learning curves, both MyoSuite bodies ----
ARMS = [("blank", "blank", "blank (model-free, 128 real)", "0.25", "-"), ("blank64", "blank64", "blank64 (model-free, 64 real)", "0.55", "--"),
        ("prior", "prior", "prior (body model)", "tab:blue", "-"), ("randprior", "prior", "randprior", "tab:cyan", "--"),
        ("coach", "coach", "coach (teacher T1)", "tab:red", "-"), ("randcoach", "coach", "randcoach", "tab:orange", "--"),
        ("priorcoach", "priorcoach", "priorcoach (T1)", "tab:purple", "-")]
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
for ax, tag, unit in ((axes[0], "elbow", "mrad"), (axes[1], "finger", "mm")):
    for lab, cond, name, col, ls in ARMS:
        band(ax, load("results_conv_%s_%s" % (tag, lab), cond), name, col, ls)
    if tag == "finger":
        band(ax, load("results_conv_finger_coachT2", "coach"), "coach (teacher T2)", "tab:green", "-")
        band(ax, load("results_conv_finger_priorcoachT2", "priorcoach"), "priorcoach (T2)", "tab:olive", "-")
    ax.set_ylim(*((0, 150) if tag == "elbow" else (20, 175)))
    ax.set_xlabel("environment steps"); ax.set_ylabel("held-out error (%s)" % unit)
    ax.set_title("myoElbow" if tag == "elbow" else "myoFinger", fontsize=7.5)
    ax.set_xlim(0, 100000); ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "converged_curves.png"), dpi=300); plt.close(fig)

# ---- Fig 2: attachment sweep at 100k, both bodies: components vs t_w ----
TWS = [2000, 3000, 4000, 5000, 6000, 7000, 8000]
S = range(12)


def sweep(body):
    Z = r"C:\Users\maurice\Desktop\world_model_zeroshot\b2_pilot"
    if body == "elbow":
        con = loadd("results_coach100k_elbow", "constant"); none = loadd("results_conv_elbow_blank", "blank")
        arms = {tw: (loadd("results_sweep100k_elbow/w%d" % tw, "abrupt"), loadd("results_sweep100k_elbow/w%d" % tw, "selfanchor")) for tw in TWS[:-1]}
        arms[8000] = (loadd("results_coach100k_elbow", "abrupt"), loadd("results_coach100k_elbow_off2", "selfanchor"))
        dip_arms = {tw: (loadd(os.path.join(Z, "results_attachment_myoelbow"), "abrupt_w%d" % tw), loadd(os.path.join(Z, "results_attachment_myoelbow"), "selfanchor_w%d" % tw)) for tw in TWS}
        dip_con = loadd(os.path.join(Z, "results_withdrawal_myoelbow"), "constant"); dip_none = loadd(os.path.join(Z, "results_withdrawal_myoelbow"), "none")
    else:
        con = loadd("results_sweep100k_finger_T2", "constant"); none = loadd("results_sweep100k_finger_T2", "none")
        arms = {tw: (loadd("results_sweep100k_finger_T2/w%d" % tw, "abrupt"), loadd("results_sweep100k_finger_T2/w%d" % tw, "selfanchor")) for tw in TWS[:-1]}
        arms[8000] = (loadd("results_coach100k_finger_T2", "abrupt"), loadd("results_coach100k_finger_T2", "selfanchor"))
        dip_arms, dip_con, dip_none = arms, con, none
    return con, none, arms, dip_arms, dip_con, dip_none


COMP = [("total (abrupt $-$ constant)", "ab", "con", "k"), ("teacher (selfanchor $-$ constant)", "sa", "con", "tab:blue"),
        ("loss term (abrupt $-$ selfanchor)", "ab", "sa", "tab:red"), ("abrupt $-$ none", "ab", "none", "tab:green")]
fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.4))
for row, (body, unit) in enumerate((("elbow", "mrad"), ("finger", "mm"))):
    con, none, arms, dip_arms, dip_con, dip_none = sweep(body)
    for col, metric in enumerate(("dip", "endpoint")):
        ax = axes[row, col]
        for k, (name, x, y, color) in enumerate(COMP):
            ms, es = [], []
            for tw in TWS:
                if metric == "dip":
                    a, sa = dip_arms[tw]; src = {"ab": a, "sa": sa, "con": dip_con, "none": dip_none}
                    f = lambda r, s: r[s][tw + 1000] - r[s][tw]
                else:
                    a, sa = arms[tw]; src = {"ab": a, "sa": sa, "con": con, "none": none}
                    f = lambda r, s: r[s][100000]
                m, e = ci([f(src[x], s) - f(src[y], s) for s in S]); ms.append(m); es.append(e)
            xs = np.array(TWS) / 1000.0 + (k - 1.5) * 0.08
            ax.errorbar(xs, ms, yerr=es, fmt="o-", ms=3, lw=1, capsize=2, color=color, label=name)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title("%s: %s" % ("myoElbow" if body == "elbow" else "myoFinger (teacher T2)", "immediate dip" if metric == "dip" else "endpoint at 100k"), fontsize=7.5)
        ax.set_xlabel("attachment duration $t_w$ (k steps)"); ax.set_ylabel("paired difference (%s)" % unit); ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "sweep100k_components.png"), dpi=300); plt.close(fig)

# ---- Fig 3: joint-count series ----
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
ax = axes[0]
CK4 = CK
for lab, cond, name, col, ls in (("blank", "blank", "blank", "0.25", "-"), ("blank64", "blank64", "blank64", "0.55", "--"),
                                 (PRIOR_DIR["arm4"], "prior", "prior (body model)", "tab:blue", "-"), ("randprior", "prior", "randprior", "tab:cyan", "--")):
    d = loadd("results_arms100k_arm4_%s" % lab, cond); seeds = sorted(d)
    y = np.array([[d[s][k] for k in CK4] for s in seeds]); band(ax, y, name, col, ls)
ax.set_title("4-joint arm ($n{=}%d$)" % len(seeds), fontsize=7.5); ax.set_xlabel("environment steps"); ax.set_ylabel("held-out error (mm)")
ax.set_xlim(0, 100000); ax.set_ylim(0, 600); ax.grid(alpha=0.3); ax.legend(fontsize=6, frameon=False)
ax = axes[1]
for k, (name, x, y, color) in enumerate((("prior $-$ blank", "prior", "blank", "tab:blue"), ("prior $-$ blank64", "prior", "blank64", "tab:purple"), ("randprior $-$ blank", "randprior", "blank", "tab:cyan"))):
    ms, es = [], []
    for b, nj in (("arm1", 1), ("arm2", 2), ("arm3", 3), ("arm4", 4)):
        A = {"blank": loadd("results_arms100k_%s_blank" % b, "blank"), "blank64": loadd("results_arms100k_%s_blank64" % b, "blank64"),
             "prior": loadd("results_arms100k_%s_%s" % (b, PRIOR_DIR[b]), "prior"), "randprior": loadd("results_arms100k_%s_randprior" % b, "prior")}
        seeds = sorted(set.intersection(*[set(v) for v in A.values()]))
        bl = np.mean([A["blank"][s][100000] for s in seeds])
        m, e = ci([(A[x][s][100000] - A[y][s][100000]) / bl for s in seeds]); ms.append(m); es.append(e)
    ax.errorbar(np.arange(1, 5) + (k - 1) * 0.08, ms, yerr=es, fmt="o-", ms=3, lw=1, capsize=2, color=color, label=name)
ax.axhline(0, color="k", lw=0.6); ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("joints"); ax.set_ylabel("paired difference at 100k\n(fraction of blank mean)")
ax.set_title("body model vs. controls by joint count", fontsize=7.5); ax.grid(alpha=0.3); ax.legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "joint_series.png"), dpi=300); plt.close(fig)
print("figs written")
