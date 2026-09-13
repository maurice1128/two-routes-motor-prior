# -*- coding: utf-8 -*-
"""Audit the things the four table audits and audit_prose_v2 do NOT cover:
  (a) significance marks (* and bold) in every table vs. recomputed CIs,
  (b) intervals quoted in prose that are not table cells,
  (c) the two-decimal sweep values quoted in prose,
  (d) the [-2.5,+1.7] replace-regime interval,
  (e) the budget-matching prediction-error numbers (from the other draft's record),
  (f) Method hyperparameters against the code."""
import json, os, re, math, io
W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
Z = r"C:\Users\maurice\Desktop\world_model_zeroshot"
TEX = os.path.join(W, "tcds_merged", "paper.tex")
tex = io.open(TEX, encoding="utf-8").read()
T = 2.201
bad = 0


def check(name, cond, detail=""):
    global bad
    print("%-64s %s %s" % (name, "OK" if cond else "MISMATCH", detail))
    bad += 0 if cond else 1


def loadd(root, d, c):
    return {s: {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(os.path.join(root, d, "%s_seed%d.json" % (c, s))))} for s in range(12)}


def pt(vals):
    n = len(vals); m = sum(vals) / n
    se = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1) / n)
    return m, m - T * se, m + T * se


def ci(a, b, f):
    return pt([f(a[s]) - f(b[s]) for s in range(12)])


def sig(t3):
    return not (t3[1] < 0 < t3[2])


def block(label):
    return re.search(r"\\label\{" + re.escape(label) + r"\}(.*?)\\end\{tabular\}", tex, re.S).group(1)


def rows(blk):
    out = {}
    for line in blk.splitlines():
        if "&" not in line or "\\\\" not in line: continue
        parts = [p.strip() for p in line.split("&")]
        out[re.sub(r"\\texttt\{|\}|\$|\\textbf\{|\^\\dagger", "", parts[0]).strip()] = parts[1:]
    return out


# ---------- (a) marks: Table conv (bold on 100k cells) ----------
ARMS = [("blank", "blank"), ("prior", "prior"), ("randprior", "prior"), ("coach", "coach"), ("randcoach", "coach"), ("priorcoach", "priorcoach")]
D = {}
for tag in ("elbow", "finger"):
    for lab, cond in ARMS:
        D[(tag, lab)] = loadd(W, "results_conv_%s_%s" % (tag, lab), cond)
R = rows(block("tab:conv"))
for a, b in (("coach", "prior"), ("prior", "blank"), ("coach", "blank"), ("priorcoach", "blank"), ("priorcoach", "coach"), ("prior", "randprior"), ("coach", "randcoach")):
    cells = R["%s-%s" % (a, b)]
    for j, tag in enumerate(("elbow", "finger")):
        bold = "\\mathbf" in cells[2 * j + 1]
        check("conv bold %s-%s %s 100k" % (a, b, tag), bold == sig(ci(D[(tag, a)], D[(tag, b)], lambda c: c[100000])))
        check("conv no mark on 12k %s-%s %s" % (a, b, tag), "\\mathbf" not in cells[2 * j] and "^*" not in cells[2 * j])

# ---------- Table withdraw (bold) ----------
M = {}
for tag in ("elbow", "finger"):
    for arm in ("keep", "soft", "purge", "matched"):
        M[(tag, arm)] = loadd(W, "results_matched_%s" % tag, arm)
R = rows(block("tab:withdraw"))
for a, b in (("purge", "keep"), ("soft", "keep"), ("matched", "keep"), ("soft", "matched"), ("purge", "matched")):
    cells = R["%s-%s" % (a, b)]
    for j, tag in enumerate(("elbow", "finger")):
        check("withdraw bold %s-%s %s" % (a, b, tag), ("\\mathbf" in cells[j]) == sig(ci(M[(tag, a)], M[(tag, b)], lambda c: c[12000])))

# ---------- Table w100k (bold on both columns) ----------
B = {a: loadd(W, "results_matched100k_elbow", a) for a in ("keep", "soft", "purge", "matched")}
C = {"constant": loadd(W, "results_coach100k_elbow", "constant"), "abrupt": loadd(W, "results_coach100k_elbow", "abrupt"),
     "selfanchor": loadd(W, "results_coach100k_elbow_off2", "selfanchor")}
R = rows(block("tab:w100k"))
for src, pairs in ((B, (("purge", "keep"), ("soft", "keep"), ("matched", "keep"), ("soft", "matched"), ("purge", "matched"))),
                   (C, (("abrupt", "constant"), ("selfanchor", "constant"), ("abrupt", "selfanchor")))):
    for a, b in pairs:
        cells = R["%s-%s" % (a, b)]
        check("w100k bold %s-%s 12k" % (a, b), ("\\mathbf" in cells[0]) == sig(ci(src[a], src[b], lambda c: c[12000])))
        check("w100k bold %s-%s 100k" % (a, b), ("\\mathbf" in cells[1]) == sig(ci(src[a], src[b], lambda c: c[100000])))

# ---------- Table decomp (stars) ----------
ZD = {}
for tag, body in (("elbow", "myoelbow"), ("finger", "myofinger")):
    for a in ("none", "constant", "abrupt", "selfanchor", "randanchor"):
        ZD[(tag, a)] = loadd(Z, "b2_pilot/results_withdrawal_%s" % body, a)
dipf = lambda c: c[9000] - c[8000]; endf = lambda c: c[12000]
blk = block("tab:decomp")
for row, (a, b) in (("abrupt $-$ constant (total)", ("abrupt", "constant")), ("selfanchor $-$ constant (teacher)", ("selfanchor", "constant")),
                    ("abrupt $-$ selfanchor (loss term)", ("abrupt", "selfanchor")), ("randanchor $-$ selfanchor", ("randanchor", "selfanchor"))):
    line = [l for l in blk.splitlines() if l.startswith(row)][0]
    cells = [p.strip() for p in line.split("&")[1:]]
    for j, tag in enumerate(("elbow", "finger")):
        check("decomp star %s %s dip" % (row[:12], tag), ("^*" in cells[2 * j]) == sig(ci(ZD[(tag, a)], ZD[(tag, b)], dipf)))
        check("decomp star %s %s end" % (row[:12], tag), ("^*" in cells[2 * j + 1]) == sig(ci(ZD[(tag, a)], ZD[(tag, b)], endf)))
line = [l for l in blk.splitlines() if l.startswith("constant $-$ none")][0]
cells = [p.strip() for p in line.split("&")[1:]]
for j, tag in enumerate(("elbow", "finger")):
    check("decomp star constant-none %s end" % tag, ("^*" in cells[2 * j + 1]) == sig(ci(ZD[(tag, "constant")], ZD[(tag, "none")], endf)))

# ---------- (b) prose intervals ----------
def near(t3, want, tol=0.006):
    return all(abs(g - w) <= tol for g, w in zip(t3, want))
check("prose selfanchor-constant elbow endpoint +9.42 [+3.13,+15.70]", near(ci(ZD[("elbow", "selfanchor")], ZD[("elbow", "constant")], endf), (9.42, 3.13, 15.70)))
check("prose selfanchor-constant finger endpoint -11.29 [-20.65,-1.93]", near(ci(ZD[("finger", "selfanchor")], ZD[("finger", "constant")], endf), (-11.29, -20.65, -1.93)))
check("prose constant-none finger +5.17 [-26.49,+36.82]", near(ci(ZD[("finger", "constant")], ZD[("finger", "none")], endf), (5.17, -26.49, 36.82)))
check("reproduction abrupt-constant elbow +16.00 [+6.49,+25.52]", near(ci(ZD[("elbow", "abrupt")], ZD[("elbow", "constant")], endf), (16.00, 6.49, 25.52)))
check("reproduction abrupt-constant finger +46.37 [+22.88,+69.87]", near(ci(ZD[("finger", "abrupt")], ZD[("finger", "constant")], endf), (46.37, 22.88, 69.87)))
ZB = {a: loadd(Z, "b1_pilot/results_budget_myoelbow", a) for a in ("prior_full", "coach")}
check("reproduction prior-coach elbow +10.89 [+2.52,+19.26]", near(ci(ZB["prior_full"], ZB["coach"], endf), (10.89, 2.52, 19.26)))
# same three cells from THIS paper's own runs (the reproduction claim)
check("own runs: coach-prior elbow 12k = -10.89", abs(ci(D[("elbow", "coach")], D[("elbow", "prior")], endf)[0] + 10.89) < 0.006)
check("own runs: abrupt-constant elbow 12k = +16.00 [+6.49,+25.52]", near(ci(C["abrupt"], C["constant"], endf), (16.00, 6.49, 25.52)))

# ---------- (c) two-decimal sweep values quoted in prose ----------
ZA = {}
for tw in (7000, 8000):
    ZA[("selfanchor", tw)] = loadd(Z, "b2_pilot/results_attachment_myofinger", "selfanchor_w%d" % tw)
v7 = ci(ZA[("selfanchor", 7000)], ZD[("finger", "constant")], lambda c: c[11000])
v8 = ci(ZA[("selfanchor", 8000)], ZD[("finger", "constant")], lambda c: c[12000])
check("sweep prose finger post-4k teacher tw=7000 -6.39*", abs(v7[0] + 6.39) < 0.006 and sig(v7), "%.2f [%.2f,%.2f]" % v7)
check("sweep prose finger post-4k teacher tw=8000 -11.29*", abs(v8[0] + 11.29) < 0.006 and sig(v8), "%.2f [%.2f,%.2f]" % v8)

# ---------- (d) replace: 'intervals as narrow as [-2.5,+1.7]' ----------
ZR = {}
for tw in (2000, 4000, 6000, 8000):
    for a in ("selfanchor", "gateon"):
        ZR[(a, tw)] = loadd(Z, "b2_pilot/results_replace_myoelbow", "%s_w%d" % (a, tw))
ivs = []
for tw in (2000, 4000, 6000, 8000):
    for name, f in (("dip", lambda c, tw=tw: c[tw + 1000] - c[tw]), ("post4k", lambda c, tw=tw: c[tw + 4000]), ("end", lambda c: c[12000])):
        ivs.append((tw, name, ci(ZR[("selfanchor", tw)], ZR[("gateon", tw)], f)))
narrow = min(ivs, key=lambda x: x[2][2] - x[2][1])
check("replace narrowest selfanchor-gateon interval is [-2.5,+1.7]", abs(narrow[2][1] + 2.5) < 0.06 and abs(narrow[2][2] - 1.7) < 0.06, "%s %s %.2f [%.2f,%.2f]" % (narrow[0], narrow[1], *narrow[2]))
check("replace 'no significant slope on any metric' selfanchor-gateon", True, "(slopes are table cells, checked by audit_merged_carried)")

# ---------- (e) budget prediction error 14.3 -> 25.1 ----------
full = io.open(os.path.join(W, "build_prior_finger.log"), encoding="utf-8", errors="ignore").read()
m1 = re.search(r"k=5 fingertip position error: mean (\d+\.\d) mm", full)
matched = io.open(os.path.join(Z, "b1_pilot", "build_prior_finger_matched.log"), encoding="utf-8", errors="ignore").read()
m2 = re.search(r"N=30000\s+k5 tip_err (\d+\.\d+) mm", matched)
check("budget finger 5-step error full 14.3 (build_prior_finger.log)", m1 is not None and m1.group(1) == "14.3", m1.group(1) if m1 else "not found")
check("budget finger 5-step error matched 25.1 (build_prior_finger_matched.log)", m2 is not None and round(float(m2.group(1)), 1) == 25.1, m2.group(1) if m2 else "not found")

# ---------- (f) Method hyperparameters vs code ----------
code = ""
for fn in ("run_reach.py", "sac_dyna.py", "arm_env.py", "babble.py", "train_teacher2.py", "wm_train.py"):
    p = os.path.join(W, fn)
    if os.path.exists(p):
        code += "\n### %s\n" % fn + io.open(p, encoding="utf-8", errors="ignore").read()
def has(pat):
    return re.search(pat, code) is not None
check("code: hidden 128", has(r"hidden\s*=\s*128"))
check("code: batch 128", has(r"batch(_size)?\s*=\s*128"))
check("code: utd 2", has(r"(utd|updates?_per_step|UTD)\w*\s*=\s*2\b"))
check("code: warm-up 1000", has(r"(warm|start|random_steps|learning_starts)\w*\s*=\s*1000\b"))
check("code: lr 3e-4", has(r"3e-4|0\.0003"))
check("code: episode 100 steps", has(r"max_steps\s*=\s*100\b"))
check("code: 8 train targets / 20 held-out", has(r"n_train_targets\s*=\s*8\b") and has(r"fixed_eval_targets\(20\)"))
check("code: 64 real + 64 imagined (batch 128, real_ratio 0.5)", has(r"real_ratio\s*=\s*0\.5") and has(r"nr = int\(batch \* real_ratio\)"))
check("code: teacher 30k steps", has(r"total_steps\s*=\s*30000") or has(r"30000"))
print("\nmarks/prose/code mismatches: %d" % bad)
