# -*- coding: utf-8 -*-
"""Audit the rounded / summary numbers quoted in the v2 prose (abstract, intro,
results, discussion) that the four table audits do not cover. Own results from
wm_prior JSON; carried results from world_model_zeroshot JSON (read only)."""
import json, os, math, re, io
W = r"C:\Users\maurice\Desktop\robotic_research\wm_prior"
Z = r"C:\Users\maurice\Desktop\world_model_zeroshot"
TEX = os.path.join(W, "tcds_merged", "paper.tex")
tex = io.open(TEX, encoding="utf-8").read()
T = 2.201
bad = 0


def load(root, d, c):
    return {s: {int(r["step"]): r["eval_dist"] * 1000 for r in json.load(open(os.path.join(root, d, "%s_seed%d.json" % (c, s))))} for s in range(12)}


def ci(a, b, st):
    d = [a[s][st] - b[s][st] for s in range(12)]; m = sum(d) / 12
    se = math.sqrt(sum((x - m) ** 2 for x in d) / 11 / 12); return m, m - T * se, m + T * se


def mean(a, st):
    return sum(a[s][st] for s in range(12)) / 12


def check(name, cond, detail=""):
    global bad
    print("%-58s %s %s" % (name, "OK" if cond else "MISMATCH", detail))
    bad += 0 if cond else 1


def intext(s):
    return s in tex


# ---- own results ----
E = {k: load(W, "results_conv_elbow_" + k, c) for k, c in (("blank", "blank"), ("prior", "prior"), ("coach", "coach"), ("randcoach", "coach"))}
F = {k: load(W, "results_conv_finger_" + k, c) for k, c in (("blank", "blank"), ("prior", "prior"), ("coach", "coach"), ("randcoach", "coach"))}
M12 = {k: load(W, "results_matched_elbow", k) for k in ("keep", "soft", "purge", "matched")}
M100 = {k: load(W, "results_matched100k_elbow", k) for k in ("keep", "soft", "purge", "matched")}
C100 = {"constant": load(W, "results_coach100k_elbow", "constant"), "abrupt": load(W, "results_coach100k_elbow", "abrupt"),
        "selfanchor": load(W, "results_coach100k_elbow_off2", "selfanchor")}

# abstract / intro rounded quotes
v = ci(F["prior"], F["blank"], 100000)[0]
check("intro 'stable 40 mm on the finger' (prior-blank 100k)", abs(round(-v) - 40) == 0, "%.2f" % v)
v = ci(F["coach"], F["blank"], 100000)[0]
check("intro 'ends 28 mm worse' (coach-blank finger 100k)", round(v) == 28, "%.2f" % v)
v = ci(F["coach"], F["randcoach"], 100000)
check("intro 'worse than one held near a random teacher' sig>0", v[1] > 0, "%.2f [%.2f,%.2f]" % v)
v = ci(M12["matched"], M12["keep"], 12000)[0]
check("abstract/intro '22 mrad' (matched-keep 12k)", round(v) == 22, "%.2f" % v)
v = ci(M100["matched"], M100["keep"], 100000)
check("intro 'dissipated by 100k' (matched-keep 100k n.s.)", v[1] < 0 < v[2], "%.2f [%.2f,%.2f]" % v)
v = ci(C100["selfanchor"], C100["constant"], 100000)[0]
check("intro 'about 8 mrad' teacher info at 100k", round(v) == 8, "%.2f" % v)
v = ci(C100["abrupt"], C100["selfanchor"], 100000)
check("'loss-term component no longer detected' at 100k", v[1] < 0 < v[2], "%.2f [%.2f,%.2f]" % v)
# 'transient on the elbow, not detected at 100k'
v = ci(E["prior"], E["blank"], 100000)
check("intro elbow prior-blank not detected at 100k", v[1] < 0 < v[2], "%.2f [%.2f,%.2f]" % v)
# coach keeps small advantage on elbow at 100k
v = ci(E["coach"], E["blank"], 100000)
check("intro elbow coach-blank sig<0 at 100k", v[2] < 0, "%.2f [%.2f,%.2f]" % v)
# withdraw section: 'within 5-7 mrad of keep (25.78)' and 'within 2.4 of one another'
means = {k: mean(M100[k], 100000) for k in M100}
diffs = [means[k] - means["keep"] for k in ("soft", "purge", "matched")]
check("withdraw100k 'within 5-7 mrad of keep'", 5 <= min(diffs) and max(diffs) <= 7.5 and round(means["keep"], 2) == 25.78, "%s keep=%.2f" % ([round(d, 2) for d in diffs], means["keep"]))
spread = max(means[k] for k in ("soft", "purge", "matched")) - min(means[k] for k in ("soft", "purge", "matched"))
check("withdraw100k 'within 2.4 of one another'", spread <= 2.4, "%.2f" % spread)
# plateau numbers in IV-A
for k, want_m, want_sd in (("blank", 2.7, 11.3), ("prior", -2.9, 9.4), ("coach", -1.7, 3.5)):
    xs = [E[k][s][100000] - E[k][s][80000] for s in range(12)]; m = sum(xs) / 12
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / 11)
    check("plateau elbow %s %+.1f (SD %.1f)" % (k, want_m, want_sd), abs(m - want_m) < 0.06 and abs(sd - want_sd) < 0.06, "%+.2f sd %.2f" % (m, sd))
RP = load(W, "results_conv_elbow_randprior", "prior")
xs = [RP[s][100000] - RP[s][80000] for s in range(12)]; m = sum(xs) / 12; sd = math.sqrt(sum((x - m) ** 2 for x in xs) / 11)
check("plateau elbow randprior -25.0 (SD 29.6)", abs(m + 25.0) < 0.06 and abs(sd - 29.6) < 0.06, "%+.2f sd %.2f" % (m, sd))
# finger coach 12k / 100k means quoted in prose
check("finger coach 122.77 at 12k, 123.00 at 100k", abs(mean(F["coach"], 12000) - 122.77) < 0.006 and abs(mean(F["coach"], 100000) - 123.00) < 0.006,
      "%.2f %.2f" % (mean(F["coach"], 12000), mean(F["coach"], 100000)))
# finger coach-prior sig>0 at every ckpt from 12k: already in audit_prose. finger coach-blank 'by 40k worse than no aid' (discussion)
v = ci(F["coach"], F["blank"], 40000)
check("discussion 'by 40k it is worse than no aid' (sig at 40k)", v[1] > 0, "%.2f [%.2f,%.2f]" % v)

# ---- carried results (zeroshot, read only) ----
zw = lambda body, arm: load(Z, "b2_pilot/results_withdrawal_%s" % body, arm)
ZE = {a: zw("myoelbow", a) for a in ("none", "constant", "abrupt", "selfanchor")}
ZF = {a: zw("myofinger", a) for a in ("none", "constant", "abrupt", "selfanchor")}
dip = lambda A, a, b: sum((A[a][s][9000] - A[a][s][8000]) - (A[b][s][9000] - A[b][s][8000]) for s in range(12)) / 12
pct = dip(ZE, "abrupt", "selfanchor") / dip(ZE, "abrupt", "constant") * 100
check("decomp '89% of the dip on the elbow is the loss term'", round(pct) == 89, "%.1f%%" % pct)
pctf = dip(ZF, "abrupt", "selfanchor") / dip(ZF, "abrupt", "constant") * 100
check("decomp 'all of it on the finger' (loss term >= total)", pctf >= 100, "%.1f%%" % pctf)
# finger abrupt drifts from 131 to 169 mm over the 4000 steps after withdrawal
a9, a12 = mean(ZF["abrupt"], 9000), mean(ZF["abrupt"], 12000)
check("finger abrupt 131 mm at t_w+1000, 169 at 12k", round(a9) == 131 and round(a12) == 169, "%.1f -> %.1f" % (a9, a12))
# constant - none at each t_w on elbow: -86, -90, -66, -55, -24, -25, -44
want = [-86, -90, -66, -55, -24, -25, -44]
got = [round(ci(ZE["constant"], ZE["none"], tw)[0]) for tw in range(2000, 8001, 1000)]
check("sweep 'constant-none at each t_w' list", got == want, str(got))
# recovery counts: selfanchor finger 12/12 recover pre-withdrawal error
rec12 = sum(1 for s in range(12) if any(v <= ZF["selfanchor"][s][8000] for st, v in ZF["selfanchor"][s].items() if st > 8000))
check("finger selfanchor: all twelve seeds recover (any ckpt after t_w)", rec12 == 12, "%d/12" % rec12)
# replacing regime prose: constant 27.86; additive constant 39.13; abrupt endpoints 133/168 at tw 4000/6000, 0/12 recover; none 70.73
ZR = {"constant": load(Z, "b2_pilot/results_replace_myoelbow", "constant")}
for tw in (4000, 6000):
    ZR[("abrupt", tw)] = load(Z, "b2_pilot/results_replace_myoelbow", "abrupt_w%d" % tw)
check("replace constant ends 27.86", abs(mean(ZR["constant"], 12000) - 27.86) < 0.006, "%.2f" % mean(ZR["constant"], 12000))
check("additive constant 39.13 = conv coach 12k", abs(mean(E["coach"], 12000) - 39.13) < 0.006, "%.2f" % mean(E["coach"], 12000))
e4, e6 = mean(ZR[("abrupt", 4000)], 12000), mean(ZR[("abrupt", 6000)], 12000)
check("replace abrupt ends 133 / 168 at tw 4000 / 6000", round(e4) == 133 and round(e6) == 168, "%.1f %.1f" % (e4, e6))
r4 = sum(1 for s in range(12) if ZR[("abrupt", 4000)][s][12000] <= ZR[("abrupt", 4000)][s][4000])
r6 = sum(1 for s in range(12) if ZR[("abrupt", 6000)][s][12000] <= ZR[("abrupt", 6000)][s][6000])
check("replace abrupt recovers 0/12 at tw 4000 and 6000", r4 == 0 and r6 == 0, "%d %d" % (r4, r6))
check("'never had a teacher (70.73)' = conv blank 12k", abs(mean(E["blank"], 12000) - 70.73) < 0.006, "%.2f" % mean(E["blank"], 12000))
# selfanchor - gateon within +-3 mrad at every tw (dip)
ZG = {}
for tw in (2000, 4000, 6000, 8000):
    ZG[("selfanchor", tw)] = load(Z, "b2_pilot/results_replace_myoelbow", "selfanchor_w%d" % tw)
    ZG[("gateon", tw)] = load(Z, "b2_pilot/results_replace_myoelbow", "gateon_w%d" % tw)
dd = [abs(sum((ZG[("selfanchor", tw)][s][tw + 1000] - ZG[("selfanchor", tw)][s][tw]) - (ZG[("gateon", tw)][s][tw + 1000] - ZG[("gateon", tw)][s][tw]) for s in range(12)) / 12) for tw in (2000, 4000, 6000, 8000)]
check("replace selfanchor-gateon dip within 3.1 mrad at every tw", max(dd) <= 3.1, str([round(x, 2) for x in dd]))
# budget: 3.3x / 6.7x  (elbow 1e5/3e4, finger 2e5/3e4)
check("budget 3.3x / 6.7x", round(1e5 / 3e4, 1) == 3.3 and round(2e5 / 3e4, 1) == 6.7)
# finger teacher 169.1 vs no-op 172.3 -> 3.2 mm: reproduced by scratchpad
# noop_teacher_check.py under the same evaluate() as wm_prior/teacher_eval.py
pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teacher_noop_finger.txt")
line = io.open(pth, encoding="utf-8").read() if os.path.exists(pth) else ""
mm = re.findall(r"(\d+\.\d) mm", line)
check("finger teacher 169.1 / no-op 172.3 reproduced by evaluate()", mm == ["169.1", "172.3"], line.strip())
# recovery counts in the decomposition table footnote (first checkpoint after t_w at or below pre-withdrawal error)
def rec(A, arm):
    return sum(1 for s in range(12) if any(v <= A[arm][s][8000] for st, v in A[arm][s].items() if st > 8000))
ZE["randanchor"] = zw("myoelbow", "randanchor"); ZF["randanchor"] = zw("myofinger", "randanchor")
got = [rec(ZE, a) for a in ("none", "constant", "abrupt", "selfanchor", "randanchor")] + [rec(ZF, a) for a in ("none", "constant", "abrupt", "selfanchor", "randanchor")]
check("decomp footnote recovery counts 11,7,6,8,5 / 12,6,6,12,3", got == [11, 7, 6, 8, 5, 12, 6, 6, 12, 3], str(got))
print("\nprose-v2 mismatches: %d" % bad)
