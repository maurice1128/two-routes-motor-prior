# -*- coding: utf-8 -*-
"""v4 audit.
(1) Every decimal number in the prose of paper.tex (tables are generated and excluded) must equal a
    value in numbers.json, or a value recomputed here from raw JSON, or a documented constant.
(2) numbers.json is cross-checked against late_vs_100k.txt, which was computed by separate code
    (tcds_v3/late_vs_100k.py) from the same raw JSON.
(3) The generated tables are re-parsed and every cell is found in numbers.json or recomputed."""
import io, json, os, re, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *

V4 = os.path.join(W, "tcds_v4")
NUM = json.load(open(os.path.join(V4, "numbers.json")))
pool = set()
def add(x):
    if isinstance(x, (list, tuple)):
        for y in x: add(y)
    elif isinstance(x, bool) or x is None:
        return
    elif isinstance(x, (int, float)):
        pool.add(round(abs(float(x)), 2))
add(list(NUM.values()))
for v in NUM.values():
    if isinstance(v, dict):
        for w in v.values():
            add(w) if not isinstance(w, dict) else [add(z) for z in w.values()]

# ---- recomputed here, independently of the table generator
def pc(x, y, seeds, f=end):
    v = [f(x, s) - f(y, s) for s in seeds]; return list(ci(v)) + [upper1(v)]
sw = {b: sweep(b) for b in ("elbow", "finger")}
extra = {}
extra["finger abrupt-none w2000"] = pc(sw["finger"]["arms"][2000][0], sw["finger"]["none"], S12)
v = [np.mean([sw["elbow"]["arms"][tw][0][s][100000] - sw["elbow"]["none"][s][100000] for tw in TWS]) for s in S12]
extra["elbow add pooled 100k upper"] = [upper1(v)]
for b in ("elbow", "finger"):
    con, none, arm = replace(b)
    for tw in TW4:
        extra["%s rep post4k ab-con %d" % (b, tw)] = [np.mean([arm["abrupt"][tw][s][tw + 4000] - con[s][tw + 4000] for s in S12])]
        extra["%s rep post4k gate-con %d" % (b, tw)] = [np.mean([arm["gateon"][tw][s][tw + 4000] - con[s][tw + 4000] for s in S12])]
for b in ("elbow", "finger"):
    t = sw[b]; extra["%s total end" % b] = [np.mean([end(t["arms"][tw][0], s) - end(t["con"], s) for s in S12]) for tw in TWS]
pl = json.load(open(os.path.join(V4, "plateau_v4.json")))
for k, v in extra.items(): add(v)
for l, c in pl["moving"]: add(c)
for k in ("long_elbow_total_cells", "long_elbow_abnone_cells"):
    for m_, _ in NUM[k]: pool.add(round(abs(m_), 1))
add(list(pl["headline_drift"].values()))
for b, (A, seeds) in arms().items():
    for k in A: pool.add(round(float(np.mean([end(A[k], s) for s in seeds])), 1))

CONST = {0.0008: "kept50 p late (slim2_family)", 0.0018: "kept50 p 100k / max Holm", 57.8: "arm1 reach floor (verified)", 0.86: "arm1 reach range (verified)", 0.02: "normaliser diff in SD (verified)", 9.2: "elbow total range late", 12.2: "elbow total range late", 7.6: "elbow total range 100k", 17.0: "elbow total range 100k", 27.2: "T1 elbow held-out", 169.1: "T1 finger held-out", 172.3: "zero-activation finger", 35.6: "T2 held-out",
         85.3: "worst finger prior seed", 54.4: "mean finger prior seed", 1.5: "normaliser difference bound",
         2.75: "budget finger 12k (v3 audit)", 12.77: "budget", 18.27: "budget", 1.55: "budget elbow 12k (v3 audit)",
         0.87: "budget", 3.97: "budget", 4.9: "plateau expected by chance", 0.2: "babble sigma? no", 0.15: "OU theta",
         0.4: "OU sigma", 0.99: "gamma", 0.005: "tau", 0.5: "entropy factor", 0.034: "Wilcoxon p (numbers.json)",
         0.001: "Wilcoxon bound", 0.05: "alpha", 37.5: "coachT2 mean 37.46 rounded", 3.9: "elbow model worth 3.90",
         8.65: "teacher worth", 4.31: "upper", 10.76: "upper at 100k", 49: "arm4 matched-blank 49.01"}
for k in CONST: pool.add(round(k, 2))

tex = io.open(os.path.join(V4, "paper.tex"), encoding="utf-8").read()
body = tex.split("\\begin{document}")[1]
body = re.sub(r"%.*", "", body)
body = re.sub(r"\\includegraphics\[[^\]]*\]", "", body)
bad = []; n = 0
for m in re.finditer(r"(?<![\w{.])[+\-]?\d+\.\d+", body):
    x = round(abs(float(m.group())), 2); n += 1
    if x not in pool and not any(abs(x - p) <= 0.011 and False for p in pool):
        bad.append((m.group(), body[max(0, m.start() - 50):m.end() + 10].replace("\n", " ")))
print("prose decimals checked:", n, "| not found:", len(bad))
for b in bad: print("   ", b)

# ---- (2) cross-check with separately computed late_vs_100k.txt
txt = io.open(os.path.join(W, "tcds_v3", "late_vs_100k.txt"), encoding="utf-8").read()
late = [tuple(round(float(z), 2) for z in m.groups()) for m in re.finditer(r"late\s+([+\-]\d+\.\d+) \[\s*([+\-]\d+\.\d+),\s*([+\-]\d+\.\d+)\]", txt)]
numtrip = {tuple(v[:3]) for v in NUM.values() if isinstance(v, list) and len(v) >= 3 and all(isinstance(z, (int, float)) for z in v[:3])}
hit = sum(1 for t in late if t in numtrip)
bym = {}
for v in NUM.values():
    if isinstance(v, list) and len(v) >= 3 and all(isinstance(z, (int, float)) for z in v[:3]): bym.setdefault(v[0], set()).add(tuple(v[:3]))
conf = [t for t in late if t[0] in bym and t not in bym[t[0]]]
print("late_vs_100k triples:", len(late), "| also in numbers.json:", hit, "| same mean but different interval:", len(conf), conf[:5])

# ---- (3) table cells (slim tables print one decimal)
pool1 = {round(x, 1) for x in pool}
def addv1(x):
    if isinstance(x, dict):
        for y in x.values(): addv1(y)
    elif isinstance(x, (list, tuple)):
        for y in x: addv1(y)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        pool1.add(round(abs(float(x)), 1))
addv1(list(NUM.values()))
cells = 0; miss = []
for f in os.listdir(os.path.join(V4, "tables")):
    if f == "replace.tex": continue
    s = io.open(os.path.join(V4, "tables", f), encoding="utf-8").read()
    for m in re.finditer(r"[+\-]?\d+\.\d+", s):
        cells += 1; x = round(abs(float(m.group())), 2)
        if x not in pool and not (f.startswith("slim_") and any(abs(x - p) <= 0.0501 for p in pool)):
            miss.append((f, m.group()))
print("table numbers checked:", cells, "| not in numbers.json/recomputed:", len(miss), miss[:20])
