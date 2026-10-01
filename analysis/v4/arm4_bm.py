# -*- coding: utf-8 -*-
"""Body-model withdrawal on the four-joint arm (24 seeds): late-mean contrasts."""
import json, os
import numpy as np
from common import *
S = list(range(24))
keep = load("results_arms100k_arm4_prior2", "prior"); blank = load("results_arms100k_arm4_blank", "blank")
M = {a: load("results_matched100k_arm4", a) for a in ("matched", "purge")}
# sanity: before withdrawal (<= 8k) the withdrawn runs must equal keep exactly
mism = [(a, s, k) for a in M for s in S for k in range(2000, 8001, 2000) if abs(M[a][s][k] - keep[s][k]) > 1e-9]
print("pre-withdrawal mismatches vs keep:", len(mism))
out = {}
def c(name, x, y):
    v = [end(x, s) - end(y, s) for s in S]; r = list(ci(v)) + [upper1(v)]
    out[name] = [round(z, 2) for z in r]
    print("%-18s %+8.2f [%+8.2f, %+8.2f]%s  up %+.2f" % (name, r[0], r[1], r[2], "*" if sig(r) else " ", r[3]))
for k, r in (("keep", keep), ("blank", blank), ("matched", M["matched"]), ("purge", M["purge"])):
    print("mean %-8s %.1f" % (k, np.mean([end(r, s) for s in S])))
c("keep-blank", keep, blank)
c("matched-blank", M["matched"], blank); c("purge-blank", M["purge"], blank)
c("matched-keep", M["matched"], keep); c("purge-keep", M["purge"], keep); c("purge-matched", M["purge"], M["matched"])
kept = 100 * out["matched-blank"][0] / out["keep-blank"][0]; print("share of model advantage kept (matched): %.0f%%" % kept)
out["kept_pct_matched"] = round(kept); out["kept_pct_purge"] = round(100 * out["purge-blank"][0] / out["keep-blank"][0])
json.dump(out, open("arm4_bm.json", "w"), indent=1)
