"""The missing adversarial control for the guidance route: a RANDOM teacher.

Matched to `coach(const)` (`results_noanneal_{elbow,finger}/`) in every respect
except what the teacher contains -- same condition, same constant weight, same
12k budget, same utd, same 8 training targets, same held-out evaluation, same
twelve seeds.  Section 4.3 records that the constant schedule was never run on
train targets, so held-out is the matched target set.

Constant weight is `--coach_anneal 100000000`: the non-abrupt branch computes
coach0 * max(0, 1 - step/anneal), which at 12k of 1e8 is 0.99988.

Writes to results_randcoach_{elbow,finger}/.  Skips any seed already present, so
it is safe to re-run after an interruption.
"""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(ROOT), ".venv_myo", "Scripts", "python.exe")
SEEDS = range(12)
WORKERS = 6

BODIES = [("myoelbow", "teacher_rand_elbow.pt", "results_randcoach_elbow"),
          ("myofinger", "teacher_rand_finger.pt", "results_randcoach_finger")]


def job(body, teacher, out, seed):
    dest = os.path.join(ROOT, out, "coach_seed%d.json" % seed)
    if os.path.exists(dest):
        return "SKIP %s s%d" % (out, seed)
    env = dict(os.environ, WM_BODY=body, PYTHONIOENCODING="utf-8")
    cmd = [PY, os.path.join(ROOT, "run_reach.py"),
           "--cond", "coach", "--seed", str(seed),
           "--steps", "12000", "--utd", "2", "--ntargets", "8", "--heldout",
           "--coach", os.path.join(ROOT, teacher),
           "--coach_anneal", "100000000",
           "--out", os.path.join(ROOT, out)]
    t0 = time.time()
    log = os.path.join(ROOT, out, "coach_s%d.log" % seed)
    with open(log, "wb") as fh:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return "%s %s s%d (%.0fs)" % ("OK  " if r.returncode == 0 else "FAIL",
                                  out, seed, time.time() - t0)


def main():
    jobs = []
    for body, teacher, out in BODIES:
        os.makedirs(os.path.join(ROOT, out), exist_ok=True)
        for s in SEEDS:
            jobs.append((body, teacher, out, s))
    print("launching %d runs, %d at a time" % (len(jobs), WORKERS), flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for msg in ex.map(lambda a: job(*a), jobs):
            print(msg, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
