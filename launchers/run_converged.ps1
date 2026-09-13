# Convergence grid: the core comparison re-run to 100,000 env steps.
#
# Why: the TMLR submission scored everything at a 12,000-step endpoint and said so
# ("not a verified asymptote"). A reviewer reading that concludes the arms were
# compared in the transient, which makes every ordering a statement about
# convergence speed rather than about what each prior buys. This re-runs the six
# core arms at 100k on both analysed bodies, held-out scored.
#
# It also closes two gaps the paper flagged: randprior was only ever scored on
# train targets, and randcoach was added late. Both are in the grid here, held-out
# like everything else.
#
# One arm per output directory, because run_reach names its JSON after --cond and
# randprior/randcoach reuse the prior/coach conditions with a different checkpoint;
# writing them into a shared directory would silently overwrite the real arms.
#
# Resumable: any seed whose JSON already exists is skipped, so this can be stopped
# and restarted. Passive about other work: it waits for the machine rather than
# killing anything, because another session is running its own sweep here.

$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
$env:PYTHONUNBUFFERED = "1"

$STEPS    = 100000
$SEEDS    = 0..11
# Every run shows up as TWO python processes: the .venv_myo shim and the uv child
# that actually does the work (the process listing shows one at 0.1 s CPU and one
# at ~80 s). Both counters below see both, so these thresholds are set at twice
# the intended number of real runs: 16 here means 8 workers, 12 means 6 foreign.
# The first night ran at half the intended parallelism because of this.
$PARALLEL = 16     # 32 logical cores; run_reach pins itself to one torch thread
$GATE     = 12     # don't start until fewer than this many foreign python jobs remain

# Count every heavy python job, mine and anyone else's. Contention is what made the
# earlier 36k probe look 15x slower than it is, so the throttle has to see all of it.
function BusyAll {
  (Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
     Where-Object { $_.CommandLine -match 'run_reach|run_attachment_sweep|run_withdrawal|run_budget|run_scaling' } |
     Measure-Object).Count
}
function BusyMine {
  (Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
     Where-Object { $_.CommandLine -like '*results_conv_*' } | Measure-Object).Count
}

# The paper's coach(const) arm (results_noanneal_*) was trained against the
# clean-retrained 30k teachers, not teacher_{elbow,finger}.pt. Verified 2026-09-10:
# with teacher_elbow.pt every seed's 12k value differs from results_noanneal_elbow;
# the other paper's `constant` arm, which reproduces 39.13 exactly, uses clean30k.
# The teacher_elbow.pt grid that ran first is kept under *_teacherElbow as the
# paper's alternative-teacher control, not as the main arm.
$bodies = @(
  @{ body = "myoelbow";  tag = "elbow";
     prior = "prior_myo.pt";    randprior = "prior_rand_elbow.pt";
     coach = "teacher_elbow_clean30k.pt"; randcoach = "teacher_rand_elbow.pt" },
  @{ body = "myofinger"; tag = "finger";
     prior = "prior_finger.pt"; randprior = "prior_rand_finger.pt";
     coach = "teacher_finger_clean30k.pt"; randcoach = "teacher_rand_finger.pt" }
)

# label -> (--cond, which prior checkpoint, which coach checkpoint)
$arms = @(
  @{ label = "blank";      cond = "blank";      prior = "";          coach = "" },
  @{ label = "prior";      cond = "prior";      prior = "prior";     coach = "" },
  @{ label = "randprior";  cond = "prior";      prior = "randprior"; coach = "" },
  @{ label = "coach";      cond = "coach";      prior = "";          coach = "coach" },
  @{ label = "randcoach";  cond = "coach";      prior = "";          coach = "randcoach" },
  @{ label = "priorcoach"; cond = "priorcoach"; prior = "prior";     coach = "coach" }
)

# The gate was written for a transient pile of 42 foreign processes that was going
# to drain. It has since been decided that this machine is shared for the
# duration -- another paper's finger sweep runs ~20 workers continuously -- so a
# gate on foreign load would never open. Start now, and let PARALLEL bound our
# own footprint. Reported rather than silently skipped so the log shows why.
Write-Output ("foreign heavy python jobs at start: " + (BusyAll) + " (gate of $GATE not enforced; machine is shared by decision)")
Write-Output "starting at $(Get-Date -Format 'HH:mm:ss')"

# A run that dies leaves no JSON and an empty .err -- on 2026-09-10 seventeen of
# them went at once when the machine hit its commit limit, and a single pass
# simply moved on past them. So: repeat passes until every JSON exists. A seed is
# in flight, not dead, if its .log was written in the last $FRESH minutes (runs
# log every 2000 steps, a few minutes apart under load), and must not be launched
# a second time onto the same output file.
$FRESH = 15
$MAXPASS = 6
$pass = 0
do {
$pass++
$launched = 0; $skipped = 0; $inflight = 0
foreach ($b in $bodies) {
  $env:WM_BODY = $b.body
  foreach ($arm in $arms) {
    $out = "results_conv_$($b.tag)_$($arm.label)"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    foreach ($s in $SEEDS) {
      $dest = Join-Path $out "$($arm.cond)_seed$s.json"
      if (Test-Path $dest) { $skipped++; continue }
      $log = Join-Path $out "$($arm.cond)_s$s.log"
      if ((Test-Path $log) -and ((Get-Item $log).LastWriteTime -gt (Get-Date).AddMinutes(-$FRESH))) { $inflight++; continue }
      while ((BusyMine) -ge $PARALLEL) { Start-Sleep -Seconds 10 }

      $a = @("run_reach.py", "--cond", $arm.cond, "--seed", "$s",
             "--steps", "$STEPS", "--utd", "2", "--ntargets", "8",
             "--heldout", "--out", $out)
      if ($arm.prior -ne "") { $a += @("--prior", $b[$arm.prior]) }
      if ($arm.coach -ne "") {
        # constant weight: the non-abrupt branch is coach0*max(0,1-step/anneal), so
        # the horizon must dwarf --steps. 1e8 gives 0.999 across a 100k budget.
        $a += @("--coach", $b[$arm.coach], "--coach_anneal", "100000000")
      }
      Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $dir `
        -RedirectStandardOutput "$out\$($arm.cond)_s$s.log" `
        -RedirectStandardError  "$out\$($arm.cond)_s$s.err" -WindowStyle Hidden
      $launched++
      Start-Sleep -Milliseconds 400
    }
  }
}
Write-Output "pass $pass at $(Get-Date -Format 'HH:mm:ss'): launched $launched, done $skipped, in flight $inflight"
while ((BusyMine) -gt 0) { Start-Sleep -Seconds 30 }
$missing = 0
foreach ($b in $bodies) { foreach ($arm in $arms) { foreach ($s in $SEEDS) {
  if (-not (Test-Path (Join-Path "results_conv_$($b.tag)_$($arm.label)" "$($arm.cond)_seed$s.json"))) { $missing++ }
} } }
Write-Output "pass $pass finished: $missing of $(2 * $arms.Count * $SEEDS.Count) still missing"
} while ($missing -gt 0 -and $pass -lt $MAXPASS)
if ($missing -gt 0) { Write-Output "GAVE UP after $MAXPASS passes with $missing missing -- inspect .err/.log" }
else { Write-Output "CONVERGENCE GRID COMPLETE at $(Get-Date -Format 'HH:mm:ss')" }
