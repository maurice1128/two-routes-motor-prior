# Both withdrawal decompositions re-scored at a converged budget: myoElbow,
# withdraw at 8000, run to 100,000, 12 seeds, held-out.
#
#   body-model side  keep / soft / purge / matched   via run_prior_matched.py
#   coach side       constant / abrupt / selfanchor  via run_coach_anchor.py
#
# Why: both decompositions were measured at 12k. The merged paper argues that
# 12k endpoints sit in transients, so its own "persistent cost" claims have to
# be shown at a budget where the arms have stopped moving.
#
# Same conventions as run_converged.ps1: every run is two python processes (shim
# + uv child), so PARALLEL is set at twice the intended worker count; multi-pass
# so a seed killed by a memory spike is retried; a seed with a .log written in
# the last $FRESH minutes is in flight and is not launched twice.

$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
$env:PYTHONUNBUFFERED = "1"
$env:WM_BODY = "myoelbow"

$STEPS = 100000; $TW = 8000; $SEEDS = 0..11
# 12 = 6 real workers (each counts twice, see run_converged.ps1). Lowered from 8
# on 2026-09-11: the prior-side arms commit ~1.3 GB each and the machine sat at
# 57/58.5 GB commit with 8, which is the state that killed 17 runs the day
# before. Six is ~8 h slower and does not court that.
$PARALLEL = 12; $FRESH = 15; $MAXPASS = 6

$jobs = @()
foreach ($arm in @("keep", "soft", "purge", "matched")) {
  $jobs += @{ script = "run_prior_matched.py"; arm = $arm; out = "results_matched100k_elbow" } }
foreach ($arm in @("constant", "abrupt", "selfanchor")) {
  $jobs += @{ script = "run_coach_anchor.py";  arm = $arm; out = "results_coach100k_elbow" } }

function BusyMine {
  (Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
     Where-Object { $_.CommandLine -match 'results_(matched|coach)100k_' } | Measure-Object).Count
}

$pass = 0
do {
  $pass++; $launched = 0; $done = 0; $inflight = 0
  foreach ($j in $jobs) {
    New-Item -ItemType Directory -Force -Path $j.out | Out-Null
    foreach ($s in $SEEDS) {
      $dest = Join-Path $j.out "$($j.arm)_seed$s.json"
      if (Test-Path $dest) { $done++; continue }
      $log = Join-Path $j.out "$($j.arm)_s$s.log"
      if ((Test-Path $log) -and ((Get-Item $log).LastWriteTime -gt (Get-Date).AddMinutes(-$FRESH))) { $inflight++; continue }
      while ((BusyMine) -ge $PARALLEL) { Start-Sleep -Seconds 10 }
      Start-Process -FilePath $py `
        -ArgumentList @($j.script, "--body", "myoelbow", "--arm", $j.arm, "--seed", "$s",
                        "--steps", "$STEPS", "--withdraw-at", "$TW", "--out", $j.out) `
        -WorkingDirectory $dir `
        -RedirectStandardOutput $log -RedirectStandardError (Join-Path $j.out "$($j.arm)_s$s.err") `
        -WindowStyle Hidden
      $launched++; Start-Sleep -Milliseconds 400
    }
  }
  Write-Output "pass $pass at $(Get-Date -Format 'HH:mm:ss'): launched $launched, done $done, in flight $inflight"
  while ((BusyMine) -gt 0) { Start-Sleep -Seconds 30 }
  $missing = 0
  foreach ($j in $jobs) { foreach ($s in $SEEDS) {
    if (-not (Test-Path (Join-Path $j.out "$($j.arm)_seed$s.json"))) { $missing++ } } }
  Write-Output "pass $pass finished: $missing of $($jobs.Count * $SEEDS.Count) still missing"
} while ($missing -gt 0 -and $pass -lt $MAXPASS)
if ($missing -gt 0) { Write-Output "GAVE UP after $MAXPASS passes with $missing missing" }
else { Write-Output "WITHDRAWAL@100K COMPLETE at $(Get-Date -Format 'HH:mm:ss')" }
