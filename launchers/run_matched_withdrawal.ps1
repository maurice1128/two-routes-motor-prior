# Replay-matched prior withdrawal: three arms x 12 seeds x two bodies, 12,000 steps.
#
# Answers the control Section 3.4 names and says was not run. `purge` reproduces the
# paper's strict withdrawal; `matched` withdraws identically but holds distinct real
# transitions per update at 64, so the imagination loss is not confounded with a
# doubling of real-data gradient signal; `keep` is the never-withdrawn reference.
#
# 12,000 steps and held-out scoring on purpose: this contrast has to sit beside the
# paper's own withdrawal table, and changing the budget at the same time would make
# it incomparable. The convergence question is answered separately by
# run_converged.ps1, which is a different axis.
#
# Passive about other work and resumable, for the same reasons as run_converged.ps1.

$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
$env:PYTHONUNBUFFERED = "1"

$STEPS    = 12000
$WITHDRAW = 8000
$SEEDS    = 0..11
$PARALLEL = 6
$GATE     = 6

function BusyAll {
  (Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
     Where-Object { $_.CommandLine -match 'run_reach|run_attachment_sweep|run_withdrawal|run_budget|run_scaling|run_prior_matched' } |
     Measure-Object).Count
}
function BusyMine {
  (Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
     Where-Object { $_.CommandLine -like '*run_prior_matched*' } | Measure-Object).Count
}

Write-Output ("gate: waiting until fewer than $GATE heavy python jobs (now " + (BusyAll) + ")")
while ((BusyAll) -ge $GATE) { Start-Sleep -Seconds 30 }
Write-Output "gate open at $(Get-Date -Format 'HH:mm:ss')"

$launched = 0; $skipped = 0
foreach ($body in @("myoelbow", "myofinger")) {
  $tag = if ($body -eq "myoelbow") { "elbow" } else { "finger" }
  $env:WM_BODY = $body
  $out = "results_matched_$tag"
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  foreach ($arm in @("keep", "purge", "matched")) {
    foreach ($s in $SEEDS) {
      if (Test-Path (Join-Path $out "$($arm)_seed$s.json")) { $skipped++; continue }
      while ((BusyMine) -ge $PARALLEL) { Start-Sleep -Seconds 10 }
      Start-Process -FilePath $py `
        -ArgumentList @("run_prior_matched.py", "--body", $body, "--arm", $arm,
                        "--seed", "$s", "--steps", "$STEPS",
                        "--withdraw-at", "$WITHDRAW", "--out", $out) `
        -WorkingDirectory $dir `
        -RedirectStandardOutput "$out\$($arm)_s$s.log" `
        -RedirectStandardError  "$out\$($arm)_s$s.err" -WindowStyle Hidden
      $launched++
      Start-Sleep -Milliseconds 400
    }
  }
}
Write-Output "launched $launched, skipped $skipped"
while ((BusyMine) -gt 0) { Start-Sleep -Seconds 30 }
Write-Output "MATCHED WITHDRAWAL COMPLETE at $(Get-Date -Format 'HH:mm:ss')"
