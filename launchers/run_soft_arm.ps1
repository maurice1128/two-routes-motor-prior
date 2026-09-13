# The fourth withdrawal arm: soft withdrawal, 12 seeds, both bodies.
#
# Completes the decomposition. With the real-data content held equal across soft,
# matched and keep, the three contrasts separate cleanly:
#   soft - matched  : what stale imagined transitions are still worth
#   purge - matched : what the real-data doubling contributed
#   keep - soft     : what fresh imagination adds over stale
# Section 6 lists the stale buffer as one of the reasons the soft arm is not a
# clean single-factor manipulation; with `matched` in hand it becomes one.
#
# Low parallelism on purpose: the convergence grid owns the machine and this is a
# cheap 24-run job that should not slow it down much.

$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
$env:PYTHONUNBUFFERED = "1"

$PARALLEL = 3
function BusyMine {
  (Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
     Where-Object { $_.CommandLine -like '*run_prior_matched*' } | Measure-Object).Count
}

$launched = 0; $skipped = 0
foreach ($body in @("myoelbow", "myofinger")) {
  $tag = if ($body -eq "myoelbow") { "elbow" } else { "finger" }
  $env:WM_BODY = $body
  $out = "results_matched_$tag"
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  foreach ($s in 0..11) {
    if (Test-Path (Join-Path $out "soft_seed$s.json")) { $skipped++; continue }
    while ((BusyMine) -ge $PARALLEL) { Start-Sleep -Seconds 10 }
    Start-Process -FilePath $py `
      -ArgumentList @("run_prior_matched.py", "--body", $body, "--arm", "soft",
                      "--seed", "$s", "--steps", "12000",
                      "--withdraw-at", "8000", "--out", $out) `
      -WorkingDirectory $dir `
      -RedirectStandardOutput "$out\soft_s$s.log" `
      -RedirectStandardError  "$out\soft_s$s.err" -WindowStyle Hidden
    $launched++
    Start-Sleep -Milliseconds 400
  }
}
Write-Output "launched $launched, skipped $skipped"
while ((BusyMine) -gt 0) { Start-Sleep -Seconds 30 }
Write-Output "SOFT ARM COMPLETE at $(Get-Date -Format 'HH:mm:ss')"
