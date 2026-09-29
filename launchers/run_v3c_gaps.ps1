# Gap-closing batch for the TCDS paper (2026-09-24). 192 runs, all 100k, held-out.
#   1. body-model withdrawal on myoFinger to 100k: soft / purge / matched x 12   -> results_matched100k_finger
#      (keep = results_conv_finger_prior; the elbow-only limitation goes away)
#   2. replacing regime on myoFinger with teacher T2: constant x 12 + abrupt/selfanchor/gateon
#      at t_w 2k/4k/6k/8k x 12 -> results_replace100k_myofinger (the elbow-only limitation goes away)
# Shares the 12-worker cap with run_arms100k_prior2.ps1 (both counted by BusyMine).
$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
$env:PYTHONUNBUFFERED = "1"
$PARALLEL = 24; $FRESH = 30; $MAXPASS = 6; $COMMIT_CAP = 60

function BusyMine { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'results_(arms100k|matched100k_finger|batch64_|replace100k_myofinger)' } | Measure-Object).Count }
function CommitGB { $o = Get-CimInstance Win32_OperatingSystem; ($o.TotalVirtualMemorySize - $o.FreeVirtualMemory) / 1MB }

$jobs = @()
foreach ($s in 0..11) { foreach ($arm in @("soft","purge","matched")) {
  $o = "results_matched100k_finger"
  $jobs += @{ json="$o\${arm}_seed$s.json"; log="$o\${arm}_s$s.log"; body="myofinger";
              args=@("run_prior_matched.py","--body","myofinger","--arm",$arm,"--seed","$s","--steps","100000","--withdraw-at","8000","--out",$o) }
} }
# (batch-64 control dropped 2026-09-24: blank64 is by construction SAC at batch 64 --
#  each of 64 transitions appears exactly twice in a mean-reduced loss -- so it adds nothing)
$ro = "$dir\results_replace100k_myofinger"
foreach ($s in 0..11) {
  $jobs += @{ json="results_replace100k_myofinger\constant_seed$s.json"; log="results_replace100k_myofinger\constant_s$s.log"; body="myofinger";
              args=@("run_replace_T2.py","--body","myofinger","--arms","constant","--seed-list","$s","--steps","100000","--eval-every","1000","--out",$ro) }
}
foreach ($tw in @(2000,4000,6000,8000)) { foreach ($arm in @("abrupt","selfanchor","gateon")) { foreach ($s in 0..11) {
  $jobs += @{ json="results_replace100k_myofinger\${arm}_w${tw}_seed$s.json"; log="results_replace100k_myofinger\${arm}_w${tw}_s$s.log"; body="myofinger";
              args=@("run_replace_T2.py","--body","myofinger","--arms",$arm,"--withdraw-at","$tw","--seed-list","$s","--steps","100000","--eval-every","1000","--out",$ro) }
} } }
Write-Output "$($jobs.Count) jobs queued at $(Get-Date -Format 'HH:mm:ss')"
foreach ($j in $jobs) { New-Item -ItemType Directory -Force -Path (Split-Path (Join-Path $dir $j.json)) | Out-Null }

$pass = 0
do {
  $pass++; $launched = 0
  foreach ($j in $jobs) {
    if (Test-Path (Join-Path $dir $j.json)) { continue }
    $log = Join-Path $dir $j.log
    if ((Test-Path $log) -and ((Get-Item $log).LastWriteTime -gt (Get-Date).AddMinutes(-$FRESH))) { continue }
    while (((BusyMine) -ge $PARALLEL) -or ((CommitGB) -gt $COMMIT_CAP)) { Start-Sleep -Seconds 20 }
    $env:WM_BODY = $j.body
    Start-Process -FilePath $py -ArgumentList $j.args -WorkingDirectory $dir -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$','.err') -WindowStyle Hidden
    $launched++; Start-Sleep -Milliseconds 800
  }
  Write-Output "pass $pass at $(Get-Date -Format 'HH:mm:ss'): launched $launched"
  while ((BusyMine) -gt 0) { Start-Sleep -Seconds 60 }
  $missing = ($jobs | Where-Object { -not (Test-Path (Join-Path $dir $_.json)) } | Measure-Object).Count
  Write-Output "pass $pass finished at $(Get-Date -Format 'HH:mm:ss'): $missing missing"
} while ($missing -gt 0 -and $pass -lt $MAXPASS)
Write-Output $(if ($missing -gt 0) { "GAVE UP with $missing missing" } else { "V3C GAPS COMPLETE at $(Get-Date -Format 'HH:mm:ss')" })
