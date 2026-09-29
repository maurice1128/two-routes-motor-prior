# Re-run of the prior (babble body model) arm on arm1 and arm4 with priors rebuilt by the
# documented recipe (build_prior.py --n 200000 --seed 0 -> prior_arm1_200k.pt / prior_arm4_200k.pt).
# The checkpoints used in the first batch (prior_arm1.pt, prior_arm4.pt) could not be reproduced
# from any recorded budget or seed, so their runs are replaced. arm2 (prior.pt) and arm3
# (prior3.pt) reproduce bit for bit from the same recipe and are kept.
# arm1 seeds 0-11, arm4 seeds 0-23 = 36 runs. Same launcher discipline as run_arms100k.ps1.
$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
$env:PYTHONUNBUFFERED = "1"
$PARALLEL = 24; $FRESH = 30; $MAXPASS = 6; $COMMIT_CAP = 60
$PRIOR = @{ arm1="prior_arm1_200k.pt"; arm4="prior_arm4_200k.pt" }

function BusyMine { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'results_arms100k' } | Measure-Object).Count }
function CommitGB { $o = Get-CimInstance Win32_OperatingSystem; ($o.TotalVirtualMemorySize - $o.FreeVirtualMemory) / 1MB }

$jobs = @()
foreach ($s in 0..23) { foreach ($b in @("arm4","arm1")) {
  if ($b -eq "arm1" -and $s -gt 11) { continue }
  $o = "results_arms100k_${b}_prior2"
  $jobs += @{ json="$o\prior_seed$s.json"; log="$o\prior_s$s.log"; body=$b;
              args=@("run_reach.py","--cond","prior","--prior",$PRIOR[$b],"--heldout","--seed","$s","--steps","100000","--out",$o) }
} }
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
Write-Output $(if ($missing -gt 0) { "GAVE UP with $missing missing" } else { "ARMS100K PRIOR2 COMPLETE at $(Get-Date -Format 'HH:mm:ss')" })
