# Joint-count series to 100k: planar muscle arms with 1..4 joints, same muscle model and units.
# Arms per body: blank, prior (babble model), randprior (random model, same machinery),
# blank64 (model-free at the Dyna arms' 64 distinct real transitions per update).
# 4 bodies x 4 arms x 12 seeds = 192 runs, held-out scored, eval every 2k.
# 12 real workers (24 python processes), commit guard at 60 GB, resumable, multi-pass.
$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
$env:PYTHONUNBUFFERED = "1"
$PARALLEL = 24; $FRESH = 30; $MAXPASS = 6; $COMMIT_CAP = 60
$PRIOR = @{ arm1="prior_arm1.pt"; arm2="prior.pt"; arm3="prior3.pt"; arm4="prior_arm4.pt" }

function BusyMine { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'results_arms100k' } | Measure-Object).Count }
function CommitGB { $o = Get-CimInstance Win32_OperatingSystem; ($o.TotalVirtualMemorySize - $o.FreeVirtualMemory) / 1MB }

$jobs = @()
foreach ($s in 12..23) { foreach ($b in @("arm4","arm3")) {
  $o = "results_arms100k_$b"
  $jobs += @{ json="${o}_blank\blank_seed$s.json"; log="${o}_blank\blank_s$s.log"; body=$b;
              args=@("run_reach.py","--cond","blank","--heldout","--seed","$s","--steps","100000","--out","${o}_blank") }
  $jobs += @{ json="${o}_prior\prior_seed$s.json"; log="${o}_prior\prior_s$s.log"; body=$b;
              args=@("run_reach.py","--cond","prior","--prior",$PRIOR[$b],"--heldout","--seed","$s","--steps","100000","--out","${o}_prior") }
  $jobs += @{ json="${o}_randprior\prior_seed$s.json"; log="${o}_randprior\prior_s$s.log"; body=$b;
              args=@("run_reach.py","--cond","prior","--prior","prior_rand_$b.pt","--heldout","--seed","$s","--steps","100000","--out","${o}_randprior") }
  $jobs += @{ json="${o}_blank64\blank64_seed$s.json"; log="${o}_blank64\blank64_s$s.log"; body=$b;
              args=@("run_blank64.py","--body",$b,"--seed","$s","--steps","100000","--out","${o}_blank64") }
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
Write-Output $(if ($missing -gt 0) { "GAVE UP with $missing missing" } else { "ARMS100K EXTRA SEEDS COMPLETE at $(Get-Date -Format 'HH:mm:ss')" })
