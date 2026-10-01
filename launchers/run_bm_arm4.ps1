# Body-model withdrawal on the four-joint planar arm (2026-09-29): the elbow's body model is worth
# too little to show what withdrawal leaves; on arm4 it is worth ~95 mm. purge / matched at 8k,
# 24 seeds, 100k, prior_arm4_200k.pt (keep = results_arms100k_arm4_prior2). 48 runs.
$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
$env:PYTHONUNBUFFERED = "1"; $env:WM_BODY = "arm4"
$PARALLEL = 16; $FRESH = 45; $MAXPASS = 6; $COMMIT_CAP = 60; $O = "results_matched100k_arm4"
function BusyMine { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'results_matched100k_arm4' } | Measure-Object).Count }
function CommitGB { $o = Get-CimInstance Win32_OperatingSystem; ($o.TotalVirtualMemorySize - $o.FreeVirtualMemory) / 1MB }
New-Item -ItemType Directory -Force -Path (Join-Path $dir $O) | Out-Null
$jobs = @()
foreach ($s in 0..23) { foreach ($arm in @("matched","purge")) {
  $jobs += @{ json="$O\${arm}_seed$s.json"; log="$O\${arm}_s$s.log";
              args=@("run_prior_matched.py","--body","arm4","--arm",$arm,"--seed","$s","--steps","100000","--withdraw-at","8000","--out",$O) }
} }
Write-Output "$($jobs.Count) jobs queued at $(Get-Date -Format 'HH:mm:ss')"
$pass = 0
do {
  $pass++; $launched = 0
  foreach ($j in $jobs) {
    if (Test-Path (Join-Path $dir $j.json)) { continue }
    $log = Join-Path $dir $j.log
    if ((Test-Path $log) -and ((Get-Item $log).LastWriteTime -gt (Get-Date).AddMinutes(-$FRESH))) { continue }
    while (((BusyMine) -ge $PARALLEL) -or ((CommitGB) -gt $COMMIT_CAP)) { Start-Sleep -Seconds 20 }
    Start-Process -FilePath $py -ArgumentList $j.args -WorkingDirectory $dir -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$','.err') -WindowStyle Hidden
    $launched++; Start-Sleep -Milliseconds 800
  }
  Write-Output "pass $pass at $(Get-Date -Format 'HH:mm:ss'): launched $launched"
  while ((BusyMine) -gt 0) { Start-Sleep -Seconds 60 }
  $missing = ($jobs | Where-Object { -not (Test-Path (Join-Path $dir $_.json)) } | Measure-Object).Count
  Write-Output "pass $pass finished at $(Get-Date -Format 'HH:mm:ss'): $missing missing"
} while ($missing -gt 0 -and $pass -lt $MAXPASS)
Write-Output $(if ($missing -gt 0) { "GAVE UP with $missing missing" } else { "BMARM4 COMPLETE at $(Get-Date -Format 'HH:mm:ss')" })
