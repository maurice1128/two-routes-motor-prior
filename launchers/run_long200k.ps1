# Council follow-up (2026-09-29): myoElbow runs extended to 200,000 steps to check that the
# withdrawal result holds past 100k. Same seeds and code as the 100k runs (deterministic on CPU,
# so 0-100k reproduces them exactly). Conditions: none (never guided), constant (teacher T1 kept),
# abrupt at t_w = 2k..8k. 9 conditions x 12 seeds = 108 runs, eval every 2k.
# Resumable, multi-pass, commit-memory guard; never touches other people's processes.
$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
$env:PYTHONUNBUFFERED = "1"
$PARALLEL = 20; $FRESH = 45; $MAXPASS = 6; $COMMIT_CAP = 60
$T1E = "teacher_elbow_clean30k.pt"; $O = "results_long200k_elbow"

function BusyMine { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'results_long200k_elbow' } | Measure-Object).Count }
function CommitGB { $o = Get-CimInstance Win32_OperatingSystem; ($o.TotalVirtualMemorySize - $o.FreeVirtualMemory) / 1MB }

$jobs = @()
foreach ($s in 0..11) {
  $jobs += @{ json="$O\none_seed$s.json"; log="$O\none_s$s.log";
              args=@("run_withdrawal100k.py","--body","myoelbow","--arm","none","--seed","$s","--steps","200000","--eval-every","2000","--out",$O) }
  $jobs += @{ json="$O\constant_seed$s.json"; log="$O\constant_s$s.log";
              args=@("run_withdrawal100k.py","--body","myoelbow","--arm","constant","--seed","$s","--steps","200000","--withdraw-at","100000000","--teacher",$T1E,"--eval-every","2000","--out",$O) }
}
foreach ($tw in @(8000,2000,5000,3000,7000,4000,6000)) { foreach ($s in 0..11) {
  $jobs += @{ json="$O\w$tw\abrupt_seed$s.json"; log="$O\w$tw\abrupt_s$s.log";
              args=@("run_withdrawal100k.py","--body","myoelbow","--arm","abrupt","--seed","$s","--steps","200000","--withdraw-at","$tw","--teacher",$T1E,"--eval-every","2000","--out","$O\w$tw") }
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
    $env:WM_BODY = "myoelbow"
    Start-Process -FilePath $py -ArgumentList $j.args -WorkingDirectory $dir -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$','.err') -WindowStyle Hidden
    $launched++; Start-Sleep -Milliseconds 800
  }
  Write-Output "pass $pass at $(Get-Date -Format 'HH:mm:ss'): launched $launched"
  while ((BusyMine) -gt 0) { Start-Sleep -Seconds 60 }
  $missing = ($jobs | Where-Object { -not (Test-Path (Join-Path $dir $_.json)) } | Measure-Object).Count
  Write-Output "pass $pass finished at $(Get-Date -Format 'HH:mm:ss'): $missing missing"
} while ($missing -gt 0 -and $pass -lt $MAXPASS)
Write-Output $(if ($missing -gt 0) { "GAVE UP with $missing missing" } else { "LONG200K COMPLETE at $(Get-Date -Format 'HH:mm:ss')" })
