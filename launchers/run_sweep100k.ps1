# Attachment sweep to 100k. Resumable, multi-pass, with a commit-memory guard.
#
#   A. myoFinger, competent teacher T2: constant + none on a 1k eval grid (the existing
#      coachT2 / blank runs are on a 2k grid, and the dip needs t_w+1000), then abrupt and
#      selfanchor at t_w = 2k..7k (8k already exists in results_coach100k_finger_T2).
#   B. myoElbow, teacher clean30k: abrupt and selfanchor at t_w = 2k..7k. Dip and post-4k fall
#      inside 12k and the runs are deterministic, so they come from the 12k sweep; these runs
#      only add the 100k endpoint, so they use a 2k eval grid (still checked against the 12k
#      sweep at 2k..12k).
# 8 real workers (16 python processes: shim + uv child). A job is not started while system
# commit is above $COMMIT_CAP GB, so other people's jobs are never pushed into OOM.
$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
$env:PYTHONUNBUFFERED = "1"
$PARALLEL = 24; $FRESH = 30; $MAXPASS = 6; $COMMIT_CAP = 60
$T2 = "teacher_finger_prior100k.pt"; $T1E = "teacher_elbow_clean30k.pt"
$FO = "results_sweep100k_finger_T2"; $EO = "results_sweep100k_elbow"

function BusyMine { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'run_withdrawal100k' } | Measure-Object).Count }
function CommitGB { $o = Get-CimInstance Win32_OperatingSystem; ($o.TotalVirtualMemorySize - $o.FreeVirtualMemory) / 1MB }

$jobs = @()
foreach ($s in 0..11) {
  foreach ($arm in @("constant","none")) {
    $a = @("run_withdrawal100k.py","--body","myofinger","--arm",$arm,"--seed","$s","--steps","100000","--eval-every","1000","--out",$FO)
    if ($arm -eq "constant") { $a += @("--withdraw-at","100000000","--teacher",$T2) }
    $jobs += @{ json="$FO\${arm}_seed$s.json"; log="$FO\${arm}_s$s.log"; body="myofinger"; args=$a }
  }
}
foreach ($tw in @(2000,3000,4000,5000,6000,7000)) { foreach ($arm in @("abrupt","selfanchor")) { foreach ($s in 0..11) {
  $jobs += @{ json="$FO\w$tw\${arm}_seed$s.json"; log="$FO\w$tw\${arm}_s$s.log"; body="myofinger";
              args=@("run_withdrawal100k.py","--body","myofinger","--arm",$arm,"--seed","$s","--steps","100000","--withdraw-at","$tw","--eval-every","1000","--teacher",$T2,"--out","$FO\w$tw") }
} } }
foreach ($tw in @(2000,3000,4000,5000,6000,7000)) { foreach ($arm in @("abrupt","selfanchor")) { foreach ($s in 0..11) {
  $jobs += @{ json="$EO\w$tw\${arm}_seed$s.json"; log="$EO\w$tw\${arm}_s$s.log"; body="myoelbow";
              args=@("run_withdrawal100k.py","--body","myoelbow","--arm",$arm,"--seed","$s","--steps","100000","--withdraw-at","$tw","--eval-every","2000","--teacher",$T1E,"--out","$EO\w$tw") }
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
Write-Output $(if ($missing -gt 0) { "GAVE UP with $missing missing" } else { "SWEEP100K COMPLETE at $(Get-Date -Format 'HH:mm:ss')" })
