# selfanchor at 100k with the swap on loop step t_w - 2, to reproduce the other
# implementation's arm seed for seed (see run_coach_anchor.py, --swap-offset).
# 12 seeds, myoElbow, 6 real workers, resumable, multi-pass.
$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
$env:PYTHONUNBUFFERED = "1"; $env:WM_BODY = "myoelbow"
$out = "results_coach100k_elbow_off2"; New-Item -ItemType Directory -Force -Path $out | Out-Null
$PARALLEL = 12; $FRESH = 15; $MAXPASS = 6
function BusyMine { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -like "*$out*" } | Measure-Object).Count }
$pass = 0
do {
  $pass++; $launched = 0
  foreach ($s in 0..11) {
    if (Test-Path (Join-Path $out "selfanchor_seed$s.json")) { continue }
    $log = Join-Path $out "selfanchor_s$s.log"
    if ((Test-Path $log) -and ((Get-Item $log).LastWriteTime -gt (Get-Date).AddMinutes(-$FRESH))) { continue }
    while ((BusyMine) -ge $PARALLEL) { Start-Sleep -Seconds 10 }
    Start-Process -FilePath $py -ArgumentList @("run_coach_anchor.py","--body","myoelbow","--arm","selfanchor","--seed","$s","--steps","100000","--withdraw-at","8000","--swap-offset","-2","--out",$out) -WorkingDirectory $dir -RedirectStandardOutput $log -RedirectStandardError (Join-Path $out "selfanchor_s$s.err") -WindowStyle Hidden
    $launched++; Start-Sleep -Milliseconds 400
  }
  Write-Output "pass $pass at $(Get-Date -Format 'HH:mm:ss'): launched $launched"
  while ((BusyMine) -gt 0) { Start-Sleep -Seconds 30 }
  $missing = (0..11 | Where-Object { -not (Test-Path (Join-Path $out "selfanchor_seed$_.json")) } | Measure-Object).Count
  Write-Output "pass $pass finished: $missing missing"
} while ($missing -gt 0 -and $pass -lt $MAXPASS)
Write-Output $(if ($missing -gt 0) { "GAVE UP with $missing missing" } else { "SELFANCHOR OFF2 COMPLETE at $(Get-Date -Format 'HH:mm:ss')" })
