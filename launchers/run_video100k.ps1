# myoFinger 100k actors for the project video: blank/prior/coach x seeds 0-3.
# 4 real workers (8 python processes: shim + uv child), resumable, multi-pass.
# Passive about other people's jobs: throttles only on its own count.
$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
$env:PYTHONUNBUFFERED = "1"; $env:WM_BODY = "myofinger"
$out = "actors_finger100k"; New-Item -ItemType Directory -Force -Path $out | Out-Null
$PARALLEL = 8; $FRESH = 20; $MAXPASS = 5
function BusyMine { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -like "*train_video100k*" } | Measure-Object).Count }
$pass = 0
do {
  $pass++; $launched = 0
  foreach ($arm in @("prior","coach","blank")) {
    foreach ($s in 0..3) {
      if (Test-Path (Join-Path $out "${arm}_s$s.pt")) { continue }
      $log = Join-Path $out "${arm}_s$s.log"
      if ((Test-Path $log) -and ((Get-Item $log).LastWriteTime -gt (Get-Date).AddMinutes(-$FRESH))) { continue }
      while ((BusyMine) -ge $PARALLEL) { Start-Sleep -Seconds 15 }
      Start-Process -FilePath $py -ArgumentList @("train_video100k.py",$arm,"$s") -WorkingDirectory $dir -RedirectStandardOutput $log -RedirectStandardError (Join-Path $out "${arm}_s$s.err") -WindowStyle Hidden
      $launched++; Start-Sleep -Milliseconds 500
    }
  }
  Write-Output "pass $pass at $(Get-Date -Format 'HH:mm:ss'): launched $launched"
  while ((BusyMine) -gt 0) { Start-Sleep -Seconds 30 }
  $missing = 0
  foreach ($arm in @("prior","coach","blank")) { foreach ($s in 0..3) { if (-not (Test-Path (Join-Path $out "${arm}_s$s.pt"))) { $missing++ } } }
  Write-Output "pass $pass finished: $missing missing"
} while ($missing -gt 0 -and $pass -lt $MAXPASS)
Write-Output $(if ($missing -gt 0) { "GAVE UP with $missing missing" } else { "VIDEO100K COMPLETE at $(Get-Date -Format 'HH:mm:ss')" })
