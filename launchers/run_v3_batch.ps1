# v3 experiments, one launcher, multi-pass, resumable. Groups in order:
#   1. blank-64 (both bodies, 12 seeds, 100k)                      -> results_conv_{elbow,finger}_blank64
#   2. randanchor on myoElbow to 100k (clean30k teacher)           -> results_coach100k_elbow
#   3. replacing regime on myoElbow to 100k (13 cells x 12 seeds)  -> results_replace100k_myoelbow
#   4. myoFinger with the competent teacher T2 (prior seed 1 @100k) -> results_conv_finger_{coach,priorcoach}T2,
#      results_coach100k_finger_T2 (abrupt/selfanchor/randanchor; constant == coachT2)
# 6 real workers (12 python processes: shim + uv child). Foreign jobs are not counted.
$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
$zs  = "C:\Users\maurice\Desktop\world_model_zeroshot\b2_pilot"
Set-Location $dir
$env:PYTHONUNBUFFERED = "1"
$PARALLEL = 8; $FRESH = 25; $MAXPASS = 6
$T2 = "teacher_finger_prior100k.pt"

function BusyMine { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'run_blank64|run_replace_sweep|run_withdrawal100k|_(coach|priorcoach)T2' } | Measure-Object).Count }

# job = @{ json=<result file>; log=<log path>; body=<WM_BODY>; exe=<python>; args=@(...) ; wd=<working dir> }
$jobs = @()
foreach ($b in @(@{body="myoelbow";tag="elbow"}, @{body="myofinger";tag="finger"})) {
  foreach ($s in 0..11) {
    $out = "results_conv_$($b.tag)_blank64"
    $jobs += @{ json="$out\blank64_seed$s.json"; log="$out\blank64_s$s.log"; body=$b.body; wd=$dir;
                args=@("run_blank64.py","--body",$b.body,"--seed","$s","--steps","100000","--out",$out) }
  }
}
foreach ($s in 0..11) {
  $out = "results_coach100k_elbow"
  $jobs += @{ json="$out\randanchor_seed$s.json"; log="$out\randanchor_s$s.log"; body="myoelbow"; wd=$dir;
              args=@("run_withdrawal100k.py","--body","myoelbow","--arm","randanchor","--seed","$s","--steps","100000","--withdraw-at","8000","--teacher","teacher_elbow_clean30k.pt","--out",$out) }
}
$rout = "$dir\results_replace100k_myoelbow"
foreach ($s in 0..11) {
  $jobs += @{ json="results_replace100k_myoelbow\constant_seed$s.json"; log="results_replace100k_myoelbow\constant_s$s.log"; body="myoelbow"; wd=$zs;
              args=@("run_replace_sweep.py","--body","myoelbow","--arms","constant","--seed-list","$s","--steps","100000","--eval-every","1000","--out",$rout) }
}
foreach ($arm in @("abrupt","selfanchor","gateon")) { foreach ($tw in @(2000,4000,6000,8000)) { foreach ($s in 0..11) {
  $jobs += @{ json="results_replace100k_myoelbow\${arm}_w${tw}_seed$s.json"; log="results_replace100k_myoelbow\${arm}_w${tw}_s$s.log"; body="myoelbow"; wd=$zs;
              args=@("run_replace_sweep.py","--body","myoelbow","--arms",$arm,"--withdraw-at","$tw","--seed-list","$s","--steps","100000","--eval-every","1000","--out",$rout) }
} } }
foreach ($s in 0..11) {
  $jobs += @{ json="results_conv_finger_coachT2\coach_seed$s.json"; log="results_conv_finger_coachT2\coach_s$s.log"; body="myofinger"; wd=$dir;
              args=@("run_reach.py","--cond","coach","--coach",$T2,"--coach_anneal","100000000","--heldout","--seed","$s","--steps","100000","--out","results_conv_finger_coachT2") }
  $jobs += @{ json="results_conv_finger_priorcoachT2\priorcoach_seed$s.json"; log="results_conv_finger_priorcoachT2\priorcoach_s$s.log"; body="myofinger"; wd=$dir;
              args=@("run_reach.py","--cond","priorcoach","--prior","prior_finger.pt","--coach",$T2,"--coach_anneal","100000000","--heldout","--seed","$s","--steps","100000","--out","results_conv_finger_priorcoachT2") }
}
foreach ($arm in @("abrupt","selfanchor","randanchor")) { foreach ($s in 0..11) {
  $jobs += @{ json="results_coach100k_finger_T2\${arm}_seed$s.json"; log="results_coach100k_finger_T2\${arm}_s$s.log"; body="myofinger"; wd=$dir;
              args=@("run_withdrawal100k.py","--body","myofinger","--arm",$arm,"--seed","$s","--steps","100000","--withdraw-at","8000","--teacher",$T2,"--out","results_coach100k_finger_T2") }
} }
Write-Output "$($jobs.Count) jobs queued at $(Get-Date -Format 'HH:mm:ss')"
foreach ($j in $jobs) { $d = Split-Path (Join-Path $dir $j.json); New-Item -ItemType Directory -Force -Path $d | Out-Null }

$pass = 0
do {
  $pass++; $launched = 0
  foreach ($j in $jobs) {
    if (Test-Path (Join-Path $dir $j.json)) { continue }
    $log = Join-Path $dir $j.log
    if ((Test-Path $log) -and ((Get-Item $log).LastWriteTime -gt (Get-Date).AddMinutes(-$FRESH))) { continue }
    while ((BusyMine) -ge $PARALLEL) { Start-Sleep -Seconds 20 }
    $env:WM_BODY = $j.body
    Start-Process -FilePath $py -ArgumentList $j.args -WorkingDirectory $j.wd -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$','.err') -WindowStyle Hidden
    $launched++; Start-Sleep -Milliseconds 700
  }
  Write-Output "pass $pass at $(Get-Date -Format 'HH:mm:ss'): launched $launched"
  while ((BusyMine) -gt 0) { Start-Sleep -Seconds 60 }
  $missing = ($jobs | Where-Object { -not (Test-Path (Join-Path $dir $_.json)) } | Measure-Object).Count
  Write-Output "pass $pass finished at $(Get-Date -Format 'HH:mm:ss'): $missing missing"
} while ($missing -gt 0 -and $pass -lt $MAXPASS)
Write-Output $(if ($missing -gt 0) { "GAVE UP with $missing missing" } else { "V3 BATCH COMPLETE at $(Get-Date -Format 'HH:mm:ss')" })
