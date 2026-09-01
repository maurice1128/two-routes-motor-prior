# Throttled seed-boost: adds seeds 5-11 (-> n=12) for arm2/arm3/arm4, 4 conditions.
# Runs at most $MAX concurrent to avoid the resource exhaustion that killed the naive launch.
param([int]$MAX = 12)
$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_mm\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir
$env:PYTHONUNBUFFERED = "1"

$bodies = @(
  @{body="arm2"; out="results_main";  prior="prior.pt"},
  @{body="arm3"; out="results_main3"; prior="prior3.pt"},
  @{body="arm4"; out="results_arm4";  prior="prior_arm4.pt"}
)
$jobs = @()
foreach ($b in $bodies) {
  foreach ($s in 5,6,7,8,9,10,11) {
    foreach ($c in @("blank","colearn","prior","warm")) {
      $jobs += @{body=$b.body; out=$b.out; prior=$b.prior; cond=$c; seed=$s}
    }
  }
}

function RunningCount { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {$_.CommandLine -like '*run_reach*--seed*'} | Measure-Object).Count }

$done = 0
foreach ($j in $jobs) {
  while ((RunningCount) -ge $MAX) { Start-Sleep -Seconds 5 }
  $env:WM_BODY = $j.body
  $pr = if ($j.cond -eq "prior" -or $j.cond -eq "warm") { @("--prior",$j.prior) } else { @() }
  $args = @("run_reach.py","--cond",$j.cond,"--seed","$($j.seed)","--steps","12000","--utd","2","--ntargets","8","--out",$j.out) + $pr
  Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $dir `
    -RedirectStandardOutput "$($j.out)\$($j.cond)_s$($j.seed).log" -RedirectStandardError "$($j.out)\$($j.cond)_s$($j.seed).err" -WindowStyle Hidden
  $done++
  Start-Sleep -Milliseconds 800
}
Write-Output "all $done jobs queued; waiting for drain"
while ((RunningCount) -gt 0) { Start-Sleep -Seconds 10 }
Write-Output "SEEDBOOST COMPLETE"
