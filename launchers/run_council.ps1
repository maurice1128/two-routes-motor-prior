# Council control experiments A (randprior) + B (held-out) on myoElbow + myoFinger.
# RC-throttled at 10 concurrent run_reach. Sets WM_BODY per launch. Priority-ordered.
$py="C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir="C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir
$env:PYTHONUNBUFFERED="1"
function RC { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {$_.CommandLine -like '*run_reach*--cond*'} | Measure-Object).Count }

# Job list: body, cond, prior file (or ""), heldout flag, coach file (or ""), out dir
$jobs=@()
# EXP A (priority 1): randprior, prior cond, eval-on-train (default)
foreach($s in 0..11){ $jobs += @{body="myoelbow"; c="prior"; prior="prior_rand_elbow.pt";  heldout=$false; coach=""; out="results_randprior_elbow"} }
foreach($s in 0..11){ $jobs += @{body="myofinger";c="prior"; prior="prior_rand_finger.pt"; heldout=$false; coach=""; out="results_randprior_finger"} }
# EXP B (priority 2): held-out blank + prior
foreach($s in 0..11){ $jobs += @{body="myoelbow"; c="blank"; prior="";              heldout=$true; coach=""; out="results_heldout_elbow"} }
foreach($s in 0..11){ $jobs += @{body="myoelbow"; c="prior"; prior="prior_myo.pt";   heldout=$true; coach=""; out="results_heldout_elbow"} }
foreach($s in 0..11){ $jobs += @{body="myofinger";c="blank"; prior="";              heldout=$true; coach=""; out="results_heldout_finger"} }
foreach($s in 0..11){ $jobs += @{body="myofinger";c="prior"; prior="prior_finger.pt";heldout=$true; coach=""; out="results_heldout_finger"} }
# EXP B (priority 3, optional full 2x2): held-out coach + priorcoach
foreach($s in 0..11){ $jobs += @{body="myoelbow"; c="coach";      prior="";              heldout=$true; coach="teacher_elbow.pt";  out="results_heldout_elbow"} }
foreach($s in 0..11){ $jobs += @{body="myoelbow"; c="priorcoach"; prior="prior_myo.pt";   heldout=$true; coach="teacher_elbow.pt";  out="results_heldout_elbow"} }
foreach($s in 0..11){ $jobs += @{body="myofinger";c="coach";      prior="";              heldout=$true; coach="teacher_finger.pt"; out="results_heldout_finger"} }
foreach($s in 0..11){ $jobs += @{body="myofinger";c="priorcoach"; prior="prior_finger.pt";heldout=$true; coach="teacher_finger.pt"; out="results_heldout_finger"} }

# expand seeds (each block above is 12 seeds 0..11)
$expanded=@()
$seedcycle=0..11
$i=0
foreach($j in $jobs){ $j.seed=$seedcycle[$i % 12]; $expanded+=$j; $i++ }

foreach($j in $expanded){
  New-Item -ItemType Directory -Force -Path $j.out | Out-Null
  $fn="$($j.out)\$($j.c)_seed$($j.seed).json"
  if(Test-Path $fn){ continue }
  while((RC) -ge 10){ Start-Sleep 5 }
  $env:WM_BODY=$j.body
  $a=@("run_reach.py","--cond",$j.c,"--seed","$($j.seed)","--steps","12000","--utd","2","--ntargets","8","--out",$j.out)
  if($j.prior){ $a+=@("--prior",$j.prior) }
  if($j.coach){ $a+=@("--coach",$j.coach) }
  if($j.heldout){ $a+=@("--heldout") }
  Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $dir `
    -RedirectStandardOutput "$($j.out)\$($j.c)_s$($j.seed).log" -RedirectStandardError "$($j.out)\$($j.c)_s$($j.seed).err" -WindowStyle Hidden
  Start-Sleep -Milliseconds 400
}
while((RC) -gt 0){ Start-Sleep 8 }
Write-Output "COUNCIL EXPERIMENTS COMPLETE"
