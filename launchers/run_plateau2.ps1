# CLEAN plateau: elbow + finger, blank+prior, seeds 0-3, to 36k, held-out eval (18 eval pts).
# Tests whether blank catches up to prior given 3x budget (sample-efficiency vs ceiling).
# Single launch, throttled <=8, skip-existing. DO NOT relaunch while running.
$py="C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir="C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir; $env:PYTHONUNBUFFERED="1"
function RC { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {$_.CommandLine -like '*run_reach.py*36000*'} | Measure-Object).Count }
$cfg=@(
  @{body="myoelbow";  out="results_plateau_elbow";  prior="prior_myo.pt"},
  @{body="myofinger"; out="results_plateau_finger"; prior="prior_finger.pt"}
)
foreach($b in $cfg){ $env:WM_BODY=$b.body
  foreach($s in 0,1,2,3){ foreach($c in @("blank","prior")){
    if(Test-Path "$($b.out)\$($c)_seed$s.json"){ continue }
    while((RC) -ge 8){Start-Sleep 5}
    $pr = if($c -eq "prior"){@("--prior",$b.prior)} else {@()}
    Start-Process -FilePath $py -ArgumentList (@("run_reach.py","--cond",$c,"--seed","$s","--steps","36000","--utd","2","--ntargets","8","--heldout","--out",$b.out)+$pr) -WorkingDirectory $dir -RedirectStandardOutput "$($b.out)\$($c)_s$s.log" -RedirectStandardError "$($b.out)\$($c)_s$s.err" -WindowStyle Hidden
    Start-Sleep -Milliseconds 600
  }}
}
while((RC) -gt 0){Start-Sleep 10}
Write-Output "PLATEAU2 COMPLETE"
