# Plateau check: elbow blank+prior to 36k steps, 3 seeds, held-out eval (18 eval points).
# Tests whether the 12k endpoint is near-asymptote (curve flattens after 12k).
$py="C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir="C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir; $env:PYTHONUNBUFFERED="1"; $env:WM_BODY="myoelbow"
$out="results_plateau_elbow"; New-Item -ItemType Directory -Force -Path $out | Out-Null
function RC { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {$_.CommandLine -like '*run_reach*36000*'} | Measure-Object).Count }
foreach($s in 0,1,2){ foreach($c in @("blank","prior")){
  if(Test-Path "$out\$($c)_seed$s.json"){ continue }
  while((RC) -ge 8){Start-Sleep 4}
  $pr = if($c -eq "prior"){@("--prior","prior_myo.pt")} else {@()}
  Start-Process -FilePath $py -ArgumentList (@("run_reach.py","--cond",$c,"--seed","$s","--steps","36000","--utd","2","--ntargets","8","--heldout","--out",$out)+$pr) -WorkingDirectory $dir -RedirectStandardOutput "$out\$($c)_s$s.log" -RedirectStandardError "$out\$($c)_s$s.err" -WindowStyle Hidden
  Start-Sleep -Milliseconds 500
}}
while((RC) -gt 0){Start-Sleep 8}
Write-Output "PLATEAU COMPLETE"
