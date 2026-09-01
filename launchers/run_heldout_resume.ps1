# Resume held-out n-boost: arm2 already complete (n=12). Only arm3 seeds 4-11 left, blank+prior.
$py="C:\Users\maurice\Desktop\robotic_research\.venv_mm\Scripts\python.exe"
$dir="C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir; $env:PYTHONUNBUFFERED="1"
$env:WM_BODY="arm3"
function RC { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {$_.CommandLine -like '*run_reach*--cond*'} | Measure-Object).Count }
$out="results_heldout_arm3"; $prior="prior3.pt"
foreach($s in 4,5,6,7,8,9,10,11){ foreach($c in @("blank","prior")){
  if(Test-Path "$out\$($c)_seed$s.json"){ continue }   # skip already done
  while((RC) -ge 10){Start-Sleep 4}
  $pr = if($c -eq "prior"){@("--prior",$prior)} else {@()}
  Start-Process -FilePath $py -ArgumentList (@("run_reach.py","--cond",$c,"--seed","$s","--steps","12000","--utd","2","--ntargets","8","--heldout","--out",$out)+$pr) -WorkingDirectory $dir -RedirectStandardOutput "$out\$($c)_s$s.log" -RedirectStandardError "$out\$($c)_s$s.err" -WindowStyle Hidden
  Start-Sleep -Milliseconds 500
}}
while((RC) -gt 0){Start-Sleep 8}
Write-Output "HELDOUT ARM3 RESUME COMPLETE"
