# Launch hand blank-vs-prior experiment. WM_BODY=myohand, 60k steps, out results_hand.
$py="C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir="C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir
$env:PYTHONUNBUFFERED="1"; $env:WM_BODY="myohand"
$out="results_hand"; $prior="prior_hand.pt"; $steps=60000
New-Item -ItemType Directory -Force $out | Out-Null
function RC { (Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -match 'run_reach' -and $_.CommandLine -match '--cond'} | Measure-Object).Count }
$seeds = 0,1,2,3,4,5
$maxpar = 11
foreach($c in @("blank","prior")){ foreach($s in $seeds){
  if(Test-Path "$out\$($c)_seed$s.json"){ continue }
  while((RC) -ge $maxpar){ Start-Sleep 5 }
  $pr = if($c -eq "prior"){@("--prior",$prior)} else {@()}
  $argl = @("run_reach.py","--cond",$c,"--seed","$s","--steps","$steps","--utd","2","--ntargets","8","--out",$out)+$pr
  Start-Process -FilePath $py -ArgumentList $argl -WorkingDirectory $dir -RedirectStandardOutput "$out\$($c)_s$s.log" -RedirectStandardError "$out\$($c)_s$s.err" -WindowStyle Hidden
  Start-Sleep -Milliseconds 800
}}
Write-Output "HAND LAUNCH COMPLETE (blank+prior seeds 0-5, 60k)"
