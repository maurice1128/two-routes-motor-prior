$py="C:\Users\maurice\Desktop\robotic_research\.venv_mm\Scripts\python.exe"
$dir="C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir; $env:PYTHONUNBUFFERED="1"
function RC { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {$_.CommandLine -like '*run_reach*--cond*'} | Measure-Object).Count }
$cfg=@(@{body="arm2";out="results_2x2_arm2";prior="prior.pt";t="teacher_arm2.pt"},@{body="arm3";out="results_2x2_arm3";prior="prior3.pt";t="teacher_arm3.pt"})
foreach($b in $cfg){ foreach($s in 5,6,7,8,9,10,11){ foreach($c in @("coach","priorcoach")){
  while((RC) -ge 10){Start-Sleep 4}
  $env:WM_BODY=$b.body
  Start-Process -FilePath $py -ArgumentList @("run_reach.py","--cond",$c,"--seed","$s","--steps","12000","--utd","2","--ntargets","8","--prior",$b.prior,"--coach",$b.t,"--out",$b.out) -WorkingDirectory $dir -RedirectStandardOutput "$($b.out)\$($c)_s$s.log" -RedirectStandardError "$($b.out)\$($c)_s$s.err" -WindowStyle Hidden
  Start-Sleep -Milliseconds 500
}}}
while((RC) -gt 0){Start-Sleep 8}
Write-Output "2x2 BOOST COMPLETE"
