# Boost held-out generalization to n=12: add seeds 3-11 for blank+prior only
# (the generalization claim = blank vs prior; colearn/warm/imag are dead lines, skip).
# Throttled to <=10 concurrent run_reach to avoid MuJoCo DLL load failures.
$py="C:\Users\maurice\Desktop\robotic_research\.venv_mm\Scripts\python.exe"
$dir="C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir; $env:PYTHONUNBUFFERED="1"
function RC { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {$_.CommandLine -like '*run_reach*--cond*'} | Measure-Object).Count }
$cfg=@(@{body="arm2";prior="prior.pt";out="results_heldout_arm2"},@{body="arm3";prior="prior3.pt";out="results_heldout_arm3"})
foreach($b in $cfg){ $env:WM_BODY=$b.body
  foreach($s in 3,4,5,6,7,8,9,10,11){ foreach($c in @("blank","prior")){
    while((RC) -ge 10){Start-Sleep 4}
    $pr = if($c -eq "prior"){@("--prior",$b.prior)} else {@()}
    Start-Process -FilePath $py -ArgumentList (@("run_reach.py","--cond",$c,"--seed","$s","--steps","12000","--utd","2","--ntargets","8","--heldout","--out",$b.out)+$pr) -WorkingDirectory $dir -RedirectStandardOutput "$($b.out)\$($c)_s$s.log" -RedirectStandardError "$($b.out)\$($c)_s$s.err" -WindowStyle Hidden
    Start-Sleep -Milliseconds 500
  }}
}
while((RC) -gt 0){Start-Sleep 8}
Write-Output "HELDOUT BOOST COMPLETE (blank+prior seeds 3-11, arm2+arm3)"
