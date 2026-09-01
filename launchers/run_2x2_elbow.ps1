# myoElbow 2x2: (body prior none|frozen) x (coach none|coach). 4 cells x 12 seeds, 12000 steps.
# Mirror of run_2x2_finger.ps1 for a SECOND recognized body, to test if the finger
# dissociation (prior->endpoint, coach->speed) replicates. Uses .venv_myo.
$py="C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir="C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir
$env:PYTHONUNBUFFERED="1"; $env:WM_BODY="myoelbow"
$out="results_2x2_elbow"; New-Item -ItemType Directory -Force -Path $out | Out-Null
$prior="prior_myo.pt"; $teacher="teacher_elbow.pt"
function RC { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {$_.CommandLine -like '*run_reach*--cond*'} | Measure-Object).Count }
$cells=@(
  @{c="blank";      usePrior=$false; useCoach=$false},
  @{c="prior";      usePrior=$true;  useCoach=$false},
  @{c="coach";      usePrior=$false; useCoach=$true},
  @{c="priorcoach"; usePrior=$true;  useCoach=$true}
)
foreach($cell in $cells){ foreach($s in 0,1,2,3,4,5,6,7,8,9,10,11){
  if(Test-Path "$out\$($cell.c)_seed$s.json"){ continue }
  while((RC) -ge 10){Start-Sleep 4}
  $a=@("run_reach.py","--cond",$cell.c,"--seed","$s","--steps","12000","--utd","2","--ntargets","8","--out",$out)
  if($cell.usePrior){ $a+=@("--prior",$prior) }
  if($cell.useCoach){ $a+=@("--coach",$teacher) }
  Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $dir `
    -RedirectStandardOutput "$out\$($cell.c)_s$s.log" -RedirectStandardError "$out\$($cell.c)_s$s.err" -WindowStyle Hidden
  Start-Sleep -Milliseconds 500
}}
while((RC) -gt 0){Start-Sleep 8}
Write-Output "ELBOW 2x2 COMPLETE"
