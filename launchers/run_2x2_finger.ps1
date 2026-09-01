# myoFinger 2x2: (body prior: none|frozen) x (coach: none|coach). 4 cells x 12 seeds.
# Uses .venv_myo (MyoSuite). Identical 12000-step budget across all cells for a clean comparison.
$py="C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir="C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir
$env:PYTHONUNBUFFERED="1"; $env:WM_BODY="myofinger"
$out="results_2x2_finger"; New-Item -ItemType Directory -Force -Path $out | Out-Null
$prior="prior_finger.pt"; $teacher="teacher_finger.pt"
function RC { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {$_.CommandLine -like '*run_reach*--cond*'} | Measure-Object).Count }
# cell -> whether it needs prior / coach
$cells=@(
  @{c="blank";      usePrior=$false; useCoach=$false},
  @{c="prior";      usePrior=$true;  useCoach=$false},
  @{c="coach";      usePrior=$false; useCoach=$true},
  @{c="priorcoach"; usePrior=$true;  useCoach=$true}
)
foreach($cell in $cells){ foreach($s in 0,1,2,3,4,5,6,7,8,9,10,11){
  while((RC) -ge 10){Start-Sleep 4}
  $a=@("run_reach.py","--cond",$cell.c,"--seed","$s","--steps","12000","--utd","2","--ntargets","8","--out",$out)
  if($cell.usePrior){ $a+=@("--prior",$prior) }
  if($cell.useCoach){ $a+=@("--coach",$teacher) }
  Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $dir `
    -RedirectStandardOutput "$out\$($cell.c)_s$s.log" -RedirectStandardError "$out\$($cell.c)_s$s.err" -WindowStyle Hidden
  Start-Sleep -Milliseconds 500
}}
while((RC) -gt 0){Start-Sleep 8}
Write-Output "FINGER 2x2 COMPLETE"
