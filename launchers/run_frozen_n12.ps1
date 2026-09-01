# Frozen-ness at n=12 on myoElbow, train targets, 12k steps.
# Mirrors run_2x2_elbow.ps1 exactly (same venv, same body, same prior, same
# concurrency); only the condition set differs.  Tests the design commitment
# the title names: `prior` (freeze=True) against `warm` (identical babble
# checkpoint, freeze=False) and `colearn` (online WM, no babble), with `blank`
# as the model-free reference.  results_myo holds the n=5 version of this and
# is left untouched; this is a clean 12-seed replication of all four arms.
$py="C:\Users\maurice\Desktop\robotic_research\.venv_myo\Scripts\python.exe"
$dir="C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir
$env:PYTHONUNBUFFERED="1"; $env:WM_BODY="myoelbow"
$out="results_frozen12"; New-Item -ItemType Directory -Force -Path $out | Out-Null
$prior="prior_myo.pt"
function RC { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {$_.CommandLine -like '*run_reach*--cond*'} | Measure-Object).Count }
$cells=@(
  @{c="blank";   usePrior=$false},
  @{c="prior";   usePrior=$true },
  @{c="warm";    usePrior=$true },
  @{c="colearn"; usePrior=$false}
)
foreach($cell in $cells){ foreach($s in 0,1,2,3,4,5,6,7,8,9,10,11){
  if(Test-Path "$out\$($cell.c)_seed$s.json"){ continue }
  while((RC) -ge 10){Start-Sleep 4}
  $a=@("run_reach.py","--cond",$cell.c,"--seed","$s","--steps","12000","--utd","2","--ntargets","8","--out",$out)
  if($cell.usePrior){ $a+=@("--prior",$prior) }
  Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $dir `
    -RedirectStandardOutput "$out\$($cell.c)_s$s.log" -RedirectStandardError "$out\$($cell.c)_s$s.err" -WindowStyle Hidden
  Start-Sleep -Milliseconds 500
}}
Write-Output "launched: 4 conditions x 12 seeds -> $out"
