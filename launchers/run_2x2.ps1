# 2x2: body-prior x coach. Runs the two NEW conditions (coach, priorcoach) on arm2/arm3;
# blank & prior already exist in results_main / results_main3. Throttled to avoid overload.
param([int]$MAX = 10, [int[]]$Seeds = @(0,1,2,3,4))
$py  = "C:\Users\maurice\Desktop\robotic_research\.venv_mm\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir
$env:PYTHONUNBUFFERED = "1"
$cfg = @(
  @{body="arm2"; out="results_2x2_arm2"; prior="prior.pt";  teacher="teacher_arm2.pt"},
  @{body="arm3"; out="results_2x2_arm3"; prior="prior3.pt"; teacher="teacher_arm3.pt"}
)
function RunningCount { (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {$_.CommandLine -like '*run_reach*--cond*'} | Measure-Object).Count }
$jobs=@()
foreach ($b in $cfg) { foreach ($s in $Seeds) { foreach ($c in @("coach","priorcoach")) { $jobs += @{b=$b;s=$s;c=$c} } } }
foreach ($j in $jobs) {
  while ((RunningCount) -ge $MAX) { Start-Sleep -Seconds 4 }
  $b=$j.b; $env:WM_BODY=$b.body; New-Item -ItemType Directory -Force $b.out | Out-Null
  $a=@("run_reach.py","--cond",$j.c,"--seed","$($j.s)","--steps","12000","--utd","2","--ntargets","8",
       "--prior",$b.prior,"--coach",$b.teacher,"--out",$b.out)
  Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $dir `
    -RedirectStandardOutput "$($b.out)\$($j.c)_s$($j.s).log" -RedirectStandardError "$($b.out)\$($j.c)_s$($j.s).err" -WindowStyle Hidden
  Start-Sleep -Milliseconds 600
}
while ((RunningCount) -gt 0) { Start-Sleep -Seconds 8 }
Write-Output "2x2 COMPLETE"
