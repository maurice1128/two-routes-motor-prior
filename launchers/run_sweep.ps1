# Launch the definitive sample-efficiency sweep: 3 conditions x N seeds, in parallel.
# Each process is capped to 1 torch thread, so ~1 core each.
param(
  [int]$Steps = 40000,
  [int]$Utd = 4,
  [int[]]$Seeds = @(0,1,2,3,4),
  [string]$Out = "results"
)
$py = "C:\Users\maurice\Desktop\robotic_research\.venv_mm\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$env:PYTHONUNBUFFERED = "1"
$conds = @("blank","colearn","prior")
foreach ($s in $Seeds) {
  foreach ($c in $conds) {
    $log = Join-Path $Out "$($c)_seed$($s).log"
    Start-Process -FilePath $py `
      -ArgumentList @("run_reach.py","--cond",$c,"--seed","$s","--steps","$Steps","--utd","$Utd","--out",$Out) `
      -WorkingDirectory $dir -RedirectStandardOutput $log -RedirectStandardError "$log.err" -WindowStyle Hidden
    Write-Output "launched $c seed $s -> $log"
  }
}
Write-Output "all jobs launched"
