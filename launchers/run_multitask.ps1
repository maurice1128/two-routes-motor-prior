# Multi-task amortization: ONE frozen prior (prior.pt) serves several downstream tasks.
# For each task, compare from-scratch blank vs imagination-pretrained (0 real steps).
param([int[]]$Seeds = @(0,1,2), [string]$Out = "results_multitask")
$py = "C:\Users\maurice\Desktop\robotic_research\.venv_mm\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir
New-Item -ItemType Directory -Force $Out | Out-Null
$env:PYTHONUNBUFFERED="1"
foreach ($s in $Seeds) {
  foreach ($t in @("reachB","hold")) {
    # from-scratch baseline
    Start-Process -FilePath $py -ArgumentList @("run_reach.py","--cond","blank","--seed","$s","--steps","15000","--utd","2","--ntargets","8","--task",$t,"--out",$Out) `
      -WorkingDirectory $dir -RedirectStandardOutput "$Out\blank_$($t)_s$($s).log" -RedirectStandardError "$Out\blank_$($t)_s$($s).err" -WindowStyle Hidden
    # imagination with the SAME frozen prior.pt
    Start-Process -FilePath $py -ArgumentList @("imagine_pretrain.py","--seed","$s","--imagined","20000","--ntargets","8","--task",$t,"--prior","prior.pt","--out",$Out) `
      -WorkingDirectory $dir -RedirectStandardOutput "$Out\imag_$($t)_s$($s).log" -RedirectStandardError "$Out\imag_$($t)_s$($s).err" -WindowStyle Hidden
  }
  Write-Output "launched seed $s (reachB, hold)"
}
Write-Output "multitask launched"
