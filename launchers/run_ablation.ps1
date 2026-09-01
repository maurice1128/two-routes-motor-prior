# Model-quality ablation: imagination-pretrain using priors of varying babble budget.
# Shows how downstream zero-real-step performance depends on model accuracy.
param([int[]]$Seeds = @(0,1,2), [string]$Out = "results_ablation")
$py = "C:\Users\maurice\Desktop\robotic_research\.venv_mm\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir
New-Item -ItemType Directory -Force $Out | Out-Null
$env:PYTHONUNBUFFERED="1"
$sizes = @(1000,3000,10000,30000,100000,200000)
foreach ($s in $Seeds) {
  foreach ($n in $sizes) {
    Start-Process -FilePath $py -ArgumentList @("imagine_pretrain.py","--seed","$s","--imagined","20000","--ntargets","8","--prior","prior_$n.pt","--out","$Out\n$n") `
      -WorkingDirectory $dir -RedirectStandardOutput "$Out\imag_n$($n)_s$($s).log" -RedirectStandardError "$Out\imag_n$($n)_s$($s).err" -WindowStyle Hidden
  }
  Write-Output "launched seed $s (6 model sizes)"
}
Write-Output "ablation launched"
