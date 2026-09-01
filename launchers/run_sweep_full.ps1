# Full multi-seed sweep: blank/colearn/prior (vs real steps) + imag (0 real steps).
param(
  [int]$Steps = 15000,
  [int]$Imagined = 20000,
  [int]$Utd = 2,
  [int[]]$Seeds = @(0,1,2,3,4),
  [string]$Out = "results_main"
)
$py = "C:\Users\maurice\Desktop\robotic_research\.venv_mm\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"
Set-Location $dir
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$env:PYTHONUNBUFFERED = "1"
foreach ($s in $Seeds) {
  foreach ($c in @("blank","colearn","prior")) {
    Start-Process -FilePath $py -ArgumentList @("run_reach.py","--cond",$c,"--seed","$s","--steps","$Steps","--utd","$Utd","--ntargets","8","--out",$Out) `
      -WorkingDirectory $dir -RedirectStandardOutput (Join-Path $Out "$($c)_s$($s).log") -RedirectStandardError (Join-Path $Out "$($c)_s$($s).err") -WindowStyle Hidden
  }
  Start-Process -FilePath $py -ArgumentList @("imagine_pretrain.py","--seed","$s","--imagined","$Imagined","--ntargets","8","--out",$Out) `
    -WorkingDirectory $dir -RedirectStandardOutput (Join-Path $Out "imag_s$($s).log") -RedirectStandardError (Join-Path $Out "imag_s$($s).err") -WindowStyle Hidden
  Write-Output "launched seed $s (blank,colearn,prior,imag)"
}
Write-Output "all launched: $($Seeds.Count) seeds x 4 conditions"
