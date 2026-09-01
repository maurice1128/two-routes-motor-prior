# Held-out generalization re-measurement (fixes the eval-on-train defect the audit found).
# Trains on the 8 training goals as before, but EVALUATES on 20 disjoint unseen goals.
# Run this when MuJoCo/CPU is free. Representative subset: arm2 + arm3, 3 seeds,
# spectrum (blank/colearn/warm/prior) + pure imagination.
param([int[]]$Seeds = @(0,1,2))
$py = "C:\Users\maurice\Desktop\robotic_research\.venv_mm\Scripts\python.exe"
$dir = "C:\Users\maurice\Desktop\robotic_research\wm_prior"; Set-Location $dir
$env:PYTHONUNBUFFERED="1"
foreach ($body in @("arm2","arm3")) {
  $env:WM_BODY = $body
  $prior = "prior$($body -replace 'arm2','').pt"; if ($body -eq "arm2") { $prior = "prior.pt" } else { $prior = "prior3.pt" }
  $out = "results_heldout_$body"
  New-Item -ItemType Directory -Force $out | Out-Null
  foreach ($s in $Seeds) {
    foreach ($c in @("blank","colearn","warm","prior")) {
      $pr = if ($c -in @("prior","warm")) { @("--prior",$prior) } else { @() }
      Start-Process -FilePath $py -ArgumentList (@("run_reach.py","--cond",$c,"--seed","$s","--steps","12000","--utd","2","--ntargets","8","--heldout","--out",$out) + $pr) -RedirectStandardOutput "$out\$($c)_s$s.log" -RedirectStandardError "$out\$($c)_s$s.err" -WindowStyle Hidden
    }
    # pure imagination, held-out eval
    Start-Process -FilePath $py -ArgumentList @("imagine_pretrain.py","--seed","$s","--imagined","20000","--ntargets","8","--prior",$prior,"--heldout","--out",$out) -RedirectStandardOutput "$out\imag_s$s.log" -RedirectStandardError "$out\imag_s$s.err" -WindowStyle Hidden
  }
  Write-Output "launched $body held-out ($($Seeds.Count) seeds x 5 conditions)"
}
Write-Output "held-out re-measurement launched. Compare results_heldout_* to results_main/results_main3."
