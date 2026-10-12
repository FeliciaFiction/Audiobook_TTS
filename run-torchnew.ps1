# Runs the per-shape warm-up benchmark in the throwaway torch-upgrade venv
# (venv-torchnew, created by setup-torchnew.ps1). Compare its output against
# the torch 2.7.1+xpu reference numbers printed by the benchmark script.
#
# To run the full GUI app on this stack instead:
#   $env:KOKORO_DEVICE = "xpu"; .\venv-torchnew\Scripts\python.exe app.py

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectDir "venv-torchnew\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: venv-torchnew not found. Run .\setup-torchnew.ps1 first." -ForegroundColor Red
    exit 1
}

$env:SYCL_CACHE_PERSISTENT = "1"
$env:KOKORO_DEVICE = "xpu"

Set-Location $ProjectDir
# One STFT path per process: the per-shape penalty is shared between paths,
# so running both in one process cross-warms and hides it (see bench header).
& $VenvPython bench_shape_cost.py
& $VenvPython bench_shape_cost.py --fast
