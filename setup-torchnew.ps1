# Throwaway venv for the torch-upgrade experiment (see AGENTS.md, XPU
# knowledge base: hexgrad/kokoro PR #362 uses torch 2.13.0+xpu; we pin
# 2.7.1+xpu and want to know whether newer oneDNN/SYCL reduced the
# ~2s per-shape warm-up cost that dominates real-book generation speed).
#
# This venv is EXPERIMENTAL - do not use it for the real app without a full
# generate + M4B test. It never touches venv-xpu or books/.
#
# Usage:  .\setup-torchnew.ps1                     (torch 2.13.0, the PR's)
#         .\setup-torchnew.ps1 -TorchVersion 2.14.1 (newest on the index)

param([string]$TorchVersion = "2.13.0")

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir "venv-torchnew"
$RequirementsFile = Join-Path $ProjectDir "requirements-xpu.txt"

Write-Host "=== Audiobook TTS - torch $TorchVersion XPU throwaway setup ===" -ForegroundColor Cyan

# --- Locate Python 3.12 (kokoro 0.9.4 requires <3.13; same ceiling as venv-xpu) ---
$PythonExe = $null
try {
    $exe = (uv python find 3.12 2>$null)
    if ($exe) { $exe = $exe.Trim(); if ($exe -and (Test-Path $exe)) { $PythonExe = $exe } }
} catch { }
if (-not $PythonExe) {
    foreach ($v in @("3.12", "3.11")) {
        try {
            $exe = (py -$v -c "import sys; print(sys.executable)" 2>$null)
            if ($exe) { $exe = $exe.Trim(); if ($exe) { $PythonExe = $exe; break } }
        } catch { }
    }
}
if (-not $PythonExe) {
    Write-Host "ERROR: no Python 3.11/3.12 found (kokoro 0.9.4 needs <3.13)." -ForegroundColor Red
    exit 1
}
Write-Host "Using Python: $PythonExe"

# --- Recreate the throwaway venv ---
if (Test-Path $VenvDir) {
    Write-Host "Removing existing venv-torchnew..." -ForegroundColor Yellow
    Remove-Item -Path $VenvDir -Recurse -Force
}
& $PythonExe -m venv $VenvDir

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython --version

& $VenvPython -m pip install --upgrade pip "setuptools>=70" --quiet

# Step 1/3: torch XPU wheel.
# Try the documented single-index form first; if dependency resolution fails
# (Intel runtime packages may live on PyPI, not the xpu index), retry with
# PyPI as an extra index. The ==pin keeps pip from grabbing a newer PyPI
# build, and the +xpu local version sorts above the plain PyPI wheel.
Write-Host "`n[1/3] Installing torch==$TorchVersion from the XPU wheel index..." -ForegroundColor Yellow
& $VenvPython -m pip install --no-cache-dir --progress-bar off "torch==$TorchVersion" --index-url https://download.pytorch.org/whl/xpu
if ($LASTEXITCODE -ne 0) {
    Write-Host "Single-index install failed - retrying with PyPI as extra index..." -ForegroundColor Yellow
    & $VenvPython -m pip install --no-cache-dir --progress-bar off "torch==$TorchVersion" --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: torch XPU install failed." -ForegroundColor Red
        exit 1
    }
}

# Step 2/3: application requirements (torch already satisfied; requirements
# pin kokoro/misaki/etc., none of them pin torch)
Write-Host "`n[2/3] Installing application requirements..." -ForegroundColor Yellow
& $VenvPython -m pip install --no-cache-dir --progress-bar off -r $RequirementsFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: requirements install failed." -ForegroundColor Red
    exit 1
}

# Step 3/3: spaCy model for misaki G2P
Write-Host "`n[3/3] Installing spaCy model en_core_web_sm..." -ForegroundColor Yellow
& $VenvPython -m spacy download en_core_web_sm

# --- Verify ---
Write-Host "`nVerifying installation..." -ForegroundColor Yellow
& $VenvPython -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'XPU available: {torch.xpu.is_available()}'); print(f'XPU device: {torch.xpu.get_device_name(0)}' if torch.xpu.is_available() else 'XPU NOT AVAILABLE')"
& $VenvPython -c "import kokoro, numpy, soundfile; print(f'kokoro {kokoro.__version__}, numpy {numpy.__version__}, soundfile {soundfile.__version__}')"

Write-Host "`n=== torch $TorchVersion XPU throwaway setup complete ===" -ForegroundColor Green
Write-Host "Run the per-shape benchmark:  .\run-torchnew.ps1" -ForegroundColor Cyan
Write-Host "Delete when done:              Remove-Item -Recurse -Force venv-torchnew" -ForegroundColor DarkGray
