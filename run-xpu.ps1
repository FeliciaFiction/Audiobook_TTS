# Run Audiobook TTS on Intel XPU (GPU)
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectDir "venv-xpu\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: XPU virtual environment not found. Run setup-xpu.ps1 first." -ForegroundColor Red
    exit 1
}

# Use XPU device for Kokoro
$env:KOKORO_DEVICE = "xpu"

# Keep JIT-compiled SYCL/oneDNN kernels on disk so the first-run compilation
# penalty is only paid once instead of on every launch.
$env:SYCL_CACHE_PERSISTENT = "1"

# Check if required spaCy models are installed
Write-Host "Checking spaCy models..." -ForegroundColor Yellow
$result = & $VenvPython -c "import spacy; models = spacy.util.get_installed_models(); print('OK' if 'en_core_web_sm' in models else 'MISSING')" 2>$null
if ($result -eq "MISSING") {
    Write-Host "Required spaCy model 'en_core_web_sm' is not installed." -ForegroundColor Yellow
    Write-Host "Installing it now..." -ForegroundColor Yellow
    & $VenvPython -m spacy download en_core_web_sm
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install spaCy model. Run setup-xpu.ps1 to fix." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Starting Audiobook TTS on Intel XPU..." -ForegroundColor Cyan
Write-Host "Open http://localhost:8087 in your browser" -ForegroundColor Green

Set-Location $ProjectDir
& $VenvPython app.py
