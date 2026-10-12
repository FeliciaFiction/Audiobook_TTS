# Intel XPU (GPU) setup script for Audiobook TTS
# Creates venv-xpu with a supported Python and installs the full XPU stack.
#
# IMPORTANT: run this only when the app is NOT running (run-xpu.ps1 holds the
# venv in use). The script replaces venv-xpu but never touches the books/
# directory, so existing projects, manifests, and chapter audio are preserved.
#
# Torch stack (qualified 2026-10-08): torch==2.13.0+xpu from the PyTorch XPU
# wheel index. The XPU wheels declare their own Intel runtime dependencies
# (mkl 2026.x, intel-openmp 2026.x, tbb, umf, tcmlib, triton-xpu 3.7.x - the
# package was renamed from pytorch-triton-xpu), so unlike the old torch 2.7.1
# flow there is NO manually pinned Intel SYCL step: pip resolves everything
# from the wheel metadata. This roughly triples synthesis throughput versus
# torch 2.7.1 (79x vs 27-28x realtime steady state on the B70; see AGENTS.md).
# torchaudio/torchvision are NOT installed - nothing in this app imports them.
#
# Target interpreter: Python 3.12 (kokoro==0.9.4 declares Requires-Python
# >=3.10,<3.13, and the torch 2.13 XPU wheels ship cp310-cp312).

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir "venv-xpu"
$RequirementsFile = Join-Path $ProjectDir "requirements-xpu.txt"
$TorchVersion = "2.13.0"

Write-Host "=== Audiobook TTS - Intel XPU Setup (torch $TorchVersion) ===" -ForegroundColor Cyan
Write-Host "Project directory: $ProjectDir"
Write-Host "Virtual environment: $VenvDir"

# --- Locate a supported Python interpreter ---
Write-Host "`nChecking Python interpreter..." -ForegroundColor Yellow
$PythonVersion = $null
$PythonExe = $null

# Prefer a uv-managed interpreter: Astral builds patched 3.12.x binaries
# (python.org stopped shipping 3.12 installers at 3.12.10, and the py
# launcher does not reliably prefer newer uv-registered builds).
try {
    $uvExe = (uv python find 3.12 2>$null)
    if ($uvExe) {
        $uvExe = $uvExe.Trim()
        if ($uvExe -and (Test-Path $uvExe)) {
            $PythonVersion = "3.12 (uv-managed)"
            $PythonExe = $uvExe
        }
    }
} catch { }

foreach ($v in @("3.12", "3.11")) {
    if ($PythonExe) { break }
    try {
        $exe = (py -$v -c "import sys; print(sys.executable)" 2>$null)
        if ($exe) {
            $exe = $exe.Trim()
            if ($exe) {
                $PythonVersion = $v
                $PythonExe = $exe
                break
            }
        }
    } catch { }
}
if (-not $PythonExe) {
    Write-Host "ERROR: no suitable Python found. This project needs Python 3.11/3.12" -ForegroundColor Red
    Write-Host "(kokoro 0.9.4 does not support 3.13+)." -ForegroundColor Red
    exit 1
}
Write-Host "Using Python $PythonVersion : $PythonExe"

# --- Check Intel GPU and Level Zero runtime ---
Write-Host "`nChecking Intel GPU..." -ForegroundColor Yellow
$gpu = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*Intel*Arc*" }
if ($gpu) {
    Write-Host "Found Intel GPU: $($gpu.Name)" -ForegroundColor Green
    Write-Host "  Driver: $($gpu.DriverVersion)"
} else {
    Write-Host "WARNING: no Intel Arc GPU detected. XPU may not work." -ForegroundColor Yellow
}

$zeLoader = Get-ChildItem -Path "C:\Windows\System32" -Filter "ze_loader.dll" -ErrorAction SilentlyContinue
if ($zeLoader) {
    Write-Host "Level Zero runtime found: $($zeLoader.FullName)" -ForegroundColor Green
} else {
    Write-Host "WARNING: ze_loader.dll not found. Install Intel GPU drivers." -ForegroundColor Yellow
}

# --- Recreate the virtual environment (books/ data is never touched) ---
if (Test-Path $VenvDir) {
    Write-Host "`nRemoving existing venv-xpu..." -ForegroundColor Yellow
    Remove-Item -Path $VenvDir -Recurse -Force
}

Write-Host "Creating virtual environment with Python $PythonVersion..." -ForegroundColor Yellow
& $PythonExe -m venv $VenvDir

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython --version

# Upgrade pip and setuptools (fresh 3.12 venvs ship no setuptools; older
# bundled versions have known CVEs).
Write-Host "`nUpgrading pip and setuptools..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip "setuptools>=70"

# Step 1/3: PyTorch XPU wheel. The ==pin keeps pip from grabbing a newer
# plain-PyPI build; the +xpu local version sorts above it anyway. The wheel
# pulls its own Intel runtime dependencies (see header).
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

# Step 2/3: Application requirements (torch is already satisfied)
Write-Host "`n[2/3] Installing application requirements..." -ForegroundColor Yellow
& $VenvPython -m pip install --no-cache-dir --progress-bar off -r $RequirementsFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: requirements install failed." -ForegroundColor Red
    exit 1
}

# Step 3/3: spaCy model for Kokoro's misaki G2P (English)
# Unpinned on purpose: spaCy's downloader resolves the model version
# compatible with the installed spaCy (a hard pin here failed on spaCy 3.8.16).
Write-Host "`n[3/3] Installing spaCy model en_core_web_sm..." -ForegroundColor Yellow
& $VenvPython -m spacy download en_core_web_sm

# --- Verify ---
Write-Host "`nVerifying installation..." -ForegroundColor Yellow
& $VenvPython -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'XPU available: {torch.xpu.is_available()}'); print(f'XPU device: {torch.xpu.get_device_name(0)}' if torch.xpu.is_available() else 'XPU not available'); print(f'BF16 supported: {torch.xpu.is_bf16_supported()}' if torch.xpu.is_available() else '')"
& $VenvPython -c "import nicegui, kokoro, ebooklib, bs4, soundfile, numpy; print(f'nicegui {nicegui.__version__}, kokoro {kokoro.__version__}, ebooklib {ebooklib.VERSION}, bs4 {bs4.__version__}, soundfile {soundfile.__version__}, numpy {numpy.__version__}')"
& $VenvPython -m spacy validate

Write-Host "`n=== XPU Setup Complete ===" -ForegroundColor Green
Write-Host "To run the app with XPU:" -ForegroundColor Cyan
Write-Host "  .\run-xpu.ps1"
