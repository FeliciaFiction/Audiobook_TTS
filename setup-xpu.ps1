# Intel XPU (GPU) requirements for Audiobook TTS
# Requires Python 3.10+
# For Intel Arc / Data Center GPU (XPU) support via PyTorch native XPU + SYCL runtime
#
# Install order:
#   1. Intel SYCL runtime packages (see setup-xpu.ps1 step 1)
#   2. PyTorch XPU wheels (see setup-xpu.ps1 step 2):
#        torch==2.7.1+xpu torchaudio==2.7.1+xpu torchvision==0.22.1+xpu
#        pytorch-triton-xpu==3.3.1
#        (install with --index-url https://download.pytorch.org/whl/xpu)
#   3. Application requirements below (setup-xpu.ps1 step 3 installs this file)
#
# Prerequisites:
#   - Intel GPU driver installed (Arc Pro B70 driver 32.0.101.8804 or newer)
#   - Level Zero runtime (ze_loader.dll in System32 - included with Intel GPU driver)

# --- Application dependencies ---
nicegui==3.18.0
kokoro>=0.9.0
ebooklib>=0.19
beautifulsoup4>=4.13
lxml>=5.0
langdetect>=1.0.9
soundfile>=0.13
numpy>=2.1

# --- spaCy models for Kokoro's misaki G2P (English) ---
# Install separately after this file:
#   python -m spacy download en_core_web_sm
#   python -m spacy download en_core_web_trf
