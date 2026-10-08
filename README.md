# Audiobook TTS - EPUB to M4B Converter

A local GUI application that converts EPUB e-books into M4B audiobooks with embedded chapter markers, using the Kokoro-82M text-to-speech model running on Intel XPU.

## Features

- **Chapter-based conversion**: Parse EPUB into individual chapters, convert single chapters or batch-select any subset
- **Resume/Append**: Stop and resume generation across sessions; rebuild M4B from completed chapter files. Chapter text is persisted per project, so resuming does not require the original EPUB
- **Voice selection**: All 54 Kokoro voices enumerated from the local voice pack, with language/gender metadata, type-to-filter search and a preview player
- **Language detection**: Auto-detect EPUB language from OPF metadata with heuristic fallback
- **M4B with chapter markers**: Final audiobook includes embedded chapter markers for player navigation
- **State persistence**: Atomic writes ensure crash resilience; all progress, settings and the output folder survive interruptions
- **Settable output folder**: Choose where the M4B is written; stored per project in the manifest

## Requirements

### Hardware
- Intel Arc GPU (tested on Arc Pro B70 with 32 GB VRAM)
- Intel GPU driver 32.0.101.8804 or newer
- Windows OS

### Software
- Python 3.10+
- FFmpeg and FFprobe on PATH

## Quick Start

### 1. Setup Environment

Run the setup script in PowerShell:

```powershell
./setup-xpu.ps1
```

This will:
- Create a `venv-xpu` virtual environment with Python 3.10
- Install Intel SYCL runtime packages
- Install PyTorch XPU wheels
- Install all application dependencies (Kokoro, NiceGUI, ebooklib, etc.)

**Note**: This requires an active internet connection and may take several minutes.

### 2. Verify XPU

After setup completes, verify XPU is available:

```powershell
./venv-xpu/Scripts/python.exe -c "import torch; print(f'XPU available: {torch.xpu.is_available()}')"
```

### 3. Run the Application

```powershell
./run-xpu.ps1
```

This will:
- Activate the XPU virtual environment
- Set `SYCL_CACHE_PERSISTENT=1` for better performance
- Launch the NiceGUI interface on http://127.0.0.1:8087 and open it in your browser

## Usage

The app is a single page with a linear workflow. Projects are saved automatically; every settings change is persisted to the project's `manifest.json` immediately.

1. **Open a book**
   - Choose an EPUB file; it is parsed and a project is created automatically
   - Or click an existing project in the left panel to resume it (no EPUB needed - chapter text is stored in the project)

2. **Chapters**
   - Pick chapters with the checkboxes (status and duration are shown per chapter)
   - "Pending only" preselects everything not yet generated; generating a done chapter regenerates it

3. **Voice**
   - Language is auto-detected from the EPUB; change it to filter the voice list
   - The voice dropdown supports typing to filter by name
   - "Preview voice" synthesizes the preview text with the selected voice and plays it

4. **Output**
   - Set the folder where the M4B will be written (defaults to the project folder)

5. **Generate**
   - "Generate selected chapters" synthesizes on XPU, then automatically rebuilds the M4B
   - "Stop" finishes the current chunk and stops; already-completed chapters stay saved
   - "Rebuild audiobook" re-assembles the M4B from all completed chapter WAVs on demand

### Resume/Append

- If you generate chapters 1-5 today and 6-10 tomorrow, both sessions are tracked in the manifest
- Rebuilding assembles the entire audiobook from all completed chapter WAVs
- Chapter markers are always accurate and the M4B is never corrupted by interrupted runs

## Project Structure

```
Audiobook_TTS/
├── AGENTS.md              # Project specification and knowledge base
├── STATUS.md             # Development checklist
├── README.md             # This file
├── app.py               # Main NiceGUI application
├── epub_parser.py       # EPUB ingestion and chapter extraction
├── text_processing.py    # Text normalization and chunking
├── kokoro_synth.py       # Kokoro-82M synthesis with XPU support
├── m4b_assembler.py      # M4B assembly with ffmpeg and chapter markers
├── state_manager.py      # State persistence and manifest management
├── setup-xpu.ps1         # Intel XPU environment setup script
├── run-xpu.ps1           # Intel XPU application launch script
├── requirements-xpu.txt  # XPU dependencies list
└── books/               # Generated book projects
    └── <book-id>/
        ├── manifest.json      # Book metadata, settings, chapter status
        ├── chapters/           # Individual chapter WAV files
        │   └── NNN.wav
        ├── text/               # Extracted chapter text (enables resume without EPUB)
        └── <title>.m4b         # Final audiobook (or a user-set output folder)
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KOKORO_DEVICE` | Device for Kokoro synthesis | `xpu` |
| `SYCL_CACHE_PERSISTENT` | Persist JIT-compiled SYCL kernels | `1` (set by run-xpu.ps1) |

## Supported Languages

Kokoro-82M supports 8 languages with 54 voices in the v1.0 voice pack (enumerated from the local cache at runtime):

| Code | Language | Voice Count |
|------|----------|-------------|
| a | American English | 20 |
| b | British English | 8 |
| e | Spanish | 3 |
| f | French | 1 |
| h | Hindi | 4 |
| i | Italian | 2 |
| j | Japanese | 5 |
| p | Portuguese (pt-BR) | 3 |
| z | Mandarin Chinese | 8 |

Language is auto-detected from EPUB metadata (OPF `dc:language` tag) with fallback to text analysis.

## Fallback Routes

If PyTorch XPU encounters issues, the application supports:

1. **OpenVINO**: Subclass KModel, execute via `ov.Core` device "GPU"
2. **ONNX Runtime**: Use `kokoro-onnx` with OpenVINOExecutionProvider

These are configured automatically when PyTorch XPU fails to initialize.

## Troubleshooting

### XPU Not Detected

1. Verify Intel GPU drivers are installed:
   ```powershell
   Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*Intel*" }
   ```

2. Verify Level Zero runtime:
   ```powershell
   Test-Path "C:\Windows\System32\ze_loader.dll"
   ```

3. Check XPU in Python:
   ```powershell
   ./venv-xpu/Scripts/python.exe -c "import torch; print(torch.xpu.is_available())"
   ```

### Kokoro Model Not Found

Kokoro-82M weights are automatically downloaded to `C:\Users\<user>\.cache\huggingface\hub\` on first use. Ensure you have ~500MB of disk space available.

### FFmpeg Not Found

FFmpeg must be on your system PATH. Install via:
```powershell
winget install Gyan.FFmpeg
```

Or download from https://ffmpeg.org and add to PATH.

## Performance Notes

- Kokoro-82M uses <2 GB VRAM on Intel Arc Pro B70
- Synthesis is deterministic: same (text, voice, speed) produces identical audio
- First run includes SYCL kernel compilation (persisted via `SYCL_CACHE_PERSISTENT=1`)
- Subsequent runs are significantly faster

## Development

See `STATUS.md` for the complete development checklist and `AGENTS.md` for detailed technical specifications.

## License

This project uses Kokoro-82M (Apache 2.0 license) and PyTorch. All code in this repository is provided as-is for local, non-commercial use.

## Open Questions

The following decisions are pending (see AGENTS.md):
- MP3 output as alternative to M4B
- Batch parallelism for chapters
- Japanese/Chinese book support in v1

Resolved: the default voice is `bf_isabella` (Isabella, British English), with
British G2P automatically selected for English books; the default speed is 1.0.
Both remain configurable per project in the UI.
