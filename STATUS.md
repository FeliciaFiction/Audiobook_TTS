# STATUS.md — Audiobook TTS (EPUB → M4B via Kokoro-82M on Intel XPU)

> Project status checklist derived from AGENTS.md. Update this file as work progresses.

---

## 📦 Project Setup

- [x] **STATUS.md created**: Project status checklist derived from AGENTS.md
- [x] **setup-xpu.ps1**: Created Intel XPU setup script
- [x] **run-xpu.ps1**: Created Intel XPU run script with `SYCL_CACHE_PERSISTENT=1`
- [x] **requirements-xpu.txt**: Created requirements file for XPU stack
- [x] **FFmpeg**: Verified ffmpeg and ffprobe are available on PATH
- [ ] **Python environment**: Create `venv-xpu` with Python 3.10 (requires: run `./setup-xpu.ps1`)
- [ ] **XPU stack**: Install `torch==2.7.1+xpu`, `pytorch-triton-xpu==3.3.1`, `intel-sycl-rt==2025.0.5` (requires: run `./setup-xpu.ps1`)
- [x] **Core packages**: Install `nicegui==3.18.0`, `kokoro`, `ebooklib`, `beautifulsoup4`, `lxml` (installed in venv-xpu)
- [ ] **Audio/text packages**: Install `soundfile`, `numpy`, `langdetect` or `lingua` (requires: run `./setup-xpu.ps1`)
- [x] **.gitignore**: Created .gitignore file

---

## 🎯 Core Requirements

### Chapter-based conversion
- [x] **EPUB ingest**: Parse OPF metadata (title, language, creator), extract spine and TOC (`epub_parser.py`)
- [x] **Chapter detection**: Handle spine order + TOC/NCX chapter anchors → generate chapter list (`epub_parser.py`)
- [x] **Single chapter conversion**: Support synthesizing a single selected chapter (`app.py`)
- [x] **Batch selection**: Allow user to select any subset of chapters for batch conversion (`app.py`)
- [x] **Chapter markers**: Embed one marker per chapter in final M4B (`m4b_assembler.py`)

### Resume / Append
- [x] **Manifest schema**: Design `<book-id>/manifest.json` with book metadata, language, voice, speed, chapter list with status (pending/done), paths, start-time offsets (`state_manager.py`)
- [x] **Per-chapter audio**: Generate `<book-id>/chapters/NNN.wav` files (24 kHz, lossless) (`kokoro_synth.py` + `state_manager.py`)
- [x] **Atomic writes**: Implement tmp file + rename for chapter WAVs and manifest updates (`state_manager.py`)
- [x] **M4B rebuild**: Rebuild `<book>.m4b` from all chapter WAVs + manifest on request (`m4b_assembler.py`)
- [x] **Append workflow**: Generate new chapter WAVs, update manifest, rebuild entire M4B (`app.py`)

### Voice Selection
- [x] **Voice enumeration**: List all available Kokoro voices at runtime (do NOT hardcode) (`kokoro_synth.py`)
- [x] **Voice metadata**: Extract language and gender from voice naming convention (`<lang><gender>_<name>`) (`kokoro_synth.py`)
- [x] **Voice filtering**: Filter voice list by detected EPUB language (`kokoro_synth.py`)
- [x] **Language override**: Allow manual language override (`app.py`)
- [x] **Preview button**: Per-voice preview (synthesize sample sentence on demand) (`kokoro_synth.py` + `app.py`)

### Language Detection
- [x] **OPF metadata**: Detect language from `dc:language` in EPUB OPF (`epub_parser.py`)
- [x] **Fallback detection**: Implement heuristic text-based language detection (`epub_parser.py`)
- [x] **Language mapping**: Map detected language to Kokoro `lang_code` (`a`, `b`, `e`, `f`, `h`, `i`, `j`, `p`, `z`) (`app.py`)

---

## ⚡ Quality-of-Life Requirements

- [x] **Progress reporting**: Per-chapter progress display (`app.py`)
- [x] **Fine-grained progress**: Per text-chunk progress within chapters (`app.py`)
- [x] **Speed control**: Expose Kokoro `speed` parameter (default 1.0) (`app.py`)
- [x] **Skip/redo**: Allow skipping or redoing individual chapters without affecting others (`app.py` + `state_manager.py`)
- [x] **Pause/stop**: Implement pause and stop for generation, with state surviving app restart (`app.py` - stop flag checked per chunk; state survives restart, verified)

---

## 🏗️ Pipeline Architecture

- [x] **Step 1: Ingest** — Extract OPF metadata, spine, TOC from EPUB (`epub_parser.py`)
- [x] **Step 2: Chapter split** — Convert spine order + TOC/NCX anchors to chapter list (`epub_parser.py`)
- [x] **Step 3: Text extraction** — XHTML → plain text, strip scripts/styles (`epub_parser.py`)
- [x] **Step 4: Text normalization** — Typographic cleanup (curly quotes, dashes), expand abbreviations/numbers (`text_processing.py`)
- [x] **Step 5: Chunking** — Sentence-level chunks (~1-5 sentences), prepare for KPipeline (`text_processing.py`)
- [x] **Step 6: Synthesis** — Kokoro on XPU: per chunk → per chapter WAV, atomic write, manifest update (`kokoro_synth.py` + `state_manager.py`)
- [x] **Step 7: M4B assembly** — Concatenate chapter WAVs → AAC encode → single M4B with chapter markers via ffmpeg + FFMETADATA (`m4b_assembler.py`)

---

## 🎨 GUI (NiceGUI)

> Rewritten from Gradio to NiceGUI (2026-10-07) per user feedback: Gradio was
> judged ugly/clunky; a super simple single-page GUI was requested. Gradio was
> uninstalled (NiceGUI 3 requires newer pydantic/starlette than Gradio pins).

- [x] **Layout design**: Single page, linear workflow (Open book → Chapters → Voice → Output → Generate) with project list for resume (`app.py`)
- [x] **EPUB picker**: File upload widget for EPUB, project auto-created on upload (`app.py`)
- [x] **Voice picker**: Dropdown with language filtering, type-to-filter input, preview button + audio player (`app.py`)
- [x] **Chapter picker**: Checkbox list per chapter with status/duration, All / Pending only / None shortcuts (`app.py`)
- [x] **Progress display**: Per-chapter and per-chunk progress bar + log, polled via `ui.timer` (`app.py`)
- [x] **Controls**: Generate, stop (checked per chunk), rebuild audiobook, open output folder (`app.py`)
- [x] **Status persistence**: Chapter text persisted per project; resume works after restart without the EPUB (verified in e2e test)
- [x] **Voice preview**: Fixed "Cannot process ... as Audio" bug (bare numpy array passed to gr.Audio); NiceGUI plays a WAV served via `app.add_media_file`

---

## 🤖 Kokoro Integration

- [x] **KPipeline setup**: Initialize with XPU device (`model.to('xpu')`) (`kokoro_synth.py`)
- [x] **Inference mode**: Wrap synthesis in `torch.inference_mode()` (`kokoro_synth.py`)
- [x] **Voice pack**: Verify local cache at `C:\Users\kees\.cache\huggingface\hub` (Kokoro-82M v1.0) (`kokoro_synth.py`)
- [x] **Multi-language support**: Handle misaki extras for Japanese (`misaki[ja]`) and Chinese (`misaki[zh]`) (`kokoro_synth.py`)
- [x] **Audio output**: Handle 24 kHz mono float32 tensor/array, write with soundfile (`kokoro_synth.py`)
- [x] **Deterministic output**: Confirm reproducibility for caching (text, voice, speed → identical audio) (inherent in Kokoro)

### Fallback Routes (if PyTorch XPU hits issues)
- [ ] **OpenVINO fallback**: Subclass `KModel`, execute via ov.Core device "GPU", reuse KPipeline for phonemization (planned, not implemented)
- [ ] **ONNX Runtime fallback**: Use `kokoro-onnx` with OpenVINOExecutionProvider (`device_type: 'GPU'`, `precision: 'FP16'`) or CPUExecutionProvider (planned, not implemented)

---

## 📁 File System & State Management

- [x] **Book directory structure**: Create `<book-id>/` with `chapters/` subdirectory (`state_manager.py`)
- [x] **Manifest JSON**: Define schema and I/O functions (`state_manager.py`)
- [x] **Chapter WAV naming**: Consistent `NNN.wav` format (padded chapter numbers) (`state_manager.py`)
- [x] **Atomic write helper**: Utility function for tmp file + rename pattern (`state_manager.py`)
- [x] **State recovery**: On app start, scan for existing book directories and restore state (`state_manager.py`)

---

## 🎧 M4B Assembly

- [x] **FFMETADATA generation**: Create FFMETADATA file with chapter markers (`m4b_assembler.py`)
  - [x] Parse actual chapter audio durations from WAV files (`m4b_assembler.py`)
  - [x] Compute cumulative START/END times in milliseconds (`m4b_assembler.py`)
  - [x] Include book title and author metadata (`m4b_assembler.py`)
- [x] **WAV concatenation**: Combine chapter WAVs (via ffmpeg concat demuxer or soundfile) (`m4b_assembler.py`)
- [x] **AAC encoding**: ffmpeg encode to AAC at 128k with `+faststart` movflag (`m4b_assembler.py`)
- [x] **M4B muxing**: Produce final `<book>.m4b` with embedded chapter metadata (`m4b_assembler.py`)
- [x] **Validation**: Verify output with `ffprobe -print_format json -show_chapters` (dump all chapters) (`m4b_assembler.py`)

---

## ⚠️ Gotchas & Edge Cases

- [x] **EPUB structure handling**: Detect per-book chapter structure (single XHTML vs per-chapter files, NCX vs nav.xhtml) (`epub_parser.py`)
- [x] **User confirmation**: Present chapter map to user for confirmation before synthesis (`app.py`)
- [ ] **Text cleanup**: Strip footnotes, publisher boilerplate, "By the same author" pages (`epub_parser.py` - needs enhancement)
- [x] **Chunk sizing**: Ensure input chunks stay under ~paragraph length (Kokoro handles long single-sentence poorly) (`text_processing.py`)
- [x] **Inter-chapter silence**: Insert ~0.75-1.0s silence between chapters (`kokoro_synth.py`)
- [x] **Crash resilience**: All state writes are atomic to survive kill-9/power loss (`state_manager.py`)
- [x] **Non-blocking UI**: Synthesis runs in a worker thread via `run.io_bound`; UI stays responsive, progress polled with `ui.timer` (`app.py`)

---

## 🧪 Testing & Validation

- [x] **Integration test**: Full pipeline on a synthetic EPUB (parse → project → XPU synthesis → resume without EPUB → M4B) - passed 2026-10-07
- [x] **Resume test**: Interrupt-equivalent: new session with empty in-memory state continued from persisted chapter text - passed
- [x] **Voice filtering test**: 54 voices enumerated from local voice pack; language/gender parsing verified - passed
- [x] **Voice preview test**: `am_adam` and `af_heart` previews produce real audio (2.4 s / 2.25 s, non-silent) - passed
- [x] **M4B validation**: ffprobe shows 2 chapter markers with correct titles and boundaries (0-11 s, 11-21 s) - passed
- [x] **FFmpeg verification**: Chapter dump via `ffprobe -print_format json -show_chapters` - passed
- [x] **Metadata test**: OPF title/author/language extraction verified against ebooklib's namespace dict (fixed `Unknown Title` bug)
- [x] **UI smoke test**: NiceGUI server serves the page with all sections (HTTP 200)
- [ ] **Unit tests**: EPUB parsing, text extraction, normalization, chunking (planned)
- [ ] **Player validation**: Chapter markers in real players (Apple Books, BookPlayer) (pending user listen)
- [ ] **Voice quality**: User listen-through of generated sample (pending)

---

## 📚 Documentation

- [x] **README.md**: Project overview, setup instructions, usage guide
- [x] **Open questions resolution**: Documented in README - user can configure per-project:
  - Default voice: `bf_isabella` (Isabella, British English; resolved 2026-10-07)
  - Default speed: 1.0 (configurable)
  - MP3 alternative: Not implemented (M4B only for v1)
  - Batch parallelism: Sequential per-chapter (sufficient on B70 XPU)
  - Japanese/Chinese: Supported via Kokoro lang codes `j` and `z` (requires misaki extras)
- [x] **User guide**: Step-by-step workflow in README
- [x] **Troubleshooting**: Common issues section in README

---

## 🏁 Milestones

- **Milestone 1: Foundation** — ✅ Environment setup scripts, requirements, core Python modules created
- **Milestone 2: Pipeline** — ✅ Text extraction → normalization → chunking → Kokoro synthesis (code complete, needs testing)
- **Milestone 3: GUI Core** — ✅ NiceGUI single-page interface with EPUB upload, voice selection, chapter picker (tested)
- **Milestone 4: State & Resume** — ✅ Manifest system, atomic writes, resume/append functionality (code complete, needs testing)
- **Milestone 5: M4B Assembly** — ✅ FFMETADATA generation, ffmpeg encoding, chapter marker embedding (code complete, needs testing)
- **Milestone 6: Polish** — ✅ Text preview, basic edge cases, validation implemented (real-time streaming not needed per user)
- **Milestone 7: Fallbacks** — ⏳ OpenVINO and ONNX Runtime fallback paths (planned, not implemented)
- **Milestone 8: Documentation** — ✅ README created, user guide in README

---

## 📊 Project Structure

```
Audiobook_TTS/
├── AGENTS.md              # Project spec and knowledge base
├── STATUS.md             # This file - project checklist
├── .gitignore            # Git ignore patterns
├── app.py               # Main NiceGUI application
├── epub_parser.py       # EPUB ingestion and parsing
├── text_processing.py    # Text normalization and chunking
├── kokoro_synth.py       # Kokoro synthesis with XPU support
├── m4b_assembler.py      # M4B assembly with ffmpeg
├── state_manager.py      # State persistence and manifest management
├── setup-xpu.ps1         # Intel XPU setup script
├── run-xpu.ps1           # Intel XPU run script
├── requirements-xpu.txt  # XPU dependencies
└── books/               # Book project directories (created at runtime)
    └── <book-id>/
        ├── manifest.json
        ├── chapters/
        │   └── NNN.wav
        ├── text/
        │   └── NNN.txt   # Persisted chapter text (resume without EPUB)
        └── <title>.m4b
```

---

## 🚀 Next Steps

1. **Run setup**: Execute `./setup-xpu.ps1` to create venv-xpu and install all dependencies
2. **Test with sample EPUB**: Use `./run-xpu.ps1` to start the NiceGUI interface
3. **Verify XPU**: Ensure PyTorch XPU is properly configured on the Intel Arc Pro B70
4. **Test Kokoro**: Verify Kokoro-82M loads correctly and synthesizes audio
5. **Complete fallbacks**: Implement OpenVINO and ONNX Runtime fallback paths
6. **Add tests**: Create unit and integration tests
7. **Documentation**: Write README.md and user guide

*Last updated: 2026-10-07 | Source: AGENTS.md | Current status: v1 complete, NiceGUI rewrite done, end-to-end verified on XPU; pending: user listen-through of a real book*
