# AGENTS.md — Audiobook TTS (EPUB → M4B via Kokoro-82M on Intel XPU)

This file is the founding spec and knowledge base for this project. It records the
product requirements, the target environment, verified knowledge about Kokoro TTS
and Intel XPU inference, and design decisions already made. Read it fully before
starting work. Update it when a decision changes or new knowledge is verified.

## Project mission

A local GUI application that converts EPUB e-books into M4B audiobooks with
embedded chapter markers, using the Kokoro-82M text-to-speech model running on the
local Intel Arc Pro B70 GPU (XPU). Target user workflow: pick an EPUB, pick a
voice, select chapters, generate — and be able to stop/resume across sessions.

## Status

v1 implemented and verified end-to-end (2026-10-07): EPUB ingest → chapter
split → text extraction → chunking → Kokoro synthesis on XPU → chaptered M4B,
with a single-page NiceGUI app, per-project state persistence and
resume-without-EPUB. The GUI was rewritten from Gradio to NiceGUI after user
feedback (Gradio was judged clunky; see Design decisions).

## Core requirements

1. **Chapter-based conversion.** Parse the EPUB into chapters. The user can
   convert a single chapter, or batch-select any subset of chapters. Each chapter
   becomes one chapter marker in the final M4B.
2. **Resume / append.** If chapters 1-5 of 20 are already done, the user can
   later generate 6+ and get a single M4B containing everything. This is
   implemented by *rebuilding* the M4B from per-chapter intermediate audio plus a
   manifest (see Design decisions) — never by binary-appending to the M4B.
3. **Voice selection.** Show all Kokoro voices available for the selected
   language, with language and gender metadata. A per-voice preview button
   (synthesize one sample sentence) is strongly desirable.
4. **Language detection + voice filtering.** Detect the EPUB's language (from
   OPF `dc:language` metadata first, with heuristic text-based fallback), then
   filter the voice list to voices that support that language (Kokoro voice
   naming convention: first letter = language, second = gender; see Kokoro
   knowledge base below). Allow manual override.

### Quality-of-life requirements (assumed, confirm with user if scope grows)

- Progress reporting per chapter (and ideally per text chunk within a chapter).
- Speed control (Kokoro `speed` parameter, default 1.0).
- Skip/redo individual chapters without touching others.
- Pause/stop generation mid-run; state must survive a restart of the app.

## Non-goals (for v1)

- No cloud services, no telemetry. Everything runs locally.
- No DRM circumvention. EPUBs with DRM are out of scope.
- No speaker diarization / multi-voice dialogue acting. One voice per book (a
  single global voice choice). Per-chapter voice mapping can come later.

## Target environment (verified 2026-10-07)

- Windows, Python 3.10 venv convention (mirror the sibling project
  `D:\git\Voxtral-AI-Demo-Local-Interface`: `venv-xpu\`, `setup-xpu.ps1`,
  `run-xpu.ps1` pattern).
- Intel Arc Pro B70, 32 GB VRAM, driver 32.0.101.8804+.
- Working XPU stack already proven on this machine: `torch==2.7.1+xpu`,
  `pytorch-triton-xpu==3.3.1`, `intel-sycl-rt==2025.0.5` family (see sibling
  project's `requirements-xpu.txt` for the exact pin set).
- Kokoro is trivially small next to this hardware: 82M params, ~327 MB weights,
  <2 GB VRAM. The B70 is enormous overkill; CPU fallback would also work, but
  XPU should be used for headroom on long books.

## Design decisions already made

- **GUI framework: NiceGUI (3.x)** (decided 2026-10-07, replacing the original
  Gradio choice after user feedback that Gradio was ugly/clunky and a "super
  simple" GUI was wanted). Single page, linear workflow: Open book → Chapters →
  Voice → Output → Generate. Served on 127.0.0.1:8087; `run-xpu.ps1` opens the
  browser automatically. Long-running synthesis runs via `run.io_bound` in a
  worker thread with a `threading.Event` stop flag; the UI polls progress with
  `ui.timer`. Note: NiceGUI 3 upgraded pydantic/starlette beyond gradio 5.38's
  pins, so Gradio was uninstalled — do not reintroduce it.
  NiceGUI API gotchas verified on 3.18: `ui.select(options)` dict is
  **{value: label}** (not label→value); `ui.slider` min/max are keyword-only;
  upload events expose `e.file.name` / `await e.file.save(path)` (no
  `e.content`); audio playback via `app.add_media_file(local_file=...)`.
- **Voice enumeration: from the HF cache, never hardcoded.** The v1.0 voice
  pack (54 voices) is globbed from
  `<HF_HUB_CACHE>/models--hexgrad--Kokoro-82M/voices/*.pt` and
  `snapshots/*/voices`. An earlier hardcoded fallback list contained invented
  voice names (e.g. `am_lee`) that 404 at synthesis time — this was a real bug.
  The hardcoded list now exists only as a last resort.
- **Idempotent per-chapter pipeline.** Never mutate a finished M4B in place.
  The canonical artifacts are:
  1. `<book-id>/manifest.json` — book metadata, detected language, chosen
     voice, speed, chapter list with status (pending/done), per-chapter output
     paths, start-time offsets computed at build time.
  2. `<book-id>/chapters/NNN.wav` (or .flac) — one lossless audio file per
     finished chapter, written atomically (tmp file + rename) so an interrupted
     run never leaves a half chapter marked done.
  3. `<book-id>/<book>.m4b` — a *derived artifact*, rebuilt from the chapter
     files + manifest whenever the user requests it. Appending chapter 6+ to an
     existing audiobook = generate the new chapter WAVs, update manifest, rebuild
     the whole M4B from all chapter files. This makes resume trivial, allows
     re-voicing a single chapter later, and avoids all M4B append corruption
     risks.
- **Lossless intermediates.** Synthesize to 24 kHz WAV/FLAC (Kokoro's native
  rate), encode to AAC in the M4B only at final assembly. Kokoro yields
  `(1, N)`-shaped tensors — flatten to mono before concatenating/writing.
- **Chapter text is persisted** in `<book-id>/text/NNN.txt` (atomic writes)
  at project creation, so resume works after an app restart without the
  original EPUB. The manifest also stores `epub_path`, `text_dir`, and the
  user-settable output folder (`m4b_path`).
- **Kokoro via the official `kokoro` PyPI package (KPipeline/KModel) running
  on PyTorch XPU** as the primary path. ONNX/OpenVINO is the fallback path if
  the PyTorch route hits issues (see Intel XPU knowledge base).

## Pipeline architecture

```
EPUB file
  ├─ 1. Ingest: extract OPF metadata (title, language, creator), spine, TOC
  ├─ 2. Chapter split: spine order + TOC/NCX chapter anchors → chapter list
  ├─ 3. Text extraction: XHTML → plain text (strip scripts/styles, footnotes
  │     handling is a real quality issue — flag link/notes sections per book)
  ├─ 4. Text normalization: typography cleanup (curly quotes, dashes), expand
  │     abbreviations/numbers where Kokoro's G2P handles them poorly
  ├─ 5. Chunking: sentence-level chunks ~1-5 sentences; KPipeline already
  │     yields per-sentence audio chunks — concatenate per chapter
  ├─ 6. Synthesis (Kokoro on XPU): per chunk → per chapter WAV, atomic write,
  │     manifest updated per chapter
  └─ 7. M4B assembly: chapter WAVs → AAC → single M4B with chapter markers
        (ffmpeg concat + FFMETADATA chapters file)
```

Libraries to prefer: `ebooklib` + `beautifulsoup4` + `lxml` for EPUB parsing;
`langdetect` or `lingua` for heuristic language fallback; `soundfile`/`numpy`
for audio handling; plain `ffmpeg` subprocess for assembly (no exotic M4B
tooling — ffmpeg alone can produce chaptered M4B, see knowledge base below).

## Kokoro knowledge base (verified 2026-10-07)

- Model: `hexgrad/Kokoro-82M`, Apache 2.0, 82M params, v1.0 (Jan 2025).
  Weights ~327 MB. Output: 24 kHz mono. StyleTTS2-style architecture.
- Python entry point: `from kokoro import KPipeline` (PyPI package `kokoro`,
  G2P via `misaki`). Pipeline usage:

  ```python
  pipeline = KPipeline(lang_code='a')           # 'a' = American English
  for graphemes, phonemes, audio in pipeline(text, voice='af_heart', speed=1.0):
      ...  # audio is a float32 tensor/array, 24 kHz; write with soundfile
  ```

  The generator yields one chunk per sentence-group — this is the natural
  chunking unit for long-book synthesis; concatenate chunks per chapter.
- **Language codes** (KPipeline `lang_code`): `a` American English, `b` British
  English, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `j` Japanese,
  `p` Portuguese (pt-BR), `z` Mandarin Chinese.
- **Voice naming**: `<lang><gender>_<name>`, e.g. `af_heart` = American-female
  "heart", `bm_george` = British-male. v1.0 ships **54 voices across 8
  languages** (verified counts from the local voice pack: a=20, b=8, z=8, j=5,
  h=4, e=3, p=3, i=2, f=1). Build the voice→language map from the prefix;
  do not hardcode voice names, enumerate at runtime from the loaded voice pack
  so new voice packs keep working.
- Japanese and Chinese require the misaki extras (`misaki[ja]`, `misaki[zh]`)
  and are heavier in text processing. English/Spanish/French/Italian/
  Portuguese/Hindi use the English-adjacent G2P pipeline.
- **Already cached locally** (`C:\Users\kees\.cache\huggingface\hub`):
  - `models--hexgrad--Kokoro-82M` — PyTorch route: `kokoro-v1_0.pth`,
    `config.json`, `voices/*.pt` (v1.0 voice pack, e.g. `af_heart.pt`).
  - `models--magicunicorn--kokoro-tts-intel` — ONNX route snapshot:
    `kokoro-v0_19.onnx` + `voices-v1.0.bin`. (Note: this is the older v0.19
    ONNX export; the ONNX route is secondary anyway.)
- Kokoro is deterministic (no sampling loop): three forward passes per chunk,
  so output is reproducible per (text, voice, speed) — good for caching and
  for re-generating a single chapter identically.

## Intel XPU knowledge base (verified on this B70, 2026-10-07)

- Working stack: `torch==2.7.1+xpu` + `pytorch-triton-xpu==3.3.1` + Intel SYCL
  runtime pins (see sibling project `requirements-xpu.txt`); bf16 is supported
  on the B70; GPU is detected as `Intel(R) Arc(TM) Pro B70 Graphics`.
- **PyTorch XPU route for Kokoro** (primary): load `KModel` (or let KPipeline
  own it), then `model.to('xpu')`. Voice style tensors go to the same device.
  Kokoro is a small conv/linear graph with **no autoregressive decode**, so the
  SDPA-with-KV-cache pathology measured on this GPU does not apply; still,
  wrap inference in `torch.inference_mode()` and call `torch.xpu.synchronize()`
  only around timing/progress boundaries.
- **Kokoro XPU performance (measured 2026-10-07, fp32, B70)**: steady-state
  synthesis is **~20-28x realtime**. The dominant first-run cost is SYCL/oneDNN
  JIT kernel compilation (CPU-heavy); it fades across sessions thanks to
  `SYCL_CACHE_PERSISTENT=1`, and the app warms up in a background thread at
  startup. Benchmark findings:
  - One pipeline call per *chapter* (KPipeline does its own sentence splitting)
    is ~3x faster than calling the pipeline per small chunk (per-call G2P +
    setup overhead). `synthesize_chapter()` implements this.
  - bf16 autocast is *slower* than fp32 on this stack (~22x vs ~26x realtime)
    and alters the output — do not switch. `model.to(bfloat16)` crashes
    (mixed dtypes in KModel's LSTMs); only autocast runs at all.
  - misaki G2P is negligible (~0.02s per 140 words); the forward pass is ~100%
    of synthesis time. CPU activity during generation is the SYCL runtime
    feeding the GPU, not the model running on CPU.
- **Env vars to set in the run script** (`run-xpu.ps1`):
  `SYCL_CACHE_PERSISTENT=1` — persists JIT-compiled SYCL/oneDNN kernels on disk
  so the first-run compilation penalty is paid once, not per launch. Verified
  beneficial on this machine during the Voxtral investigation.
- **Fallback route 1 — OpenVINO**: an established pattern exists of subclassing
  `KModel` and executing an OpenVINO graph (ov.Core, device "GPU") while
  reusing KPipeline for phonemization. OpenVINO's GPU plugin works on Arc on
  Windows with the standard Intel driver. Good if PyTorch XPU gives trouble.
- **Fallback route 2 — ONNX Runtime**: `kokoro-onnx` package with the
  `OpenVINOExecutionProvider` (`device_type: 'GPU'`, `precision: 'FP16'`) or
  `CPUExecutionProvider`. Requires the ONNX export files (already in local
  cache, see above).
- **VRAM**: Kokoro fits in <2 GB; on this 32 GB card memory pressure is a
  non-issue. Do not add device offloading complexity.
- Pitfall learned from the Voxtral investigation: on Windows/WDDM, allocations
  beyond VRAM *silently* spill to shared system memory with catastrophic
  slowdowns. Irrelevant at Kokoro's size, but do not load other large models
  alongside without expecting interference.

## M4B / ffmpeg knowledge base

- M4B = MP4 audio container (AAC) + chapter metadata. ffmpeg alone can build it.
- Chapter markers come from an FFMETADATA file passed as a second input:

  ```
  ;FFMETADATA1
  title=<book title>
  artist=<author>

  [CHAPTER]
  TIMEBASE=1/1000
  START=0
  END=<chapter 1 length in ms>
  title=Chapter 1: <name>

  [CHAPTER]
  TIMEBASE=1/1000
  START=<cumulative ms>
  END=<next boundary>
  title=Chapter 2: <name>
  ```

- Assembly: concatenate chapter WAVs (ffmpeg concat demuxer or a WAV concat via
  soundfile, then single encode), then mux with chapters:

  ```
  ffmpeg -i combined.wav -i chapters.txt -map_metadata 1 -c:a aac -b:a 128k \
         -movflags +faststart <book>.m4b
  ```

  Chapter START/END are computed from actual per-chapter audio durations read
  from the files (never from text length estimates).
- `faststart` matters: many players (BookPlayer, Apple Books) behave better with
  the moov atom at the front.
- Verify the result with `ffprobe -print_format json -show_chapters <file>`.
  Do NOT use `ffprobe -f ffmetadata <file>` — that flag parses the *input
  file as* an ffmetadata file, it does not dump embedded chapters (a real bug
  this caused).
- The ffmpeg concat demuxer resolves relative paths against the **concat list
  file's location**, not the process CWD — always write absolute paths into
  the list file.
- ebooklib (0.20) exposes OPF metadata as
  `book.metadata = {namespace: {prop: [(value, attrs), ...]}}` — e.g.
  `book.metadata['http://purl.org/dc/elements/1.1/']['title'][0][0]`. Do not
  look up keys like `[title]` or `DC:title`.
- On Windows, always pass `encoding='utf-8', errors='replace'` to
  `subprocess.run(..., text=True)`; the default cp1252 codec crashes on
  UTF-8 output from ffmpeg/ffprobe.
- **Legacy manifests**: projects created by the original Gradio app can store
  `voice` as a raw dropdown dict (`{"value": "af_heart", "label": ...}`) and have
  no `text/` directory. `BookManifest.from_dict` normalizes dict voices, and
  project loading restores chapter text from `epub_path` when none is stored
  (or warns the user to re-open the EPUB).

## Gotchas checklist

- EPUB chapter detection is book-specific and messy: some books put all content
  in one XHTML file, some have per-chapter files, TOC lives in NCX and/or
  nav.xhtml. Detect structure per book and let the user confirm the chapter map
  before synthesizing.
- Strip footnotes, publisher boilerplate, and "By the same author" pages, or
  chapters will be polluted. Make the pre-flight text preview a first-class
  GUI step.
- **Drop caps**: many EPUBs render the first letter of a chapter in its own
  inline element (`<span class="big">H</span>alla of...`). Text extraction
  must not insert a separator at inline-element boundaries or the word is
  split ("H alla"). `_get_text` in epub_parser keeps inline content joined
  (no separator for inline tags; newlines only around block-level tags).
  Distinguish from legit single-letter words like "I" at a chapter start.
- Kokoro handles very long single-sentence input poorly; keep chunks under
  roughly a paragraph. Insert short inter-chapter silence (~0.75-1.0 s).
- All writes to manifest and chapter WAVs must be atomic (tmp + rename) so
  kill -9 / power loss never corrupts resume state.
- NiceGUI long-running work: run synthesis via `run.io_bound` (thread), signal
  stop with a `threading.Event` checked per chunk, and refresh the UI from a
  `ui.timer` callback; never block the event loop with model inference.

## Open questions (ask the user before/while building)

- Preferred default voice: **resolved 2026-10-07 — `bf_isabella`** (Isabella,
  British English; user choice after listening to test output). Narration
  speed stays at the 1.0 default (user-configurable per project).
- MP3 output as an alternative to M4B, or M4B only?
- Should the GUI offer batch parallelism (multiple chapters in flight), or is
  sequential per-chapter generation with progress sufficient? (On the B70,
  sequential Kokoro synthesis is already far faster than real-time.)
- Japanese/Chinese book support in v1, or Latin-script languages only?
