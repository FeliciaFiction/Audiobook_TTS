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

- Windows, Python 3.12 venv convention (updated 2026-10-08): `venv-xpu\`,
  `setup-xpu.ps1`, `run-xpu.ps1` pattern. The interpreter is the **uv-managed
  3.12.13** (`uv python install 3.12.13`; setup-xpu.ps1 resolves it via
  `uv python find 3.12` and falls back to the py launcher). Why 3.12.13 and
  not the python.org 3.12.10: python.org stopped shipping 3.12 binary
  installers at 3.12.10 (source-only since), so 3.12.10 misses all later
  security fixes; Astral's prebuilt 3.12.13 is patched to April 2026 —
  3.12.14/15 remain source-only. Why not 3.13+: **kokoro 0.9.4 declares
  Requires-Python >=3.10,<3.13** (verified: `pip install kokoro==0.9.4` on
  3.13 finds no matching distribution), and the pinned torch==2.7.1+xpu
  wheels stop at cp313 (3.14/3.15 need newer torch). Revisit when kokoro
  raises its Python ceiling. The sibling project
  `D:\git\Voxtral-AI-Demo-Local-Interface` still uses 3.10 — do not mirror
  it for new venvs.
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
- **stft/istft XPU fallback (measured 2026-10-08)**: torch 2.7.1+xpu has no
  XPU kernel for `torch.stft`/`torch.istft`. Kokoro's default decoder path
  (`disable_complex=False` → `TorchSTFT` in `kokoro/istftnet.py`) calls both
  once per sentence chunk on XPU-resident tensors, so ATen silently copies
  to CPU, runs the FFT, and copies back — every chunk. This shows up as the
  one-time `UserWarning: Aten Op fallback from XPU to CPU ... _VF.stft`
  at warm-up. Measured cost on the same fixed text, fp32, B70, steady state:
  default path **14.9x realtime** vs `KModel(disable_complex=True)`
  (`CustomSTFT`, conv-based, XPU-native) **27.5x realtime** — i.e. the
  fallback roughly halves throughput. Caveat: `CustomSTFT.inverse` is an
  explicitly *approximate* iSTFT (no proper COLA/DC-Nyquist handling, see
  its own comments in `kokoro/custom_stft.py`); its audio diverges from the
  reference path (time-domain corr 0.93 on a test sentence), so it is NOT a
  free drop-in switch — chapters synthesized with it would sound slightly
  different from reference-path chapters. A correct fix would be an
  XPU-native istft with proper window-sum normalization.
- **Fast GPU synthesis switch (added 2026-10-08)**: the user A/B-listened to
  both paths (compare_stft_*.wav, since deleted) and heard no difference, so
  the GUI now exposes the choice: a switch in the Voice card persists
  `fast_stft` in the manifest (default **False** = reference TorchSTFT
  path). `KokoroSynthesizer.fast_stft` mirrors it; `load_pipeline()` now
  builds `KModel(repo_id, disable_complex=fast_stft)` itself and reloads when
  the flag changes. `warm_up()` warms both decoder paths on XPU so toggling
  mid-session doesn't pay a load/JIT penalty. Enabling shows a warning
  notification about switching mid-book. (See the per-shape bullet below:
  at real-book scale this switch only helps ~20%, because the per-chunk
  shape cost dominates.)
- **CORRECTED AGAIN 2026-10-08 (late): per-shape penalty hits BOTH paths;
  cross-warming invalidated the "no penalty on fast path" claim.** The
  per-shape cost lives in the conv stack shared by both STFT paths, so
  benchmarking the fast path after the default path in the SAME process
  hides its penalty (the default run already paid it). Clean, one-path-per-
  process measurements (`.eval()` fixed):
  - torch 2.7.1+xpu: ~1.9-2.4s per NEW chunk length (small chunks), warm
    0.25-0.39s (~27x); whole ch. 2 (88s, 5 chunks) ~13s cold / ~3.2s warm.
  - torch 2.13.0+xpu: ~1.9-2.2s per new small-chunk length (~2.7s per chunk
    on whole chapters), warm 0.10-0.14s (**76-79x**); whole ch. 2 ~14.5s
    cold / ~1.2s warm.
  - The penalty is exact-length granular, per-process (does not survive
    restart even for the same text; `SYCL_CACHE_PERSISTENT` does not cover
    it), and every sentence group in a real book is a new length.
  - **Real-book effective speed is therefore ~9x on EITHER torch version**
    for ~26s chunks (wall ≈ penalty×n_chunks + audio/warm_x). torch 2.13
    wins for longer chunks, repeated text, previews, and chapter re-dos;
    it is a wash at this book's chunk size. The real remaining lever is
    eliminating the per-shape cost itself (profile a cold call; possible
    oneDNN knobs or shape bucketing) - not attempted yet.
  1. A `.eval()` bug: when `load_pipeline()` started building `KModel`
     itself (for the fast_stft switch), it dropped the `.eval()` that
     `KPipeline` applies to models it creates. Until fixed the same
     evening, the model ran in *training* mode — slower LSTM kernels on
     2.7.1, and a hard crash on torch 2.13 ("oneDNN LSTM on XPU only
     supports inference mode"). Fixed in `load_pipeline()`. All
     benchmarks run between the fast_stft change and the fix are suspect.
  2. Measurement-pollution lessons: repeated-sentence texts keep shapes
     warm and overstate speed, and running both STFT paths in one process
     cross-warms them (see above). NOTE: an earlier claim that "two Kokoro
     processes share the GPU at half speed each" is DISPROVEN on the fixed
     stack — see the parallel-scaling bullet below.
- **Parallel chapter workers scale near-linearly (measured 2026-10-08
  night, torch 2.13, fast path, speed 0.85, real chapters, 16-core CPU)**:
  synthesizing different chapters in N independent processes
  simultaneously gives ~N x combined throughput, and each individual
  chapter even runs FASTER than single-process (the per-shape building is
  single-threaded per process; multiple processes use more cores while the
  GPU interleaves their short forward passes):
    1 worker: ch6+ch7 sequential: 6.4x combined
    2 workers: 11.8x (both chapters at near-unchanged per-chapter wall)
    3 workers: 19.7x
    4 workers: 29.1x (ch6 dropped 99.7s -> 75.1s wall)
  VRAM: <2 GB per model instance, 4 instances = ~8 GB of 32 GB. Worker
  pool of 3-4 chapter-level workers is the obvious app feature; chapters
  are the natural parallel unit (independent WAVs, per-chapter manifest
  updates, resume stays trivial). Use separate PROCESSES (proven);
  in-process threads sharing one pipeline are unproven and likely unsafe.
- **Multicore trial copy (2026-10-08 night)**: the parallel-workers
  feature is implemented and end-to-end tested in
  `D:\git\Audiobook_TTS_multicore` (a code copy sharing this project's
  venv-xpu, own books/, port 8088): `synth_worker_proc.py` subprocess
  worker with a file-based job protocol, `generate_worker_parallel()` in
  its app.py, `n_workers` (1-4, default 3) in the manifest + GUI select.
  Verified: 3 workers on real chapters (correct durations, M4B OK) and
  mid-run stop (no partial WAV, chapter stays pending). VALIDATED IN REAL
  USE (same night): full book (58 chapters, 12.6 h audio) at 7 workers -
  42 chapters in 16 min (~34x effective vs ~9x single-worker); 7 is the
  optimum on this machine (GPU memory mostly saturated; more risks WDDM
  VRAM spill). Ported back into this repo in commit 883a9ed:
  `generate_worker_parallel()` + `synth_worker_proc.py`, `n_workers`
  1-10 (default 3) in the manifest + GUI select.
- **torch 2.13.0+xpu is the main stack since 2026-10-08 (evening).**
  `setup-xpu.ps1` now installs `torch==2.13.0` from the XPU wheel index —
  the wheel declares its own Intel runtime deps (mkl 2026.x,
  intel-openmp 2026.x, tbb, umf, tcmlib, triton-xpu 3.7.x; the package
  renamed from pytorch-triton-xpu), so the old manually pinned Intel SYCL
  step is gone. torchaudio/torchvision are not installed (nothing imports
  them). Qualified end-to-end via `qualify_stack.py` (real book text →
  fast-path synthesis at speed 0.85 → WAVs → chaptered M4B → ffprobe
  verification: PASSED). Re-run `qualify_stack.py` after any future pin
  bump. Kept in the tree: `qualify_stack.py`. The other throwaway
  experiment artifacts (`setup-torchnew.ps1` / `run-torchnew.ps1`
  fresh-venv harness, `bench_shape_cost.py`, `bench_bert_share.py`)
  were removed from the tree on 2026-10-13 and live in git history.
  The experiment venv `venv-torchnew` was deleted after the upgrade.
- **Defaults changed 2026-10-08 (user decisions)**: `fast_stft` now defaults
  to **True** everywhere (manifest dataclass, `create_book`, synthesizer,
  GUI switch; legacy manifests missing the key also load as True) — the
  user A/B-listened and heard no quality difference. Default narration
  speed is **0.85** (`DEFAULT_SPEED` in app.py; per-project speed still
  persisted in the manifest, existing projects keep their stored 1.0).
  Note: their Paladin's Faith project's finished chapters were generated
  at speed 1.0; new chapters at 0.85 will read slower than the existing
  ones - regenerate (redo selected chapters) if uniform pacing matters.
- **Flash Attention / xpu-kernels skill: NOT applicable to Kokoro (investigated
  2026-10-12).** The HF blog "intel-xpu-kernels-skill" (Xe-Forge) reports up to
  13.3x on Flash Attention forward on this exact GPU (Arc Pro B70) — but at
  sequence lengths 2K-16K, batched, where the stock XPU Triton FA kernel
  collapses. Kokoro's forward contains exactly one attention-bearing
  component: the PL-BERT (CustomAlbert, 12 layers, hidden 768, 12 heads,
  max_position_embeddings 512 — so S ≤ 512, batch 1), which transformers 5.19
  already routes through SDPA. Everything else is LSTM + weight-normed conv
  stacks (TextEncoder, ProsodyPredictor, iSTFTNet decoder). Measured (fast
  path, fp32, 510-token synthetic chunk = the maximum possible, ~14 s audio):
  warm full forward ~465 ms; BERT alone ~21-35 ms (~7 % at the extreme, less
  on typical chunks); attention is only ~20 % of the BERT's FLOPs at S=510
  (MLP dominates) → **a hypothetically free attention kernel saves ~1-2 % of
  warm forward time and 0 % of the per-shape cold penalty** (which lives in
  the conv stack, see above). Verified with `bench_bert_share.py` (BERT-stub
  delta measurement; note `KModel.forward` takes a phoneme STRING — passing a
  token tensor silently produces a 2-token degenerate input, use
  `forward_with_tokens` for synthetic ids). The xpu-kernels skill itself could
  in principle be pointed at Kokoro's conv/LSTM ops instead of attention, but
  it would not touch the per-shape JIT penalty (a compile cost, not a slow
  kernel) — parallel workers remain the real lever.
- **External leads for the per-shape penalty (web research 2026-10-12, NOT
  yet verified locally).** The per-shape cost is oneDNN JIT primitive
  creation (~300-400 ms per GPU primitive per oneDNN's own numbers, PR
  oneapi-src/oneDNN#1105); their **persistent cache** (blob to disk, ~2 ms
  reload) is the direct cure, but PyTorch XPU does not expose it — on
  Windows oneDNN is embedded in the XPU runtime with no supported runtime
  interface (pytorch#196231). No env-var fix exists on our stack. Two
  workable-looking levers, in order:
  1. `torch.compile(mode="max-autotune", dynamic=True)` on XPU is supported
     on Windows since torch 2.7. Precedent on THIS exact GPU + TTS:
     klopstack/xpu_qwen3_tts (B70, inductor max-autotune dynamic=True on a
     transformer talker + XPUGraph capture of the code predictor → 2.4x
     realtime; ~16 min autotune cold start amortized by persisting
     TORCHINDUCTOR_CACHE_DIR). Caveats for Kokoro: LSTMs /
     pack_padded_sequence cause dynamo graph breaks, so coverage would be
     partial (conv stacks compile, LSTM sections stay eager); dynamic=True
     is what would neutralize the per-length penalty for compiled regions.
  2. XPUGraph capture (torch.xpu.XPUGraph since 2.11; torch.accelerator.Graph
     in 2.12) removes per-launch overhead — our forward does ~2,100 kernel
     launches plus ~525 aten::item D2H syncs (profiler run 2026-10-12).
     BUT: capture is static-shape (per-shape capture still pays the oneDNN
     JIT once), the .item()s break capture (pack_padded_sequence sorts on
     CPU, repeat_interleave), and XPU capture has open correctness bugs in
     vLLM (vllm#54785, vllm#48946). `mode="reduce-overhead"` silently
     no-ops on XPU even on 2.16 nightly (pytorch#200478); cudagraph-trees
     enablement for XPU is in flight (pytorch PR #200279). Watch for a
     future torch where reduce-overhead works on XPU.
  Next experiment if pursued: recreate the per-shape benchmark
  (`bench_shape_cost.py`, removed 2026-10-13 - recover from git history)
  with a compiled variant (one path per process) and measure the
  new-shape penalty.
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

- **Corrupted config files (fixed 2026-10-08):** `requirements-xpu.txt` had
  been overwritten with a copy of `run-xpu.ps1`, and `setup-xpu.ps1` contained
  the requirements list — the two had effectively swapped contents, so
  `pip install -r requirements-xpu.txt` failed. Both were restored;
  `requirements-xpu.txt` now pins the exact versions verified in the working
  venv (kokoro 0.9.4, ebooklib 0.20, beautifulsoup4 4.15.0, lxml 6.1.3,
  soundfile 0.14.0, numpy 2.2.6, nicegui 3.18.0). The spaCy model is
  downloaded unpinned (`python -m spacy download en_core_web_sm` — a hard
  `==3.8.0` pin is rejected by spaCy 3.8.16's downloader). setuptools must be
  >=70 in the venv
  (CVE-2024-6345); 65.5.0 shipped with Python 3.10.11 and was upgraded in
  place on 2026-10-08. Rebuilding the venv with `setup-xpu.ps1` preserves
  `books/` (projects, manifests, chapter audio) — only `venv-xpu` is deleted.
- EPUB chapter detection is book-specific and messy: some books put all content
  in one XHTML file, some have per-chapter files, TOC lives in NCX and/or
  nav.xhtml. Detect structure per book and let the user confirm the chapter map
  before synthesizing.
- **Project Gutenberg EPUBs** (verified 2026-10-08 with pg23910 封神演義): the
  NCX contains only 2 nav points (title page + license), so TOC-based
  extraction yields 2 boilerplate chapters and misses every real content file.
  `_extract_chapters` now discards a TOC that maps to fewer than half of the
  spine's HTML documents and falls back to the spine; spine documents whose
  extracted text is empty (covers, pages that are entirely `pg-boilerplate`)
  are skipped. The ebooklib constant is `ebooklib.ITEM_DOCUMENT` (top-level) —
  `epub.ITEM_DOCUMENT` does not exist (latent AttributeError that only fired
  once the spine fallback ran).
- **CJK word counts**: `len(text.split())` counts ~0 words for Chinese text
  (no whitespace); `_count_words` counts CJK characters as words.
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
