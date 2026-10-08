#!/usr/bin/env python3
"""
Audiobook TTS - EPUB to M4B Converter

A local GUI application that converts EPUB e-books into M4B audiobooks
with embedded chapter markers, using Kokoro-82M on Intel XPU.

Usage:
    .\\run-xpu.ps1          (recommended, enables XPU)
    python app.py          (direct, CPU or XPU auto-detected)
"""

import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import soundfile as sf

from nicegui import app, events, run, ui

from epub_parser import parse_epub, BookMetadata, ChapterInfo
from kokoro_synth import KokoroSynthesizer
from m4b_assembler import M4BAssembler, build_m4b, verify_m4b
from state_manager import StateManager, BookManifest


# =============================================================================
# Configuration
# =============================================================================

BOOKS_DIR = "books"
SAMPLE_RATE = 24000
DEFAULT_SPEED = 1.0
DEFAULT_VOICE = "bf_isabella"  # Isabella, British English
INTERCHAPTER_SILENCE_S = 0.75

# Kokoro language codes by ISO 639-1
ISO_TO_KOKORO = {
    'en': ['a', 'b'],
    'es': ['e'],
    'fr': ['f'],
    'hi': ['h'],
    'it': ['i'],
    'pt': ['p'],
    'ja': ['j'],
    'zh': ['z'],
}

KOKORO_LANG_NAMES = {
    'a': 'American English',
    'b': 'British English',
    'e': 'Spanish',
    'f': 'French',
    'h': 'Hindi',
    'i': 'Italian',
    'j': 'Japanese',
    'p': 'Portuguese',
    'z': 'Mandarin Chinese',
}


# =============================================================================
# Components
# =============================================================================

# Device: XPU if requested and available, otherwise CPU.
REQUESTED_DEVICE = os.environ.get("KOKORO_DEVICE", "xpu")
synthesizer = KokoroSynthesizer(device=REQUESTED_DEVICE)

assembler = M4BAssembler()
sm = StateManager(BOOKS_DIR)
Path(BOOKS_DIR).mkdir(parents=True, exist_ok=True)

# Voice list (enumerated from the local voice pack; does not load the model)
ALL_VOICES = synthesizer.get_available_voices()

# Preview audio served to the browser
PREVIEW_DIR = Path(BOOKS_DIR) / "_previews"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_WAV = PREVIEW_DIR / "preview.wav"
if not PREVIEW_WAV.exists():
    sf.write(PREVIEW_WAV, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
PREVIEW_URL = app.add_media_file(local_file=str(PREVIEW_WAV), url_path='/voice_preview.wav')


# =============================================================================
# Session state (single-user local app)
# =============================================================================

class Session:
    def __init__(self):
        self.manifest: Optional[BookManifest] = None
        self.chapter_texts: Dict[int, str] = {}
        self.selected: Set[int] = set()
        self.generating = False
        self.stop_event = threading.Event()
        self.progress = {
            "chapter": 0, "chapters": 0,
            "chunk": 0, "chunks": 0,
            "message": "", "m4b_message": "",
        }
        self.log_lines: List[str] = []
        self._log_shown = 0

    def reset(self):
        self.manifest = None
        self.chapter_texts = {}
        self.selected = set()


S = Session()


# =============================================================================
# Helpers
# =============================================================================

def iso_to_kokoro(iso: str) -> str:
    codes = ISO_TO_KOKORO.get((iso or "en").lower(), ['a'])
    return codes[0]


def voice_options_for(kokoro_codes: List[str]) -> Dict[str, str]:
    """Map of voice name -> display label, for the languages given."""
    voices = [v for v in ALL_VOICES if v.lang_code in kokoro_codes]
    return {v.name: v.display_name for v in voices}


def fmt_duration(ms: int) -> str:
    seconds = ms // 1000
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def save_manifest() -> None:
    if S.manifest:
        sm._save_manifest(S.manifest)


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    S.log_lines.append(f"[{timestamp}] {message}")


# =============================================================================
# Generation worker (runs in a thread via run.io_bound)
# =============================================================================

def generate_worker(indices: List[int]) -> str:
    m = S.manifest
    total = len(indices)

    for n, idx in enumerate(indices):
        if S.stop_event.is_set():
            log(f"Stopped by user at chapter {idx + 1}")
            return "stopped"

        chapter = m.chapters[idx]
        S.progress.update(chapter=n + 1, chapters=total, chunk=0, chunks=0,
                           message=f"Chapter {idx + 1}: {chapter.title}")
        log(f"Generating chapter {idx + 1}: {chapter.title}")

        # Chapter text: in-memory (fresh EPUB) or persisted (resumed project)
        text = S.chapter_texts.get(idx) or sm.load_chapter_text(m, idx)
        if not text or not text.strip():
            sm.update_chapter_status(m, idx, "error", 0, "No stored chapter text")
            log(f"Chapter {idx + 1}: no stored text - open the book's EPUB in step 1 to enable generation")
            continue

        # Redo: drop any previous audio for a chapter selected for generation
        if chapter.status == "done":
            sm.reset_chapter(m, idx)

        # One pipeline call per chapter; KPipeline yields per-sentence chunks.
        # Estimated chunk count is only for the progress bar.
        est_chunks = max(1, len(re.findall(r'[.!?…]+', text)))
        S.progress.update(chunk=0, chunks=est_chunks)

        audio_parts = []
        chunk_count = 0
        t_start = time.perf_counter()

        try:
            for audio in synthesizer.synthesize_chapter(
                text, voice=m.voice, speed=m.speed, lang_code=m.kokoro_lang_code,
            ):
                if S.stop_event.is_set():
                    log(f"Stopped by user at chapter {idx + 1}, chunk {chunk_count + 1}")
                    return "stopped"
                audio_parts.append(audio)
                chunk_count += 1
                S.progress.update(chunk=chunk_count)
        except Exception as e:
            sm.update_chapter_status(m, idx, "error", 0, str(e))
            log(f"Chapter {idx + 1} failed: {e}")
            continue

        if not audio_parts:
            sm.update_chapter_status(m, idx, "error", 0, "No audio generated")
            continue

        final_audio = np.concatenate(audio_parts)

        # Short silence between chapters (not after the final one)
        if idx < len(m.chapters) - 1:
            silence = synthesizer.add_silence(INTERCHAPTER_SILENCE_S)
            final_audio = np.concatenate([final_audio, silence])

        import io
        buffer = io.BytesIO()
        sf.write(buffer, final_audio, SAMPLE_RATE, format='WAV')
        sm.atomic_write_wav(m, idx, buffer.getvalue())

        duration_ms = int(len(final_audio) / SAMPLE_RATE * 1000)
        sm.update_chapter_status(m, idx, "done", duration_ms)

        wall = time.perf_counter() - t_start
        audio_s = len(final_audio) / SAMPLE_RATE
        realtime = audio_s / wall if wall > 0 else 0
        log(f"Chapter {idx + 1} done ({fmt_duration(duration_ms)}, {chunk_count} chunks) "
            f"in {wall:.1f}s ({realtime:.0f}x realtime)")

    return "complete"


def build_m4b_now() -> str:
    m = S.manifest
    wav_paths, titles = [], []
    for ch in m.chapters:
        if ch.status == "done":
            path = sm.get_chapter_path(m, ch.index)
            if os.path.exists(path):
                wav_paths.append(path)
                titles.append(ch.title)

    if not wav_paths:
        return "No completed chapters yet - generate at least one chapter first."

    try:
        build_m4b(wav_paths, m.m4b_path, m.title, m.author, titles)
        verification = verify_m4b(m.m4b_path)
        if verification.get("valid"):
            n = len(verification.get("chapters", []))
            return f"Audiobook built: {m.m4b_path} ({n} chapter markers)"
        return f"Built but verification failed: {verification.get('error', 'unknown')}"
    except Exception as e:
        return f"Error building M4B: {e}"


# =============================================================================
# UI
# =============================================================================

@ui.page('/')
def main_page():
    # ------------------------------------------------------------------ state
    chapter_container = None
    audio_container = None
    voice_select = None
    language_select = None
    speed_slider = None
    output_input = None
    generate_btn = None
    stop_btn = None
    build_btn = None
    progress_bar = None
    progress_label = None

    # ------------------------------------------------------------- refreshables
    @ui.refreshable
    def render_projects():
        books = sm.list_books()
        if not books:
            ui.label('No projects yet - open an EPUB to start one.').classes('text-gray-500')
            return
        for b in books:
            with ui.card().classes('w-full cursor-pointer no-shadow border').on(
                'click', lambda _, bid=b['book_id']: on_load_project(bid)
            ):
                ui.label(b['title'] or b['book_id']).classes('font-semibold')
                with ui.row().classes('items-center gap-2 text-sm text-gray-500'):
                    ui.label(f"{b['done_chapters']}/{b['total_chapters']} chapters")
                    ui.separator().props('vertical')
                    ui.label(b.get('voice', ''))

    @ui.refreshable
    def render_chapters():
        m = S.manifest
        if m is None:
            ui.label('No book loaded.').classes('text-gray-500')
            return
        for ch in m.chapters:
            with ui.row().classes('w-full items-center gap-2'):
                cb = ui.checkbox(
                    f"{ch.index + 1}. {ch.title}",
                    value=ch.index in S.selected,
                    on_change=lambda e, idx=ch.index: on_select_chapter(idx, e.value),
                ).classes('flex-grow')
                if ch.status == 'done':
                    badge = ui.badge('done', color='green')
                    ui.label(fmt_duration(ch.duration_ms)).classes('text-xs text-gray-500')
                elif ch.status == 'error':
                    ui.badge('error', color='red').tooltip(ch.error or '')
                else:
                    ui.badge('pending', color='grey')

    # -------------------------------------------------------------- UI actions
    def on_select_chapter(idx: int, checked: bool):
        if checked:
            S.selected.add(idx)
        else:
            S.selected.discard(idx)

    def on_select_all():
        m = S.manifest
        if m:
            S.selected = {ch.index for ch in m.chapters}
            render_chapters.refresh()

    def on_select_none():
        S.selected = set()
        render_chapters.refresh()

    def on_select_pending():
        m = S.manifest
        if m:
            S.selected = {ch.index for ch in m.chapters if ch.status != 'done'}
            render_chapters.refresh()

    def on_language_change(e):
        m = S.manifest
        if m is None:
            return
        m.kokoro_lang_code = e.value
        iso = next((iso for iso, codes in ISO_TO_KOKORO.items() if e.value in codes), 'en')
        m.language = iso
        options = voice_options_for(ISO_TO_KOKORO.get(iso, ['a']))
        voice_select.set_options(options)
        if m.voice not in options:
            m.voice = next(iter(options), DEFAULT_VOICE)
            voice_select.set_value(m.voice)
        save_manifest()
        ui.notify('Saved', type='positive', position='bottom-right')

    def on_voice_change(e):
        if S.manifest:
            S.manifest.voice = e.value
            save_manifest()
            ui.notify('Saved', type='positive', position='bottom-right')

    def on_speed_change(e):
        if S.manifest:
            S.manifest.speed = e.value
            save_manifest()

    def on_output_change(e):
        m = S.manifest
        if m and e.value:
            try:
                sm.set_output_folder(m, e.value)
                ui.notify(f"Audiobook will be written to {m.m4b_path}",
                          type='positive', position='bottom-right')
            except Exception as ex:
                ui.notify(f"Could not use that folder: {ex}", type='negative')

    async def on_upload(e: events.UploadEventArguments):
        try:
            epub_path = Path(BOOKS_DIR) / "_incoming" / e.file.name
            epub_path.parent.mkdir(parents=True, exist_ok=True)
            await e.file.save(epub_path)
            metadata, chapters = await run.io_bound(parse_epub, str(epub_path))
        except Exception as ex:
            ui.notify(f"Could not read EPUB: {ex}", type='negative')
            return

        lang_code = iso_to_kokoro(metadata.language)
        # Prefer the default voice's language variant when the book language
        # supports it (e.g. bf_isabella -> British G2P for English books)
        voice_lang = DEFAULT_VOICE[0]
        if voice_lang in ISO_TO_KOKORO.get(metadata.language.lower(), []):
            lang_code = voice_lang
        m = sm.create_book(
            title=metadata.title,
            author=metadata.author,
            language=metadata.language,
            kokoro_lang_code=lang_code,
            voice=DEFAULT_VOICE,
            speed=DEFAULT_SPEED,
            epub_path=str(epub_path),
        )
        sm.add_chapters(m, [
            {"title": c.title, "word_count": c.word_count} for c in chapters
        ])
        for c in chapters:
            if c.content:
                sm.save_chapter_text(m, c.index, c.content)

        load_project_into_ui(m)
        S.chapter_texts = {c.index: c.content for c in chapters}
        ui.notify(f"Project created: {m.title} ({len(chapters)} chapters)",
                  type='positive')

    async def on_load_project(book_id: str):
        m = sm.load_manifest(book_id)
        if not m:
            ui.notify(f"Project {book_id} not found", type='negative')
            return
        S.chapter_texts = {}

        # Re-extract chapter text from the source EPUB when available:
        # self-healing for parser fixes and for legacy projects without text/
        refreshed = False
        if m.epub_path and os.path.exists(m.epub_path):
            try:
                _, chapters = await run.io_bound(parse_epub, m.epub_path)
                if len(chapters) == len(m.chapters):
                    for c in chapters:
                        if c.content:
                            sm.save_chapter_text(m, c.index, c.content)
                    S.chapter_texts = {c.index: c.content for c in chapters if c.content}
                    refreshed = True
                    ui.notify("Chapter text refreshed from the EPUB", type='positive')
                else:
                    ui.notify(f"EPUB has {len(chapters)} chapters but the project has "
                              f"{len(m.chapters)} - keeping stored text", type='warning')
            except Exception as ex:
                ui.notify(f"Could not re-read the EPUB ({m.epub_path}): {ex}",
                          type='warning')

        has_text = any(sm.load_chapter_text(m, ch.index) for ch in m.chapters)
        if m.chapters and not has_text and not refreshed:
            ui.notify("This project has no stored chapter text. Open its EPUB "
                      "in step 1 to continue generating.", type='warning')

        load_project_into_ui(m)
        ui.notify(f"Loaded project: {m.title}", type='positive')

    def load_project_into_ui(m: BookManifest):
        S.reset()
        S.manifest = m
        # Pre-select chapters that still need work
        S.selected = {ch.index for ch in m.chapters if ch.status != 'done'}

        book_title.set_content(f"**{m.title}** — {m.author or 'unknown author'}")
        has_text = bool(S.chapter_texts) or any(
            sm.load_chapter_text(m, ch.index) for ch in m.chapters
        ) if m.chapters else False
        meta = (
            f"Language: {KOKORO_LANG_NAMES.get(m.kokoro_lang_code, m.kokoro_lang_code)} "
            f"&nbsp;·&nbsp; {len(m.chapters)} chapters &nbsp;·&nbsp; "
            f"{sum(1 for c in m.chapters if c.status == 'done')} done"
        )
        if not has_text:
            meta += (" &nbsp;·&nbsp; **no stored chapter text — open the EPUB "
                     "in step 1 to enable generation**")
        book_meta.set_content(meta)

        language_select.set_value(m.kokoro_lang_code)
        options = voice_options_for(ISO_TO_KOKORO.get(m.language, [m.kokoro_lang_code]))
        if m.voice not in options:
            options[m.voice] = f"{m.voice} (custom)"
        voice_select.set_options(options)
        voice_select.set_value(m.voice)
        speed_slider.set_value(m.speed)
        output_input.set_value(str(Path(m.m4b_path).parent))

        render_chapters.refresh()

    async def on_preview():
        m = S.manifest
        voice = voice_select.value or DEFAULT_VOICE
        text = preview_text.value or "The quick brown fox jumps over the lazy dog."
        preview_btn.set_text('Generating...')
        try:
            audio = await run.io_bound(synthesizer.preview_voice, voice, text, 1.0)
        except Exception as ex:
            ui.notify(f"Preview failed: {ex}", type='negative')
            return
        finally:
            preview_btn.set_text('Preview voice')
        if audio is None or len(audio) == 0:
            ui.notify("Preview produced no audio", type='negative')
            return
        sf.write(PREVIEW_WAV, audio, SAMPLE_RATE)
        with audio_container:
            audio_container.clear()
            ui.audio(src=f"{PREVIEW_URL}?t={time.time()}", controls=True).classes('w-full')

    async def on_generate():
        if S.generating:
            return
        if S.manifest is None:
            ui.notify("Open an EPUB or load a project first", type='negative')
            return
        if not S.selected:
            ui.notify("Select at least one chapter", type='negative')
            return

        indices = sorted(S.selected)
        S.stop_event.clear()
        S.generating = True
        generate_btn.disable()
        build_btn.disable()
        S.progress.update(chapter=0, chapters=len(indices), chunk=0, chunks=0,
                           message="Starting...", m4b_message="")
        log(f"Generating {len(indices)} chapter(s) with voice {S.manifest.voice}")

        try:
            result = await run.io_bound(generate_worker, indices)
            generated = any(S.manifest.chapters[i].status == "done" for i in indices)
            if result == "complete" and generated:
                log("All chapters generated, building audiobook...")
                m4b_message = await run.io_bound(build_m4b_now)
                S.progress["m4b_message"] = m4b_message
                log(m4b_message)
                ui.notify(m4b_message, type='positive')
            elif result == "complete":
                log("No chapters were generated - nothing to build")
                ui.notify("No chapters were generated - see the log for details",
                          type='negative')
            elif result == "stopped":
                ui.notify("Generation stopped", type='warning')
        except Exception as ex:
            log(f"Generation error: {ex}")
            ui.notify(f"Generation error: {ex}", type='negative')
        finally:
            S.generating = False
            generate_btn.enable()
            build_btn.enable()
            S.progress.update(message="Idle", chapter=0, chapters=0, chunk=0, chunks=0)
            render_chapters.refresh()
            render_projects.refresh()

    def on_stop():
        if S.generating:
            S.stop_event.set()
            log("Stop requested - finishing current chunk...")
            ui.notify("Stopping after the current chunk", type='warning')

    async def on_build():
        if S.manifest is None:
            ui.notify("Open a project first", type='negative')
            return
        build_btn.disable()
        try:
            message = await run.io_bound(build_m4b_now)
            S.progress["m4b_message"] = message
            log(message)
            ui.notify(message, type='positive' if not message.startswith(("No", "Error")) else 'warning')
        finally:
            build_btn.enable()

    def on_open_folder():
        folder = str(Path(S.manifest.m4b_path).parent) if S.manifest else BOOKS_DIR
        try:
            os.startfile(folder)  # Windows
        except Exception:
            ui.notify(f"Output folder: {folder}")

    def update_progress_ui():
        p = S.progress
        if p["chapters"] > 0 and p["chunks"] > 0:
            fraction = (p["chapter"] - 1 + (p["chunk"] / p["chunks"])) / p["chapters"]
            progress_bar.set_value(min(fraction, 1.0))
            progress_label.set_text(
                f"Chapter {p['chapter']}/{p['chapters']} — chunk {p['chunk']}"
                + (f" — {p['message']}" if p['message'] and p['message'] != 'Idle' else "")
            )
        elif p["message"]:
            progress_label.set_text(p["message"])

        # Append new log lines
        while S._log_shown < len(S.log_lines):
            event_log.push(S.log_lines[S._log_shown])
            S._log_shown += 1

    # ------------------------------------------------------------------ layout
    with ui.header().classes('items-center justify-between'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('menu_book', size='2rem')
            ui.label('Audiobook TTS').classes('text-xl font-semibold')
        try:
            import torch
            if REQUESTED_DEVICE == 'xpu' and torch.xpu.is_available():
                ui.badge('Intel XPU (GPU)', color='green')
            else:
                ui.badge('CPU', color='blue')
        except ImportError:
            pass

    with ui.row().classes('w-full no-wrap items-start gap-4 p-4'):

        # ---- left column: projects ----
        with ui.card().classes('w-72 shrink-0'):
            ui.label('Projects').classes('font-semibold text-lg')
            ui.button('Refresh', icon='refresh', on_click=render_projects.refresh) \
                .props('flat dense')
            render_projects()

        # ---- main column ----
        with ui.column().classes('flex-grow gap-4 min-w-0'):

            # 1. Open book
            with ui.card().classes('w-full'):
                ui.label('1 · Open a book').classes('font-semibold text-lg')
                ui.label('Pick an EPUB to start a new project, or click a project on the left to resume.'
                         ).classes('text-sm text-gray-500')
                ui.upload(on_upload=on_upload, auto_upload=True,
                          label='Choose an EPUB file') \
                    .props('accept=.epub').classes('w-full')
                book_title = ui.markdown('').classes('text-lg')
                book_meta = ui.markdown('').classes('text-sm text-gray-500')

            # 2. Chapters
            with ui.card().classes('w-full'):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('2 · Chapters').classes('font-semibold text-lg')
                    with ui.row():
                        ui.button('All', on_click=on_select_all).props('outline dense')
                        ui.button('Pending only', on_click=on_select_pending).props('outline dense')
                        ui.button('None', on_click=on_select_none).props('outline dense')
                with ui.scroll_area().classes('w-full h-56'):
                    chapter_container = ui.column().classes('w-full gap-1')
                    with chapter_container:
                        render_chapters()

            # 3. Voice
            with ui.card().classes('w-full'):
                ui.label('3 · Voice').classes('font-semibold text-lg')
                with ui.row().classes('w-full items-start gap-4'):
                    with ui.column().classes('flex-grow gap-2'):
                        language_select = ui.select(
                            {code: name for code, name in KOKORO_LANG_NAMES.items()},
                            value='b', label='Language',
                            on_change=on_language_change,
                        ).classes('w-full')
                        voice_select = ui.select(
                            voice_options_for(['b']),
                            value=DEFAULT_VOICE, label='Voice (type to filter)',
                            with_input=True, clearable=False,
                            on_change=on_voice_change,
                        ).classes('w-full')
                        speed_slider = ui.slider(
                            min=0.5, max=2.0, step=0.05, value=DEFAULT_SPEED,
                            on_change=on_speed_change,
                        ).props('label-always').classes('w-full')
                        ui.label('Speed').classes('text-xs text-gray-500')
                    with ui.column().classes('w-80 gap-2'):
                        preview_text = ui.input(
                            value='The quick brown fox jumps over the lazy dog.',
                            label='Preview text',
                        ).classes('w-full')
                        preview_btn = ui.button('Preview voice', icon='play_arrow',
                                                on_click=on_preview).classes('w-full')
                        audio_container = ui.column().classes('w-full')
                        with audio_container:
                            ui.audio(src=PREVIEW_URL, controls=True).classes('w-full')

            # 4. Output
            with ui.card().classes('w-full'):
                ui.label('4 · Output').classes('font-semibold text-lg')
                with ui.row().classes('w-full items-end gap-2'):
                    output_input = ui.input(
                        label='Audiobook folder',
                        value=str(Path(BOOKS_DIR).resolve()),
                        on_change=on_output_change,
                    ).classes('flex-grow')
                    ui.button('Open folder', icon='folder_open',
                              on_click=on_open_folder).props('outline')

            # 5. Generate
            with ui.card().classes('w-full'):
                ui.label('5 · Generate').classes('font-semibold text-lg')
                with ui.row():
                    generate_btn = ui.button('Generate selected chapters',
                                              icon='play_circle', on_click=on_generate) \
                        .props('color=primary')
                    stop_btn = ui.button('Stop', icon='stop',
                                         on_click=on_stop).props('outline color=red')
                    build_btn = ui.button('Rebuild audiobook', icon='build',
                                          on_click=on_build).props('outline')
                progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full')
                progress_label = ui.label('Idle').classes('text-sm text-gray-500')
                ui.label('Per-chapter WAV files are written as they finish; the M4B is '
                         'rebuilt from all finished chapters, so you can stop and resume anytime.'
                         ).classes('text-xs text-gray-400')
                with ui.card().classes('w-full bg-gray-100 no-shadow'):
                    event_log = ui.log(max_lines=300).classes('w-full h-40 bg-gray-100')

    # progress polling (runs on the UI thread)
    ui.timer(0.5, update_progress_ui)


# =============================================================================
# Main entry point
# =============================================================================

if __name__ in {"__main__", "__mp_main__"}:
    print(f"Starting Audiobook TTS (device requested: {REQUESTED_DEVICE})")
    print(f"Found {len(ALL_VOICES)} voices in the local voice pack")

    # Warm up the model and JIT kernels in the background so the first
    # Generate click doesn't pay the model-load + compile penalty
    threading.Thread(
        target=synthesizer.warm_up, args=("b", DEFAULT_VOICE), daemon=True,
    ).start()

    ui.run(
        host="127.0.0.1",
        port=8087,
        title="Audiobook TTS",
        show=True,
        reload=False,
    )
