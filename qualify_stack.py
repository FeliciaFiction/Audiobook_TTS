#!/usr/bin/env python3
"""Full-stack qualification test: XPU synthesis -> WAV -> chaptered M4B.

Run this after any pin bump or stack change (see requirements-xpu.txt:
"Do not bump pins without a full generate + M4B test"). It exercises the
same code paths the app uses:

  1. KokoroSynthesizer.synthesize_to_wav() on XPU with the project's
     default settings (fast_stft=True, voice bf_isabella, speed 0.85)
  2. m4b_assembler.build_m4b() (ffmpeg concat + AAC + FFMETADATA chapters)
  3. m4b_assembler.verify_m4b() (ffprobe chapter/duration validation)

Uses the first stored chapter text from books/ (falling back to a fixed
paragraph), writes everything into a temp directory, and prints a clear
PASS/FAIL summary. Exits non-zero on failure.

Usage:  python qualify_stack.py
"""

import glob
import shutil
import sys
import tempfile
import time
from pathlib import Path

from kokoro_synth import KokoroSynthesizer
from m4b_assembler import build_m4b, verify_m4b

CHAPTER_COUNT = 3
VOICE = "bf_isabella"
SPEED = 0.85
FALLBACK_TEXT = ("The harbour lights blinked twice and then went out altogether, "
                 "and the town decided that the night had gone on long enough. "
                 "She counted the coins again and put them away.")


def find_texts():
    """Real stored chapter texts when available, else the fixed paragraph."""
    for path in sorted(glob.glob("books/*/text/*.txt"))[:5]:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if 50 < len(text.split()) < 250:
            yield text
    while True:
        yield FALLBACK_TEXT


def main() -> int:
    print("=== Audiobook TTS stack qualification ===")
    synthesizer = KokoroSynthesizer(device="xpu", lang_code="b")
    synthesizer.fast_stft = True
    print("fast_stft=True, voice=" + VOICE + f", speed={SPEED}")

    tmp = Path(tempfile.mkdtemp(prefix="audiobook_qualify_"))
    wav_paths, titles = [], []
    failures = []

    try:
        texts = find_texts()
        for i in range(CHAPTER_COUNT):
            text = next(texts)
            wav = tmp / f"{i + 1:03d}.wav"
            t0 = time.perf_counter()
            synthesizer.synthesize_to_wav(
                text, str(wav), voice=VOICE, speed=SPEED, lang_code="b")
            wall = time.perf_counter() - t0
            import soundfile as sf
            audio, sr = sf.read(str(wav))
            duration = len(audio) / sr
            print(f"chapter {i + 1}: {duration:5.1f}s audio, {wall:5.2f}s wall "
                  f"({duration / wall:4.1f}x realtime)")
            if duration < 3.0:
                failures.append(f"chapter {i + 1} suspiciously short ({duration:.1f}s)")
            wav_paths.append(str(wav))
            titles.append(f"Qualification Chapter {i + 1}")

        m4b_path = str(tmp / "qualification.m4b")
        print("building M4B...")
        build_m4b(wav_paths, m4b_path, "Stack Qualification", "Audiobook TTS", titles)

        result = verify_m4b(m4b_path)
        print(f"verify_m4b: {result}")
        if not result.get("valid"):
            failures.append(f"M4B verification failed: {result.get('error', 'unknown')}")
        else:
            chapters = result.get("chapters", [])
            if len(chapters) != CHAPTER_COUNT:
                failures.append(f"expected {CHAPTER_COUNT} chapter markers, got {len(chapters)}")
            for ch in chapters:
                if not ch.get("title"):
                    failures.append(f"chapter marker without title: {ch}")

        if failures:
            print("\n=== QUALIFICATION FAILED ===")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("\n=== QUALIFICATION PASSED ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
