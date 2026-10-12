"""Per-shape warm-up cost benchmark for Kokoro on Intel XPU.

Measures the two things that determine real-book generation speed
(see AGENTS.md, XPU knowledge base):

1. Per-shape penalty: the first synthesis of a given chunk length in a
   process costs extra (oneDNN/SYCL primitive + descriptor building,
   ~2s per chunk, exact-length granular, per-process - NOT covered by
   SYCL_CACHE_PERSISTENT, and NOT shared across processes).
2. Steady-state speed: repeat synthesis of an already-warm shape.

IMPORTANT methodology note (learned the hard way 2026-10-08): the penalty
is shared between the two STFT paths (it lives in the common conv stack),
so benchmarking one path AFTER the other in the same process cross-warms
it and hides the penalty. Run ONE path per process:

    python bench_shape_cost.py           # default TorchSTFT path
    python bench_shape_cost.py --fast     # fast CustomSTFT path

Reference numbers on this machine (Arc Pro B70, fp32, GPU otherwise idle,
bf_isabella / lang 'b', .eval()-fixed code, fast path in a FRESH process):
    torch 2.7.1+xpu: new-shape call ~1.9-2.4s, warm 0.25-0.39s (~27x);
        whole ch. 2 (88s, 5 chunks): ~13s cold / ~3.2s warm (27x)
    torch 2.13.0+xpu: new-shape call ~1.9-2.2s (small chunks) / ~2.7s per
        chunk on chapters, warm 0.10-0.14s (~76-79x); whole ch. 2:
        ~14.5s cold / ~1.2s warm
    Real books (every chunk a new length): wall ~= 2-2.7s * n_chunks +
    audio/warm_x  -> ~9x effective on either torch for ~26s chunks;
    longer chunks or repeats favor torch 2.13.
"""

import glob
import sys
import time

import torch

from kokoro_synth import KokoroSynthesizer

BASE = ("The harbour lights blinked twice and then went out altogether, "
        "and the town decided the night had gone on long enough.")
LADDER = [
    ("base, 1st call (new shape)", BASE),
    ("base, 2nd call (warm)      ", BASE),
    ("+1 word (new shape)        ", BASE + " Rain"),
    ("+1 word, repeat            ", BASE + " Rain"),
    ("+3 words (new shape)       ", BASE + " Rain fell steadily"),
    ("+3 words, repeat           ", BASE + " Rain fell steadily"),
    ("long variant (new shape)   ", BASE + " Rain fell steadily all night and the streets emptied one by one"),
    ("long variant, repeat       ", BASE + " Rain fell steadily all night and the streets emptied one by one"),
]


def find_chapter_text():
    """First stored chapter text of ~200 words (like Paladin's ch. 2)."""
    for path in sorted(glob.glob("books/*/text/002.txt")):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if 100 < len(text.split()) < 500:
            return path, text
    return "(no book text found - synthetic fallback)", BASE + " " + BASE + " " + BASE


def time_call(s, text):
    t0 = time.perf_counter()
    parts, chunks = [], 0
    for a in s.synthesize_chapter(text, voice="bf_isabella", speed=1.0, lang_code="b"):
        parts.append(a)
        chunks += 1
    wall = time.perf_counter() - t0
    total = sum(len(p) for p in parts) / 24000
    return wall, total, chunks


def main():
    fast = "--fast" in sys.argv
    print(f"torch {torch.__version__}, XPU available: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"device: {torch.xpu.get_device_name(0)}")
    print(f"path: {'fast_stft=True (CustomSTFT)' if fast else 'fast_stft=False (TorchSTFT default)'}"
          " - run the other path in a separate process to avoid cross-warming")
    chapter_path, chapter_text = find_chapter_text()
    print(f"chapter text: {chapter_path} ({len(chapter_text.split())} words)")

    s = KokoroSynthesizer(device="xpu", lang_code="b")
    s.fast_stft = fast

    # Load + warm the pipeline on a small different shape
    wall, audio, chunks = time_call(s, "Warm up now, please.")
    print(f"warm-up call: {wall:.2f}s ({audio:.1f}s audio)")

    print("-- shape ladder (per-call wall time) --")
    for name, text in LADDER:
        wall, audio, chunks = time_call(s, text)
        print(f"  {name}: {wall:6.2f}s wall, {audio:5.1f}s audio, {chunks} chunk(s)")

    print("-- whole chapter, one pipeline call --")
    wall, audio, chunks = time_call(s, chapter_text)
    print(f"  1st (new shapes): {wall:6.2f}s wall, {audio:5.1f}s audio, "
          f"{chunks} chunks -> {audio / wall:5.1f}x realtime")
    wall, audio2, _ = time_call(s, chapter_text)
    print(f"  2nd (warm)      : {wall:6.2f}s wall, {audio2:5.1f}s audio, "
          f"{chunks} chunks -> {audio2 / wall:5.1f}x realtime")

    print("\nCompare against the reference numbers in the header of this file.")


if __name__ == "__main__":
    main()
