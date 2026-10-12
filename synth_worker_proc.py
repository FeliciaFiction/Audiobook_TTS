#!/usr/bin/env python3
"""Subprocess worker for parallel chapter synthesis (multicore trial).

The GUI app (parent) never synthesizes in parallel in-process: the Kokoro
pipeline is not thread-safe and each worker needs its own XPU model
instance. Instead the parent spawns N of these subprocess workers.

File-based job protocol (crash-safe, no IPC machinery):

- Parent writes one job file per chapter:  <jobs>/<NNN>.job
- A worker claims a job atomically:       rename <NNN>.job -> <NNN>.claim
  (os.rename fails for the second claimer on Windows - no double work)
- On completion the worker writes         <NNN>.result (JSON)
  and deletes the claim file.
- Parent touches                          <jobs>/stop
  -> workers abort the current chapter between chunks and exit; the
  chapter stays pending (its WAV is only ever written atomically at the
  end, so an aborted chapter leaves no partial file).

Worker responsibilities: read the chapter text, synthesize with the
manifest's voice/speed/lang/fast mode, append inter-chapter silence when
needed, write the chapter WAV atomically, report duration via the result
file. The parent owns manifest.json - workers never write it.

Measured (B70, torch 2.13, fast path): N workers ~= N x combined
throughput; the per-shape building is single-threaded per process and the
GPU interleaves the (short) forward passes. See AGENTS.md.
"""

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from kokoro_synth import KokoroSynthesizer

SAMPLE_RATE = 24000
INTERCHAPTER_SILENCE_S = 0.75


def atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def find_job(jobs_dir: Path):
    """Claim the next pending job, or None if none are left."""
    for job in sorted(jobs_dir.glob("*.job")):
        claim = job.with_suffix(".claim")
        try:
            os.rename(job, claim)
            return claim
        except OSError:
            continue  # another worker claimed it first
    return None


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: synth_worker_proc.py <jobs_dir> <manifest.json>", file=sys.stderr)
        return 2
    jobs_dir = Path(sys.argv[1])
    manifest_path = Path(sys.argv[2])

    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)

    stop_file = jobs_dir / "stop"
    lang = m.get("kokoro_lang_code", "b")
    voice = m.get("voice", "af_heart")
    speed = float(m.get("speed", 1.0))
    n_chapters = len(m.get("chapters", []))

    synth = KokoroSynthesizer(device=os.environ.get("KOKORO_DEVICE", "xpu"))
    synth.fast_stft = bool(m.get("fast_stft", True))

    # Load the model and warm the kernels BEFORE claiming jobs so a
    # slow-starting worker does not hold a chapter hostage.
    try:
        for _ in synth.synthesize_chapter(
                "Warm up now, please.", voice=voice, speed=speed, lang_code=lang):
            pass
    except Exception as e:
        print(f"worker warm-up failed: {e}", file=sys.stderr, flush=True)

    processed = 0
    while True:
        if stop_file.exists():
            break
        claim = find_job(jobs_dir)
        if claim is None:
            break
        idx = int(claim.stem)
        result = {"index": idx, "status": "error", "duration_ms": 0, "error": None}
        t0 = time.perf_counter()
        try:
            text_path = Path(m["text_dir"]) / f"{idx + 1:03d}.txt"
            text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
            if not text or not text.strip():
                raise RuntimeError("No stored chapter text")

            wav_path = str(Path(m["chapters_dir"]) / m["chapters"][idx]["file"])
            audio_parts, chunks, stopped = [], 0, False
            for a in synth.synthesize_chapter(
                    text, voice=voice, speed=speed, lang_code=lang):
                if stop_file.exists():
                    stopped = True
                    break
                audio_parts.append(a)
                chunks += 1

            if stopped:
                result = {"index": idx, "status": "stopped",
                          "duration_ms": 0, "error": None}
            elif not audio_parts:
                result = {"index": idx, "status": "error", "duration_ms": 0,
                          "error": "No audio generated"}
            else:
                audio = np.concatenate(audio_parts).astype(np.float32)
                if idx < n_chapters - 1:
                    silence = np.zeros(
                        int(INTERCHAPTER_SILENCE_S * SAMPLE_RATE), dtype=np.float32)
                    audio = np.concatenate([audio, silence])
                buffer = io.BytesIO()
                sf.write(buffer, audio, SAMPLE_RATE, format="WAV")
                atomic_write_bytes(wav_path, buffer.getvalue())
                wall = time.perf_counter() - t0
                result = {"index": idx, "status": "done",
                          "duration_ms": int(len(audio) / SAMPLE_RATE * 1000),
                          "chunks": chunks, "wall": round(wall, 1)}
        except Exception as e:
            result = {"index": idx, "status": "error", "duration_ms": 0,
                      "error": str(e)}

        (jobs_dir / f"{idx:03d}.result").write_text(
            json.dumps(result), encoding="utf-8")
        try:
            claim.unlink()
        except OSError:
            pass
        processed += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
