"""
M4B Assembler for Audiobook TTS

Assembles chapter WAV files into a single M4B audiobook with chapter markers
using ffmpeg.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import soundfile as sf


@dataclass
class ChapterMarker:
    """Chapter marker for M4B."""
    title: str
    start_ms: int
    end_ms: int


class M4BAssembler:
    """
    Assembles chapter WAV files into M4B with chapter markers.
    
    Uses ffmpeg for:
    1. Concatenating WAV files
    2. Encoding to AAC
    3. Embedding chapter metadata via FFMETADATA
    """
    
    # Default bitrate for AAC encoding
    DEFAULT_BITRATE = "128k"
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        """
        Initialize assembler.
        
        Args:
            ffmpeg_path: Path to ffmpeg executable
            ffprobe_path: Path to ffprobe executable
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self._verified = False
        self._verify_ffmpeg()
    
    def _verify_ffmpeg(self) -> None:
        """Verify ffmpeg and ffprobe are available."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise FileNotFoundError(f"ffmpeg not found at {self.ffmpeg_path}")
        except Exception as e:
            raise RuntimeError(f"ffmpeg verification failed: {e}")
        
        try:
            result = subprocess.run(
                [self.ffprobe_path, "-version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise FileNotFoundError(f"ffprobe not found at {self.ffprobe_path}")
        except Exception as e:
            raise RuntimeError(f"ffprobe verification failed: {e}")
        
        self._verified = True
    
    def get_wav_duration_ms(self, wav_path: str) -> int:
        """
        Get duration of a WAV file in milliseconds using ffprobe.
        
        Args:
            wav_path: Path to WAV file
            
        Returns:
            Duration in milliseconds
        """
        try:
            result = subprocess.run(
                [
                    self.ffprobe_path,
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    wav_path
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                print(f"ffprobe error: {result.stderr}")
                return 0
            
            duration_seconds = float(result.stdout.strip())
            return int(duration_seconds * 1000)
        except Exception as e:
            print(f"Error getting WAV duration: {e}")
            return 0
    
    def get_wav_sample_rate(self, wav_path: str) -> int:
        """
        Get sample rate of a WAV file using soundfile.
        
        Args:
            wav_path: Path to WAV file
            
        Returns:
            Sample rate in Hz
        """
        try:
            with sf.SoundFile(wav_path) as f:
                return f.samplerate
        except Exception as e:
            print(f"Error getting sample rate: {e}")
            return 24000  # Kokoro default
    
    def create_ffmetadata(self, title: str, author: str, 
                         chapter_markers: List[ChapterMarker]) -> str:
        """
        Create FFMETADATA file content for chapter markers.
        
        Args:
            title: Book title
            author: Book author
            chapter_markers: List of ChapterMarker objects
            
        Returns:
            FFMETADATA file content as string
        """
        lines = []
        
        lines.append(";FFMETADATA1")
        lines.append(f"title={title}")
        lines.append(f"artist={author}")
        lines.append("")
        
        for i, marker in enumerate(chapter_markers):
            lines.append(f"[CHAPTER]")
            lines.append("TIMEBASE=1/1000")
            lines.append(f"START={marker.start_ms}")
            lines.append(f"END={marker.end_ms}")
            lines.append(f"title=Chapter {i+1}: {marker.title}")
            lines.append("")
        
        return "\n".join(lines)
    
    def concatenate_wavs(self, wav_paths: List[str], output_path: str) -> None:
        """
        Concatenate WAV files into a single WAV using ffmpeg concat demuxer.
        
        Args:
            wav_paths: List of WAV file paths in order
            output_path: Output concatenated WAV file path
        """
        # Create concat list file
        concat_list_path = output_path + ".concat.txt"
        
        with open(concat_list_path, 'w', encoding='utf-8') as f:
            for path in wav_paths:
                # The concat demuxer resolves relative paths against the concat
                # list file's location, so always write absolute paths.
                f.write(f"file '{Path(path).resolve().as_posix()}'\n")
        
        # Run ffmpeg
        cmd = [
            self.ffmpeg_path,
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path,
            "-y",  # Overwrite output
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")
        except Exception as e:
            raise RuntimeError(f"Error concatenating WAVs: {e}")
        finally:
            # Clean up concat list file
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)
    
    def encode_to_aac(self, wav_path: str, aac_path: str, bitrate: str = None) -> None:
        """
        Encode WAV to AAC using ffmpeg.
        
        Args:
            wav_path: Input WAV file
            aac_path: Output AAC file
            bitrate: AAC bitrate (default: 128k)
        """
        if bitrate is None:
            bitrate = self.DEFAULT_BITRATE
        
        cmd = [
            self.ffmpeg_path,
            "-i", wav_path,
            "-c:a", "aac",
            "-b:a", bitrate,
            "-movflags", "+faststart",
            aac_path,
            "-y",
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg AAC encoding failed: {result.stderr}")
        except Exception as e:
            raise RuntimeError(f"Error encoding to AAC: {e}")
    
    def build_m4b(self, wav_paths: List[str], output_m4b_path: str, 
                  title: str, author: str, chapter_titles: List[str],
                  bitrate: str = None) -> None:
        """
        Build M4B from chapter WAV files with chapter markers.
        
        Steps:
        1. Get durations of each WAV file
        2. Calculate cumulative start/end times
        3. Create FFMETADATA file
        4. Concatenate WAVs
        5. Encode to AAC with chapter metadata
        
        Args:
            wav_paths: List of chapter WAV file paths in order
            output_m4b_path: Output M4B file path
            title: Book title
            author: Book author
            chapter_titles: List of chapter titles
            bitrate: AAC bitrate (default: 128k)
        """
        if bitrate is None:
            bitrate = self.DEFAULT_BITRATE
        
        if len(wav_paths) == 0:
            raise ValueError("No WAV files provided")
        
        if len(chapter_titles) != len(wav_paths):
            raise ValueError(f"Chapter titles count ({len(chapter_titles)}) "
                           f"doesn't match WAV paths count ({len(wav_paths)})")
        
        # Step 1: Get durations
        durations_ms = []
        for wav_path in wav_paths:
            duration = self.get_wav_duration_ms(wav_path)
            durations_ms.append(duration)
        
        # Step 2: Calculate cumulative times
        cumulative_time = 0
        chapter_markers = []
        
        for i, (title_str, duration) in enumerate(zip(chapter_titles, durations_ms)):
            start_ms = cumulative_time
            end_ms = cumulative_time + duration
            
            chapter_markers.append(ChapterMarker(
                title=title_str,
                start_ms=start_ms,
                end_ms=end_ms,
            ))
            
            cumulative_time = end_ms
        
        # Step 3: Create FFMETADATA
        ffmetadata_content = self.create_ffmetadata(title, author, chapter_markers)
        ffmetadata_path = output_m4b_path + ".ffmetadata"
        
        with open(ffmetadata_path, 'w', encoding='utf-8') as f:
            f.write(ffmetadata_content)
        
        # Step 4: Concatenate WAVs
        concat_wav_path = output_m4b_path + ".concat.wav"
        self.concatenate_wavs(wav_paths, concat_wav_path)
        
        # Step 5: Encode to M4B with chapter metadata
        cmd = [
            self.ffmpeg_path,
            "-i", concat_wav_path,
            "-i", ffmetadata_path,
            "-map_metadata", "1",
            "-c:a", "aac",
            "-b:a", bitrate,
            "-movflags", "+faststart",
            output_m4b_path,
            "-y",
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg M4B assembly failed: {result.stderr}")
            
            print(f"M4B created: {output_m4b_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error building M4B: {e}")
        finally:
            # Clean up temporary files
            for tmp_path in [concat_wav_path, ffmetadata_path]:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    def build_m4b_from_manifest(self, manifest_dict: Dict, wav_dir: str, 
                               output_m4b_path: str, bitrate: str = None) -> None:
        """
        Build M4B from manifest information.
        
        Args:
            manifest_dict: Dictionary with book metadata and chapters
            wav_dir: Directory containing chapter WAV files
            output_m4b_path: Output M4B file path
            bitrate: AAC bitrate
        """
        title = manifest_dict.get("title", "Unknown")
        author = manifest_dict.get("author", "Unknown")
        chapters = manifest_dict.get("chapters", [])
        
        wav_paths = []
        chapter_titles = []
        
        for chapter in chapters:
            if chapter.get("status") == "done":
                wav_file = chapter.get("file", "")
                wav_path = os.path.join(wav_dir, wav_file)
                if os.path.exists(wav_path):
                    wav_paths.append(wav_path)
                    chapter_titles.append(chapter.get("title", f"Chapter {chapter.get('index', 0) + 1}"))
        
        if not wav_paths:
            raise ValueError("No completed chapter WAV files found")
        
        self.build_m4b(wav_paths, output_m4b_path, title, author, chapter_titles, bitrate)
    
    def verify_m4b(self, m4b_path: str) -> Dict:
        """
        Verify M4B file and extract chapter metadata using ffprobe.
        
        Args:
            m4b_path: Path to M4B file
            
        Returns:
            Dictionary with verification info
        """
        result = {
            "valid": False,
            "duration_seconds": 0,
            "bitrate": "",
            "chapters": [],
            "error": ""
        }
        
        try:
            # Check if file exists
            if not os.path.exists(m4b_path):
                result["error"] = "File not found"
                return result
            
            # Get basic info and chapter metadata as JSON
            import json as _json
            cmd = [
                self.ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_chapters",
                m4b_path
            ]
            
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10)
            if proc.returncode != 0:
                result["error"] = proc.stderr
                return result
            
            info = _json.loads(proc.stdout)
            
            if 'format' in info:
                result["duration_seconds"] = float(info['format'].get('duration', 0))
                result["bitrate"] = info['format'].get('bit_rate', '')
            
            for chapter in info.get('chapters', []):
                tags = chapter.get('tags', {})
                result["chapters"].append({
                    "id": chapter.get("id"),
                    "title": tags.get("title", ""),
                    "start_ms": int(float(chapter.get("start_time", 0))),
                    "end_ms": int(float(chapter.get("end_time", 0))),
                })
            
            result["valid"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result


# Global assembler instance
assembler = M4BAssembler()


# Convenience functions
def build_m4b(wav_paths: List[str], output_m4b_path: str, title: str, author: str, 
              chapter_titles: List[str]) -> None:
    """Build M4B from WAV files."""
    assembler.build_m4b(wav_paths, output_m4b_path, title, author, chapter_titles)


def verify_m4b(m4b_path: str) -> Dict:
    """Verify M4B file."""
    return assembler.verify_m4b(m4b_path)
