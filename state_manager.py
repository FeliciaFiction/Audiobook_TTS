"""
State Manager for Audiobook TTS

Handles book directories, manifest.json, atomic writes, and state persistence
for resume/append functionality.

Manifest schema:
{
    "book_id": "unique_book_id",
    "title": "Book Title",
    "author": "Author Name",
    "language": "en",  # ISO 639-1 code
    "detected_language": "en",  # From OPF or heuristic
    "kokoro_lang_code": "a",  # Kokoro language code (a, b, e, f, h, i, j, p, z)
    "voice": "af_heart",
    "speed": 1.0,
    "epub_path": "path/to/original.epub",  # Original EPUB, if known
    "output_dir": "absolute/path/to/book-id",
    "m4b_path": "absolute/path/to/output-folder/book.m4b",
    "chapters_dir": "absolute/path/to/book-id/chapters",
    "text_dir": "absolute/path/to/book-id/text",
    "chapters": [
        {
            "index": 0,
            "title": "Chapter 1",
            "file": "001.wav",
            "status": "pending" | "done" | "error",
            "duration_ms": 123456,
            "start_time_ms": 0,
            "word_count": 1500
        },
        ...
    ],
    "created_at": "2026-10-07T12:00:00Z",
    "updated_at": "2026-10-07T12:00:00Z"
}
"""

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import re


@dataclass
class ChapterState:
    """State for a single chapter."""
    index: int
    title: str
    file: str  # Filename in chapters_dir, e.g., "001.wav"
    status: str = "pending"  # pending, done, error
    duration_ms: int = 0
    start_time_ms: int = 0
    word_count: int = 0
    error: Optional[str] = None


@dataclass
class BookManifest:
    """Manifest for a book conversion project."""
    book_id: str
    title: str
    author: str = ""
    language: str = ""  # ISO 639-1
    detected_language: str = ""
    kokoro_lang_code: str = "a"  # Default: American English
    voice: str = "af_heart"
    speed: float = 1.0
    epub_path: str = ""
    output_dir: str = ""
    m4b_path: str = ""
    chapters_dir: str = ""
    text_dir: str = ""
    chapters: List[ChapterState] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "book_id": self.book_id,
            "title": self.title,
            "author": self.author,
            "language": self.language,
            "detected_language": self.detected_language,
            "kokoro_lang_code": self.kokoro_lang_code,
            "voice": self.voice,
            "speed": self.speed,
            "epub_path": self.epub_path,
            "output_dir": self.output_dir,
            "m4b_path": self.m4b_path,
            "chapters_dir": self.chapters_dir,
            "text_dir": self.text_dir,
            "chapters": [asdict(c) for c in self.chapters],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BookManifest":
        """Create from dictionary (JSON deserialization)."""
        chapters = [
            ChapterState(
                index=c["index"],
                title=c["title"],
                file=c["file"],
                status=c.get("status", "pending"),
                duration_ms=c.get("duration_ms", 0),
                start_time_ms=c.get("start_time_ms", 0),
                word_count=c.get("word_count", 0),
                error=c.get("error"),
            )
            for c in data.get("chapters", [])
        ]
        # Normalize voice: legacy (Gradio-era) manifests could store the raw
        # dropdown value as a dict like {"value": "af_heart", "label": "..."}
        voice = data.get("voice", "af_heart")
        if isinstance(voice, dict):
            voice = voice.get("value") or voice.get("name") or "af_heart"
        return cls(
            book_id=data.get("book_id", ""),
            title=data.get("title", ""),
            author=data.get("author", ""),
            language=data.get("language", ""),
            detected_language=data.get("detected_language", ""),
            kokoro_lang_code=data.get("kokoro_lang_code", "a"),
            voice=voice,
            speed=data.get("speed", 1.0),
            epub_path=data.get("epub_path", ""),
            output_dir=data.get("output_dir", ""),
            m4b_path=data.get("m4b_path", ""),
            chapters_dir=data.get("chapters_dir", ""),
            text_dir=data.get("text_dir", ""),
            chapters=chapters,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


class StateManager:
    """Manages book conversion state, manifest, and file system operations."""
    
    MANIFEST_FILE = "manifest.json"
    CHAPTERS_DIR = "chapters"
    TEXT_DIR = "text"
    
    def __init__(self, base_dir: str):
        """
        Initialize state manager.
        
        Args:
            base_dir: Base directory for all book projects
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as a filename."""
        clean = re.sub(r'[<>:"/\\|?*]', '_', name).strip()
        return clean or "audiobook"
    
    def _generate_book_id(self, title: str, author: str = "") -> str:
        """Generate a unique book ID from title and author."""
        # Clean title for use in directory name
        clean_title = re.sub(r'[^\w\s-]', '_', title).strip()
        clean_title = re.sub(r'[\s]+', '_', clean_title)[:50]
        clean_author = re.sub(r'[^\w\s-]', '_', author).strip()
        clean_author = re.sub(r'[\s]+', '_', clean_author)[:30] if clean_author else ""
        
        if clean_author:
            id_base = f"{clean_title}_by_{clean_author}"
        else:
            id_base = clean_title
        
        # Add UUID suffix to ensure uniqueness
        return f"{id_base}_{uuid.uuid4().hex[:8]}"
    
    def _get_timestamp(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    def create_book(self, title: str, author: str = "", language: str = "", 
                    kokoro_lang_code: str = "a", voice: str = "af_heart", 
                    speed: float = 1.0, epub_path: str = "",
                    output_folder: str = "") -> BookManifest:
        """
        Create a new book project with directory structure and manifest.
        
        Args:
            title: Book title
            author: Author name
            language: ISO 639-1 language code
            kokoro_lang_code: Kokoro language code
            voice: Selected voice ID
            speed: Narration speed
            epub_path: Path to the source EPUB (may be empty)
            output_folder: Folder where the M4B is written (defaults to the
                           project directory)
            
        Returns:
            BookManifest for the new project
        """
        book_id = self._generate_book_id(title, author)
        output_dir = self.base_dir / book_id
        chapters_dir = output_dir / self.CHAPTERS_DIR
        text_dir = output_dir / self.TEXT_DIR
        
        # Create directories
        output_dir.mkdir(parents=True, exist_ok=True)
        chapters_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self._get_timestamp()
        
        if output_folder:
            m4b_path = str(Path(output_folder) / f"{self._sanitize_filename(title)}.m4b")
        else:
            m4b_path = str(output_dir / f"{self._sanitize_filename(title)}.m4b")
        
        manifest = BookManifest(
            book_id=book_id,
            title=title,
            author=author,
            language=language,
            detected_language=language,
            kokoro_lang_code=kokoro_lang_code,
            voice=voice,
            speed=speed,
            epub_path=epub_path,
            output_dir=str(output_dir),
            m4b_path=m4b_path,
            chapters_dir=str(chapters_dir),
            text_dir=str(text_dir),
            created_at=timestamp,
            updated_at=timestamp,
        )
        
        # Save manifest atomically
        self._save_manifest(manifest)
        
        return manifest
    
    def _save_manifest(self, manifest: BookManifest) -> None:
        """Save manifest to file atomically (tmp + rename)."""
        manifest_path = Path(manifest.output_dir) / self.MANIFEST_FILE
        tmp_path = manifest_path.with_suffix('.tmp')
        
        # Update timestamp
        manifest.updated_at = self._get_timestamp()
        
        # Write to temp file
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        tmp_path.replace(manifest_path)
    
    def load_manifest(self, book_id: str) -> Optional[BookManifest]:
        """Load manifest for a book."""
        manifest_path = self.base_dir / book_id / self.MANIFEST_FILE
        
        if not manifest_path.exists():
            return None
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return BookManifest.from_dict(data)
    
    def list_books(self) -> List[Dict[str, Any]]:
        """List all book projects with their status."""
        books = []
        
        for book_dir in self.base_dir.iterdir():
            if not book_dir.is_dir():
                continue
            
            manifest_path = book_dir / self.MANIFEST_FILE
            if not manifest_path.exists():
                continue
            
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Count chapter statuses
                total = len(data.get("chapters", []))
                done = sum(1 for c in data.get("chapters", []) if c.get("status") == "done")
                error = sum(1 for c in data.get("chapters", []) if c.get("status") == "error")
                
                books.append({
                    "book_id": data.get("book_id", book_dir.name),
                    "title": data.get("title", book_dir.name),
                    "author": data.get("author", ""),
                    "voice": data.get("voice", ""),
                    "language": data.get("language", ""),
                    "total_chapters": total,
                    "done_chapters": done,
                    "error_chapters": error,
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        
        return books
    
    def update_chapter_status(self, manifest: BookManifest, chapter_index: int, 
                              status: str, duration_ms: int = 0,
                              error: Optional[str] = None) -> None:
        """
        Update a chapter's status in the manifest and save.
        
        Args:
            manifest: The book manifest
            chapter_index: Index of the chapter to update
            status: New status (pending, done, error)
            duration_ms: Chapter duration in milliseconds
            error: Error message if status is error
        """
        if 0 <= chapter_index < len(manifest.chapters):
            manifest.chapters[chapter_index].status = status
            manifest.chapters[chapter_index].duration_ms = duration_ms
            manifest.chapters[chapter_index].error = error
            
            # Recalculate start times
            cumulative_time = 0
            for i, chapter in enumerate(manifest.chapters):
                chapter.start_time_ms = cumulative_time
                cumulative_time += chapter.duration_ms
            
            self._save_manifest(manifest)
    
    def get_chapter_path(self, manifest: BookManifest, chapter_index: int) -> str:
        """Get the file path for a chapter's WAV file."""
        if 0 <= chapter_index < len(manifest.chapters):
            return str(Path(manifest.chapters_dir) / manifest.chapters[chapter_index].file)
        return ""
    
    def atomic_write_wav(self, manifest: BookManifest, chapter_index: int, 
                        audio_data: bytes) -> str:
        """
        Atomically write chapter WAV file (tmp + rename).
        
        Args:
            manifest: The book manifest
            chapter_index: Chapter index
            audio_data: Raw WAV file bytes
            
        Returns:
            Path to the written WAV file
        """
        chapter_path = self.get_chapter_path(manifest, chapter_index)
        tmp_path = chapter_path + ".tmp"
        
        # Write to temp file
        with open(tmp_path, 'wb') as f:
            f.write(audio_data)
        
        # Atomic rename
        Path(tmp_path).rename(chapter_path)
        
        return chapter_path
    
    def delete_book(self, book_id: str) -> bool:
        """Delete a book project and all its files."""
        book_dir = self.base_dir / book_id
        
        if not book_dir.exists():
            return False
        
        try:
            shutil.rmtree(book_dir)
            return True
        except Exception as e:
            print(f"Error deleting book {book_id}: {e}")
            return False
    
    def add_chapters(self, manifest: BookManifest, chapters: List[Dict[str, Any]]) -> None:
        """
        Add chapters to a book manifest.
        
        Args:
            manifest: The book manifest
            chapters: List of chapter dicts with 'title' key
        """
        for i, chapter in enumerate(chapters):
            chapter_state = ChapterState(
                index=len(manifest.chapters),
                title=chapter.get("title", f"Chapter {len(manifest.chapters) + 1}"),
                file=f"{len(manifest.chapters) + 1:03d}.wav",
                status="pending",
            )
            manifest.chapters.append(chapter_state)
        
        self._save_manifest(manifest)
    
    def reset_chapter(self, manifest: BookManifest, chapter_index: int) -> None:
        """Reset a chapter to pending status, removing its WAV file."""
        if 0 <= chapter_index < len(manifest.chapters):
            chapter_path = self.get_chapter_path(manifest, chapter_index)
            if os.path.exists(chapter_path):
                os.remove(chapter_path)
            
            manifest.chapters[chapter_index].status = "pending"
            manifest.chapters[chapter_index].duration_ms = 0
            manifest.chapters[chapter_index].error = None
            
            self._save_manifest(manifest)
    
    def reset_all_chapters(self, manifest: BookManifest) -> None:
        """Reset all chapters to pending status."""
        for i in range(len(manifest.chapters)):
            self.reset_chapter(manifest, i)
    
    def set_output_folder(self, manifest: BookManifest, output_folder: str) -> None:
        """
        Set the folder where the M4B is written and persist it in the manifest.
        
        Args:
            manifest: The book manifest
            output_folder: Absolute path to the output folder
        """
        folder = Path(output_folder)
        folder.mkdir(parents=True, exist_ok=True)
        manifest.m4b_path = str(folder / f"{self._sanitize_filename(manifest.title)}.m4b")
        self._save_manifest(manifest)
    
    def save_chapter_text(self, manifest: BookManifest, chapter_index: int,
                          text: str) -> None:
        """
        Persist a chapter's extracted text (atomic tmp + rename).
        
        This allows resuming generation after an app restart without the
        original EPUB file.
        """
        if not manifest.text_dir:
            manifest.text_dir = str(Path(manifest.output_dir) / self.TEXT_DIR)
        text_dir = Path(manifest.text_dir)
        text_dir.mkdir(parents=True, exist_ok=True)
        
        text_path = text_dir / f"{chapter_index + 1:03d}.txt"
        tmp_path = text_dir / f"{chapter_index + 1:03d}.txt.tmp"
        
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(text)
        tmp_path.replace(text_path)
    
    def load_chapter_text(self, manifest: BookManifest, chapter_index: int) -> Optional[str]:
        """Load a chapter's persisted text, or None if not stored."""
        text_dir = manifest.text_dir or str(Path(manifest.output_dir) / self.TEXT_DIR)
        text_path = Path(text_dir) / f"{chapter_index + 1:03d}.txt"
        if not text_path.exists():
            return None
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read()


# Global state manager instance
# Default base directory is ./books relative to project root
DEFAULT_BASE_DIR = "books"
state_manager = StateManager(DEFAULT_BASE_DIR)
