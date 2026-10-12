"""
EPUB Parser for Audiobook TTS

Handles EPUB file ingestion, metadata extraction, and chapter detection.
Uses ebooklib for EPUB parsing and beautifulsoup4/lxml for HTML processing.
"""

import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning, Comment
import lxml

# Suppress XMLParsedAsHTMLWarning from BeautifulSoup
# This occurs when parsing XML documents (like EPUB nav.xhtml) with the HTML parser
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass
class ChapterInfo:
    """Information about a chapter extracted from EPUB."""
    index: int
    title: str
    file_path: str  # Internal EPUB path
    content: str  # Extracted plain text
    word_count: int = 0


@dataclass
class BookMetadata:
    """Metadata extracted from EPUB."""
    title: str
    author: str = ""
    language: str = ""
    identifier: str = ""
    creator: str = ""
    publisher: str = ""
    date: str = ""
    description: str = ""


class EPUBParser:
    """Parses EPUB files and extracts metadata and chapters."""
    
    # Patterns for cleaning text
    WHITESPACE_PATTERN = re.compile(r'\s+')
    NEWLINE_PATTERN = re.compile(r'(\r\n|\r|\n)+')

    # HTML tags treated as block boundaries (inline tags stay joined)
    BLOCK_TAGS = [
        'p', 'div', 'section', 'article', 'header', 'footer',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'li',
    ]
    
    # HTML tags to completely remove
    REMOVE_TAGS = ['script', 'style', 'noscript', 'head', 'meta', 'link', 'img']
    
    # CSS selectors for elements to remove (footnotes, boilerplate)
    # NOTE: '.pg-boilerplate' is the Project Gutenberg header/footer/license
    # wrapper. PG title pages consist *entirely* of that div, so after removal
    # such documents have no text; the spine fallback skips empty chapters.
    REMOVE_SELECTORS = [
        'footnote',
        '.footnote',
        '[role="footnote"]',
        '[class*="footnote"]',
        '[class*="note"]',
        'aside',
        '.pg-boilerplate',
        '[class*="copyright"]',
        '[class*="legal"]',
        '[class*="same-author"]',
        '[class*="also-by"]',
    ]
    
    # Patterns for detecting boilerplate text
    BOILERPLATE_PATTERNS = [
        re.compile(r'\b(also by|by the same author|other books by|more from)\b', re.IGNORECASE),
        re.compile(r'\b(copyright|all rights reserved|published by)\b', re.IGNORECASE),
        re.compile(r'\b(table of contents|toc|index)\b', re.IGNORECASE),
        re.compile(r'\b(page \d+|\d+ of \d+)\b', re.IGNORECASE),
    ]
    
    # CJK characters (Han, kana, hangul): languages written without spaces,
    # so whitespace word-splitting counts ~0 words for them
    CJK_CHAR_PATTERN = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')
    
    def _count_words(self, text: str) -> int:
        """Count words: whitespace-delimited for spaced languages, one per
        CJK character for Chinese/Japanese/Korean."""
        if not text:
            return 0
        cjk = len(self.CJK_CHAR_PATTERN.findall(text))
        non_cjk_words = sum(
            1 for w in text.split() if not self.CJK_CHAR_PATTERN.search(w)
        )
        return cjk + non_cjk_words
    
    def __init__(self):
        pass
    
    def parse_epub(self, epub_path: str, extract_text: bool = True) -> Tuple[BookMetadata, List[ChapterInfo]]:
        """
        Parse an EPUB file and extract metadata and chapters.
        
        Args:
            epub_path: Path to the EPUB file
            extract_text: Whether to extract text content from chapters
            
        Returns:
            Tuple of (BookMetadata, List[ChapterInfo])
        """
        epub_path = Path(epub_path)
        
        # Read EPUB
        with open(epub_path, 'rb') as f:
            book = epub.read_epub(f)
        
        # Extract metadata
        metadata = self._extract_metadata(book)
        
        # Detect spine order and TOC
        chapters = self._extract_chapters(book, metadata, extract_text)
        
        return metadata, chapters
    
    def _extract_metadata(self, book: epub.EpubBook) -> BookMetadata:
        """Extract metadata from EPUB book object."""
        def get_meta(prop: str, default: str = "") -> str:
            """Get a Dublin Core metadata property.

            ebooklib stores metadata as {namespace: {prop: [(value, attrs), ...]}}.
            """
            for namespace in book.metadata.values():
                values = namespace.get(prop)
                if isinstance(values, list) and values:
                    value = values[0]
                    if isinstance(value, (tuple, list)) and value:
                        value = value[0]
                    if value is not None:
                        return str(value)
                elif values not in (None, "", []):
                    return str(values)
            return default
        
        title = get_meta('title', 'Unknown Title')
        author = get_meta('creator') or get_meta('author') or get_meta('DC:creator') or "Unknown Author"
        language = get_meta('language', 'en')
        identifier = get_meta('identifier', '')
        creator = get_meta('creator', '')
        publisher = get_meta('publisher', '')
        date = get_meta('date', '')
        description = get_meta('description', '')
        
        # Clean author (remove trailing " (Author)" patterns)
        author = re.sub(r'\s*\([^)]*\)', '', author).strip()
        
        # Convert language to ISO 639-1 if needed
        language = self._normalize_language(language)
        
        return BookMetadata(
            title=title,
            author=author,
            language=language,
            identifier=identifier,
            creator=creator,
            publisher=publisher,
            date=date,
            description=description,
        )
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language code to ISO 639-1."""
        if not language:
            return "en"
        
        # Already ISO 639-1
        if len(language) == 2 and language.isalpha():
            return language.lower()
        
        # Handle ISO 639-2 or other formats
        lang_map = {
            'eng': 'en',
            'fra': 'fr',
            'spa': 'es',
            'deu': 'de',
            'ita': 'it',
            'por': 'pt',
            'rus': 'ru',
            'jpn': 'ja',
            'zho': 'zh',
            'hin': 'hi',
        }
        
        return lang_map.get(language.lower(), 'en')
    
    def _extract_chapters(self, book: epub.EpubBook, metadata: BookMetadata, 
                          extract_text: bool) -> List[ChapterInfo]:
        """
        Extract chapters from EPUB spine and TOC.
        
        Strategy:
        1. Try to use NCX TOC (legacy EPUB2)
        2. Try to use nav.xhtml TOC (EPUB3)
        3. Fall back to spine order
        
        For Calibre-generated EPUBs with split files (partXXX_split_YYY.html),
        we need to concatenate text from all split parts of each chapter.
        """
        chapters = []
        
        # Build mapping of spine IDs to their content and preserve spine order
        # Handle both list (of tuples) and dict spine formats
        spine_id_to_item = {}
        spine_order = []  # Ordered list of (spine_id, item) tuples
        
        if isinstance(book.spine, dict):
            # spine is a dict: {spine_id: item}
            for item_id, item in book.spine.items():
                spine_id_to_item[item_id] = item
                spine_order.append((item_id, item))
        elif isinstance(book.spine, list):
            # spine is a list of tuples: [(spine_id, item_or_linear), ...]
            for spine_entry in book.spine:
                if isinstance(spine_entry, tuple) and len(spine_entry) >= 2:
                    spine_id = spine_entry[0]
                    item_or_linear = spine_entry[1]
                    # Check if the second element is the item or a linear flag
                    # In ebooklib, spine entries are (item, is_linear) tuples
                    if isinstance(item_or_linear, str):
                        # It's (spine_id, linear_flag), need to get the actual item
                        item = book.get_item_with_id(spine_id)
                        if item:
                            spine_id_to_item[spine_id] = item
                            spine_order.append((spine_id, item))
                    elif hasattr(item_or_linear, 'get_type'):
                        # It's the actual item (second element)
                        spine_id_to_item[spine_id] = item_or_linear
                        spine_order.append((spine_id, item_or_linear))
                    else:
                        # Try to get item by ID
                        item = book.get_item_with_id(spine_id)
                        if item:
                            spine_id_to_item[spine_id] = item
                            spine_order.append((spine_id, item))
                else:
                    # Direct item reference
                    item = book.get_item_with_id(spine_entry)
                    if item:
                        spine_id_to_item[spine_entry] = item
                        spine_order.append((spine_entry, item))
        
        # Try to get TOC from NCX (EPUB2)
        ncx_toc = self._extract_ncx_toc(book)
        
        # Try to get TOC from nav.xhtml (EPUB3)
        nav_toc = self._extract_nav_toc(book)
        
        # Use whichever TOC we found, prefer nav.xhtml for EPUB3
        toc_items = nav_toc if nav_toc else ncx_toc
        
        if toc_items:
            # TOC-based chapter extraction
            for toc_item in toc_items:
                chapter_info = self._create_chapter_from_toc(
                    toc_item, spine_id_to_item, spine_order, len(chapters), extract_text
                )
                if chapter_info:
                    chapters.append(chapter_info)
        
        # A TOC can be far coarser than the spine: Project Gutenberg books
        # often ship a minimal NCX with just two nav points (title page and
        # license). If the TOC maps to only a small minority of the spine's
        # HTML documents, it does not describe the book - discard it and use
        # the spine so the real content files become chapters.
        spine_docs = [i for i in spine_id_to_item.values() if i is not None]
        if chapters and spine_docs:
            covered_files = {c.file_path for c in chapters}
            if len(covered_files) < max(2, len(spine_docs) // 2):
                chapters = []
        
        # If TOC didn't give us a usable chapter map, fall back to spine
        # Group consecutive split files together
        if not chapters and spine_order:
            # Process spine in order, grouping split files
            current_split_base = None
            current_split_items = []
            
            for spine_id, item in spine_order:
                if item.get_type() != ebooklib.ITEM_DOCUMENT:
                    continue
                
                file_path = item.get_name()
                split_base = self._get_split_file_base(file_path)
                
                # If we're already collecting split files for this base
                if split_base == current_split_base:
                    current_split_items.append(item)
                else:
                    # Finish the previous group
                    if current_split_items:
                        chapter_info = self._create_chapter_from_spine_items(
                            current_split_items, len(chapters), extract_text
                        )
                        if chapter_info and (not extract_text or chapter_info.content.strip()):
                            chapters.append(chapter_info)
                    
                    # Start a new group
                    current_split_base = split_base
                    current_split_items = [item]
            
            # Don't forget the last group
            if current_split_items:
                chapter_info = self._create_chapter_from_spine_items(
                    current_split_items, len(chapters), extract_text
                )
                if chapter_info and (not extract_text or chapter_info.content.strip()):
                    chapters.append(chapter_info)
        
        # If we still have no chapters, try all items
        if not chapters:
            for idx, item in enumerate(book.get_items()):
                if item.get_type() != ebooklib.ITEM_DOCUMENT:
                    continue
                chapter_info = self._create_chapter_from_spine(
                    item, len(chapters), extract_text
                )
                if chapter_info and (not extract_text or chapter_info.content.strip()):
                    chapters.append(chapter_info)
        
        # Keep indices sequential even when documents were skipped
        for i, chapter in enumerate(chapters):
            chapter.index = i
        
        return chapters
    
    def _extract_ncx_toc(self, book: epub.EpubBook) -> List[Dict]:
        """Extract TOC from NCX file (EPUB2)."""
        toc_items = []
        
        for item in book.get_items():
            if item.get_name().lower().endswith('.ncx'):
                try:
                    ncx_content = item.get_content().decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(ncx_content, 'lxml')
                    
                    nav_points = soup.find_all('navpoint')
                    for nav_point in nav_points:
                        label = nav_point.find('navlabel')
                        text = label.find('text').get_text() if label and label.find('text') else ""
                        
                        content = nav_point.find('content')
                        src = content.get('src', '') if content else ''
                        
                        # Remove fragment from src
                        src = src.split('#')[0]
                        
                        toc_items.append({
                            'title': text.strip(),
                            'src': src,
                        })
                except Exception as e:
                    print(f"Error parsing NCX: {e}")
        
        return toc_items
    
    def _extract_nav_toc(self, book: epub.EpubBook) -> List[Dict]:
        """Extract TOC from nav.xhtml (EPUB3)."""
        toc_items = []
        
        for item in book.get_items():
            if 'nav' in item.get_name().lower() and item.get_name().endswith(('.xhtml', '.html')):
                try:
                    nav_content = item.get_content().decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(nav_content, 'lxml')
                    
                    # Find all <a> tags in <nav> elements with epub:type="toc"
                    nav_elements = soup.find_all('nav', {'epub:type': 'toc'})
                    if not nav_elements:
                        nav_elements = soup.find_all('nav')
                    
                    for nav in nav_elements:
                        links = nav.find_all('a', href=True)
                        for link in links:
                            text = link.get_text().strip()
                            href = link.get('href', '')
                            
                            # Resolve relative href
                            src = href.split('#')[0]
                            
                            if text:
                                toc_items.append({
                                    'title': text,
                                    'src': src,
                                })
                except Exception as e:
                    print(f"Error parsing nav.xhtml: {e}")
        
        return toc_items
    
    def _get_split_file_base(self, file_path: str) -> Optional[str]:
        """
        Extract the base part number from a Calibre split file path.
        
        Calibre splits large HTML files into parts like:
        - text/part0015_split_000.html
        - text/part0015_split_001.html
        - text/part0015_split_002.html
        
        Returns the base (e.g., 'text/part0015') or None if not a split file.
        """
        # Match pattern: .../partXXXX_split_YYY.html or partXXXX_split_YYY.html
        import re
        match = re.search(r'(.*/)?part(\d+)_split_(\d+)(\.html)?$', file_path, re.IGNORECASE)
        if match:
            # Return the base path with part number
            prefix = match.group(1) or ''
            part_num = match.group(2)
            return f"{prefix}part{part_num}"
        return None
    
    def _find_all_split_items(self, item: epub.EpubItem, spine_order: List[tuple],
                               spine_id_to_item: Dict) -> List[epub.EpubItem]:
        """
        Find all split file items that belong to the same chapter.
        
        For Calibre EPUBs, chapters may be split across multiple files:
        part0015_split_000.html, part0015_split_001.html, etc.
        
        Returns a list of all items that are part of the same split chapter.
        """
        items = [item]
        file_path = item.get_name()
        base_path = self._get_split_file_base(file_path)
        
        if base_path is None:
            return items
        
        # Find the position of the current item in the spine
        current_idx = -1
        for idx, (spine_id, spine_item) in enumerate(spine_order):
            if spine_item.get_name() == file_path:
                current_idx = idx
                break
        
        if current_idx == -1:
            return items
        
        # Look at subsequent spine items for matching split files
        for idx in range(current_idx + 1, len(spine_order)):
            spine_id, spine_item = spine_order[idx]
            next_path = spine_item.get_name()
            next_base = self._get_split_file_base(next_path)
            
            # If it has the same base, it's part of the same chapter
            if next_base == base_path:
                items.append(spine_item)
            else:
                # Once we hit a different base, we're done
                break
        
        return items
    
    def _create_chapter_from_toc(self, toc_item: Dict, spine_id_to_item: Dict,
                                  spine_order: List[tuple], index: int, 
                                  extract_text: bool) -> Optional[ChapterInfo]:
        """Create ChapterInfo from TOC entry by finding matching spine item."""
        src = toc_item.get('src', '')
        title = toc_item.get('title', f"Chapter {index + 1}")
        
        # Find spine item that matches this TOC entry
        for spine_id, item in spine_id_to_item.items():
            item_href = item.get_name()
            # Remove fragments and query params
            item_href_clean = item_href.split('#')[0].split('?')[0]
            src_clean = src.split('#')[0].split('?')[0]
            
            if item_href_clean == src_clean or item_href_clean.endswith(src_clean):
                # Found matching item - check for Calibre split files
                all_items = self._find_all_split_items(item, spine_order, spine_id_to_item)
                return self._create_chapter_from_spine_items(all_items, index, extract_text, title)
        
        # If we couldn't find a matching spine item, try by path
        for spine_id, item in spine_id_to_item.items():
            if src in item.get_name():
                # Found matching item - check for Calibre split files
                all_items = self._find_all_split_items(item, spine_order, spine_id_to_item)
                return self._create_chapter_from_spine_items(all_items, index, extract_text, title)
        
        return None
    
    def _create_chapter_from_spine(self, item: epub.EpubItem, index: int, 
                                    extract_text: bool, title: Optional[str] = None) -> ChapterInfo:
        """Create ChapterInfo from spine item."""
        # For backward compatibility, delegate to the list version
        return self._create_chapter_from_spine_items([item], index, extract_text, title)
    
    def _create_chapter_from_spine_items(self, items: List[epub.EpubItem], index: int, 
                                         extract_text: bool, title: Optional[str] = None) -> ChapterInfo:
        """Create ChapterInfo from a list of spine items (for handling split files)."""
        # Use the first item's path as the file_path
        file_path = items[0].get_name() if items else ""
        
        # Default title
        if not title:
            title = f"Chapter {index + 1}"
        
        # Extract text from all items and concatenate
        content = ""
        if extract_text:
            for item in items:
                content += self._extract_text_from_item(item)
                # Add a separator between split parts, but not excessive whitespace
                if len(content) > 0 and not content.endswith('\n'):
                    content += ' '
        
        return ChapterInfo(
            index=index,
            title=title,
            file_path=file_path,
            content=content,
            word_count=self._count_words(content),
        )
    
    def _extract_text_from_item(self, item: epub.EpubItem) -> str:
        """Extract plain text from an EPUB item (HTML/XHTML)."""
        try:
            html_content = item.get_content().decode('utf-8', errors='ignore')
        except:
            return ""
        
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove unwanted tags
        for tag in self.REMOVE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove by selectors
        for selector in self.REMOVE_SELECTORS:
            try:
                for element in soup.select(selector):
                    element.decompose()
            except:
                pass
        
        # Get body or root
        body = soup.find('body') or soup
        
        # Extract text
        text = self._get_text(body)
        
        # Clean up
        text = self._clean_text(text)
        
        return text
    
    def _get_text(self, element: Tag) -> str:
        """
        Recursively extract text from an element.

        Inline content must stay joined: a drop cap like
        ``<span class="big">H</span>alla of Rutger's Howe`` is one word in
        print, so no separator may be inserted between the span and the
        following text. Whitespace inside text nodes is kept as-is (the
        cleaner collapses it later); only block-level tags get newline
        separators.
        """
        parts = []

        for child in element.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                parts.append(str(child))
            elif isinstance(child, Tag):
                tag_name = child.name.lower()

                if tag_name == 'br':
                    parts.append('\n')
                elif tag_name == 'li':
                    parts.append('\n- ' + self._get_text(child).strip() + '\n')
                elif tag_name in self.BLOCK_TAGS:
                    parts.append('\n' + self._get_text(child) + '\n')
                else:
                    # Inline tag (span, em, i, b, a, ...): join without separator
                    parts.append(self._get_text(child))

        return ''.join(parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove boilerplate sections
        for pattern in self.BOILERPLATE_PATTERNS:
            text = pattern.sub('', text)
        
        # Remove excessive whitespace
        text = self.NEWLINE_PATTERN.sub('\n\n', text)
        text = self.WHITESPACE_PATTERN.sub(' ', text)
        
        # Trim
        text = text.strip()
        
        # Remove leading/trailing newlines
        text = text.strip('\n')
        
        return text


# Global parser instance
parser = EPUBParser()


# Convenience function
def parse_epub(epub_path: str) -> Tuple[BookMetadata, List[ChapterInfo]]:
    """Parse an EPUB file and return metadata and chapters."""
    return parser.parse_epub(epub_path)
