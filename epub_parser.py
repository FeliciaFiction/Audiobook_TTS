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
    REMOVE_SELECTORS = [
        'footnote',
        '.footnote',
        '[role="footnote"]',
        '[class*="footnote"]',
        '[class*="note"]',
        'aside',
        '. boilerplate',  # Common class names
        '[class*="boilerplate"]',
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
        """
        chapters = []
        
        # Build mapping of spine IDs to their content
        # Handle both list (of tuples) and dict spine formats
        spine_id_to_item = {}
        if isinstance(book.spine, dict):
            # spine is a dict: {spine_id: item}
            spine_id_to_item = {item_id: item for item_id, item in book.spine.items()}
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
                        # This shouldn't happen, but handle it
                        item = book.get_item_with_id(spine_id)
                        if item:
                            spine_id_to_item[spine_id] = item
                    elif hasattr(item_or_linear, 'get_type'):
                        # It's the actual item (second element)
                        spine_id_to_item[spine_id] = item_or_linear
                    else:
                        # Try to get item by ID
                        item = book.get_item_with_id(spine_id)
                        if item:
                            spine_id_to_item[spine_id] = item
                else:
                    # Direct item reference
                    spine_id_to_item[spine_entry] = book.get_item_with_id(spine_entry)
        
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
                    toc_item, spine_id_to_item, len(chapters), extract_text
                )
                if chapter_info:
                    chapters.append(chapter_info)
        
        # If TOC didn't give us chapters, fall back to spine
        if not chapters and book.spine:
            # Handle both list and dict spine formats
            if isinstance(book.spine, dict):
                spine_items = book.spine.items()
            elif isinstance(book.spine, list):
                # Convert list of tuples to (id, item) pairs
                spine_items = []
                for entry in book.spine:
                    if isinstance(entry, tuple) and len(entry) >= 2:
                        spine_id = entry[0]
                        item_or_linear = entry[1]
                        if hasattr(item_or_linear, 'get_type'):
                            spine_items.append((spine_id, item_or_linear))
                        else:
                            item = book.get_item_with_id(spine_id)
                            if item:
                                spine_items.append((spine_id, item))
            else:
                spine_items = []
            
            for idx, (spine_id, item) in enumerate(spine_items):
                # Skip non-HTML items (CSS, images, etc.)
                if not item.get_type() == epub.ITEM_DOCUMENT:
                    continue
                    
                chapter_info = self._create_chapter_from_spine(
                    item, idx, extract_text
                )
                if chapter_info:
                    chapters.append(chapter_info)
        
        # If we still have no chapters, try all items
        if not chapters:
            for idx, item in enumerate(book.get_items()):
                if not item.get_type() == epub.ITEM_DOCUMENT:
                    continue
                chapter_info = self._create_chapter_from_spine(
                    item, idx, extract_text
                )
                if chapter_info:
                    chapters.append(chapter_info)
        
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
    
    def _create_chapter_from_toc(self, toc_item: Dict, spine_id_to_item: Dict, 
                                  index: int, extract_text: bool) -> Optional[ChapterInfo]:
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
                return self._create_chapter_from_spine(item, index, extract_text, title)
        
        # If we couldn't find a matching spine item, try by path
        for spine_id, item in spine_id_to_item.items():
            if src in item.get_name():
                return self._create_chapter_from_spine(item, index, extract_text, title)
        
        return None
    
    def _create_chapter_from_spine(self, item: epub.EpubItem, index: int, 
                                    extract_text: bool, title: Optional[str] = None) -> ChapterInfo:
        """Create ChapterInfo from spine item."""
        file_path = item.get_name()
        
        # Default title
        if not title:
            title = f"Chapter {index + 1}"
        
        # Extract text if requested
        content = ""
        if extract_text:
            content = self._extract_text_from_item(item)
        
        return ChapterInfo(
            index=index,
            title=title,
            file_path=file_path,
            content=content,
            word_count=len(content.split()),
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
