"""
Text Processing for Audiobook TTS

Handles text normalization, typography cleanup, and chunking for Kokoro synthesis.
"""

import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TextChunk:
    """A chunk of text ready for synthesis."""
    text: str
    chunk_index: int
    chapter_index: int


class TextProcessor:
    """Processes text for TTS synthesis."""
    
    # Typography replacements
    TYPOGRAPHY_MAP = {
        # Curly quotes
        r'"': '"',
        r'"': '"',
        r'\u2018': "'",  # Left single quotation mark
        r'\u2019': "'",  # Right single quotation mark
        r'\u201A': "'",  # Single low-9 quotation mark
        r'\u201B': "'",  # Single high-reversed-9 quotation mark
        r'\u201C': '"',  # Left double quotation mark
        r'\u201D': '"',  # Right double quotation mark
        r'\u201E': '"',  # Double low-9 quotation mark
        r'\u201F': '"',  # Double high-reversed-9 quotation mark
        
        # Dashes
        r'\u2013': '-',  # En dash
        r'\u2014': '--',  # Em dash
        r'\u2015': '-',  # Horizontal bar
        
        # Ellipsis
        r'\u2026': '...',  # Horizontal ellipsis
        
        # Other
        r'\u2009': ' ',  # Thin space
        r'\u202F': ' ',  # Narrow no-break space
        r'\u205F': ' ',  # Medium mathematical space
        r'\u3000': ' ',  # Ideographic space
    }
    
    # Number and abbreviation expansions for better TTS
    NUMBER_EXPANSIONS = {
        # Ordinals
        r'\b(\d+)st\b': r'\1st',
        r'\b(\d+)nd\b': r'\1nd',
        r'\b(\d+)rd\b': r'\1rd',
        r'\b(\d+)th\b': r'\1th',
        
        # Simple numbers that might be better spoken as words
        # (Kokoro generally handles numbers well, but we can add exceptions)
    }
    
    # Patterns for splitting into sentences
    SENTENCE_SPLITTERS = [
        r'(?<=[.!?])\s+',  # Split after sentence-ending punctuation
        r'(?<=\n\n)',  # Split after double newline (paragraph break)
    ]
    
    def __init__(self, max_chunk_length: int = 500, max_sentences: int = 5):
        """
        Initialize text processor.
        
        Args:
            max_chunk_length: Maximum characters per chunk
            max_sentences: Maximum sentences per chunk
        """
        self.max_chunk_length = max_chunk_length
        self.max_sentences = max_sentences
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text: fix typography, clean up special characters.
        
        Args:
            text: Raw text from EPUB
            
        Returns:
            Normalized text
        """
        # Replace typography
        for pattern, replacement in self.TYPOGRAPHY_MAP.items():
            text = re.sub(pattern, replacement, text)
        
        # Fix common spacing issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = re.sub(r'([.,!?;:])(\w)', r'\1 \2', text)  # Space after punctuation
        text = re.sub(r'(\w)([.,!?;:])', r'\1\2 ', text)  # Space before punctuation
        
        # Clean up double spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix quote spacing
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        
        return text.strip()
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expand abbreviations and numbers for better TTS pronunciation.
        
        Args:
            text: Normalized text
            
        Returns:
            Text with expanded abbreviations
        """
        # Apply number/abbreviation expansions
        for pattern, replacement in self.NUMBER_EXPANSIONS.items():
            text = re.sub(pattern, replacement, text)
        
        # Common abbreviations
        expansions = {
            r'\bDr\.\b': 'Doctor',
            r'\bMr\.\b': 'Mister',
            r'\bMrs\.\b': 'Missus',
            r'\bMs\.\b': 'Miss',
            r'\bProf\.\b': 'Professor',
            r'\bPres\.\b': 'President',
            r'\bSen\.\b': 'Senator',
            r'\bRep\.\b': 'Representative',
            r'\bGen\.\b': 'General',
            r'\bCol\.\b': 'Colonel',
            r'\bCapt\.\b': 'Captain',
            r'\bLt\.\b': 'Lieutenant',
            r'\bSgt\.\b': 'Sergeant',
            r'\bRev\.\b': 'Reverend',
            r'\bHon\.\b': 'Honorable',
            r'\bSt\.\b': 'Saint',
            r'\bAve\.\b': 'Avenue',
            r'\bBlvd\.\b': 'Boulevard',
            r'\bRd\.\b': 'Road',
            r'\bSt\.\b': 'Street',
            r'\bLn\.\b': 'Lane',
            r'\bDr\.\b': 'Drive',
            r'\bCt\.\b': 'Court',
            r'\bPl\.\b': 'Place',
            r'\bSq\.\b': 'Square',
            r'\bApt\.\b': 'Apartment',
            r'\bSuite\.\b': 'Suite',
            r'\bNo\.\b': 'Number',
            r'\bU\.S\.\b': 'United States',
            r'\bU\.K\.\b': 'United Kingdom',
            r'\bE\.U\.\b': 'European Union',
        }
        
        for pattern, replacement in expansions.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Normalized text
            
        Returns:
            List of sentences
        """
        sentences = []
        current = text
        
        # Split by sentence-ending punctuation followed by whitespace or newline
        parts = re.split(r'(?<=[.!?])\s+', text)
        
        for part in parts:
            part = part.strip()
            if part:
                # Handle cases where sentence ends with quote
                if part.endswith('"') or part.endswith("'") or part.endswith(')') or part.endswith(']'):
                    # Don't split if this is a quoted sentence
                    if part in ['"', "'", ')', ']']:
                        if sentences:
                            sentences[-1] = sentences[-1] + part
                        else:
                            sentences.append(part)
                    else:
                        sentences.append(part)
                else:
                    sentences.append(part)
        
        # Merge sentences that are too short
        final_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 0:
                final_sentences.append(sentence)
        
        return final_sentences
    
    def chunk_text(self, text: str, chapter_index: int = 0) -> List[TextChunk]:
        """
        Split text into chunks suitable for Kokoro synthesis.
        
        Args:
            text: Text to chunk
            chapter_index: Chapter index for tracking
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        
        # Normalize
        text = self.normalize_text(text)
        text = self.expand_abbreviations(text)
        
        # Split into sentences
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return chunks
        
        # Build chunks from sentences
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed limits
            if (len(current_chunk) >= self.max_sentences or 
                current_length + sentence_length > self.max_chunk_length):
                # Start a new chunk
                if current_chunk:
                    chunks.append(TextChunk(
                        text=' '.join(current_chunk),
                        chunk_index=chunk_index,
                        chapter_index=chapter_index,
                    ))
                    chunk_index += 1
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunks.append(TextChunk(
                text=' '.join(current_chunk),
                chunk_index=chunk_index,
                chapter_index=chapter_index,
            ))
        
        return chunks
    
    def process_chapter(self, text: str, chapter_index: int = 0) -> List[TextChunk]:
        """
        Process a full chapter: normalize, expand, chunk.
        
        Args:
            text: Chapter text
            chapter_index: Chapter index
            
        Returns:
            List of TextChunk objects ready for synthesis
        """
        return self.chunk_text(text, chapter_index)


# Global processor instance
processor = TextProcessor()


# Convenience functions
def normalize_text(text: str) -> str:
    """Normalize text."""
    return processor.normalize_text(text)


def chunk_text(text: str, chapter_index: int = 0) -> List[TextChunk]:
    """Chunk text for synthesis."""
    return processor.chunk_text(text, chapter_index)
