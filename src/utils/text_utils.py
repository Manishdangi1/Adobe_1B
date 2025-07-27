"""
Text processing utilities for PDF analysis
"""
import re
import string
from typing import List, Dict, Any
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class TextSection:
    """Represents a text section with metadata"""
    title: str
    content: str
    page_number: int
    start_pos: int
    end_pos: int
    section_type: str = "text"

def clean_text(text: str) -> str:
    """
    Clean and normalize text content
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
    
    # Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace('–', '-').replace('—', '-')
    
    # Remove page numbers and headers/footers
    text = re.sub(r'\b\d+\s*of\s*\d+\b', '', text)  # "Page X of Y"
    text = re.sub(r'\bPage\s+\d+\b', '', text)      # "Page X"
    
    # Clean up multiple periods
    text = re.sub(r'\.{2,}', '.', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_sections(text: str, page_number: int) -> List[TextSection]:
    """
    Extract logical sections from text based on headers and structure
    
    Args:
        text: Text content to analyze
        page_number: Page number for reference
        
    Returns:
        List of extracted sections
    """
    sections = []
    
    if not text.strip():
        return sections
    
    # Split by potential section headers
    # Look for patterns like: "1. Title", "Chapter X", "Section Name", etc.
    section_patterns = [
        r'^\d+\.\s+([A-Z][^.\n]+)',  # "1. Section Title"
        r'^[A-Z][A-Z\s]{2,}:',       # "SECTION NAME:"
        r'^Chapter\s+\d+',           # "Chapter 1"
        r'^Section\s+\d+',           # "Section 1"
        r'^[A-Z][^.\n]{3,}$',        # ALL CAPS titles
    ]
    
    lines = text.split('\n')
    current_section = []
    current_title = f"Page {page_number} Content"
    start_pos = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check if line is a section header
        is_header = False
        for pattern in section_patterns:
            if re.match(pattern, line, re.MULTILINE):
                # Save current section if it has content
                if current_section:
                    section_content = '\n'.join(current_section)
                    if section_content.strip():
                        sections.append(TextSection(
                            title=current_title,
                            content=section_content,
                            page_number=page_number,
                            start_pos=start_pos,
                            end_pos=start_pos + len(section_content)
                        ))
                
                # Start new section
                current_title = line
                current_section = [line]
                start_pos = text.find(line, start_pos)
                is_header = True
                break
        
        if not is_header:
            current_section.append(line)
    
    # Add the last section
    if current_section:
        section_content = '\n'.join(current_section)
        if section_content.strip():
            sections.append(TextSection(
                title=current_title,
                content=section_content,
                page_number=page_number,
                start_pos=start_pos,
                end_pos=start_pos + len(section_content)
            ))
    
    # If no sections were found, create one section with all content
    if not sections and text.strip():
        sections.append(TextSection(
            title=f"Page {page_number} Content",
            content=text,
            page_number=page_number,
            start_pos=0,
            end_pos=len(text)
        ))
    
    return sections

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract key terms from text using TF-IDF approach
    
    Args:
        text: Text to analyze
        top_n: Number of keywords to return
        
    Returns:
        List of keywords
    """
    # Tokenize and clean
    words = word_tokenize(text.lower())
    
    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
    
    # Count frequencies
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, freq in sorted_words[:top_n]]

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text segments
    
    Args:
        text1: First text segment
        text2: Second text segment
        
    Returns:
        Similarity score (0-1)
    """
    # Simple Jaccard similarity
    words1 = set(word_tokenize(text1.lower()))
    words2 = set(word_tokenize(text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of entity types and their values
    """
    entities = {
        'locations': [],
        'organizations': [],
        'dates': [],
        'numbers': [],
        'urls': []
    }
    
    # Extract URLs
    url_pattern = r'https?://[^\s]+'
    entities['urls'] = re.findall(url_pattern, text)
    
    # Extract dates
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
    ]
    
    for pattern in date_patterns:
        entities['dates'].extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Extract numbers (prices, quantities, etc.)
    number_patterns = [
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Currency
        r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',  # Numbers with commas
    ]
    
    for pattern in number_patterns:
        entities['numbers'].extend(re.findall(pattern, text))
    
    return entities

def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except for sentence boundaries
    text = re.sub(r'[^\w\s\.\!\?]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending
            for i in range(end, max(start + chunk_size - 200, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def calculate_readability_score(text: str) -> float:
    """
    Calculate Flesch Reading Ease score
    
    Args:
        text: Text to analyze
        
    Returns:
        Readability score (0-100, higher is easier to read)
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Count syllables (simplified)
    syllables = 0
    for word in words:
        word_lower = word.lower()
        if word_lower.endswith(('es', 'ed')):
            word_lower = word_lower[:-2]
        syllables += len(re.findall(r'[aeiouy]+', word_lower))
    
    if not sentences or not words:
        return 0.0
    
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = syllables / len(words)
    
    # Flesch Reading Ease formula
    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    return max(0.0, min(100.0, score)) 