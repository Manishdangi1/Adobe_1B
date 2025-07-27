"""
High-performance PDF processing module with async capabilities
"""
import asyncio
import fitz  # PyMuPDF
import pdfplumber
import PyPDF2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from dataclasses import dataclass
import re
import json

from ..utils.text_utils import clean_text, extract_sections
from ...config.config import processing_config

logger = logging.getLogger(__name__)

@dataclass
class PDFSection:
    """Represents a section extracted from a PDF"""
    title: str
    content: str
    page_number: int
    start_pos: int
    end_pos: int
    importance_score: float = 0.0
    section_type: str = "text"

@dataclass
class PDFDocument:
    """Represents a processed PDF document"""
    filename: str
    title: str
    sections: List[PDFSection]
    metadata: Dict[str, Any]
    total_pages: int
    processing_time: float

class PDFProcessor:
    """High-performance PDF processor with multiple engines"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or processing_config.max_workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.engines = {
            "pymupdf": self._extract_with_pymupdf,
            "pdfplumber": self._extract_with_pdfplumber,
            "PyPDF2": self._extract_with_pypdf2
        }
    
    async def process_pdf(self, pdf_path: Path) -> PDFDocument:
        """Process a single PDF file asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            start_time = asyncio.get_event_loop().time()
            
            # Use the configured PDF engine
            engine = processing_config.pdf_engine
            if engine not in self.engines:
                engine = "pymupdf"  # fallback
            
            # Run extraction in thread pool
            sections, metadata, total_pages = await loop.run_in_executor(
                self.executor, 
                self.engines[engine], 
                pdf_path
            )
            
            # Post-process sections
            processed_sections = await self._post_process_sections(sections)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return PDFDocument(
                filename=pdf_path.name,
                title=metadata.get("title", pdf_path.stem),
                sections=processed_sections,
                metadata=metadata,
                total_pages=total_pages,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            raise
    
    async def process_multiple_pdfs(self, pdf_paths: List[Path]) -> List[PDFDocument]:
        """Process multiple PDF files concurrently"""
        tasks = [self.process_pdf(pdf_path) for pdf_path in pdf_paths]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> Tuple[List[PDFSection], Dict, int]:
        """Extract content using PyMuPDF (fastest)"""
        sections = []
        metadata = {}
        
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    # Extract sections from page
                    page_sections = extract_sections(text, page_num + 1)
                    sections.extend(page_sections)
                
                # Extract tables if enabled
                if processing_config.extract_tables:
                    tables = page.get_tables()
                    for i, table in enumerate(tables):
                        table_text = self._table_to_text(table)
                        if table_text.strip():
                            sections.append(PDFSection(
                                title=f"Table {i+1}",
                                content=table_text,
                                page_number=page_num + 1,
                                start_pos=0,
                                end_pos=len(table_text),
                                section_type="table"
                            ))
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            raise
        
        return sections, metadata, total_pages
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> Tuple[List[PDFSection], Dict, int]:
        """Extract content using pdfplumber (better for complex layouts)"""
        sections = []
        metadata = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata = pdf.metadata
                total_pages = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    if text and text.strip():
                        page_sections = extract_sections(text, page_num + 1)
                        sections.extend(page_sections)
                    
                    # Extract tables
                    if processing_config.extract_tables:
                        tables = page.extract_tables()
                        for i, table in enumerate(tables):
                            table_text = self._table_to_text(table)
                            if table_text.strip():
                                sections.append(PDFSection(
                                    title=f"Table {i+1}",
                                    content=table_text,
                                    page_number=page_num + 1,
                                    start_pos=0,
                                    end_pos=len(table_text),
                                    section_type="table"
                                ))
        
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
            raise
        
        return sections, metadata, total_pages
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> Tuple[List[PDFSection], Dict, int]:
        """Extract content using PyPDF2 (basic extraction)"""
        sections = []
        metadata = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata = reader.metadata
                total_pages = len(reader.pages)
                
                for page_num in range(total_pages):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text and text.strip():
                        page_sections = extract_sections(text, page_num + 1)
                        sections.extend(page_sections)
        
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            raise
        
        return sections, metadata, total_pages
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table data to readable text"""
        if not table:
            return ""
        
        lines = []
        for row in table:
            # Clean and join row data
            clean_row = [str(cell).strip() if cell else "" for cell in row]
            lines.append(" | ".join(clean_row))
        
        return "\n".join(lines)
    
    async def _post_process_sections(self, sections: List[PDFSection]) -> List[PDFSection]:
        """Post-process extracted sections"""
        processed_sections = []
        
        for section in sections:
            # Clean text
            section.content = clean_text(section.content)
            
            # Skip empty sections
            if not section.content.strip():
                continue
            
            # Calculate basic importance score
            section.importance_score = self._calculate_importance(section)
            
            # Limit section size
            if len(section.content) > processing_config.max_chunk_size:
                # Split large sections
                chunks = self._split_section(section)
                processed_sections.extend(chunks)
            else:
                processed_sections.append(section)
        
        # Sort by importance
        processed_sections.sort(key=lambda x: x.importance_score, reverse=True)
        
        return processed_sections
    
    def _calculate_importance(self, section: PDFSection) -> float:
        """Calculate importance score for a section"""
        score = 0.0
        
        # Title-based scoring
        title_lower = section.title.lower()
        if any(keyword in title_lower for keyword in ["introduction", "overview", "summary"]):
            score += 0.3
        elif any(keyword in title_lower for keyword in ["conclusion", "recommendation", "next steps"]):
            score += 0.4
        
        # Content-based scoring
        content_lower = section.content.lower()
        
        # Length scoring (moderate length is preferred)
        length_score = min(len(section.content) / 1000, 1.0)
        score += length_score * 0.2
        
        # Keyword density
        keyword_count = sum(1 for word in content_lower.split() if len(word) > 5)
        score += min(keyword_count / 100, 0.3)
        
        # Section type scoring
        if section.section_type == "table":
            score += 0.2
        
        return min(score, 1.0)
    
    def _split_section(self, section: PDFSection) -> List[PDFSection]:
        """Split large sections into smaller chunks"""
        chunks = []
        content = section.content
        chunk_size = processing_config.max_chunk_size
        overlap = processing_config.chunk_overlap
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk_content = content[i:i + chunk_size]
            if chunk_content.strip():
                chunks.append(PDFSection(
                    title=f"{section.title} (Part {len(chunks) + 1})",
                    content=chunk_content,
                    page_number=section.page_number,
                    start_pos=section.start_pos + i,
                    end_pos=section.start_pos + i + len(chunk_content),
                    importance_score=section.importance_score * 0.8,  # Slightly lower for chunks
                    section_type=section.section_type
                ))
        
        return chunks
    
    def __del__(self):
        """Cleanup executor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False) 