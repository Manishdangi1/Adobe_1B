#!/usr/bin/env python3
"""
Challenge 1b Solution Generator
Generates solutions in the exact format expected by the challenge
"""
import json
import sys
from pathlib import Path
import PyPDF2
import re
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Section:
    """Represents a section from a PDF"""
    title: str
    content: str
    page_number: int
    start_pos: int
    end_pos: int
    importance_score: float = 0.0

class Challenge1bSolutionGenerator:
    """Generates solutions in the exact Challenge 1b format"""
    
    def __init__(self):
        self.sections = []
        self.metadata = {}
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = {
                    "title": reader.metadata.get('/Title', pdf_path.stem),
                    "author": reader.metadata.get('/Author', ''),
                    "subject": reader.metadata.get('/Subject', ''),
                    "creator": reader.metadata.get('/Creator', ''),
                    "total_pages": len(reader.pages)
                }
                
                # Extract text page by page
                full_text = ""
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    full_text += f"\n--- PAGE {i+1} ---\n{page_text}\n"
                
                return full_text, metadata
                
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return "", {}
    
    def extract_sections(self, text: str) -> List[Section]:
        """Extract sections from text with proper importance scoring"""
        sections = []
        lines = text.split('\n')
        current_section = []
        current_title = "Main Content"
        current_page = 1
        start_pos = 0
        
        # Section detection patterns
        section_patterns = [
            r'^[A-Z][A-Z\s]{3,}$',  # ALL CAPS titles
            r'^\d+\.\s+[A-Z][^.\n]+',  # Numbered sections
            r'^[A-Z][^.\n]{3,}:$',  # Title with colon
            r'^Chapter\s+\d+',  # Chapter headers
            r'^Section\s+\d+',  # Section headers
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check for page markers
            if line.startswith('--- PAGE'):
                page_match = re.search(r'PAGE (\d+)', line)
                if page_match:
                    current_page = int(page_match.group(1))
                continue
            
            if not line:
                continue
            
            # Check if line is a section header
            is_header = False
            for pattern in section_patterns:
                if re.match(pattern, line):
                    # Save current section
                    if current_section:
                        section_content = " ".join(current_section)
                        if section_content.strip():
                            sections.append(Section(
                                title=current_title,
                                content=section_content,
                                page_number=current_page,
                                start_pos=start_pos,
                                end_pos=start_pos + len(section_content),
                                importance_score=self._calculate_importance(current_title, section_content)
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
            section_content = " ".join(current_section)
            if section_content.strip():
                sections.append(Section(
                    title=current_title,
                    content=section_content,
                    page_number=current_page,
                    start_pos=start_pos,
                    end_pos=start_pos + len(section_content),
                    importance_score=self._calculate_importance(current_title, section_content)
                ))
        
        return sections
    
    def _calculate_importance(self, title: str, content: str) -> float:
        """Calculate importance score for a section"""
        score = 0.0
        
        # Title-based scoring
        title_lower = title.lower()
        if any(keyword in title_lower for keyword in ["introduction", "overview", "summary", "welcome"]):
            score += 0.3
        elif any(keyword in title_lower for keyword in ["challenge", "mission", "objective"]):
            score += 0.4
        elif any(keyword in title_lower for keyword in ["requirement", "guideline", "rule"]):
            score += 0.35
        elif any(keyword in title_lower for keyword in ["technology", "platform", "architecture"]):
            score += 0.3
        
        # Content-based scoring
        content_lower = content.lower()
        
        # Length scoring (moderate length is preferred)
        length_score = min(len(content) / 1000, 1.0)
        score += length_score * 0.2
        
        # Keyword density
        important_keywords = ["adobe", "hackathon", "challenge", "pdf", "technology", "requirement", "mission"]
        keyword_count = sum(1 for keyword in important_keywords if keyword in content_lower)
        score += min(keyword_count / len(important_keywords), 1.0) * 0.3
        
        return min(score, 1.0)
    
    def refine_text_for_persona(self, content: str, persona: str, task: str) -> str:
        """Refine text content based on persona and task"""
        # Clean and format the text
        lines = content.split('\n')
        refined_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Filter out very short lines
                # Remove excessive whitespace
                line = re.sub(r'\s+', ' ', line)
                refined_lines.append(line)
        
        refined_text = '\n'.join(refined_lines)
        
        # For HR Professional persona, focus on actionable insights
        if "hr" in persona.lower():
            # Add context for HR professionals
            if "hackathon" in content.lower():
                refined_text = f"Hackathon Challenge Context: {refined_text}"
            if "requirement" in content.lower():
                refined_text = f"Requirements Analysis: {refined_text}"
        
        return refined_text
    
    def generate_solution(self, input_file: Path, output_file: Path) -> bool:
        """Generate solution in the exact Challenge 1b format"""
        try:
            # Load input configuration
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            # Extract configuration
            challenge_info = input_data.get("challenge_info", {})
            documents = input_data.get("documents", [])
            persona = input_data.get("persona", {}).get("role", "")
            job_to_be_done = input_data.get("job_to_be_done", {}).get("task", "")
            
            print(f"🔍 Processing challenge: {challenge_info.get('challenge_id', 'Unknown')}")
            print(f"👥 Persona: {persona}")
            print(f"📋 Task: {job_to_be_done}")
            
            # Process each document
            all_sections = []
            input_documents = []
            
            for doc in documents:
                filename = doc.get("filename", "")
                title = doc.get("title", "")
                
                # Find PDF file
                pdf_path = input_file.parent / "PDFs" / filename
                if not pdf_path.exists():
                    print(f"⚠️ PDF not found: {pdf_path}")
                    continue
                
                print(f"📄 Processing: {filename}")
                
                # Extract text and metadata
                text, metadata = self.extract_text_from_pdf(pdf_path)
                if not text:
                    print(f"❌ No text extracted from {filename}")
                    continue
                
                # Extract sections
                sections = self.extract_sections(text)
                print(f"✅ Extracted {len(sections)} sections from {filename}")
                
                # Add document info to sections
                for section in sections:
                    section.document = filename
                
                all_sections.extend(sections)
                input_documents.append(filename)
            
            if not all_sections:
                print("❌ No sections extracted from any documents")
                return False
            
            # Sort sections by importance
            all_sections.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Generate extracted_sections (top sections with importance ranking)
            extracted_sections = []
            for i, section in enumerate(all_sections[:20]):  # Top 20 sections
                extracted_sections.append({
                    "document": section.document,
                    "section_title": section.title,
                    "importance_rank": i + 1,
                    "page_number": section.page_number
                })
            
            # Generate subsection_analysis (detailed analysis of top sections)
            subsection_analysis = []
            for section in all_sections[:10]:  # Top 10 for detailed analysis
                refined_text = self.refine_text_for_persona(section.content, persona, job_to_be_done)
                subsection_analysis.append({
                    "document": section.document,
                    "refined_text": refined_text[:1000],  # Limit to 1000 characters
                    "page_number": section.page_number
                })
            
            # Create the solution in exact Challenge 1b format
            solution = {
                "metadata": {
                    "input_documents": input_documents,
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "challenge_id": challenge_info.get("challenge_id", ""),
                    "test_case_name": challenge_info.get("test_case_name", ""),
                    "total_sections_extracted": len(all_sections),
                    "total_documents_processed": len(input_documents)
                },
                "extracted_sections": extracted_sections,
                "subsection_analysis": subsection_analysis
            }
            
            # Save solution
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(solution, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Solution generated: {output_file}")
            print(f"📊 Extracted {len(extracted_sections)} sections")
            print(f"🔍 Detailed analysis of {len(subsection_analysis)} sections")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            return False

def main():
    """Main function to generate Challenge 1b solution"""
    print("🚀 Challenge 1b Solution Generator")
    print("=" * 60)
    
    # Find input files
    collection_dirs = [d for d in Path(".").iterdir() if d.is_dir() and d.name.startswith("Collection")]
    
    if not collection_dirs:
        print("❌ No collection directories found")
        return
    
    print(f"📁 Found {len(collection_dirs)} collection directories")
    
    success_count = 0
    
    for collection_dir in collection_dirs:
        input_file = collection_dir / "challenge1b_input.json"
        output_file = collection_dir / "challenge1b_output.json"
        
        if not input_file.exists():
            print(f"⚠️ No input file found in {collection_dir.name}")
            continue
        
        print(f"\n🔧 Processing {collection_dir.name}...")
        
        # Generate solution
        generator = Challenge1bSolutionGenerator()
        success = generator.generate_solution(input_file, output_file)
        
        if success:
            success_count += 1
            print(f"✅ {collection_dir.name} processed successfully")
        else:
            print(f"❌ Failed to process {collection_dir.name}")
    
    print(f"\n🎉 Processing complete: {success_count}/{len(collection_dirs)} collections successful")

if __name__ == "__main__":
    main() 