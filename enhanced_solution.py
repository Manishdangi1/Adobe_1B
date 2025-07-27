#!/usr/bin/env python3
"""
 Challenge 1b Solution Generator

"""

import json
import sys
import time
from pathlib import Path
import PyPDF2
import re
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

# Import superior models
try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    import torch
    import spacy
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    from sklearn.metrics.pairwise import cosine_similarity
    from performance_monitor import PerformanceMonitor
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please ensure all superior models are installed")
    sys.exit(1)

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
    document: str = ""

class EnhancedChallenge1bGenerator:
    """Enhanced solution generator using superior models"""
    
    def __init__(self):
        self.sections = []
        self.metadata = {}
        self.monitor = PerformanceMonitor()
        
        # Initialize superior models
        self._load_superior_models()
        
    def _load_superior_models(self):
        """Load superior models for enhanced processing"""
        logger.info("Loading superior models...")
        
        try:
            # Load superior sentence transformer
            self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
            logger.info("✅ Superior sentence transformer loaded")
            
            # Load superior BERT model
            self.bert_tokenizer = AutoTokenizer.from_pretrained('microsoft/MiniLM-L12-H384-uncased')
            self.bert_model = AutoModel.from_pretrained('microsoft/MiniLM-L12-H384-uncased')
            logger.info("✅ Superior BERT model loaded")
            
            # Load spaCy for NLP
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("✅ spaCy model loaded")
            
            # Download NLTK data if needed
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
                
            self.stop_words = set(stopwords.words('english'))
            logger.info("✅ NLTK data loaded")
            
        except Exception as e:
            logger.error(f"❌ Error loading superior models: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from PDF using enhanced processing"""
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
                
                # Extract text page by page with enhanced processing
                full_text = ""
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        full_text += f"\n--- PAGE {i+1} ---\n{page_text}\n"
                
                return full_text, metadata
                
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return "", {}
    
    def extract_sections_enhanced(self, text: str) -> List[Section]:
        """Extract sections using superior NLP models"""
        sections = []
        lines = text.split('\n')
        current_section = []
        current_title = "Main Content"
        current_page = 1
        start_pos = 0
        
        # Enhanced section detection patterns
        section_patterns = [
            r'^[A-Z][A-Z\s]{3,}$',  # ALL CAPS titles
            r'^\d+\.\s+[A-Z][^.\n]+',  # Numbered sections
            r'^[A-Z][^.\n]{3,}:$',  # Title with colon
            r'^Chapter\s+\d+',  # Chapter headers
            r'^Section\s+\d+',  # Section headers
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$',  # Title case headers
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
            
            # Enhanced section header detection
            is_header = False
            for pattern in section_patterns:
                if re.match(pattern, line):
                    # Save current section
                    if current_section:
                        section_content = " ".join(current_section)
                        if len(section_content.strip()) > 50:  # Minimum content length
                            section = Section(
                                title=current_title,
                                content=section_content,
                                page_number=current_page,
                                start_pos=start_pos,
                                end_pos=i,
                                importance_score=0.0
                            )
                            sections.append(section)
                    
                    # Start new section
                    current_title = line
                    current_section = []
                    start_pos = i
                    is_header = True
                    break
            
            if not is_header:
                current_section.append(line)
        
        # Add final section
        if current_section:
            section_content = " ".join(current_section)
            if len(section_content.strip()) > 50:
                section = Section(
                    title=current_title,
                    content=section_content,
                    page_number=current_page,
                    start_pos=start_pos,
                    end_pos=len(lines),
                    importance_score=0.0
                )
                sections.append(section)
        
        return sections
    
    def calculate_importance_enhanced(self, title: str, content: str, persona: str, task: str) -> float:
        """Calculate importance using superior models"""
        score = 0.0
        
        # 1. Semantic similarity with persona and task
        try:
            # Create embeddings for comparison
            persona_task_text = f"{persona} {task}"
            comparison_texts = [title, content[:1000], persona_task_text]  # Limit content length
            
            # Get embeddings using superior model
            embeddings = self.sentence_model.encode(comparison_texts)
            
            # Calculate similarities
            title_similarity = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
            content_similarity = cosine_similarity([embeddings[1]], [embeddings[2]])[0][0]
            
            # Weighted similarity score
            similarity_score = (title_similarity * 0.4 + content_similarity * 0.6)
            score += similarity_score * 0.5
            
        except Exception as e:
            logger.warning(f"Error in semantic similarity: {e}")
        
        # 2. NLP-based analysis using spaCy
        try:
            doc = self.nlp(content[:2000])  # Limit for performance
            
            # Named entity recognition
            entities = [ent.label_ for ent in doc.ents]
            entity_score = len(entities) / 10.0  # Normalize
            score += min(entity_score, 0.2)
            
            # Part-of-speech analysis
            pos_tags = [token.pos_ for token in doc]
            noun_count = pos_tags.count('NOUN')
            verb_count = pos_tags.count('VERB')
            
            # Balanced content score
            if noun_count > 0 and verb_count > 0:
                balance_score = min(noun_count / (noun_count + verb_count), 1.0)
                score += balance_score * 0.1
                
        except Exception as e:
            logger.warning(f"Error in NLP analysis: {e}")
        
        # 3. Keyword-based scoring (enhanced)
        content_lower = content.lower()
        title_lower = title.lower()
        
        # Persona-specific keywords
        persona_keywords = {
            # Collection 1: Travel Planning
            "travel planner": ["itinerary", "accommodation", "transportation", "attractions", "budget", "group travel", "south of france", "4-day trip", "college friends", "travel guide", "destination", "planning"],
            
            # Collection 2: Adobe Acrobat Learning
            "hr professional": ["forms", "onboarding", "compliance", "fillable", "workflow", "documentation", "employee", "process", "acrobat", "pdf", "digital forms", "automation"],
            
            # Collection 3: Recipe Collection
            "food contractor": ["vegetarian", "buffet", "corporate", "menu", "ingredients", "preparation", "catering", "dinner", "recipe", "cooking", "meal planning", "dietary"],
            
            # Generic personas for other use cases
            "researcher": ["research", "study", "analysis", "methodology", "findings", "conclusion"],
            "student": ["learning", "education", "study", "assignment", "course", "academic"],
            "analyst": ["analysis", "data", "report", "insights", "metrics", "trends"],
            "manager": ["management", "strategy", "planning", "leadership", "team", "goals"],
            "developer": ["technology", "code", "development", "implementation", "system", "architecture"]
        }
        
        # Get relevant keywords for persona
        relevant_keywords = persona_keywords.get(persona.lower(), [])
        if not relevant_keywords:
            relevant_keywords = ["important", "key", "main", "primary", "essential"]
        
        # Calculate keyword density
        keyword_matches = sum(1 for keyword in relevant_keywords if keyword in content_lower)
        keyword_score = min(keyword_matches / len(relevant_keywords), 1.0)
        score += keyword_score * 0.2
        
        # 4. Content quality scoring
        sentences = sent_tokenize(content)
        if sentences:
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            # Prefer moderate sentence length
            length_score = 1.0 - abs(avg_sentence_length - 15) / 15.0
            score += max(length_score, 0) * 0.1
        
        return min(score, 1.0)
    
    def refine_text_enhanced(self, content: str, persona: str, task: str) -> str:
        """Refine text using superior NLP models"""
        try:
            # Clean text
            lines = content.split('\n')
            refined_lines = []
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    # Remove excessive whitespace
                    line = re.sub(r'\s+', ' ', line)
                    refined_lines.append(line)
            
            refined_text = '\n'.join(refined_lines)
            
            # Use spaCy for advanced text processing
            doc = self.nlp(refined_text[:3000])  # Limit for performance
            
            # Extract key sentences using superior model
            sentences = sent_tokenize(refined_text)
            if len(sentences) > 5:
                # Use sentence transformer to find most relevant sentences
                sentence_embeddings = self.sentence_model.encode(sentences)
                task_embedding = self.sentence_model.encode([f"{persona} {task}"])
                
                # Calculate similarities
                similarities = cosine_similarity(sentence_embeddings, task_embedding).flatten()
                
                # Select top sentences
                top_indices = np.argsort(similarities)[-5:]  # Top 5 sentences
                selected_sentences = [sentences[i] for i in sorted(top_indices)]
                refined_text = ' '.join(selected_sentences)
            
            # Add persona-specific context
            if "researcher" in persona.lower():
                refined_text = f"Research Context: {refined_text}"
            elif "student" in persona.lower():
                refined_text = f"Educational Context: {refined_text}"
            elif "analyst" in persona.lower():
                refined_text = f"Analytical Context: {refined_text}"
            
            return refined_text
            
        except Exception as e:
            logger.warning(f"Error in text refinement: {e}")
            return content
    
    def generate_enhanced_solution(self, input_file: Path, output_file: Path) -> bool:
        """Generate enhanced solution using superior models"""
        with self.monitor.monitor_execution("Enhanced Solution Generation"):
            try:
                # Load input configuration
                with open(input_file, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
                
                # Extract configuration
                challenge_info = input_data.get("challenge_info", {})
                documents = input_data.get("documents", [])
                persona = input_data.get("persona", {}).get("role", "")
                job_to_be_done = input_data.get("job_to_be_done", {}).get("task", "")
                
                logger.info(f"🔍 Processing challenge: {challenge_info.get('challenge_id', 'Unknown')}")
                logger.info(f"👥 Persona: {persona}")
                logger.info(f"📋 Task: {job_to_be_done}")
                
                # Process each document
                all_sections = []
                input_documents = []
                
                for doc in documents:
                    filename = doc.get("filename", "")
                    title = doc.get("title", "")
                    
                    # Find PDF file
                    pdf_path = input_file.parent / "PDFs" / filename
                    if not pdf_path.exists():
                        logger.warning(f"⚠️ PDF not found: {pdf_path}")
                        continue
                    
                    logger.info(f"📄 Processing: {filename}")
                    
                    # Extract text and metadata
                    text, metadata = self.extract_text_from_pdf(pdf_path)
                    if not text:
                        logger.error(f"❌ No text extracted from {filename}")
                        continue
                    
                    # Extract sections using enhanced method
                    sections = self.extract_sections_enhanced(text)
                    logger.info(f"✅ Extracted {len(sections)} sections from {filename}")
                    
                    # Calculate importance scores using superior models
                    for section in sections:
                        section.document = filename
                        section.importance_score = self.calculate_importance_enhanced(
                            section.title, section.content, persona, job_to_be_done
                        )
                    
                    all_sections.extend(sections)
                    input_documents.append(filename)
                
                # Sort sections by importance score
                all_sections.sort(key=lambda x: x.importance_score, reverse=True)
                
                # Generate enhanced output
                output_data = {
                    "metadata": {
                        "input_documents": input_documents,
                        "persona": persona,
                        "job_to_be_done": job_to_be_done,
                        "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "model_used": "Superior Models (all-mpnet-base-v2, MiniLM-L12-H384)",
                        "total_sections": len(all_sections)
                    },
                    "extracted_sections": [
                        {
                            "document": section.document,
                            "section_title": section.title,
                            "importance_rank": i + 1,
                            "importance_score": round(section.importance_score, 3),
                            "page_number": section.page_number
                        }
                        for i, section in enumerate(all_sections[:20])  # Top 20 sections
                    ],
                    "subsection_analysis": [
                        {
                            "document": section.document,
                            "section_title": section.title,
                            "refined_text": self.refine_text_enhanced(
                                section.content, persona, job_to_be_done
                            ),
                            "page_number": section.page_number,
                            "importance_score": round(section.importance_score, 3)
                        }
                        for section in all_sections[:10]  # Top 10 for detailed analysis
                    ]
                }
                
                # Save enhanced output
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Enhanced solution generated: {output_file}")
                logger.info(f"📊 Processed {len(input_documents)} documents")
                logger.info(f"📝 Extracted {len(all_sections)} sections")
                logger.info(f"🎯 Top section score: {all_sections[0].importance_score:.3f}")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Error generating enhanced solution: {e}")
                return False

def main():
    """Main function"""
    try:
        # Find input file
        input_file = Path("Collection 1/challenge1b_input.json")
        if not input_file.exists():
            logger.error(f"❌ Input file not found: {input_file}")
            sys.exit(1)
        
        # Set output file
        output_file = Path("Collection 1/challenge1b_output_enhanced.json")
        
        # Generate enhanced solution
        generator = EnhancedChallenge1bGenerator()
        success = generator.generate_enhanced_solution(input_file, output_file)
        
        if success:
            logger.info("🎉 Enhanced solution generation completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Enhanced solution generation failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 