"""
Main analysis engine for Challenge 1b PDF analysis
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

from .pdf_processor import PDFProcessor, PDFDocument, PDFSection
from ..models.embedding_model import EmbeddingModel
from ..utils.text_utils import extract_keywords, calculate_text_similarity, extract_entities
from ...config.config import model_config, collection_config, COLLECTION_MODELS

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Result of PDF analysis"""
    metadata: Dict[str, Any]
    extracted_sections: List[Dict[str, Any]]
    subsection_analysis: List[Dict[str, Any]]
    processing_time: float
    total_documents: int
    total_sections: int

class AnalysisEngine:
    """Main analysis engine for Challenge 1b"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_model = None  # Lazy initialization
        self._initialized = False
    
    async def initialize(self):
        """Initialize the analysis engine"""
        if self._initialized:
            return
        
        # Initialize embedding model
        self.embedding_model = EmbeddingModel()
        self._initialized = True
        logger.info("Analysis engine initialized")
    
    async def analyze_collection(
        self, 
        collection_path: Path, 
        challenge_id: str,
        persona: str,
        job_to_be_done: str
    ) -> AnalysisResult:
        """
        Analyze a collection of PDFs based on persona and task
        
        Args:
            collection_path: Path to collection directory
            challenge_id: Challenge identifier
            persona: User persona
            job_to_be_done: Task description
            
        Returns:
            AnalysisResult with extracted content and analysis
        """
        await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        # Find PDF files
        pdf_dir = collection_path / "PDFs"
        if not pdf_dir.exists():
            raise ValueError(f"PDF directory not found: {pdf_dir}")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_dir}")
        
        logger.info(f"Processing {len(pdf_files)} PDF files for challenge {challenge_id}")
        
        # Process PDFs
        documents = await self.pdf_processor.process_multiple_pdfs(pdf_files)
        
        # Filter out failed documents
        successful_docs = [doc for doc in documents if isinstance(doc, PDFDocument)]
        failed_docs = [doc for doc in documents if isinstance(doc, Exception)]
        
        if failed_docs:
            logger.warning(f"Failed to process {len(failed_docs)} documents")
        
        # Extract all sections
        all_sections = []
        for doc in successful_docs:
            for section in doc.sections:
                all_sections.append({
                    "document": doc.filename,
                    "section": section,
                    "doc_metadata": doc.metadata
                })
        
        # Analyze sections based on persona and task
        analyzed_sections = await self._analyze_sections_for_persona(
            all_sections, persona, job_to_be_done, challenge_id
        )
        
        # Rank sections by importance
        ranked_sections = await self._rank_sections_by_importance(
            analyzed_sections, persona, job_to_be_done
        )
        
        # Generate subsection analysis
        subsection_analysis = await self._generate_subsection_analysis(
            ranked_sections, persona, job_to_be_done
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Prepare metadata
        metadata = {
            "input_documents": [doc.filename for doc in successful_docs],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "challenge_id": challenge_id,
            "processing_time": processing_time,
            "total_documents": len(successful_docs),
            "total_sections": len(all_sections),
            "failed_documents": len(failed_docs),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return AnalysisResult(
            metadata=metadata,
            extracted_sections=ranked_sections,
            subsection_analysis=subsection_analysis,
            processing_time=processing_time,
            total_documents=len(successful_docs),
            total_sections=len(all_sections)
        )
    
    async def _analyze_sections_for_persona(
        self, 
        sections: List[Dict], 
        persona: str, 
        job_to_be_done: str,
        challenge_id: str
    ) -> List[Dict]:
        """Analyze sections based on persona and task requirements"""
        analyzed_sections = []
        
        # Get collection-specific configuration
        collection_config = COLLECTION_MODELS.get(challenge_id, {})
        keywords = collection_config.get("keywords", [])
        
        # Build search index for similarity search
        section_texts = [item["section"].content for item in sections]
        section_metadata = [
            {
                "document": item["document"],
                "section_title": item["section"].title,
                "page_number": item["section"].page_number,
                "section_type": item["section"].section_type,
                "original_index": i
            }
            for i, item in enumerate(sections)
        ]
        
        await self.embedding_model.build_search_index(section_texts, section_metadata)
        
        # Analyze each section
        for i, item in enumerate(sections):
            section = item["section"]
            
            # Extract keywords and entities
            keywords_found = extract_keywords(section.content, top_n=10)
            entities = extract_entities(section.content)
            
            # Calculate relevance to persona and task
            relevance_score = await self._calculate_relevance_score(
                section.content, persona, job_to_be_done, keywords
            )
            
            # Find similar sections
            similar_sections = await self.embedding_model.search_similar(
                section.content, top_k=5, threshold=0.6
            )
            
            analyzed_sections.append({
                "document": item["document"],
                "section_title": section.title,
                "content": section.content,
                "page_number": section.page_number,
                "importance_rank": section.importance_score,
                "relevance_score": relevance_score,
                "keywords": keywords_found,
                "entities": entities,
                "similar_sections": similar_sections,
                "section_type": section.section_type,
                "analysis_index": i
            })
        
        return analyzed_sections
    
    async def _calculate_relevance_score(
        self, 
        content: str, 
        persona: str, 
        job_to_be_done: str, 
        keywords: List[str]
    ) -> float:
        """Calculate relevance score for content based on persona and task"""
        score = 0.0
        
        # Convert to lowercase for matching
        content_lower = content.lower()
        persona_lower = persona.lower()
        job_lower = job_to_be_done.lower()
        
        # Keyword matching
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content_lower)
        score += min(keyword_matches / len(keywords), 1.0) * 0.4
        
        # Persona-specific scoring
        if "travel" in persona_lower and any(word in content_lower for word in ["destination", "accommodation", "itinerary", "travel"]):
            score += 0.3
        elif "hr" in persona_lower and any(word in content_lower for word in ["form", "compliance", "onboarding", "workflow"]):
            score += 0.3
        elif "food" in persona_lower and any(word in content_lower for word in ["recipe", "ingredient", "cooking", "menu"]):
            score += 0.3
        
        # Task-specific scoring
        task_keywords = job_lower.split()
        task_matches = sum(1 for word in task_keywords if len(word) > 3 and word in content_lower)
        score += min(task_matches / len([w for w in task_keywords if len(w) > 3]), 1.0) * 0.3
        
        return min(score, 1.0)
    
    async def _rank_sections_by_importance(
        self, 
        sections: List[Dict], 
        persona: str, 
        job_to_be_done: str
    ) -> List[Dict]:
        """Rank sections by importance and relevance"""
        for section in sections:
            # Combine importance and relevance scores
            combined_score = (
                section["importance_rank"] * 0.4 +
                section["relevance_score"] * 0.6
            )
            section["combined_score"] = combined_score
        
        # Sort by combined score
        ranked_sections = sorted(sections, key=lambda x: x["combined_score"], reverse=True)
        
        # Add ranking
        for i, section in enumerate(ranked_sections):
            section["importance_rank"] = i + 1
        
        return ranked_sections
    
    async def _generate_subsection_analysis(
        self, 
        sections: List[Dict], 
        persona: str, 
        job_to_be_done: str
    ) -> List[Dict]:
        """Generate detailed analysis for top sections"""
        subsection_analysis = []
        
        # Take top sections for detailed analysis
        top_sections = sections[:model_config.max_sections_per_doc]
        
        for section in top_sections:
            # Refine text based on persona and task
            refined_text = await self._refine_text_for_persona(
                section["content"], persona, job_to_be_done
            )
            
            # Generate insights
            insights = await self._generate_insights(
                section, persona, job_to_be_done
            )
            
            subsection_analysis.append({
                "document": section["document"],
                "refined_text": refined_text,
                "page_number": section["page_number"],
                "insights": insights,
                "relevance_score": section["relevance_score"],
                "keywords": section["keywords"],
                "entities": section["entities"]
            })
        
        return subsection_analysis
    
    async def _refine_text_for_persona(
        self, 
        content: str, 
        persona: str, 
        job_to_be_done: str
    ) -> str:
        """Refine text content based on persona and task requirements"""
        # For now, return cleaned content
        # In a more advanced implementation, this could use summarization models
        lines = content.split('\n')
        refined_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Filter out very short lines
                refined_lines.append(line)
        
        return '\n'.join(refined_lines)
    
    async def _generate_insights(
        self, 
        section: Dict, 
        persona: str, 
        job_to_be_done: str
    ) -> List[str]:
        """Generate insights for a section based on persona and task"""
        insights = []
        
        # Extract key information based on persona
        if "travel" in persona.lower():
            # Look for travel-related insights
            if any(word in section["content"].lower() for word in ["accommodation", "hotel", "stay"]):
                insights.append("Contains accommodation information")
            if any(word in section["content"].lower() for word in ["transportation", "transport", "travel"]):
                insights.append("Contains transportation details")
            if any(word in section["content"].lower() for word in ["activity", "attraction", "visit"]):
                insights.append("Contains activity recommendations")
        
        elif "hr" in persona.lower():
            # Look for HR-related insights
            if any(word in section["content"].lower() for word in ["form", "document", "template"]):
                insights.append("Contains form templates or examples")
            if any(word in section["content"].lower() for word in ["compliance", "regulation", "policy"]):
                insights.append("Contains compliance information")
            if any(word in section["content"].lower() for word in ["workflow", "process", "procedure"]):
                insights.append("Contains workflow procedures")
        
        elif "food" in persona.lower():
            # Look for food-related insights
            if any(word in section["content"].lower() for word in ["recipe", "ingredient", "cooking"]):
                insights.append("Contains recipe information")
            if any(word in section["content"].lower() for word in ["vegetarian", "vegan", "dietary"]):
                insights.append("Contains dietary restriction information")
            if any(word in section["content"].lower() for word in ["buffet", "menu", "serving"]):
                insights.append("Contains menu planning information")
        
        # Add general insights
        if section["relevance_score"] > 0.8:
            insights.append("Highly relevant to the task")
        elif section["relevance_score"] > 0.6:
            insights.append("Moderately relevant to the task")
        
        if len(section["keywords"]) > 5:
            insights.append("Contains multiple relevant keywords")
        
        return insights
    
    def save_analysis_result(self, result: AnalysisResult, output_path: Path) -> None:
        """Save analysis result to JSON file"""
        # Convert to serializable format
        output_data = {
            "metadata": result.metadata,
            "extracted_sections": [
                {
                    "document": section["document"],
                    "section_title": section["section_title"],
                    "importance_rank": section["importance_rank"],
                    "page_number": section["page_number"]
                }
                for section in result.extracted_sections
            ],
            "subsection_analysis": [
                {
                    "document": analysis["document"],
                    "refined_text": analysis["refined_text"],
                    "page_number": analysis["page_number"]
                }
                for analysis in result.subsection_analysis
            ]
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis result saved to {output_path}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.embedding_model:
            self.embedding_model.clear_cache()
        logger.info("Analysis engine cleaned up") 