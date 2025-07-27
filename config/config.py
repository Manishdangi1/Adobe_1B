"""
Configuration settings for Challenge 1b PDF Analysis Application
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """ML Model Configuration"""
    # Sentence Transformer Models
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_model: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Text Processing
    max_chunk_size: int = 512
    chunk_overlap: int = 50
    max_sections_per_doc: int = 20
    
    # Vector Search
    faiss_index_type: str = "IVF100,Flat"
    similarity_threshold: float = 0.7
    top_k_results: int = 10
    
    # Caching
    cache_embeddings: bool = True
    cache_duration: int = 3600  # 1 hour

@dataclass
class ProcessingConfig:
    """Document Processing Configuration"""
    # PDF Processing
    pdf_engine: str = "pymupdf"  # pymupdf, pdfplumber, PyPDF2
    enable_ocr: bool = True
    ocr_language: str = "eng"
    
    # Text Extraction
    extract_images: bool = False
    extract_tables: bool = True
    preserve_formatting: bool = True
    
    # Async Processing
    max_workers: int = 4
    batch_size: int = 5
    timeout_seconds: int = 300

@dataclass
class CollectionConfig:
    """Collection-specific configurations"""
    collections: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.collections is None:
            self.collections = {
                "round_1b_001": {  # Recipe Collection
                    "name": "Recipe Collection",
                    "persona": "Food Contractor",
                    "task": "Prepare vegetarian buffet-style dinner menu for corporate gathering",
                    "keywords": ["vegetarian", "buffet", "corporate", "dinner", "menu", "recipes"],
                    "model_focus": "culinary"
                },
                "round_1b_002": {  # Travel Planning
                    "name": "Travel Planning",
                    "persona": "Travel Planner", 
                    "task": "Plan a 4-day trip for 10 college friends to South of France",
                    "keywords": ["travel", "France", "itinerary", "accommodation", "activities", "budget"],
                    "model_focus": "travel"
                },
                "round_1b_003": {  # Adobe Acrobat Learning
                    "name": "Adobe Acrobat Learning",
                    "persona": "HR Professional",
                    "task": "Create and manage fillable forms for onboarding and compliance",
                    "keywords": ["forms", "onboarding", "compliance", "Adobe", "Acrobat", "HR"],
                    "model_focus": "technical"
                }
            }

@dataclass
class AppConfig:
    """Main Application Configuration"""
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    collections_dir: Path = base_dir / "Collection"
    output_dir: Path = base_dir / "outputs"
    cache_dir: Path = base_dir / "cache"
    models_dir: Path = base_dir / "models"
    
    # API Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "app.log"
    
    # Performance
    enable_gpu: bool = False
    memory_limit_gb: int = 8
    max_file_size_mb: int = 100
    
    def __post_init__(self):
        # Create directories if they don't exist
        for path in [self.output_dir, self.cache_dir, self.models_dir]:
            path.mkdir(parents=True, exist_ok=True)

# Global configuration instances
model_config = ModelConfig()
processing_config = ProcessingConfig()
collection_config = CollectionConfig()
app_config = AppConfig()

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "production":
    app_config.debug = False
    app_config.log_level = "WARNING"
    processing_config.max_workers = 8
    model_config.cache_embeddings = True

if os.getenv("ENABLE_GPU") == "true":
    app_config.enable_gpu = True
    model_config.embedding_model = "all-mpnet-base-v2"  # Better model for GPU

# Collection-specific model configurations
COLLECTION_MODELS = {
    "round_1b_001": {  # Recipe Collection
        "embedding_model": "all-MiniLM-L6-v2",
        "keywords": ["recipe", "ingredients", "cooking", "vegetarian", "buffet"],
        "ner_labels": ["INGREDIENT", "COOKING_METHOD", "CUISINE_TYPE", "DIETARY_RESTRICTION"]
    },
    "round_1b_002": {  # Travel Planning
        "embedding_model": "all-mpnet-base-v2", 
        "keywords": ["destination", "accommodation", "transportation", "activities", "budget"],
        "ner_labels": ["LOCATION", "DURATION", "COST", "ACTIVITY", "ACCOMMODATION"]
    },
    "round_1b_003": {  # Adobe Acrobat Learning
        "embedding_model": "all-mpnet-base-v2",
        "keywords": ["forms", "workflow", "compliance", "onboarding", "Adobe"],
        "ner_labels": ["SOFTWARE_FEATURE", "WORKFLOW_STEP", "COMPLIANCE_REQUIREMENT", "FORM_FIELD"]
    }
} 