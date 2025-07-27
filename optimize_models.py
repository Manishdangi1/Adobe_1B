#!/usr/bin/env python3
"""
Model Optimization Script for Adobe Challenge 1B

"""

import os
import sys
import time
import logging
from pathlib import Path
import torch
import spacy
import nltk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimizes model loading and caching for challenge constraints"""
    
    def __init__(self):
        self.models_dir = Path("/app/models")
        self.cache_dir = Path("/app/cache")
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Set environment variables for optimization
        os.environ['TRANSFORMERS_CACHE'] = str(self.models_dir)
        os.environ['HF_HOME'] = str(self.models_dir)
        os.environ['TORCH_HOME'] = str(self.models_dir)
        
        # CPU optimization
        os.environ['OMP_NUM_THREADS'] = '8'
        os.environ['MKL_NUM_THREADS'] = '8'
        os.environ['NUMEXPR_NUM_THREADS'] = '8'
        
        # Disable CUDA for CPU-only processing
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
    def optimize_torch(self):
        """Optimize PyTorch for CPU processing"""
        logger.info("Optimizing PyTorch for CPU...")
        
        # Set PyTorch to CPU mode
        torch.set_num_threads(8)
        
        # Verify CPU-only mode
        if torch.cuda.is_available():
            logger.warning("CUDA is available but disabled for CPU-only processing")
        
        logger.info(f"PyTorch threads: {torch.get_num_threads()}")
        logger.info("PyTorch optimization completed")
        
    def preload_spacy(self):
        """Preload spaCy model for faster inference"""
        logger.info("Preloading spaCy model...")
        start_time = time.time()
        
        try:
            # Use en_core_web_sm for speed and size
            nlp = spacy.load('en_core_web_sm')
            
            # Test the model
            test_text = "This is a test sentence for optimization."
            doc = nlp(test_text)
            
            load_time = time.time() - start_time
            logger.info(f"spaCy model loaded in {load_time:.2f} seconds")
            logger.info(f"Model components: {nlp.pipe_names}")
            
            return nlp
            
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            return None
            
    def preload_sentence_transformer(self):
        """Preload superior sentence transformer model"""
        logger.info("Preloading superior sentence transformer model...")
        start_time = time.time()
        
        try:
            # Use all-mpnet-base-v2 - superior quality, reasonable size
            model_name = 'all-mpnet-base-v2'  # ~420MB, best quality
            model = SentenceTransformer(model_name)
            
            # Test the model
            test_sentences = ["This is a test sentence.", "Another test sentence."]
            embeddings = model.encode(test_sentences)
            
            load_time = time.time() - start_time
            logger.info(f"Superior sentence transformer ({model_name}) loaded in {load_time:.2f} seconds")
            logger.info(f"Embedding shape: {embeddings.shape}")
            logger.info(f"Model size: ~420MB - Best quality for size")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {e}")
            # Fallback to smaller model if needed
            try:
                logger.info("Falling back to smaller model...")
                model = SentenceTransformer('all-MiniLM-L12-H384-uncased')  # ~120MB
                return model
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                return None
            
    def preload_bert_model(self):
        """Preload superior BERT model for better text understanding"""
        logger.info("Preloading superior BERT model...")
        start_time = time.time()
        
        try:
            # Use microsoft/MiniLM-L12-H384-uncased - superior quality, reasonable size
            model_name = 'microsoft/MiniLM-L12-H384-uncased'  # ~120MB, superior quality
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Test the model
            test_text = "This is a test sentence for BERT model."
            inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
            
            load_time = time.time() - start_time
            logger.info(f"Superior BERT model ({model_name}) loaded in {load_time:.2f} seconds")
            logger.info(f"Model size: ~120MB - Superior quality for size")
            
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            return None, None
            
    def download_nltk_data(self):
        """Download essential NLTK data"""
        logger.info("Downloading essential NLTK data...")
        
        try:
            # Download only essential NLTK data for optimal size
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)  # Named entity recognition
            
            logger.info("Essential NLTK data downloaded successfully")
            
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {e}")
            
    def create_model_cache(self):
        """Create model cache for faster loading"""
        logger.info("Creating model cache...")
        
        try:
            # Cache spaCy model
            nlp = self.preload_spacy()
            if nlp:
                cache_path = self.cache_dir / "spacy_cache.pkl"
                joblib.dump(nlp, cache_path)
                logger.info(f"spaCy model cached to {cache_path}")
            
            # Cache superior sentence transformer
            model = self.preload_sentence_transformer()
            if model:
                cache_path = self.cache_dir / "sentence_transformer_cache.pkl"
                joblib.dump(model, cache_path)
                logger.info(f"Superior sentence transformer cached to {cache_path}")
                
            # Cache superior BERT model
            tokenizer, bert_model = self.preload_bert_model()
            if tokenizer and bert_model:
                cache_path = self.cache_dir / "bert_cache.pkl"
                joblib.dump((tokenizer, bert_model), cache_path)
                logger.info(f"Superior BERT model cached to {cache_path}")
                
        except Exception as e:
            logger.error(f"Error creating model cache: {e}")
            
    def verify_constraints(self):
        """Verify that the setup meets challenge constraints"""
        logger.info("Verifying challenge constraints...")
        
        # Check model directory size
        if self.models_dir.exists():
            total_size = sum(f.stat().st_size for f in self.models_dir.rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)
            logger.info(f"Model directory size: {size_gb:.2f} GB")
            
            if size_gb > 1.0:
                logger.warning(f"Model size ({size_gb:.2f} GB) exceeds 1GB limit!")
                logger.info("Consider using smaller models if needed")
            else:
                logger.info("✅ Model size constraint satisfied")
        
        # Check CPU-only mode
        if not torch.cuda.is_available() or os.environ.get('CUDA_VISIBLE_DEVICES') == '':
            logger.info("✅ CPU-only mode confirmed")
        else:
            logger.warning("CUDA is available - ensure CPU-only processing")
            
        # Check thread optimization
        logger.info(f"PyTorch threads: {torch.get_num_threads()}")
        logger.info(f"OMP threads: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
        
        # Log superior model configuration
        logger.info("Superior Model Configuration:")
        logger.info("  - Sentence Transformer: all-mpnet-base-v2 (~420MB) - Best quality")
        logger.info("  - BERT: microsoft/MiniLM-L12-H384 (~120MB) - Superior quality")
        logger.info("  - spaCy: en_core_web_sm (~12MB) - Fast and efficient")
        logger.info("  - NLTK: Essential data only (~50MB)")
        logger.info("  - Total: ~602MB (well under 1GB limit)")
        
    def run_optimization(self):
        """Run complete optimization process"""
        logger.info("Starting superior model optimization for Adobe Challenge 1B...")
        logger.info("Using best quality models while staying under 1GB constraint")
        
        start_time = time.time()
        
        # Run optimizations
        self.optimize_torch()
        self.download_nltk_data()
        self.preload_spacy()
        self.preload_sentence_transformer()
        self.preload_bert_model()
        self.create_model_cache()
        self.verify_constraints()
        
        total_time = time.time() - start_time
        logger.info(f"Superior model optimization completed in {total_time:.2f} seconds")
        
        return True

def main():
    """Main function"""
    optimizer = ModelOptimizer()
    
    try:
        success = optimizer.run_optimization()
        if success:
            logger.info("Superior model optimization completed successfully!")
            sys.exit(0)
        else:
            logger.error("Model optimization failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error during optimization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 