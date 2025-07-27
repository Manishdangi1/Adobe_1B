"""
High-performance embedding model with caching and vector search
"""
import asyncio
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from cachetools import TTLCache
import logging
from dataclasses import dataclass

from ...config.config import model_config, app_config

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding computation"""
    embeddings: np.ndarray
    texts: List[str]
    metadata: Dict[str, Any]

class EmbeddingModel:
    """High-performance embedding model with caching and vector search"""
    
    def __init__(self, model_name: str = None, cache_size: int = 1000):
        self.model_name = model_name or model_config.embedding_model
        self.device = "cuda" if torch.cuda.is_available() and app_config.enable_gpu else "cpu"
        
        # Initialize model
        self.model = None
        self._load_model()
        
        # Initialize cache
        self.cache = TTLCache(
            maxsize=cache_size,
            ttl=model_config.cache_duration
        )
        
        # FAISS index for similarity search
        self.index = None
        self.index_metadata = []
        
        logger.info(f"Initialized embedding model: {self.model_name} on {self.device}")
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to default model
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
            logger.info("Using fallback model: all-MiniLM-L6-v2")
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key for texts"""
        text_hash = hashlib.md5(''.join(texts).encode()).hexdigest()
        return f"{self.model_name}_{text_hash}"
    
    async def get_embeddings(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        """
        Get embeddings for a list of texts with caching
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            return EmbeddingResult(embeddings=np.array([]), texts=[], metadata={})
        
        # Check cache first
        cache_key = self._get_cache_key(texts)
        if cache_key in self.cache:
            logger.debug(f"Cache hit for {len(texts)} texts")
            return self.cache[cache_key]
        
        # Process in batches
        embeddings = []
        processed_texts = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Run embedding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                None, 
                self._embed_batch, 
                batch_texts
            )
            
            embeddings.extend(batch_embeddings)
            processed_texts.extend(batch_texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        result = EmbeddingResult(
            embeddings=embeddings_array,
            texts=processed_texts,
            metadata={
                "model": self.model_name,
                "device": self.device,
                "batch_size": batch_size,
                "total_texts": len(texts)
            }
        )
        
        # Cache the result
        if model_config.cache_embeddings:
            self.cache[cache_key] = result
        
        return result
    
    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts"""
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=len(texts)
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            # Return zero embeddings for failed texts
            return [np.zeros(self.model.get_sentence_embedding_dimension()) for _ in texts]
    
    async def build_search_index(self, texts: List[str], metadata: List[Dict] = None) -> None:
        """
        Build FAISS index for similarity search
        
        Args:
            texts: List of texts to index
            metadata: Optional metadata for each text
        """
        if not texts:
            return
        
        # Get embeddings
        embedding_result = await self.get_embeddings(texts)
        
        # Build FAISS index
        dimension = embedding_result.embeddings.shape[1]
        
        # Use IVF index for better performance with large datasets
        if len(texts) > 1000:
            nlist = min(100, len(texts) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Train the index
            self.index.train(embedding_result.embeddings)
        else:
            # Use simple index for smaller datasets
            self.index = faiss.IndexFlatIP(dimension)
        
        # Add vectors to index
        self.index.add(embedding_result.embeddings)
        
        # Store metadata
        self.index_metadata = metadata or [{"text": text, "index": i} for i, text in enumerate(texts)]
        
        logger.info(f"Built FAISS index with {len(texts)} texts, dimension {dimension}")
    
    async def search_similar(
        self, 
        query: str, 
        top_k: int = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar texts using FAISS
        
        Args:
            query: Query text
            top_k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of similar texts with scores
        """
        if not self.index or not query:
            return []
        
        top_k = top_k or model_config.top_k_results
        threshold = threshold or model_config.similarity_threshold
        
        # Get query embedding
        query_result = await self.get_embeddings([query])
        query_embedding = query_result.embeddings[0].reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:  # FAISS returns -1 for invalid indices
                continue
            
            if idx < len(self.index_metadata):
                result = {
                    "text": self.index_metadata[idx].get("text", ""),
                    "score": float(score),
                    "index": int(idx),
                    **self.index_metadata[idx]
                }
                results.append(result)
        
        return results
    
    async def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a list of texts
        
        Args:
            texts: List of texts
            
        Returns:
            Similarity matrix
        """
        if not texts:
            return np.array([])
        
        # Get embeddings
        embedding_result = await self.get_embeddings(texts)
        embeddings = embedding_result.embeddings
        
        # Compute similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        return similarity_matrix
    
    def save_index(self, filepath: Path) -> None:
        """Save FAISS index to file"""
        if self.index:
            faiss.write_index(self.index, str(filepath))
            
            # Save metadata
            metadata_file = filepath.with_suffix('.pkl')
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.index_metadata, f)
            
            logger.info(f"Saved index to {filepath}")
    
    def load_index(self, filepath: Path) -> None:
        """Load FAISS index from file"""
        try:
            self.index = faiss.read_index(str(filepath))
            
            # Load metadata
            metadata_file = filepath.with_suffix('.pkl')
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    self.index_metadata = pickle.load(f)
            
            logger.info(f"Loaded index from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load index from {filepath}: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "cache_size": len(self.cache),
            "index_size": len(self.index_metadata) if self.index else 0
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        self.cache.clear()
        logger.info("Cleared embedding cache")
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'cache'):
            self.cache.clear() 