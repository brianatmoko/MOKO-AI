from moko_config import settings
from moko_agents.core_node import CoreNode
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import threading
import os
from pathlib import Path
import faiss
from enum import Enum
import json

class IndexType(Enum):
    HNSW = "hnsw"
    IVF = "ivf"
    FLAT = "flat"
    SQ = "sq"

class StorageStrategy(Enum):
    MEMORY_MAPPED = "memory_mapped"
    COMPRESSED = "compressed"
    HYBRID = "hybrid"

@dataclass
class VectorMetadata:
    vector_id: int
    metadata: Dict[str, Any]
    created_at: float
    updated_at: float
    source: str
    file_path: Optional[str] = None

@dataclass
class SearchResult:
    vector_id: int
    similarity_score: float
    metadata: Dict[str, Any]
    file_path: Optional[str] = None

class VectorDatabase:
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.index = None
        self.dimension = 0
        self.index_type = IndexType.HNSW
        self.storage_strategy = StorageStrategy.HYBRID
        self.metadata_store: Dict[int, VectorMetadata] = {}
        self.vector_ids: List[int] = []
        self.id_to_index_map: Dict[int, int] = {}
        self.index_to_id_map: Dict[int, int] = {}
        self.is_trained = False
        self.stats = {
            "vectors_inserted": 0,
            "searches_performed": 0,
            "average_search_time_ms": 0.0,
            "last_maintenance": time.time()
        }
        
        # Setup FAISS index
        self._setup_faiss_index()
        
        print("  ✅ [VectorDatabase] FAISS vector search engine initialized")
    
    def _setup_faiss_index(self):
        """Setup FAISS index based on configuration"""
        try:
            if self.config_path:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                self.dimension = config.get("dimension", 768)
                self.index_type = IndexType(config.get("index_type", "hnsw"))
                self.storage_strategy = StorageStrategy(config.get("storage_strategy", "hybrid"))
            
            # Create index based on type
            if self.index_type == IndexType.HNSW:
                # HNSW for approximate nearest neighbor search
                self.index = faiss.IndexHNSWFlat(self.dimension)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 50
            elif self.index_type == IndexType.IVF:
                # IVF index for exact search with quantization
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 256)
                self.is_trained = False
            elif self.index_type == IndexType.SQ:
                # Scalar quantization for compressed storage
                self.index = faiss.IndexSQ(self.dimension)
            else:
                # Flat index for exact search
                self.index = faiss.IndexFlatL2(self.dimension)
            
            # Set metadata
            if hasattr(self.index, 'is_trained'):
                self.is_trained = self.index.is_trained
                
        except Exception as e:
            print(f"  ⚠️  [VectorDatabase] FAISS setup warning: {e}")
            # Create basic flat index as fallback
            self.index = faiss.IndexFlatL2(768)
            self.dimension = 768
    
    def insert_vectors(self, vectors: np.ndarray, 
                      metadata: Optional[List[VectorMetadata]] = None,
                      ids: Optional[List[int]] = None):
        """
        Insert vectors into FAISS index with metadata
        
        Args:
            vectors: numpy array of vector embeddings (n_vectors x dimension)
            metadata: Optional metadata for each vector
            ids: Optional vector IDs (auto-generated if not provided)
        """
        if self.index is None:
            raise ValueError("FAISS index not initialized")
        
        n_vectors, dim = vectors.shape
        
        if dim != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {dim}")
        
        # Generate IDs if not provided
        if ids is None:
            ids = list(range(self.stats["vectors_inserted"], self.stats["vectors_inserted"] + n_vectors))
        elif len(ids) != n_vectors:
            raise ValueError("Number of IDs must match number of vectors")
        
        # Insert vectors into FAISS index
        self.index.add(sparse_vectors)
        
        # Store metadata
        for i, vector_id in enumerate(ids):
            self.vector_ids.append(vector_id)
            self.index_to_id_map[len(self.index_to_id_map)] = vector_id
            self.id_to_index_map[vector_id] = len(self.index_to_id_map)
            
            if metadata and i < len(metadata):
                self.metadata_store[vector_id] = metadata[i]
            else:
                # Create default metadata
                self.metadata_store[vector_id] = VectorMetadata(
                    vector_id=vector_id,
                    metadata={
                        "created": time.time(),
                        "updated": time.time(),
                        "source": "api_insert",
                        "dimension": dim
                    },
                    created_at=time.time(),
                    updated_at=time.time(),
                    source="api_insert"
                )
        
        # Update statistics
        self.stats["vectors_inserted"] += n_vectors
        self.is_trained = getattr(self.index, 'is_trained', False)
        
        print(f"  ✅ [VectorDatabase] Inserted {n_vectors} vectors")
    
    def search_similar(self, query_vector: np.ndarray, top_k: int = 10,
                      threshold: float = 0.0, include_distances: bool = True) -> List[SearchResult]:
        """
        Search for similar vectors using FAISS index
        
        Args:
            query_vector: Query vector (dimension must match index)
            top_k: Number of similar vectors to return
            threshold: Similarity threshold for filtering (cosine similarity)
            include_distances: Whether to include similarity scores
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            raise ValueError("FAISS index not initialized")
        
        start_time = time.time()
        
        # Convert query to FAISS format
        if isinstance(query_vector, np.ndarray):
            sparse_vectors = faiss.vector_to_array(query_vector)
        else:
            sparse_vectors = query_vector
        
        # Perform search
        distances, indices = self.index.search(sparse_vectors, top_k)
        
        # Convert to ID-based results
        search_results = []
        
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1:  # Invalid index
                continue
            
            # Convert FAISS index back to vector ID
            vector_id = self.index_to_id_map.get(idx, idx)
            
            # Calculate cosine similarity from distances
            if include_distances and len(distances) > 0:
                similarity = self._faiss_distance_to_cosine(distances[0][i])
            else:
                similarity = 0.0
            
            # Get metadata if available
            metadata = self.metadata_store.get(vector_id)
            
            result = SearchResult(
                vector_id=vector_id,
                similarity_score=similarity,
                metadata=metadata.metadata if metadata else {},
                file_path=metadata.file_path if metadata else None
            )
            
            # Apply threshold filter
            if similarity >= threshold:
                search_results.append(result)
        
        # Update statistics
        end_time = time.time()
        search_time_ms = (end_time - start_time) * 1000
        self.stats["searches_performed"] += 1
        
        # Update average search time
        old_avg = self.stats["average_search_time_ms"]
        new_avg = (old_avg * (self.stats["searches_performed"] - 1) + search_time_ms) / self.stats["searches_performed"]
        self.stats["average_search_time_ms"] = new_avg
        
        return search_results
    
    def _faiss_distance_to_cosine(self, faiss_distance: float) -> float:
        """
        Convert FAISS distance to cosine similarity
        
        Note: Different FAISS index types use different distance metrics:
        - L2: Euclidean distance -> cosine similarity via normalization
        - Inner product: Direct negative similarity (because FAISS uses dot product)
        """
        if self.index_type == IndexType.SQ or self.index_type == IndexType.FLAT:
            # Euclidean distance (L2)
            # Normalize to cosine similarity using approximate relationship
            # For high-dimensional vectors, this approximation is reasonable
            try:
                # Normalize distance to similarity (0 to 1 range)
                similarity = max(0.0, min(1.0, 1.0 - (faiss_distance / 2.0)))
                return similarity
            except:
                return 0.0
        else:
            # Inner product distance (dot product with sign)
            # FAISS uses HNSW with inner product by default
            # Convert dot product to cosine similarity
            similarity = 1.0 - faiss_distance / 2.0
            return max(0.0, min(1.0, similarity))
    
    def search_batch(self, query_vectors: List[np.ndarray], 
                    top_k: int = 10, threshold: float = 0.0) -> List[List[SearchResult]]:
        """
        Batch search for multiple query vectors
        
        Args:
            query_vectors: List of query vectors
            top_k: Number of similar vectors to return per query
            threshold: Similarity threshold for filtering
            
        Returns:
            List of search result lists (one per query)
        """
        results = []
        
        for query_vector in query_vectors:
            query_results = self.search_similar(query_vector, top_k, threshold)
            results.append(query_results)
        
        return results
    
    def get_vector_metadata(self, vector_id: int) -> Optional[VectorMetadata]:
        """Get metadata for a specific vector ID"""
        return self.metadata_store.get(vector_id)
    
    def update_vector_metadata(self, vector_id: int, 
                              updated_metadata: Dict[str, Any]) -> bool:
        """Update metadata for a specific vector ID"""
        if vector_id in self.metadata_store:
            self.metadata_store[vector_id].metadata.update(updated_metadata)
            self.metadata_store[vector_id].updated_at = time.time()
            return True
        return False
    
    def remove_vector(self, vector_id: int) -> bool:
        """Remove a vector from the database"""
        if vector_id in self.metadata_store:
            del self.metadata_store[vector_id]
            
            # Remove from mappings (may need to rebuild index for efficiency)
            index_idx = self.id_to_index_map.get(vector_id)
            if index_idx is not None:
                del self.id_to_index_map[vector_id]
            
            self.stats["vectors_inserted"] -= 1
            return True
        return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and diagnostics"""
        return {
            "index_type": self.index_type.value,
            "dimension": self.dimension,
            "total_vectors": self.stats["vectors_inserted"],
            "total_searches": self.stats["searches_performed"],
            "average_search_time_ms": self.stats["average_search_time_ms"],
            "is_trained": self.is_trained,
            "metadata_count": len(self.metadata_store),
            "last_maintenance": self.stats["last_maintenance"]
        }
    
    def optimize_index(self, reindex: bool = False):
        """
        Optimize FAISS index for performance
        
        Args:
            reindex: Whether to rebuild index from scratch
        """
        start_time = time.time()
        
        if reindex and hasattr(self.index, 'reconstruct_index'):
            # Reconstruct index from vector database
            self._reconstruct_index()
        elif hasattr(self.index, 'hnsw'):
            # Optimize HNSW parameters
            self.index.hnsw.efSearch = max(10, min(200, self.index.hnsw.efConstruction))
        
        self.stats["last_maintenance"] = time.time()
        
        print(f"  ✅ [VectorDatabase] Index optimization completed in {(time.time() - start_time) * 1000:.2f}ms")
    
    def _reconstruct_index(self):
        """Reconstruct FAISS index from all vectors"""
        if not self.metadata_store:
            return
        
        # Prepare vectors for reconstruction
        vectors = []
        vector_ids = []
        
        for vector_id, metadata in self.metadata_store.items():
            # This is a simplified reconstruction
            # In practice, you'd store actual vectors or implement persistence
            pass
        
        # Note: Full reconstruction requires vector persistence
        # For now, this is a placeholder for future implementation
        print("  ⚠️  [VectorDatabase] Full reconstruction not implemented - requires vector persistence")

# Global vector database service instance
vector_db_instance = None

def get_vector_database(config_path: str = None) -> VectorDatabase:
    """
    Get or create global vector database instance
    
    Args:
        config_path: Optional path to FAISS configuration file
        
    Returns:
        VectorDatabase instance
    """
    global vector_db_instance
    
    if vector_db_instance is None:
        vector_db_instance = VectorDatabase(config_path)
    
    return vector_db_instance

# Convenience functions for vector database operations
def insert_vector_batch(vectors: np.ndarray, metadata: Optional[List[Dict]] = None,
                       ids: Optional[List[int]] = None):
    """Global vector database insert function"""
    db = get_vector_database()
    if db:
        vector_metadata = [VectorMetadata(
            vector_id=i if ids is None else ids[i],
            metadata=meta if metadata is not None else {},
            created_at=time.time(),
            updated_at=time.time(),
            source="batch_insert"
        ) for i in range(len(vectors))]
        return db.insert_vectors(vectors, vector_metadata, ids)
    return False

def search_similar_vectors(query_vector: np.ndarray, top_k: int = 10,
                           threshold: float = 0.0) -> List[SearchResult]:
    """Global vector database search function"""
    db = get_vector_database()
    if db:
        return db.search_similar(query_vector, top_k, threshold)
    return []