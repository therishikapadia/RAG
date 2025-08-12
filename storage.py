import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import faiss


class FaissStore:
    """FAISS-based vector store for document retrieval."""
    
    def __init__(self, dim: int, index_path: str = "./faiss.index", 
                 doc_store_path: str = "./doc_store.json", 
                 id_map_path: str = "./id_map.json"):
        """
        Initialize FAISS store.
        
        Args:
            dim: Vector dimension
            index_path: Path to save/load FAISS index
            doc_store_path: Path to save/load document store
            id_map_path: Path to save/load ID mapping
        """
        self.dim = dim
        self.index_path = index_path
        self.doc_store_path = doc_store_path
        self.id_map_path = id_map_path
        
        # Initialize FAISS index (using Inner Product for normalized vectors)
        self.index = faiss.IndexFlatIP(dim)
        
        # Document storage: doc_id -> {"text": str, "metadata": dict}
        self.doc_store: Dict[str, Dict[str, Any]] = {}
        
        # ID mapping: faiss_idx -> doc_id
        self.id_map: Dict[int, str] = {}
        
        # Load existing data if available
        self.load()
    
    def add(self, doc_ids: List[str], vectors: np.ndarray, 
            texts: List[str], metadatas: List[Dict[str, Any]]):
        """
        Add documents to the store.
        
        Args:
            doc_ids: List of document IDs
            vectors: Numpy array of vectors (shape: [n, dim])
            texts: List of text chunks
            metadatas: List of metadata dictionaries
        """
        if len(doc_ids) != len(texts) != len(metadatas) != vectors.shape[0]:
            raise ValueError("All inputs must have the same length")
        
        # Add vectors to FAISS index
        vectors = vectors.astype('float32')
        start_idx = self.index.ntotal
        self.index.add(vectors)
        
        # Update document store and ID mapping
        for i, (doc_id, text, meta) in enumerate(zip(doc_ids, texts, metadatas)):
            faiss_idx = start_idx + i
            
            # Store document
            self.doc_store[doc_id] = {
                "text": text,
                "metadata": meta
            }
            
            # Update ID mapping
            self.id_map[faiss_idx] = doc_id
        
        # Save to disk
        self.save()
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents.
        
        Args:
            query_vector: Query vector (shape: [dim])
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score, metadata) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure query vector is the right shape and type
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype('float32')
        
        # Search FAISS index
        scores, faiss_indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        results = []
        for score, faiss_idx in zip(scores[0], faiss_indices[0]):
            if faiss_idx == -1:  # No more results
                break
                
            doc_id = self.id_map.get(faiss_idx)
            if doc_id and doc_id in self.doc_store:
                metadata = self.doc_store[doc_id]["metadata"]
                results.append((doc_id, float(score), metadata))
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        return self.doc_store.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete document by ID.
        Note: This doesn't remove from FAISS index (would require rebuilding),
        just from doc_store and id_map.
        """
        if doc_id not in self.doc_store:
            return False
        
        # Remove from doc store
        del self.doc_store[doc_id]
        
        # Remove from ID mapping
        faiss_idx_to_remove = None
        for faiss_idx, stored_doc_id in self.id_map.items():
            if stored_doc_id == doc_id:
                faiss_idx_to_remove = faiss_idx
                break
        
        if faiss_idx_to_remove is not None:
            del self.id_map[faiss_idx_to_remove]
        
        self.save()
        return True
    
    def save(self):
        """Save index and metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save document store
        with open(self.doc_store_path, 'w', encoding='utf-8') as f:
            json.dump(self.doc_store, f, ensure_ascii=False, indent=2)
        
        # Save ID mapping (convert int keys to strings for JSON)
        id_map_serializable = {str(k): v for k, v in self.id_map.items()}
        with open(self.id_map_path, 'w', encoding='utf-8') as f:
            json.dump(id_map_serializable, f, ensure_ascii=False, indent=2)
    
    def load(self):
        """Load index and metadata from disk."""
        # Load FAISS index
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Failed to load FAISS index: {e}")
                self.index = faiss.IndexFlatIP(self.dim)
        
        # Load document store
        if os.path.exists(self.doc_store_path):
            try:
                with open(self.doc_store_path, 'r', encoding='utf-8') as f:
                    self.doc_store = json.load(f)
                print(f"Loaded {len(self.doc_store)} documents")
            except Exception as e:
                print(f"Failed to load document store: {e}")
                self.doc_store = {}
        
        # Load ID mapping
        if os.path.exists(self.id_map_path):
            try:
                with open(self.id_map_path, 'r', encoding='utf-8') as f:
                    id_map_str = json.load(f)
                    # Convert string keys back to integers
                    self.id_map = {int(k): v for k, v in id_map_str.items()}
                print(f"Loaded ID mapping with {len(self.id_map)} entries")
            except Exception as e:
                print(f"Failed to load ID mapping: {e}")
                self.id_map = {}
    
    def clear(self):
        """Clear all data."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.doc_store = {}
        self.id_map = {}
        
        # Remove files
        for path in [self.index_path, self.doc_store_path, self.id_map_path]:
            if os.path.exists(path):
                os.remove(path)
    
    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_vectors": self.index.ntotal,
            "total_documents": len(self.doc_store),
            "dimension": self.dim,
            "id_mappings": len(self.id_map)
        }