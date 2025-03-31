#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document embedding module for converting text to vector representations.
This module handles creating and saving embeddings for document chunks.
"""

import os
import sys
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import glob
import pickle
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    import torch
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not installed. Will use placeholder embeddings.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import FAISS for vector similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not installed. Will use slower numpy-based search.")
    FAISS_AVAILABLE = False

class DocumentEmbedder:
    """Class for creating and managing document embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the document embedder.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            device: Device to run the model on (cuda, cpu)
        """
        self.model_name = model_name
        self.embedding_size = None
        self.embeddings = {}  # chunk_id -> embedding
        self.chunk_data = {}  # chunk_id -> chunk info
        self.index = None
        
        # Set up the device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu" if SENTENCE_TRANSFORMERS_AVAILABLE else None
        else:
            self.device = device
        
        # Initialize the model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Initializing sentence-transformer model: {model_name} on {self.device}")
                self.model = SentenceTransformer(model_name, device=self.device)
                self.embedding_size = self.model.get_sentence_embedding_dimension()
            except Exception as e:
                logger.error(f"Error initializing sentence-transformer model: {str(e)}")
                self.model = None
        else:
            logger.warning("Using placeholder embeddings as sentence-transformers is not available")
            self.model = None
            self.embedding_size = 384  # Default dimension for placeholder embeddings
    
    def _generate_placeholder_embedding(self, text: str) -> np.ndarray:
        """
        Generate a placeholder embedding for text.
        This is used when sentence-transformers is not available.
        
        Args:
            text: Text to generate a placeholder embedding for
            
        Returns:
            A numpy array as a placeholder embedding
        """
        # Create a deterministic but different embedding for each text
        text_hash = sum(ord(c) for c in text)
        np.random.seed(text_hash)
        return np.random.randn(self.embedding_size).astype(np.float32)
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as a numpy array
        """
        if not text or text.isspace():
            # Return a zero vector for empty text
            return np.zeros(self.embedding_size, dtype=np.float32)
        
        if self.model is not None:
            try:
                with torch.no_grad():
                    embedding = self.model.encode(text, convert_to_numpy=True)
                return embedding
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                # Fall back to placeholder
                return self._generate_placeholder_embedding(text)
        else:
            return self._generate_placeholder_embedding(text)
    
    def embed_chunks(self, processed_docs_dir: str, embeddings_dir: str):
        """
        Create embeddings for all chunks in a processed documents directory.
        
        Args:
            processed_docs_dir: Directory with processed documents
            embeddings_dir: Directory to save embeddings
        """
        logger.info(f"Creating embeddings for chunks in {processed_docs_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Load chunks
        chunks_dir = os.path.join(processed_docs_dir, "chunks")
        if not os.path.exists(chunks_dir):
            logger.error(f"Chunks directory not found: {chunks_dir}")
            return
        
        # Get list of chunk files
        chunk_files = glob.glob(os.path.join(chunks_dir, "*.json"))
        logger.info(f"Found {len(chunk_files)} chunks to embed")
        
        # Process chunks
        embeddings_list = []
        chunk_ids = []
        metadata_list = []
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r') as f:
                    chunk_data = json.load(f)
                
                chunk_id = chunk_data.get("chunk_id")
                text = chunk_data.get("text", "")
                metadata = chunk_data.get("metadata", {})
                
                # Skip if no text content
                if not text or not chunk_id:
                    continue
                
                # Create embedding
                embedding = self.embed_text(text)
                
                # Store data
                self.embeddings[chunk_id] = embedding
                self.chunk_data[chunk_id] = chunk_data
                
                # Add to lists for index creation
                embeddings_list.append(embedding)
                chunk_ids.append(chunk_id)
                metadata_list.append(metadata)
                
            except Exception as e:
                logger.error(f"Error processing chunk file {chunk_file}: {str(e)}")
        
        # Create FAISS index
        if embeddings_list and FAISS_AVAILABLE:
            self._create_faiss_index(np.array(embeddings_list))
        
        # Save embeddings and metadata
        self._save_embeddings(embeddings_dir, chunk_ids, embeddings_list, metadata_list)
        
        logger.info(f"Created embeddings for {len(embeddings_list)} chunks")
    
    def _create_faiss_index(self, embeddings_array: np.ndarray):
        """
        Create a FAISS index for fast similarity search.
        
        Args:
            embeddings_array: Array of embeddings
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available. Skipping index creation.")
            return
        
        try:
            # Ensure the embeddings are in float32 format
            embeddings_array = embeddings_array.astype(np.float32)
            
            # Create and train the index
            self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
            self.index.add(embeddings_array)
            
            logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            self.index = None
    
    def _save_embeddings(self, 
                         output_dir: str, 
                         chunk_ids: List[str], 
                         embeddings_list: List[np.ndarray],
                         metadata_list: List[Dict]):
        """
        Save embeddings and metadata to disk.
        
        Args:
            output_dir: Directory to save embeddings
            chunk_ids: List of chunk IDs
            embeddings_list: List of embeddings
            metadata_list: List of chunk metadata
        """
        # Save embeddings in the required format
        try:
            # Convert to a single numpy array
            embeddings_array = np.array(embeddings_list)
            
            # Save the embeddings array
            np.save(os.path.join(output_dir, "embeddings.npy"), embeddings_array)
            
            # Save chunk IDs and metadata
            with open(os.path.join(output_dir, "chunk_ids.json"), 'w') as f:
                json.dump(chunk_ids, f)
            
            with open(os.path.join(output_dir, "chunk_metadata.json"), 'w') as f:
                json.dump(metadata_list, f)
            
            # Save FAISS index if available
            if self.index is not None:
                faiss.write_index(self.index, os.path.join(output_dir, "faiss_index.bin"))
            
            # Save model info
            model_info = {
                "model_name": self.model_name,
                "embedding_size": self.embedding_size,
                "num_chunks": len(chunk_ids),
                "created_at": time.time()
            }
            with open(os.path.join(output_dir, "model_info.json"), 'w') as f:
                json.dump(model_info, f)
            
            logger.info(f"Saved embeddings and metadata to {output_dir}")
        
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
    
    @classmethod
    def load_embeddings(cls, embeddings_dir: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict], Any]:
        """
        Load embeddings from disk.
        
        Args:
            embeddings_dir: Directory with saved embeddings
            
        Returns:
            Tuple of (embeddings dict, chunk data dict, FAISS index)
        """
        try:
            # Load embeddings
            embeddings_path = os.path.join(embeddings_dir, "embeddings.npy")
            chunk_ids_path = os.path.join(embeddings_dir, "chunk_ids.json")
            chunk_metadata_path = os.path.join(embeddings_dir, "chunk_metadata.json")
            
            if not os.path.exists(embeddings_path) or not os.path.exists(chunk_ids_path):
                logger.error(f"Embeddings or chunk IDs not found in {embeddings_dir}")
                return {}, {}, None
            
            # Load embeddings array
            embeddings_array = np.load(embeddings_path)
            
            # Load chunk IDs
            with open(chunk_ids_path, 'r') as f:
                chunk_ids = json.load(f)
            
            # Load chunk metadata if available
            chunk_data = {}
            if os.path.exists(chunk_metadata_path):
                with open(chunk_metadata_path, 'r') as f:
                    metadata_list = json.load(f)
                for i, chunk_id in enumerate(chunk_ids):
                    if i < len(metadata_list):
                        chunk_data[chunk_id] = {"metadata": metadata_list[i]}
            
            # Create embeddings dictionary
            embeddings = {}
            for i, chunk_id in enumerate(chunk_ids):
                if i < len(embeddings_array):
                    embeddings[chunk_id] = embeddings_array[i]
            
            # Load FAISS index if available
            index = None
            if FAISS_AVAILABLE and os.path.exists(os.path.join(embeddings_dir, "faiss_index.bin")):
                try:
                    index = faiss.read_index(os.path.join(embeddings_dir, "faiss_index.bin"))
                except Exception as e:
                    logger.error(f"Error loading FAISS index: {str(e)}")
            
            logger.info(f"Loaded {len(embeddings)} embeddings from {embeddings_dir}")
            return embeddings, chunk_data, index
        
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            return {}, {}, None

def main():
    """Main function to embed documents."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create embeddings for documents')
    parser.add_argument('--input', required=True, help='Directory with processed documents')
    parser.add_argument('--output', default='embeddings', help='Output directory for embeddings')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Sentence transformer model name')
    parser.add_argument('--device', default=None, help='Device to use (cuda, cpu)')
    
    args = parser.parse_args()
    
    # Initialize embedder
    embedder = DocumentEmbedder(
        model_name=args.model,
        device=args.device
    )
    
    # Create embeddings
    embedder.embed_chunks(args.input, args.output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
