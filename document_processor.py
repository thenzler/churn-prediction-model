#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document processing and indexing for question answering system.
This module handles document loading, preprocessing, and vectorization
to prepare for retrieval and question answering.
"""

import os
import re
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.warning("SentenceTransformer not available, falling back to simple TF-IDF")
    HAVE_SENTENCE_TRANSFORMERS = False
    from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class Document:
    """Class to represent a document or document chunk"""
    doc_id: str
    text: str
    metadata: dict
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self):
        """Convert to dictionary (embeddings saved separately)"""
        return {
            'doc_id': self.doc_id,
            'text': self.text,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[np.ndarray] = None):
        """Create a Document from a dictionary"""
        return cls(
            doc_id=data['doc_id'],
            text=data['text'],
            metadata=data['metadata'],
            embedding=embedding
        )

class DocumentCollection:
    """Class to store and manage a collection of documents"""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
    
    def add_document(self, document: Document):
        """Add a document to the collection"""
        self.documents[document.doc_id] = document
        if document.embedding is not None:
            self.embeddings[document.doc_id] = document.embedding
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID"""
        return self.documents.get(doc_id)
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents"""
        return list(self.documents.values())
    
    def get_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Get embedding for a document"""
        return self.embeddings.get(doc_id)
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all embeddings"""
        return self.embeddings
    
    def save(self, directory: str):
        """Save the document collection to files"""
        os.makedirs(directory, exist_ok=True)
        
        # Save documents as JSON
        docs_data = {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()}
        with open(os.path.join(directory, 'documents.json'), 'w') as f:
            json.dump(docs_data, f, indent=2)
        
        # Save embeddings as NumPy arrays
        if self.embeddings:
            np.save(os.path.join(directory, 'embeddings.npy'), {
                doc_id: embedding for doc_id, embedding in self.embeddings.items()
            }, allow_pickle=True)
        
        logger.info(f"Saved {len(self.documents)} documents to {directory}")
    
    @classmethod
    def load(cls, directory: str) -> 'DocumentCollection':
        """Load a document collection from files"""
        collection = cls()
        
        # Load documents
        try:
            with open(os.path.join(directory, 'documents.json'), 'r') as f:
                docs_data = json.load(f)
            
            # Load embeddings if they exist
            embeddings = {}
            embedding_path = os.path.join(directory, 'embeddings.npy')
            if os.path.exists(embedding_path):
                embeddings = np.load(embedding_path, allow_pickle=True).item()
            
            # Create documents
            for doc_id, doc_data in docs_data.items():
                embedding = embeddings.get(doc_id)
                collection.add_document(Document.from_dict(doc_data, embedding))
            
            logger.info(f"Loaded {len(collection.documents)} documents from {directory}")
            return collection
            
        except FileNotFoundError:
            logger.warning(f"No document collection found in {directory}")
            return collection

class DocumentProcessor:
    """Process and index documents for question answering"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_gpu: bool = False):
        """
        Initialize the document processor
        
        Args:
            model_name: Name of the SentenceTransformer model to use for embeddings
            use_gpu: Whether to use GPU for embeddings (if available)
        """
        self.collection = DocumentCollection()
        
        # Initialize the embedding model
        if HAVE_SENTENCE_TRANSFORMERS:
            logger.info(f"Initializing embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            if use_gpu and self.embedding_model.device.type != 'cuda':
                logger.info("Moving model to CUDA")
                self.embedding_model.to('cuda')
        else:
            logger.info("Using TF-IDF for document vectorization")
            self.embedding_model = TfidfVectorizer(
                max_features=1024,
                stop_words='english'
            )
            self._fitted = False
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            chunk_size: Maximum chunk size (characters)
            overlap: Overlap between chunks (characters)
            
        Returns:
            List of text chunks
        """
        # Simple approach: split by newlines first, then combine
        paragraphs = text.split('\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            # If adding this paragraph would exceed chunk size, save current chunk and start a new one
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from the end of the previous chunk
                if len(current_chunk) > overlap:
                    # Find the start of a sentence within the overlap area if possible
                    overlap_text = current_chunk[-overlap:]
                    sentence_start = overlap_text.find('. ') + 2
                    if sentence_start >= 2:  # Found a sentence boundary
                        current_chunk = overlap_text[sentence_start:]
                    else:
                        current_chunk = overlap_text
                else:
                    current_chunk = ""
            
            # Add paragraph to current chunk
            if current_chunk and not current_chunk.endswith(' '):
                current_chunk += ' '
            current_chunk += paragraph
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_file(self, file_path: str, document_id: Optional[str] = None, chunk_size: int = 1000, 
                   overlap: int = 200, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Process a text file into document chunks
        
        Args:
            file_path: Path to the file
            document_id: ID for the document (default: file name)
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            metadata: Additional metadata for the document
            
        Returns:
            List of chunk IDs added to the collection
        """
        logger.info(f"Processing file: {file_path}")
        
        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        # Determine document ID
        if document_id is None:
            document_id = os.path.basename(file_path)
        
        # Initialize metadata
        if metadata is None:
            metadata = {}
            
        # Add file metadata
        file_metadata = {
            'source': file_path,
            'file_type': file_path.split('.')[-1].lower(),
            'file_size': os.path.getsize(file_path),
            'modified_time': os.path.getmtime(file_path)
        }
        metadata.update(file_metadata)
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Unicode decode error, trying with latin-1 encoding: {file_path}")
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
        
        # Process chunks
        return self.process_text(text, document_id, chunk_size, overlap, metadata)
    
    def process_text(self, text: str, document_id: str, chunk_size: int = 1000, 
                   overlap: int = 200, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Process text into document chunks
        
        Args:
            text: Text content
            document_id: ID for the document
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            metadata: Additional metadata for the document
            
        Returns:
            List of chunk IDs added to the collection
        """
        # Initialize metadata
        if metadata is None:
            metadata = {}
            
        # Split text into chunks
        chunks = self._chunk_text(text, chunk_size, overlap)
        
        # Create document objects for each chunk
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i+1}"
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'document_id': document_id,
                'chunk_index': i,
                'chunk_count': len(chunks)
            })
            
            # Create document object
            document = Document(
                doc_id=chunk_id,
                text=chunk,
                metadata=chunk_metadata
            )
            
            # Add to collection
            self.collection.add_document(document)
            chunk_ids.append(chunk_id)
        
        logger.info(f"Processed {document_id} into {len(chunks)} chunks")
        return chunk_ids
    
    def process_directory(self, directory: str, file_extensions: List[str] = None, 
                        recursive: bool = True, chunk_size: int = 1000, 
                        overlap: int = 200) -> Dict[str, List[str]]:
        """
        Process all files in a directory
        
        Args:
            directory: Directory path
            file_extensions: List of file extensions to process (e.g., ['.txt', '.md'])
            recursive: Whether to process subdirectories
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            Dictionary mapping file paths to lists of chunk IDs
        """
        logger.info(f"Processing directory: {directory}")
        
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return {}
        
        # Default file extensions
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.csv', '.json']
            
        # Make sure all extensions start with a dot
        file_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in file_extensions]
        
        # Process files
        result = {}
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext in file_extensions:
                    chunk_ids = self.process_file(
                        file_path=file_path,
                        chunk_size=chunk_size,
                        overlap=overlap
                    )
                    result[file_path] = chunk_ids
            
            # Skip subdirectories if not recursive
            if not recursive:
                break
        
        logger.info(f"Processed {len(result)} files in {directory}")
        return result
    
    def generate_embeddings(self, batch_size: int = 32, use_existing: bool = True) -> None:
        """
        Generate embeddings for all documents in the collection
        
        Args:
            batch_size: Batch size for embedding generation
            use_existing: Whether to use existing embeddings
        """
        # Collect documents that need embeddings
        documents_to_embed = []
        doc_ids = []
        
        for doc_id, doc in self.collection.documents.items():
            if not use_existing or doc.embedding is None:
                documents_to_embed.append(doc.text)
                doc_ids.append(doc_id)
        
        if not documents_to_embed:
            logger.info("No documents need embeddings")
            return
        
        logger.info(f"Generating embeddings for {len(documents_to_embed)} documents")
        
        # Generate embeddings
        if HAVE_SENTENCE_TRANSFORMERS:
            # Process in batches
            embeddings = []
            for i in range(0, len(documents_to_embed), batch_size):
                batch_texts = documents_to_embed[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(batch_texts)
                embeddings.extend(batch_embeddings)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(documents_to_embed)-1)//batch_size + 1}")
        else:
            # Fit the vectorizer if not already fitted
            if not hasattr(self, '_fitted') or not self._fitted:
                self.embedding_model.fit(documents_to_embed)
                self._fitted = True
            
            # Transform documents to embeddings
            embeddings_matrix = self.embedding_model.transform(documents_to_embed)
            embeddings = [embeddings_matrix[i].toarray()[0] for i in range(embeddings_matrix.shape[0])]
        
        # Update document embeddings
        for i, doc_id in enumerate(doc_ids):
            doc = self.collection.documents[doc_id]
            doc.embedding = embeddings[i]
            self.collection.embeddings[doc_id] = embeddings[i]
        
        logger.info(f"Generated embeddings for {len(doc_ids)} documents")
    
    def save_collection(self, directory: str = 'document_index'):
        """
        Save the document collection to files
        
        Args:
            directory: Directory to save the collection
        """
        self.collection.save(directory)
    
    def load_collection(self, directory: str = 'document_index'):
        """
        Load a document collection from files
        
        Args:
            directory: Directory to load the collection from
        """
        self.collection = DocumentCollection.load(directory)
        
        # Make sure the embedding model is fitted if using TF-IDF
        if not HAVE_SENTENCE_TRANSFORMERS and self.collection.documents:
            # Extract texts for fitting
            texts = [doc.text for doc in self.collection.documents.values()]
            self.embedding_model.fit(texts)
            self._fitted = True
    
    def get_collection(self) -> DocumentCollection:
        """Get the document collection"""
        return self.collection

def main():
    """
    Main function to demonstrate the document processor
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Process documents for question answering')
    parser.add_argument('--input', required=True, help='Input file or directory')
    parser.add_argument('--output', default='document_index', help='Output directory for document index')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model name')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Maximum chunk size in characters')
    parser.add_argument('--overlap', type=int, default=200, help='Overlap between chunks in characters')
    parser.add_argument('--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('--extensions', default='.txt,.md,.csv,.json', help='Comma-separated list of file extensions to process')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DocumentProcessor(model_name=args.model)
    
    # Process input
    if os.path.isdir(args.input):
        file_extensions = args.extensions.split(',')
        processor.process_directory(
            directory=args.input,
            file_extensions=file_extensions,
            recursive=args.recursive,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
    elif os.path.isfile(args.input):
        processor.process_file(
            file_path=args.input,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
    else:
        logger.error(f"Input not found: {args.input}")
        return 1
    
    # Generate embeddings
    processor.generate_embeddings()
    
    # Save collection
    processor.save_collection(args.output)
    
    logger.info(f"Document processing complete. Index saved to {args.output}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
