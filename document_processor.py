#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document processing module for text extraction and vectorization.
This module handles loading different document types, extracting text,
and converting it into vector representations for search and retrieval.
"""

import os
import re
import logging
import json
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd

# Text extraction libraries
import PyPDF2
import docx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Text embedding and indexing
import torch
from sentence_transformers import SentenceTransformer
import faiss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class Document:
    """Class to represent a document with its content and metadata."""
    
    def __init__(
        self, 
        doc_id: str, 
        text: str, 
        source: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        date: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize a document.
        
        Args:
            doc_id: Unique identifier for the document
            text: The text content of the document
            source: Source file path or identifier
            title: Document title (if available)
            author: Document author (if available)
            date: Document date (if available)
            metadata: Additional metadata as dictionary
        """
        self.doc_id = doc_id
        self.text = text
        self.source = source
        self.title = title or os.path.basename(source)
        self.author = author
        self.date = date
        self.metadata = metadata or {}
        
        # Extract chunks for better retrieval
        self.chunks = self._chunk_text()
        
    def _chunk_text(self, max_chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Split document text into overlapping chunks.
        
        Args:
            max_chunk_size: Maximum character length for each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of dictionaries containing chunks and their positions
        """
        chunks = []
        sentences = sent_tokenize(self.text)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_size + sentence_len <= max_chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_len
            else:
                # Create chunk from current sentences
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'doc_id': self.doc_id,
                        'text': chunk_text,
                        'start_idx': len(''.join(current_chunk[:max(0, len(current_chunk)-overlap)])),
                        'end_idx': len(chunk_text)
                    })
                
                # Start new chunk, potentially including overlap
                overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_size = len(' '.join(current_chunk))
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'doc_id': self.doc_id,
                'text': chunk_text,
                'start_idx': len(''.join(current_chunk[:max(0, len(current_chunk)-overlap)])),
                'end_idx': len(chunk_text)
            })
        
        return chunks
    
    def to_dict(self) -> Dict:
        """Convert document to dictionary representation."""
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'source': self.source,
            'author': self.author,
            'date': self.date,
            'metadata': self.metadata,
            'text_length': len(self.text),
            'num_chunks': len(self.chunks)
        }
    
    @classmethod
    def from_dict(cls, data: Dict, text: str) -> 'Document':
        """Create document from dictionary and text."""
        return cls(
            doc_id=data['doc_id'],
            text=text,
            source=data['source'],
            title=data['title'],
            author=data['author'],
            date=data['date'],
            metadata=data['metadata']
        )

class DocumentProcessor:
    """Class to process documents for text extraction and vectorization."""
    
    def __init__(self, model_name: str = 'distilbert-base-nli-stsb-mean-tokens'):
        """
        Initialize the document processor.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model = None  # Lazy loading
        self.index = None
        self.documents = {}
        self.chunks = []
        
        logger.info(f"Initialized DocumentProcessor with model {model_name}")
    
    def _load_model(self):
        """Load the embedding model if not already loaded."""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
    
    def extract_text_from_file(self, file_path: str) -> Tuple[str, Dict]:
        """
        Extract text from a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (extracted text, metadata)
        """
        logger.info(f"Extracting text from {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        metadata = {}
        
        try:
            if file_ext == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_ext in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
            elif file_ext in ['.txt', '.md', '.html', '.htm']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read(), metadata
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return "", metadata
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return "", metadata
    
    def _extract_from_pdf(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from PDF file."""
        text = ""
        metadata = {}
        
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extract metadata
            if reader.metadata:
                metadata = {
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'creation_date': reader.metadata.get('/CreationDate', '')
                }
            
            # Extract text from all pages
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        return text, metadata
    
    def _extract_from_docx(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from DOCX file."""
        doc = docx.Document(file_path)
        metadata = {
            'title': doc.core_properties.title or '',
            'author': doc.core_properties.author or '',
            'created': str(doc.core_properties.created) if doc.core_properties.created else ''
        }
        
        # Extract text from paragraphs
        text = "\n".join([para.text for para in doc.paragraphs])
        
        return text, metadata
    
    def add_document(self, file_path: str, doc_id: Optional[str] = None) -> Optional[Document]:
        """
        Process and add a document.
        
        Args:
            file_path: Path to the document file
            doc_id: Optional document ID (default: file name)
            
        Returns:
            Document object if successful, None otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        # Use filename as ID if not provided
        if doc_id is None:
            doc_id = os.path.basename(file_path)
        
        # Extract text and metadata
        text, metadata = self.extract_text_from_file(file_path)
        
        if not text:
            logger.warning(f"No text extracted from {file_path}")
            return None
        
        # Create document
        doc = Document(
            doc_id=doc_id,
            text=text,
            source=file_path,
            title=metadata.get('title', None),
            author=metadata.get('author', None),
            date=metadata.get('creation_date', metadata.get('created', None)),
            metadata=metadata
        )
        
        # Add to documents
        self.documents[doc_id] = doc
        
        # Add chunks to the list
        self.chunks.extend(doc.chunks)
        
        logger.info(f"Added document {doc_id} with {len(doc.chunks)} chunks")
        return doc
    
    def add_documents_from_directory(self, dir_path: str, extensions: List[str] = None) -> List[Document]:
        """
        Process and add all documents from a directory.
        
        Args:
            dir_path: Path to the directory
            extensions: List of file extensions to process (default: ['.pdf', '.docx', '.txt'])
            
        Returns:
            List of added documents
        """
        if extensions is None:
            extensions = ['.pdf', '.docx', '.txt', '.md']
        
        added_docs = []
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    file_path = os.path.join(root, file)
                    doc = self.add_document(file_path)
                    if doc:
                        added_docs.append(doc)
        
        logger.info(f"Added {len(added_docs)} documents from {dir_path}")
        return added_docs
    
    def build_index(self):
        """Build search index from document chunks."""
        if not self.chunks:
            logger.warning("No documents to index")
            return
        
        logger.info(f"Building index from {len(self.chunks)} chunks")
        
        # Load model if not already loaded
        self._load_model()
        
        # Get text from all chunks
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Generate embeddings
        chunk_embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create Faiss index
        dimension = chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(chunk_embeddings).astype('float32'))
        
        logger.info(f"Index built with dimension {dimension}")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for chunks relevant to a query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of dictionaries with search results and source documents
        """
        if not self.index:
            logger.warning("No index available. Call build_index() first.")
            return []
        
        # Load model if not already loaded
        self._load_model()
        
        # Encode query
        query_vector = self.model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_vector, k=min(k, len(self.chunks)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
                
            chunk = self.chunks[idx]
            doc_id = chunk['doc_id']
            document = self.documents.get(doc_id)
            
            if document:
                results.append({
                    'chunk': chunk,
                    'document': document.to_dict(),
                    'score': float(1.0 / (1.0 + distances[0][i])),
                    'distance': float(distances[0][i])
                })
        
        return results
    
    def save(self, output_dir: str):
        """
        Save processed documents and index.
        
        Args:
            output_dir: Directory to save to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save documents metadata
        docs_metadata = {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()}
        with open(os.path.join(output_dir, 'documents_metadata.json'), 'w') as f:
            json.dump(docs_metadata, f, indent=2)
        
        # Save documents text
        for doc_id, doc in self.documents.items():
            with open(os.path.join(output_dir, f"{doc_id}.txt"), 'w', encoding='utf-8') as f:
                f.write(doc.text)
        
        # Save chunks
        with open(os.path.join(output_dir, 'chunks.json'), 'w') as f:
            json.dump(self.chunks, f, indent=2)
        
        # Save index if available
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(output_dir, 'document_index.faiss'))
        
        logger.info(f"Saved documents and index to {output_dir}")
    
    @classmethod
    def load(cls, input_dir: str) -> 'DocumentProcessor':
        """
        Load processed documents and index.
        
        Args:
            input_dir: Directory to load from
            
        Returns:
            DocumentProcessor object
        """
        processor = cls()
        
        # Load documents metadata
        metadata_path = os.path.join(input_dir, 'documents_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                docs_metadata = json.load(f)
            
            # Load documents
            for doc_id, metadata in docs_metadata.items():
                text_path = os.path.join(input_dir, f"{doc_id}.txt")
                if os.path.exists(text_path):
                    with open(text_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    doc = Document.from_dict(metadata, text)
                    processor.documents[doc_id] = doc
        
        # Load chunks
        chunks_path = os.path.join(input_dir, 'chunks.json')
        if os.path.exists(chunks_path):
            with open(chunks_path, 'r') as f:
                processor.chunks = json.load(f)
        
        # Load index if available
        index_path = os.path.join(input_dir, 'document_index.faiss')
        if os.path.exists(index_path):
            processor.index = faiss.read_index(index_path)
        
        logger.info(f"Loaded {len(processor.documents)} documents and {len(processor.chunks)} chunks from {input_dir}")
        return processor

def main():
    """
    Main function to process documents.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Process documents for question answering')
    parser.add_argument('--input', required=True, help='Input directory or file')
    parser.add_argument('--output', required=True, help='Output directory for processed documents')
    parser.add_argument('--model', default='paraphrase-MiniLM-L6-v2', help='SentenceTransformer model name')
    
    args = parser.parse_args()
    
    processor = DocumentProcessor(model_name=args.model)
    
    if os.path.isdir(args.input):
        processor.add_documents_from_directory(args.input)
    else:
        processor.add_document(args.input)
    
    processor.build_index()
    processor.save(args.output)
    
    logger.info("Document processing completed")

if __name__ == "__main__":
    main()
