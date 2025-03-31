#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document-based question answering system.
This module handles document retrieval and question answering
based on a collection of indexed documents.
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
import heapq
from collections import Counter
import string

# Import document processor
from document_processor import DocumentProcessor, Document, DocumentCollection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer, util
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.warning("SentenceTransformer not available, falling back to simple TF-IDF")
    HAVE_SENTENCE_TRANSFORMERS = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class QueryResult:
    """Class to represent a document retrieval result"""
    doc_id: str
    document: Document
    score: float
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'doc_id': self.doc_id,
            'text': self.document.text,
            'metadata': self.document.metadata,
            'score': self.score
        }

@dataclass
class AnswerResult:
    """Class to represent a QA result"""
    question: str
    answer: str
    sources: List[QueryResult]
    confidence: float
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'question': self.question,
            'answer': self.answer,
            'sources': [source.to_dict() for source in self.sources],
            'confidence': self.confidence
        }
    
    def get_source_references(self) -> List[str]:
        """Get formatted source references"""
        references = []
        
        for i, source in enumerate(self.sources):
            doc_id = source.doc_id
            score = source.score
            metadata = source.document.metadata
            
            if 'source' in metadata:
                source_file = os.path.basename(metadata['source'])
                reference = f"[{i+1}] {source_file} (score: {score:.2f})"
            else:
                reference = f"[{i+1}] {doc_id} (score: {score:.2f})"
                
            references.append(reference)
            
        return references
    
    def get_formatted_answer(self, include_sources: bool = True) -> str:
        """Get a formatted answer with sources"""
        result = self.answer
        
        if include_sources and self.sources:
            result += "\n\nSources:\n"
            result += "\n".join(self.get_source_references())
            
        return result

class DocumentRetriever:
    """Retrieve relevant documents for a query"""
    
    def __init__(self, document_processor: DocumentProcessor):
        """
        Initialize the document retriever
        
        Args:
            document_processor: Document processor with indexed documents
        """
        self.document_processor = document_processor
        self.collection = document_processor.get_collection()
        
        # Check if we have embeddings
        if HAVE_SENTENCE_TRANSFORMERS:
            self.embedding_model = document_processor.embedding_model
        else:
            self.tfidf_vectorizer = document_processor.embedding_model
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[QueryResult]:
        """
        Search for documents relevant to a query
        
        Args:
            query: Search query
            top_k: Number of top results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of query results
        """
        logger.info(f"Searching for: {query}")
        
        if not self.collection.documents:
            logger.warning("No documents in collection")
            return []
        
        # Get query embedding
        if HAVE_SENTENCE_TRANSFORMERS:
            query_embedding = self.embedding_model.encode(query)
            
            # Calculate similarity with all documents
            results = []
            for doc_id, doc in self.collection.documents.items():
                doc_embedding = self.collection.get_embedding(doc_id)
                if doc_embedding is not None:
                    # Calculate cosine similarity
                    similarity = util.pytorch_cos_sim(
                        query_embedding.reshape(1, -1), 
                        doc_embedding.reshape(1, -1)
                    ).item()
                    
                    if similarity >= threshold:
                        results.append(QueryResult(
                            doc_id=doc_id,
                            document=doc,
                            score=similarity
                        ))
            
            # Sort by score (descending) and take top k
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]
            
        else:
            # Transform query to TF-IDF vector
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarity with all documents
            results = []
            for doc_id, doc in self.collection.documents.items():
                doc_embedding = self.collection.get_embedding(doc_id)
                if doc_embedding is not None:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        query_vector,
                        doc_embedding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity >= threshold:
                        results.append(QueryResult(
                            doc_id=doc_id,
                            document=doc,
                            score=similarity
                        ))
            
            # Sort by score (descending) and take top k
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]
        
        logger.info(f"Found {len(results)} relevant documents")
        return results

class QuestionAnswerer:
    """Answer questions based on retrieved documents"""
    
    def __init__(self, document_retriever: DocumentRetriever):
        """
        Initialize the question answerer
        
        Args:
            document_retriever: Document retriever for finding relevant documents
        """
        self.document_retriever = document_retriever
        
        # Try to import transformers for better QA
        try:
            from transformers import pipeline
            
            logger.info("Initializing QA pipeline with transformers")
            self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            self.use_transformers = True
        except ImportError:
            logger.warning("Transformers not available, falling back to simple answer extraction")
            self.use_transformers = False
    
    def extract_answer_simple(self, question: str, context: str) -> Tuple[str, float]:
        """
        Extract an answer from the context using simple heuristics
        
        Args:
            question: The question to answer
            context: Text context to extract answer from
            
        Returns:
            Tuple of (answer, confidence)
        """
        # Preprocess the question and context
        def preprocess(text):
            text = text.lower()
            text = re.sub(r'\s+', ' ', text)
            return text
        
        question = preprocess(question)
        context = preprocess(context)
        
        # Extract question words
        question_words = set(question.split())
        
        # Remove common words and stop words
        stop_words = {'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
                     'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                     'into', 'through', 'during', 'before', 'after', 'above', 'below',
                     'to', 'from', 'up', 'down', 'of', 'off', 'over', 'under', 'again',
                     'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both',
                     'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                     'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
                     'will', 'just', 'should', 'now'}
        
        important_words = question_words - stop_words
        
        # Split context into sentences
        sentences = re.split(r'[.!?]', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Score sentences based on word overlap with question
        sentence_scores = []
        for sentence in sentences:
            # Count overlapping words
            sentence_words = set(sentence.split())
            overlap = len(important_words.intersection(sentence_words))
            
            # Score is percentage of important words covered
            if len(important_words) > 0:
                score = overlap / len(important_words)
            else:
                score = 0
                
            sentence_scores.append((sentence, score))
        
        # Sort by score (descending)
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the highest-scoring sentence as the answer
        if sentence_scores and sentence_scores[0][1] > 0:
            return sentence_scores[0][0], sentence_scores[0][1]
        else:
            # If no good match, return the first sentence with a low confidence
            return sentences[0] if sentences else "", 0.1
    
    def answer_question(self, question: str, top_k: int = 5) -> AnswerResult:
        """
        Answer a question based on retrieved documents
        
        Args:
            question: The question to answer
            top_k: Number of documents to retrieve
            
        Returns:
            Answer result with sources
        """
        logger.info(f"Answering question: {question}")
        
        # Retrieve relevant documents
        results = self.document_retriever.search(question, top_k=top_k)
        
        if not results:
            logger.warning("No relevant documents found")
            return AnswerResult(
                question=question,
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                confidence=0.0
            )
        
        # Generate answer using transformers if available
        if self.use_transformers:
            # Combine contexts from top documents
            context = " ".join([result.document.text for result in results])
            
            # If context is too long, limit it
            max_context_length = 512  # Typical limit for transformer models
            if len(context) > max_context_length:
                context = context[:max_context_length]
            
            # Get answer
            qa_result = self.qa_pipeline(question=question, context=context)
            answer = qa_result["answer"]
            confidence = qa_result["score"]
            
        else:
            # Use simple answer extraction
            best_answer = ""
            best_confidence = 0.0
            
            for result in results:
                answer, confidence = self.extract_answer_simple(question, result.document.text)
                
                # Weight confidence by document score
                confidence *= result.score
                
                if confidence > best_confidence:
                    best_answer = answer
                    best_confidence = confidence
            
            answer = best_answer
            confidence = best_confidence
        
        logger.info(f"Generated answer with confidence {confidence:.2f}")
        
        return AnswerResult(
            question=question,
            answer=answer,
            sources=results,
            confidence=confidence
        )

def main():
    """
    Main function to demonstrate the QA system
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Answer questions using indexed documents')
    parser.add_argument('--index', default='document_index', help='Directory with document index')
    parser.add_argument('--question', required=True, help='Question to answer')
    parser.add_argument('--top-k', type=int, default=3, help='Number of documents to retrieve')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model name')
    
    args = parser.parse_args()
    
    # Initialize document processor and load collection
    processor = DocumentProcessor(model_name=args.model)
    processor.load_collection(args.index)
    
    # Initialize document retriever
    retriever = DocumentRetriever(processor)
    
    # Initialize question answerer
    qa = QuestionAnswerer(retriever)
    
    # Answer question
    result = qa.answer_question(args.question, top_k=args.top_k)
    
    # Print formatted answer
    print("\n" + "="*80)
    print(f"Question: {result.question}")
    print("="*80)
    print(f"Answer: {result.answer}")
    print("="*80)
    print("Sources:")
    
    for i, source in enumerate(result.sources):
        print(f"[{i+1}] {source.document.metadata.get('source', source.doc_id)} (score: {source.score:.2f})")
        print(f"    {source.document.text[:100]}...")
    
    print("="*80)
    print(f"Confidence: {result.confidence:.2f}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
