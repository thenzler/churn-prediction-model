#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Question answering module for document-based queries.
This module provides functionality to answer questions based on document content.
"""

import os
import logging
import json
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import torch

# NLP components
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Import document processor
from document_processor import DocumentProcessor, Document

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

class QuestionAnswerer:
    """Class to answer questions based on document content."""
    
    def __init__(
        self, 
        document_processor: DocumentProcessor,
        qa_model_name: str = 'deepset/roberta-base-squad2',
        device: str = None
    ):
        """
        Initialize the question answerer.
        
        Args:
            document_processor: DocumentProcessor object with indexed documents
            qa_model_name: Name of the QA model to use
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.document_processor = document_processor
        self.qa_model_name = qa_model_name
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize QA model
        logger.info(f"Initializing QA model {qa_model_name} on {self.device}")
        self.qa_pipeline = pipeline(
            'question-answering',
            model=qa_model_name,
            tokenizer=qa_model_name,
            device=0 if self.device == 'cuda' else -1
        )
        
        # Initialize BM25 search for fallback
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """Initialize BM25 search as fallback for semantic search."""
        if not self.document_processor.chunks:
            logger.warning("No document chunks available for BM25 initialization")
            self.bm25 = None
            self.bm25_corpus = []
            return
        
        logger.info("Initializing BM25 search")
        
        # Create tokenized corpus from chunks
        self.bm25_corpus = []
        for chunk in self.document_processor.chunks:
            tokenized_text = word_tokenize(chunk['text'].lower())
            self.bm25_corpus.append(tokenized_text)
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.bm25_corpus)
    
    def answer_question(
        self, 
        question: str, 
        num_chunks: int = 5,
        use_semantic_search: bool = True,
        confidence_threshold: float = 0.1,
        return_sources: bool = True
    ) -> Dict:
        """
        Answer a question based on document content.
        
        Args:
            question: Question to answer
            num_chunks: Number of chunks to retrieve
            use_semantic_search: Whether to use semantic search (vs BM25)
            confidence_threshold: Minimum confidence for answers
            return_sources: Whether to return source documents in response
            
        Returns:
            Dictionary with answer, confidence, and sources
        """
        logger.info(f"Answering question: {question}")
        
        # Retrieve relevant chunks
        if use_semantic_search and self.document_processor.index is not None:
            relevant_chunks = self.document_processor.search(question, k=num_chunks)
            chunks_text = [result['chunk']['text'] for result in relevant_chunks]
            chunks_info = [
                {
                    'doc_id': result['chunk']['doc_id'],
                    'source': result['document']['source'],
                    'title': result['document']['title'],
                    'score': result['score']
                }
                for result in relevant_chunks
            ]
        elif self.bm25 is not None:
            # Fallback to BM25 search
            tokenized_query = word_tokenize(question.lower())
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_n = np.argsort(bm25_scores)[-num_chunks:][::-1]
            
            chunks_text = []
            chunks_info = []
            
            for i in top_n:
                if i < len(self.document_processor.chunks):
                    chunk = self.document_processor.chunks[i]
                    doc_id = chunk['doc_id']
                    document = self.document_processor.documents.get(doc_id)
                    
                    if document:
                        chunks_text.append(chunk['text'])
                        chunks_info.append({
                            'doc_id': doc_id,
                            'source': document.source,
                            'title': document.title,
                            'score': float(bm25_scores[i])
                        })
        else:
            logger.error("No search index or BM25 available. Cannot answer question.")
            return {
                'answer': 'I could not find an answer because no document index is available.',
                'confidence': 0.0,
                'sources': []
            }
        
        if not chunks_text:
            return {
                'answer': 'I could not find relevant information to answer your question.',
                'confidence': 0.0,
                'sources': []
            }
        
        # Combine chunks for context
        context = "\n\n".join(chunks_text)
        
        # Run QA model
        try:
            result = self.qa_pipeline(question=question, context=context)
            
            answer = result['answer']
            confidence = result['score']
            
            if confidence < confidence_threshold:
                answer = 'I could not find a confident answer to your question in the provided documents.'
                confidence = 0.0
            
            response = {
                'answer': answer,
                'confidence': float(confidence)
            }
            
            # Add sources if requested
            if return_sources:
                # Normalize scores for readability
                max_score = max([info['score'] for info in chunks_info]) if chunks_info else 1.0
                for info in chunks_info:
                    info['normalized_score'] = info['score'] / max_score
                
                # Add sources to response
                response['sources'] = chunks_info
            
            return response
        
        except Exception as e:
            logger.error(f"Error running QA model: {e}")
            return {
                'answer': 'An error occurred while trying to answer your question.',
                'confidence': 0.0,
                'sources': chunks_info if return_sources else []
            }
    
    def save(self, output_dir: str):
        """
        Save QA model configuration.
        
        Args:
            output_dir: Directory to save to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        config = {
            'qa_model_name': self.qa_model_name,
            'device': self.device
        }
        
        with open(os.path.join(output_dir, 'qa_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved QA configuration to {output_dir}")
    
    @classmethod
    def load(cls, input_dir: str, document_processor: DocumentProcessor) -> 'QuestionAnswerer':
        """
        Load QA model configuration.
        
        Args:
            input_dir: Directory to load from
            document_processor: DocumentProcessor object with indexed documents
            
        Returns:
            QuestionAnswerer object
        """
        config_path = os.path.join(input_dir, 'qa_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            return cls(
                document_processor=document_processor,
                qa_model_name=config.get('qa_model_name', 'deepset/roberta-base-squad2'),
                device=config.get('device', None)
            )
        else:
            logger.warning(f"QA configuration not found at {config_path}, using defaults")
            return cls(document_processor=document_processor)

def interactive_qa(qa_system: QuestionAnswerer, exit_commands: List[str] = None):
    """
    Run an interactive QA session.
    
    Args:
        qa_system: QuestionAnswerer object
        exit_commands: List of commands to exit the session
    """
    if exit_commands is None:
        exit_commands = ['exit', 'quit', 'q']
    
    print("\nDocument-based Question Answering System")
    print("---------------------------------------")
    print(f"Type your questions or '{exit_commands[0]}' to exit.")
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in exit_commands:
                print("Exiting QA session.")
                break
            
            if not question:
                continue
            
            # Process question
            result = qa_system.answer_question(question, return_sources=True)
            
            # Display answer
            print(f"\nAnswer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            # Display sources
            if 'sources' in result and result['sources']:
                print("\nSources:")
                for i, source in enumerate(result['sources']):
                    rel_score = source.get('normalized_score', 0.0)
                    print(f"  {i+1}. {source['title']} (Relevance: {rel_score:.2f})")
        
        except KeyboardInterrupt:
            print("\nExiting QA session.")
            break
        
        except Exception as e:
            print(f"Error: {e}")

def main():
    """
    Main function to run QA system.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Answer questions based on document content')
    parser.add_argument('--docs-dir', required=True, help='Directory with processed documents')
    parser.add_argument('--qa-model', default='deepset/roberta-base-squad2', help='QA model name')
    parser.add_argument('--interactive', action='store_true', help='Run interactive QA session')
    parser.add_argument('--question', help='Question to answer (single-shot mode)')
    
    args = parser.parse_args()
    
    # Load document processor
    processor = DocumentProcessor.load(args.docs_dir)
    
    # Initialize QA system
    qa_system = QuestionAnswerer(
        document_processor=processor,
        qa_model_name=args.qa_model
    )
    
    if args.interactive:
        interactive_qa(qa_system)
    elif args.question:
        result = qa_system.answer_question(args.question, return_sources=True)
        
        print(f"Question: {args.question}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        if 'sources' in result and result['sources']:
            print("\nSources:")
            for i, source in enumerate(result['sources']):
                rel_score = source.get('normalized_score', 0.0)
                print(f"  {i+1}. {source['title']} (Relevance: {rel_score:.2f})")
    else:
        print("Please provide a question with --question or use --interactive for an interactive session.")

if __name__ == "__main__":
    main()
