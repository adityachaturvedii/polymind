"""
Retriever module for the AI Agent System.

This module implements the retrieval component of the RAG system,
responsible for retrieving relevant information from various knowledge sources.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
from sentence_transformers import SentenceTransformer

from src.core.config import RAGConfig

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retriever for the AI Agent System.

    The Retriever is responsible for retrieving relevant information from
    various knowledge sources, including:
    - Vector database of documents
    - Knowledge graph
    - Code repository
    - External sources (if enabled)
    """

    def __init__(self, config: RAGConfig):
        """
        Initialize the Retriever.

        Args:
            config: RAG system configuration
        """
        self.config = config
        
        # Initialize embedding model
        self.embedding_model = self._initialize_embedding_model()
        
        # Initialize vector database
        self.vector_db = self._initialize_vector_db()
        
        # Initialize collections for different knowledge domains
        self.collections = self._initialize_collections()
        
        logger.info("Retriever initialized")

    def _initialize_embedding_model(self) -> SentenceTransformer:
        """
        Initialize the embedding model.

        Returns:
            SentenceTransformer: The embedding model
        """
        try:
            # Use the same embedding model as the memory system
            model = SentenceTransformer("all-mpnet-base-v2")
            logger.info("Embedding model loaded")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            logger.warning("Falling back to default embedding model")
            return SentenceTransformer("all-MiniLM-L6-v2")

    def _initialize_vector_db(self) -> chromadb.Client:
        """
        Initialize the vector database.

        Returns:
            chromadb.Client: The vector database client
        """
        # Create directory for knowledge base if it doesn't exist
        knowledge_base_path = self.config.knowledge_base_path
        os.makedirs(knowledge_base_path, exist_ok=True)
        
        # Initialize persistent client
        try:
            client = chromadb.PersistentClient(path=str(knowledge_base_path))
            logger.info(f"Vector database initialized at {knowledge_base_path}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            logger.warning("Falling back to in-memory vector database")
            return chromadb.Client()

    def _initialize_collections(self) -> Dict[str, chromadb.Collection]:
        """
        Initialize collections for different knowledge domains.

        Returns:
            Dict[str, chromadb.Collection]: Dictionary of collections
        """
        collections = {}
        
        # Create collections for each domain
        domains = ["general", "cv", "nlp", "tabular"]
        
        for domain in domains:
            collections[domain] = self.vector_db.get_or_create_collection(domain)
            logger.info(f"Collection '{domain}' initialized")
        
        return collections

    def retrieve(
        self, 
        query: str, 
        domains: Optional[List[str]] = None,
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant information for a query.

        Args:
            query: The query string
            domains: Optional list of domains to search
            n_results: Number of results to retrieve
            filters: Optional filters to apply

        Returns:
            List[Dict[str, Any]]: List of retrieved items
        """
        # Default to all domains if not specified
        if domains is None:
            domains = list(self.collections.keys())
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = []
        
        # Search each specified domain
        for domain in domains:
            if domain not in self.collections:
                logger.warning(f"Unknown domain: {domain}")
                continue
            
            collection = self.collections[domain]
            
            # Perform search
            search_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters
            )
            
            # Process results
            if search_results["ids"] and search_results["ids"][0]:
                for i in range(len(search_results["ids"][0])):
                    result = {
                        "id": search_results["ids"][0][i],
                        "text": search_results["documents"][0][i],
                        "metadata": search_results["metadatas"][0][i],
                        "domain": domain,
                        "distance": search_results["distances"][0][i] if "distances" in search_results else None,
                    }
                    results.append(result)
        
        # Sort by distance (similarity)
        results.sort(key=lambda x: x["distance"] if x["distance"] is not None else float("inf"))
        
        # Apply reranking if enabled
        if self.config.reranking_enabled and len(results) > 1:
            results = self._rerank_results(query, results)
        
        # Return top n results
        return results[:n_results]

    def _rerank_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank results based on additional criteria.

        Args:
            query: The original query
            results: The initial results

        Returns:
            List[Dict[str, Any]]: Reranked results
        """
        # In a real implementation, this would use a more sophisticated
        # reranking algorithm, possibly with a cross-encoder model
        
        # For now, just return the original results
        return results

    def add_document(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        domain: str = "general"
    ) -> str:
        """
        Add a document to the knowledge base.

        Args:
            text: The document text
            metadata: Document metadata
            domain: The domain to add the document to

        Returns:
            str: The document ID
        """
        # Check if domain exists
        if domain not in self.collections:
            logger.warning(f"Unknown domain: {domain}, creating new collection")
            self.collections[domain] = self.vector_db.get_or_create_collection(domain)
        
        # Generate document ID
        doc_id = f"{domain}_{len(self.collections[domain].get()['ids']) + 1}"
        
        # Generate embedding
        embedding = self.embedding_model.encode(text).tolist()
        
        # Add to collection
        self.collections[domain].add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text]
        )
        
        logger.info(f"Added document to {domain} with ID {doc_id}")
        return doc_id

    def delete_document(self, doc_id: str, domain: str = "general") -> bool:
        """
        Delete a document from the knowledge base.

        Args:
            doc_id: The document ID
            domain: The domain the document belongs to

        Returns:
            bool: True if successful, False otherwise
        """
        # Check if domain exists
        if domain not in self.collections:
            logger.warning(f"Unknown domain: {domain}")
            return False
        
        try:
            # Delete from collection
            self.collections[domain].delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id} from {domain}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return False

    def update_document(
        self, 
        doc_id: str, 
        text: str, 
        metadata: Dict[str, Any], 
        domain: str = "general"
    ) -> bool:
        """
        Update a document in the knowledge base.

        Args:
            doc_id: The document ID
            text: The updated document text
            metadata: Updated document metadata
            domain: The domain the document belongs to

        Returns:
            bool: True if successful, False otherwise
        """
        # Check if domain exists
        if domain not in self.collections:
            logger.warning(f"Unknown domain: {domain}")
            return False
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(text).tolist()
            
            # Update in collection
            self.collections[domain].update(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )
            
            logger.info(f"Updated document {doc_id} in {domain}")
            return True
        except Exception as e:
            logger.error(f"Failed to update document: {str(e)}")
            return False

    def get_document(self, doc_id: str, domain: str = "general") -> Optional[Dict[str, Any]]:
        """
        Get a document from the knowledge base.

        Args:
            doc_id: The document ID
            domain: The domain the document belongs to

        Returns:
            Optional[Dict[str, Any]]: The document or None if not found
        """
        # Check if domain exists
        if domain not in self.collections:
            logger.warning(f"Unknown domain: {domain}")
            return None
        
        try:
            # Get from collection
            results = self.collections[domain].get(ids=[doc_id])
            
            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "text": results["documents"][0],
                    "metadata": results["metadatas"][0],
                    "domain": domain,
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get document: {str(e)}")
            return None

    def build_context(
        self, 
        query: str, 
        task_type: str, 
        domains: Optional[List[str]] = None,
        n_results: int = 5
    ) -> str:
        """
        Build a context string for a query.

        This method retrieves relevant information and formats it into a
        context string that can be used as input for an LLM.

        Args:
            query: The query string
            task_type: The type of task (e.g., "modeling", "coding")
            domains: Optional list of domains to search
            n_results: Number of results to include

        Returns:
            str: The formatted context string
        """
        # Retrieve relevant information
        results = self.retrieve(query, domains, n_results)
        
        # Format context based on task type
        if task_type == "modeling":
            return self._format_modeling_context(results)
        elif task_type == "coding":
            return self._format_coding_context(results)
        else:
            return self._format_general_context(results)

    def _format_modeling_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format context for modeling tasks.

        Args:
            results: The retrieved results

        Returns:
            str: The formatted context string
        """
        context_parts = ["# Relevant Information for Modeling\n"]
        
        for i, result in enumerate(results):
            context_parts.append(f"## Source {i+1}: {result.get('metadata', {}).get('title', 'Untitled')}\n")
            context_parts.append(result["text"])
            context_parts.append("\n---\n")
        
        return "\n".join(context_parts)

    def _format_coding_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format context for coding tasks.

        Args:
            results: The retrieved results

        Returns:
            str: The formatted context string
        """
        context_parts = ["# Relevant Code Examples and Documentation\n"]
        
        for i, result in enumerate(results):
            context_parts.append(f"## Source {i+1}: {result.get('metadata', {}).get('title', 'Untitled')}\n")
            
            # If it's code, format as code block
            if result.get('metadata', {}).get('type') == 'code':
                language = result.get('metadata', {}).get('language', '')
                context_parts.append(f"```{language}")
                context_parts.append(result["text"])
                context_parts.append("```\n")
            else:
                context_parts.append(result["text"])
            
            context_parts.append("\n---\n")
        
        return "\n".join(context_parts)

    def _format_general_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format context for general tasks.

        Args:
            results: The retrieved results

        Returns:
            str: The formatted context string
        """
        context_parts = ["# Relevant Information\n"]
        
        for i, result in enumerate(results):
            context_parts.append(f"## Source {i+1}: {result.get('metadata', {}).get('title', 'Untitled')}\n")
            context_parts.append(result["text"])
            context_parts.append("\n---\n")
        
        return "\n".join(context_parts)
