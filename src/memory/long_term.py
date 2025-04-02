"""
Long-Term Memory module for the AI Agent System.

This module implements the long-term memory component that handles
persistent knowledge and past interactions using vector embeddings.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.config import MemoryConfig

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    Long-Term Memory for the AI Agent System.

    The LongTermMemory handles:
    - Episodic memory: Record of past interactions and tasks
    - Semantic memory: Factual knowledge and concepts
    - Procedural memory: Methods, algorithms, and procedures
    
    It uses vector embeddings for efficient similarity search and retrieval.
    """

    def __init__(self, config: MemoryConfig):
        """
        Initialize the LongTermMemory.

        Args:
            config: Memory system configuration
        """
        self.config = config
        
        # Initialize embedding model
        self.embedding_model = self._initialize_embedding_model()
        
        # Initialize vector database
        self.vector_db = self._initialize_vector_db()
        
        # Initialize collections for different memory types
        self.episodic_collection = self.vector_db.get_or_create_collection("episodic")
        self.semantic_collection = self.vector_db.get_or_create_collection("semantic")
        self.procedural_collection = self.vector_db.get_or_create_collection("procedural")
        
        logger.info("Long-Term Memory initialized")

    def _initialize_embedding_model(self) -> SentenceTransformer:
        """
        Initialize the embedding model.

        Returns:
            SentenceTransformer: The embedding model
        """
        model_name = self.config.embedding_model
        try:
            model = SentenceTransformer(model_name)
            logger.info(f"Embedding model '{model_name}' loaded")
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
        # Create directory for vector database if it doesn't exist
        vector_db_path = self.config.vector_db_path
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Initialize persistent client
        try:
            client = chromadb.PersistentClient(path=str(vector_db_path))
            logger.info(f"Vector database initialized at {vector_db_path}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            logger.warning("Falling back to in-memory vector database")
            return chromadb.Client()

    def _embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: The text to embed

        Returns:
            List[float]: The embedding vector
        """
        return self.embedding_model.encode(text).tolist()

    def store(self, data: Dict[str, Any]) -> str:
        """
        Store data in long-term memory.

        This method determines the appropriate memory type (episodic, semantic,
        or procedural) based on the data content and stores it with vector
        embeddings for efficient retrieval.

        Args:
            data: The data to store

        Returns:
            str: The ID of the stored data
        """
        # Generate a unique ID for the data
        data_id = f"{int(time.time())}_{hash(json.dumps(data, sort_keys=True))}"
        
        # Determine memory type based on data content
        memory_type = self._determine_memory_type(data)
        
        # Prepare data for storage
        text_representation = self._data_to_text(data)
        embedding = self._embed_text(text_representation)
        metadata = {
            "timestamp": time.time(),
            "type": memory_type,
            **{k: str(v) for k, v in data.items() if isinstance(v, (str, int, float, bool))},
        }
        
        # Store in appropriate collection
        if memory_type == "episodic":
            self.episodic_collection.add(
                ids=[data_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text_representation]
            )
        elif memory_type == "semantic":
            self.semantic_collection.add(
                ids=[data_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text_representation]
            )
        elif memory_type == "procedural":
            self.procedural_collection.add(
                ids=[data_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text_representation]
            )
        
        logger.info(f"Stored data in {memory_type} memory with ID {data_id}")
        return data_id

    def _determine_memory_type(self, data: Dict[str, Any]) -> str:
        """
        Determine the memory type based on data content.

        Args:
            data: The data to analyze

        Returns:
            str: The memory type ("episodic", "semantic", or "procedural")
        """
        # Check for task-related information (episodic)
        if any(key in data for key in ["task_id", "task", "conversation", "interaction"]):
            return "episodic"
        
        # Check for procedural information
        if any(key in data for key in ["code", "algorithm", "procedure", "function", "method"]):
            return "procedural"
        
        # Default to semantic for factual knowledge
        return "semantic"

    def _data_to_text(self, data: Dict[str, Any]) -> str:
        """
        Convert data dictionary to text representation.

        Args:
            data: The data to convert

        Returns:
            str: Text representation of the data
        """
        # For simple data, just convert to JSON string
        if all(isinstance(v, (str, int, float, bool, list, dict)) for v in data.values()):
            return json.dumps(data, indent=2)
        
        # For complex data, create a structured text representation
        lines = []
        
        # Add title if available
        if "title" in data:
            lines.append(f"# {data['title']}")
            lines.append("")
        
        # Add task information if available
        if "task" in data:
            lines.append(f"Task: {data['task']}")
            lines.append("")
        
        # Add other fields
        for key, value in data.items():
            if key not in ["title", "task"]:
                if isinstance(value, (str, int, float, bool)):
                    lines.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    lines.append(f"{key}: {json.dumps(value)}")
                else:
                    lines.append(f"{key}: {str(value)}")
        
        return "\n".join(lines)

    def retrieve(
        self, 
        query: str, 
        n: int = 5, 
        memory_types: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant information from long-term memory.

        Args:
            query: The query string
            n: Number of results to retrieve
            memory_types: Optional list of memory types to search
            filters: Optional filters to apply

        Returns:
            List[Dict[str, Any]]: List of retrieved items
        """
        # Default to all memory types if not specified
        if memory_types is None:
            memory_types = ["episodic", "semantic", "procedural"]
        
        # Generate query embedding
        query_embedding = self._embed_text(query)
        
        results = []
        
        # Search each specified memory type
        for memory_type in memory_types:
            collection = self._get_collection_by_type(memory_type)
            
            # Skip if collection doesn't exist
            if collection is None:
                continue
            
            # Perform search
            search_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                where=filters
            )
            
            # Process results
            if search_results["ids"]:
                for i in range(len(search_results["ids"][0])):
                    result = {
                        "id": search_results["ids"][0][i],
                        "text": search_results["documents"][0][i],
                        "metadata": search_results["metadatas"][0][i],
                        "memory_type": memory_type,
                        "distance": search_results["distances"][0][i] if "distances" in search_results else None,
                    }
                    results.append(result)
        
        # Sort by distance (similarity)
        results.sort(key=lambda x: x["distance"] if x["distance"] is not None else float("inf"))
        
        # Return top n results
        return results[:n]

    def _get_collection_by_type(self, memory_type: str) -> Optional[chromadb.Collection]:
        """
        Get collection by memory type.

        Args:
            memory_type: The memory type

        Returns:
            Optional[chromadb.Collection]: The collection or None if not found
        """
        if memory_type == "episodic":
            return self.episodic_collection
        elif memory_type == "semantic":
            return self.semantic_collection
        elif memory_type == "procedural":
            return self.procedural_collection
        else:
            logger.warning(f"Unknown memory type: {memory_type}")
            return None

    def store_task(self, task_id: str, task_data: Dict[str, Any]) -> str:
        """
        Store task data in episodic memory.

        Args:
            task_id: The task ID
            task_data: The task data

        Returns:
            str: The ID of the stored data
        """
        # Ensure task_id is included in the data
        if "task_id" not in task_data:
            task_data["task_id"] = task_id
        
        # Store in episodic memory
        return self.store(task_data)

    def retrieve_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve task data from episodic memory.

        Args:
            task_id: The task ID

        Returns:
            Optional[Dict[str, Any]]: The task data or None if not found
        """
        # Search for task by ID
        results = self.episodic_collection.query(
            query_embeddings=None,
            where={"task_id": task_id},
            n_results=1
        )
        
        if results["ids"] and results["ids"][0]:
            return {
                "id": results["ids"][0][0],
                "text": results["documents"][0][0],
                "metadata": results["metadatas"][0][0],
                "memory_type": "episodic",
            }
        else:
            return None

    def store_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """
        Store knowledge in semantic memory.

        Args:
            knowledge: The knowledge to store

        Returns:
            str: The ID of the stored data
        """
        # Ensure type is set to semantic
        knowledge["_memory_type"] = "semantic"
        
        # Store in semantic memory
        return self.store(knowledge)

    def store_procedure(self, procedure: Dict[str, Any]) -> str:
        """
        Store procedure in procedural memory.

        Args:
            procedure: The procedure to store

        Returns:
            str: The ID of the stored data
        """
        # Ensure type is set to procedural
        procedure["_memory_type"] = "procedural"
        
        # Store in procedural memory
        return self.store(procedure)

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of long-term memory.

        Returns:
            Dict[str, Any]: The current state
        """
        # In a real implementation, this would return statistics or metadata
        # about the long-term memory state
        return {
            "episodic_count": self.episodic_collection.count(),
            "semantic_count": self.semantic_collection.count(),
            "procedural_count": self.procedural_collection.count(),
            "vector_db_path": str(self.config.vector_db_path),
        }

    def clear(self) -> None:
        """Clear all long-term memory."""
        # Delete all collections
        self.vector_db.delete_collection("episodic")
        self.vector_db.delete_collection("semantic")
        self.vector_db.delete_collection("procedural")
        
        # Recreate collections
        self.episodic_collection = self.vector_db.get_or_create_collection("episodic")
        self.semantic_collection = self.vector_db.get_or_create_collection("semantic")
        self.procedural_collection = self.vector_db.get_or_create_collection("procedural")
        
        logger.info("Long-term memory cleared")
