"""
Short-Term Memory module for the AI Agent System.

This module implements the short-term memory component that handles
recent conversations, working memory, and task context.
"""

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Union

from src.core.config import MemoryConfig

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    Short-Term Memory for the AI Agent System.

    The ShortTermMemory handles:
    - Conversation buffer: Recent messages and exchanges
    - Working memory: Active information for the current task
    - Task context: Specific context for ongoing tasks
    """

    def __init__(self, config: MemoryConfig):
        """
        Initialize the ShortTermMemory.

        Args:
            config: Memory system configuration
        """
        self.config = config
        
        # Initialize conversation buffer as a deque with max size
        self.conversation_buffer = deque(maxlen=config.conversation_buffer_size)
        
        # Initialize working memory as a dictionary with priority weights
        self.working_memory = {}
        self.working_memory_weights = {}
        
        # Initialize task context storage
        self.task_contexts = {}
        
        logger.info("Short-Term Memory initialized")

    def update(self, input_data: Dict[str, Any]) -> None:
        """
        Update short-term memory with new information.

        Args:
            input_data: The data to store in memory
        """
        # Handle different types of updates based on the input data
        
        # If it's a conversation message, add to conversation buffer
        if "message" in input_data:
            self._update_conversation(input_data)
        
        # If it's task-related information, update working memory
        if "task_id" in input_data:
            self._update_working_memory(input_data)
            
            # If it contains task context, update task contexts
            if "context" in input_data:
                self.update_task_context(input_data["task_id"], input_data["context"])
    
    def _update_conversation(self, message_data: Dict[str, Any]) -> None:
        """
        Update the conversation buffer with a new message.

        Args:
            message_data: The message data to add
        """
        # Add timestamp if not present
        if "timestamp" not in message_data:
            message_data["timestamp"] = time.time()
        
        # Add to conversation buffer
        self.conversation_buffer.append(message_data)
        
        # If buffer is full, log that oldest message was removed
        if len(self.conversation_buffer) == self.config.conversation_buffer_size:
            logger.debug("Conversation buffer full, oldest message removed")
    
    def _update_working_memory(self, input_data: Dict[str, Any]) -> None:
        """
        Update working memory with new information.

        Args:
            input_data: The data to store in working memory
        """
        # Extract task ID
        task_id = input_data.get("task_id")
        if not task_id:
            logger.warning("No task_id provided for working memory update")
            return
        
        # Initialize task in working memory if not present
        if task_id not in self.working_memory:
            self.working_memory[task_id] = {}
            self.working_memory_weights[task_id] = {}
        
        # Update working memory with new data
        for key, value in input_data.items():
            if key != "task_id" and key != "timestamp":
                self.working_memory[task_id][key] = value
                
                # Set or update weight for this item
                self.working_memory_weights[task_id][key] = 1.0  # Initial weight
        
        # Apply decay to existing weights
        self._apply_decay(task_id)
        
        # Prune working memory if it exceeds capacity
        self._prune_working_memory(task_id)
    
    def _apply_decay(self, task_id: str) -> None:
        """
        Apply decay to working memory weights.

        Args:
            task_id: The task ID
        """
        # Skip if task not in working memory
        if task_id not in self.working_memory_weights:
            return
        
        # Apply decay factor to all weights
        for key in self.working_memory_weights[task_id]:
            self.working_memory_weights[task_id][key] *= self.config.decay_rate
    
    def _prune_working_memory(self, task_id: str) -> None:
        """
        Prune working memory if it exceeds capacity.

        Args:
            task_id: The task ID
        """
        # Skip if task not in working memory
        if task_id not in self.working_memory:
            return
        
        # Check if working memory exceeds capacity
        if len(self.working_memory[task_id]) <= self.config.working_memory_capacity:
            return
        
        # Sort items by weight
        sorted_items = sorted(
            self.working_memory_weights[task_id].items(),
            key=lambda x: x[1]
        )
        
        # Remove lowest-weight items until within capacity
        items_to_remove = len(self.working_memory[task_id]) - self.config.working_memory_capacity
        for i in range(items_to_remove):
            key_to_remove = sorted_items[i][0]
            del self.working_memory[task_id][key_to_remove]
            del self.working_memory_weights[task_id][key_to_remove]
            
            logger.debug(f"Pruned item '{key_to_remove}' from working memory for task {task_id}")
    
    def update_task_context(self, task_id: str, context: Dict[str, Any]) -> None:
        """
        Update the context for a specific task.

        Args:
            task_id: The task ID
            context: The task context to store
        """
        # Store or update task context
        self.task_contexts[task_id] = context
        logger.debug(f"Updated context for task {task_id}")
    
    def get_conversation(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent conversation history.

        Args:
            n: Optional number of most recent messages to retrieve

        Returns:
            List[Dict[str, Any]]: List of conversation messages
        """
        if n is None:
            return list(self.conversation_buffer)
        else:
            # Return the n most recent messages
            return list(self.conversation_buffer)[-n:]
    
    def get_working_memory(self, task_id: str) -> Dict[str, Any]:
        """
        Get working memory for a specific task.

        Args:
            task_id: The task ID

        Returns:
            Dict[str, Any]: Working memory for the task
        """
        return self.working_memory.get(task_id, {})
    
    def get_task_context(self, task_id: str) -> Dict[str, Any]:
        """
        Get context for a specific task.

        Args:
            task_id: The task ID

        Returns:
            Dict[str, Any]: Task context
        """
        return self.task_contexts.get(task_id, {})
    
    def clear(self) -> None:
        """Clear all short-term memory."""
        self.conversation_buffer.clear()
        self.working_memory.clear()
        self.working_memory_weights.clear()
        self.task_contexts.clear()
        logger.info("Short-term memory cleared")
    
    def clear_task(self, task_id: str) -> None:
        """
        Clear memory for a specific task.

        Args:
            task_id: The task ID
        """
        if task_id in self.working_memory:
            del self.working_memory[task_id]
        
        if task_id in self.working_memory_weights:
            del self.working_memory_weights[task_id]
        
        if task_id in self.task_contexts:
            del self.task_contexts[task_id]
            
        logger.info(f"Memory cleared for task {task_id}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of short-term memory.

        Returns:
            Dict[str, Any]: The current state
        """
        return {
            "conversation_buffer": list(self.conversation_buffer),
            "working_memory": self.working_memory,
            "working_memory_weights": self.working_memory_weights,
            "task_contexts": self.task_contexts,
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the state of short-term memory.

        Args:
            state: The state to set
        """
        if "conversation_buffer" in state:
            self.conversation_buffer = deque(
                state["conversation_buffer"],
                maxlen=self.config.conversation_buffer_size
            )
        
        if "working_memory" in state:
            self.working_memory = state["working_memory"]
        
        if "working_memory_weights" in state:
            self.working_memory_weights = state["working_memory_weights"]
        
        if "task_contexts" in state:
            self.task_contexts = state["task_contexts"]
            
        logger.info("Short-term memory state restored")
