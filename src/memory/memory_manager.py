"""
Memory Manager module for the AI Agent System.

This module implements the memory management system that coordinates
short-term, long-term, and agent-specific memory components.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from src.core.config import MemoryConfig
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Memory Manager for the AI Agent System.

    The MemoryManager coordinates the different memory components:
    - Short-term memory: Recent conversations and working memory
    - Long-term memory: Persistent knowledge and past interactions
    - Agent-specific memory: Context for each agent role
    """

    def __init__(self, config: MemoryConfig):
        """
        Initialize the MemoryManager.

        Args:
            config: Memory system configuration
        """
        self.config = config
        
        # Initialize memory components
        self.short_term = ShortTermMemory(config)
        self.long_term = LongTermMemory(config)
        self.agent_memory = AgentMemory(config)
        
        logger.info("Memory Manager initialized")

    def update(self, input_data: Dict[str, Any], agent_id: Optional[str] = None) -> None:
        """
        Update memory with new information.

        This method distributes the update to the appropriate memory components
        based on the input data and context.

        Args:
            input_data: The data to store in memory
            agent_id: Optional agent identifier for agent-specific memory
        """
        # Update short-term memory
        self.short_term.update(input_data)
        
        # Update agent-specific memory if agent_id is provided
        if agent_id:
            self.agent_memory.update(agent_id, input_data)
        
        # Determine if the information should be stored in long-term memory
        if self._should_store_long_term(input_data):
            self.long_term.store(input_data)
    
    def get_context(
        self, 
        query: str, 
        agent_id: Optional[str] = None, 
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build context for a query from various memory sources.

        This method retrieves relevant information from different memory components
        and builds a comprehensive context for the query.

        Args:
            query: The query or task for which to build context
            agent_id: Optional agent identifier for agent-specific context
            task_id: Optional task identifier for task-specific context

        Returns:
            Dict[str, Any]: The constructed context
        """
        context = {}
        
        # Get recent conversation history
        context["conversation"] = self.short_term.get_conversation()
        
        # Get task-specific context if task_id is provided
        if task_id:
            context["task"] = self.short_term.get_task_context(task_id)
        
        # Get agent-specific context if agent_id is provided
        if agent_id:
            context["agent"] = self.agent_memory.get_context(agent_id)
        
        # Get relevant long-term memories
        context["long_term"] = self.long_term.retrieve(query)
        
        # Optimize context based on agent role and query
        return self._optimize_context(context, agent_id, query)
    
    def _optimize_context(
        self, 
        context: Dict[str, Any], 
        agent_id: Optional[str] = None,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize context for a specific agent and query.

        This method applies various optimization strategies to ensure the context
        fits within the agent's context window and prioritizes the most relevant
        information.

        Args:
            context: The raw context from different memory sources
            agent_id: Optional agent identifier for role-specific optimization
            query: Optional query for relevance-based optimization

        Returns:
            Dict[str, Any]: The optimized context
        """
        # If no agent_id is provided, use default optimization
        if not agent_id:
            return self._default_optimize_context(context)
        
        # Get context window size for the agent
        context_window_size = self.config.context_window_sizes.get(
            agent_id, 6000  # Default size if not specified
        )
        
        # Apply hierarchical summarization based on agent role
        if agent_id == "team_leader":
            # Team leader needs high-level overview of everything
            return self._summarize_for_team_leader(context, context_window_size)
        elif agent_id in ["ml_architect", "sw_architect"]:
            # Architects need detailed technical information
            return self._summarize_for_architect(context, agent_id, context_window_size)
        elif agent_id in ["ml_engineer", "sw_engineer", "data_engineer"]:
            # Engineers need implementation details
            return self._summarize_for_engineer(context, agent_id, context_window_size)
        elif agent_id == "product_manager":
            # Product manager needs user requirements and high-level info
            return self._summarize_for_product_manager(context, context_window_size)
        elif agent_id == "qa_engineer":
            # QA engineer needs testing criteria and implementation details
            return self._summarize_for_qa(context, context_window_size)
        else:
            # Default optimization for unknown roles
            return self._default_optimize_context(context)
    
    def _default_optimize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default context optimization strategy."""
        # Simple strategy: keep most recent conversations and most relevant long-term memories
        optimized = {}
        
        # Keep up to 10 most recent conversation entries
        if "conversation" in context:
            optimized["conversation"] = context["conversation"][-10:]
        
        # Keep task context as is
        if "task" in context:
            optimized["task"] = context["task"]
        
        # Keep up to 5 most relevant long-term memories
        if "long_term" in context:
            optimized["long_term"] = context["long_term"][:5]
        
        # Keep agent context as is
        if "agent" in context:
            optimized["agent"] = context["agent"]
        
        return optimized
    
    def _summarize_for_team_leader(
        self, context: Dict[str, Any], context_window_size: int
    ) -> Dict[str, Any]:
        """Optimize context for the team leader role."""
        # Team leader needs high-level summaries of all components
        # Implementation would include summarization techniques
        return self._default_optimize_context(context)  # Placeholder
    
    def _summarize_for_architect(
        self, context: Dict[str, Any], architect_type: str, context_window_size: int
    ) -> Dict[str, Any]:
        """Optimize context for architect roles."""
        # Architects need detailed technical information in their domain
        # Implementation would include domain-specific filtering
        return self._default_optimize_context(context)  # Placeholder
    
    def _summarize_for_engineer(
        self, context: Dict[str, Any], engineer_type: str, context_window_size: int
    ) -> Dict[str, Any]:
        """Optimize context for engineer roles."""
        # Engineers need implementation details in their domain
        # Implementation would include code-focused filtering
        return self._default_optimize_context(context)  # Placeholder
    
    def _summarize_for_product_manager(
        self, context: Dict[str, Any], context_window_size: int
    ) -> Dict[str, Any]:
        """Optimize context for the product manager role."""
        # Product manager needs user requirements and high-level info
        # Implementation would prioritize user needs and constraints
        return self._default_optimize_context(context)  # Placeholder
    
    def _summarize_for_qa(
        self, context: Dict[str, Any], context_window_size: int
    ) -> Dict[str, Any]:
        """Optimize context for the QA engineer role."""
        # QA engineer needs testing criteria and implementation details
        # Implementation would focus on requirements and edge cases
        return self._default_optimize_context(context)  # Placeholder
    
    def _should_store_long_term(self, input_data: Dict[str, Any]) -> bool:
        """
        Determine if information should be stored in long-term memory.

        This method applies heuristics to decide if the information is important
        enough to be stored in long-term memory.

        Args:
            input_data: The data to evaluate

        Returns:
            bool: True if the data should be stored in long-term memory
        """
        # Simple heuristic: store if it contains certain key information
        # In a real implementation, this would use more sophisticated criteria
        important_keys = ["task", "requirements", "architecture", "implementation", "results"]
        
        for key in important_keys:
            if key in input_data:
                return True
        
        # Check if it's a completed task
        if input_data.get("status") == "completed":
            return True
        
        # Default to not storing in long-term memory
        return False
    
    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term.clear()
        logger.info("Short-term memory cleared")
    
    def save_state(self, path: str) -> None:
        """
        Save the current memory state to disk.

        Args:
            path: Path to save the memory state
        """
        state = {
            "short_term": self.short_term.get_state(),
            "long_term": self.long_term.get_state(),
            "agent_memory": self.agent_memory.get_state(),
        }
        
        # In a real implementation, this would serialize the state to disk
        logger.info(f"Memory state saved to {path}")
    
    def load_state(self, path: str) -> None:
        """
        Load memory state from disk.

        Args:
            path: Path to load the memory state from
        """
        # In a real implementation, this would deserialize the state from disk
        logger.info(f"Memory state loaded from {path}")
