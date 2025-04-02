"""
Base Agent module for the PolyMind AI Agent System.

This module implements the base agent class that serves as the foundation
for all specialized agent roles.

Author: Aditya Chaturvedi
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from anthropic import Anthropic

from src.core.config import AgentConfig
from src.memory.memory_manager import MemoryManager
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base Agent for the AI Agent System.

    The BaseAgent defines the common functionality and interface for all
    specialized agent roles. It handles communication with the LLM,
    context management, and interaction with memory and RAG systems.
    """

    def __init__(
        self,
        client: Anthropic,
        memory: MemoryManager,
        retriever: Retriever,
        config: AgentConfig,
        role: str,
    ):
        """
        Initialize the BaseAgent.

        Args:
            client: Anthropic API client
            memory: Memory manager
            retriever: RAG retriever
            config: Agent configuration
            role: Agent role name
        """
        self.client = client
        self.memory = memory
        self.retriever = retriever
        self.config = config
        self.role = role
        
        # Get model version and temperature for this role
        self.model = config.model_versions.get(role, "claude-3-5-sonnet")
        self.temperature = config.temperature.get(role, 0.7)
        
        logger.info(f"{role.capitalize()} agent initialized")

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent role.

        Returns:
            str: The system prompt
        """
        # Base system prompt for all agents
        base_prompt = f"""You are an AI assistant acting as a {self.role.replace('_', ' ')} in a multi-agent system.
Your goal is to collaborate with other specialized agents to solve complex tasks.
Focus on your specific responsibilities and expertise as a {self.role.replace('_', ' ')}.
Provide clear, concise, and actionable outputs that can be used by other agents.
"""
        
        # Role-specific additions to the system prompt
        role_prompt = self._get_role_specific_prompt()
        
        return f"{base_prompt}\n\n{role_prompt}"

    @abstractmethod
    def _get_role_specific_prompt(self) -> str:
        """
        Get the role-specific part of the system prompt.

        This method should be implemented by each specialized agent class.

        Returns:
            str: The role-specific prompt
        """
        pass

    def _build_context(
        self, query: str, task_id: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Build context for a query.

        This method retrieves relevant information from memory and RAG systems
        to build a comprehensive context for the query.

        Args:
            query: The query string
            task_id: Optional task identifier
            **kwargs: Additional parameters for context building

        Returns:
            Dict[str, Any]: The built context
        """
        # Get context from memory
        memory_context = self.memory.get_context(query, self.role, task_id)
        
        # Get relevant information from RAG system
        task_type = kwargs.get("task_type", "general")
        domains = kwargs.get("domains", None)
        rag_context = self.retriever.build_context(query, task_type, domains)
        
        # Combine contexts
        context = {
            "memory": memory_context,
            "rag": rag_context,
            "query": query,
            "task_id": task_id,
            **kwargs,
        }
        
        return context

    def _format_prompt(self, context: Dict[str, Any], template: str) -> str:
        """
        Format a prompt using a template and context.

        Args:
            context: The context to use for formatting
            template: The prompt template

        Returns:
            str: The formatted prompt
        """
        # Basic string formatting
        try:
            return template.format(**context)
        except KeyError as e:
            logger.warning(f"Missing key in context: {e}")
            # Fall back to a simpler prompt if formatting fails
            return f"Given the following context:\n\n{context.get('rag', '')}\n\n{context.get('query', '')}"

    def _call_llm(
        self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None
    ) -> str:
        """
        Call the LLM with a prompt.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt
            temperature: Optional temperature override

        Returns:
            str: The LLM response
        """
        # Use default system prompt if not provided
        if system_prompt is None:
            system_prompt = self._get_system_prompt()
        
        # Use default temperature if not provided
        if temperature is None:
            temperature = self.temperature
        
        try:
            # Call the LLM
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4000,
            )
            
            # Extract and return the response text
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            return f"Error: {str(e)}"

    def _store_in_memory(
        self, data: Dict[str, Any], task_id: Optional[str] = None
    ) -> None:
        """
        Store data in memory.

        Args:
            data: The data to store
            task_id: Optional task identifier
        """
        # Add task_id if provided
        if task_id:
            data["task_id"] = task_id
        
        # Add agent role
        data["agent_role"] = self.role
        
        # Store in memory
        self.memory.update(data, self.role)
        
        logger.debug(f"Stored data in memory for {self.role}")

    def _create_summary(self, content: Dict[str, Any], level: int = 0) -> Dict[str, Any]:
        """
        Create a summary of agent work.

        Args:
            content: The content to summarize
            level: Summary level (0 = most detailed, higher = more abstract)

        Returns:
            Dict[str, Any]: The summary
        """
        # In a real implementation, this would use the LLM to generate
        # a hierarchical summary at the specified level of abstraction
        
        # For now, just return a simple summary
        return {
            "agent": self.role,
            "action": content.get("action", "unknown"),
            "result": "Summary of " + str(content.get("result", "unknown")),
            "level": level,
        }

    @abstractmethod
    def process(
        self, task: str, context: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a task with the given context.

        This method should be implemented by each specialized agent class.

        Args:
            task: The task description
            context: The task context
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The processing result
        """
        pass
