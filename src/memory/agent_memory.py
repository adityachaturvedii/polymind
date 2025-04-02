"""
Agent Memory module for the AI Agent System.

This module implements the agent-specific memory component that manages
context for each agent role.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from src.core.config import MemoryConfig

logger = logging.getLogger(__name__)


class AgentMemory:
    """
    Agent-specific Memory for the AI Agent System.

    The AgentMemory handles:
    - Agent states: Internal state of each agent
    - Agent context windows: Context for each agent role
    - Agent work summaries: Compact representation of agent outputs
    """

    def __init__(self, config: MemoryConfig):
        """
        Initialize the AgentMemory.

        Args:
            config: Memory system configuration
        """
        self.config = config
        
        # Initialize agent states
        self.agent_states = {}
        
        # Initialize agent context windows
        self.agent_contexts = {}
        
        # Initialize agent work summaries
        self.agent_summaries = {}
        
        logger.info("Agent Memory initialized")

    def update(self, agent_id: str, data: Dict[str, Any]) -> None:
        """
        Update memory for a specific agent.

        Args:
            agent_id: The agent identifier
            data: The data to store
        """
        # Initialize agent memory if not present
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = {}
        
        if agent_id not in self.agent_contexts:
            self.agent_contexts[agent_id] = []
        
        if agent_id not in self.agent_summaries:
            self.agent_summaries[agent_id] = []
        
        # Update agent state
        if "state" in data:
            self._update_agent_state(agent_id, data["state"])
        
        # Update agent context
        if "context" in data:
            self._update_agent_context(agent_id, data["context"])
        
        # Update agent work summary
        if "summary" in data:
            self._update_agent_summary(agent_id, data["summary"])
        
        # Update other agent-specific data
        for key, value in data.items():
            if key not in ["state", "context", "summary"]:
                self.agent_states[agent_id][key] = value

    def _update_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        """
        Update the state for a specific agent.

        Args:
            agent_id: The agent identifier
            state: The state to store
        """
        self.agent_states[agent_id].update(state)
        logger.debug(f"Updated state for agent {agent_id}")

    def _update_agent_context(self, agent_id: str, context: Dict[str, Any]) -> None:
        """
        Update the context for a specific agent.

        Args:
            agent_id: The agent identifier
            context: The context to store
        """
        # Get context window size for the agent
        context_window_size = self.config.context_window_sizes.get(
            agent_id, 6000  # Default size if not specified
        )
        
        # Add new context
        self.agent_contexts[agent_id].append(context)
        
        # Prune context if it exceeds the window size
        # In a real implementation, this would use token counting and more sophisticated pruning
        if len(self.agent_contexts[agent_id]) > context_window_size // 100:  # Rough approximation
            self.agent_contexts[agent_id].pop(0)
            logger.debug(f"Pruned context for agent {agent_id}")

    def _update_agent_summary(self, agent_id: str, summary: Dict[str, Any]) -> None:
        """
        Update the work summary for a specific agent.

        Args:
            agent_id: The agent identifier
            summary: The summary to store
        """
        # Add new summary
        self.agent_summaries[agent_id].append(summary)
        
        # Keep only the most recent summaries
        max_summaries = self.config.summary_levels
        if len(self.agent_summaries[agent_id]) > max_summaries:
            self.agent_summaries[agent_id].pop(0)
            logger.debug(f"Pruned summary for agent {agent_id}")

    def get_state(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the state for a specific agent.

        Args:
            agent_id: The agent identifier

        Returns:
            Dict[str, Any]: The agent state
        """
        return self.agent_states.get(agent_id, {})

    def get_context(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get the context for a specific agent.

        Args:
            agent_id: The agent identifier

        Returns:
            List[Dict[str, Any]]: The agent context
        """
        return self.agent_contexts.get(agent_id, [])

    def get_summaries(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get the work summaries for a specific agent.

        Args:
            agent_id: The agent identifier

        Returns:
            List[Dict[str, Any]]: The agent work summaries
        """
        return self.agent_summaries.get(agent_id, [])

    def get_all_agent_data(self, agent_id: str) -> Dict[str, Any]:
        """
        Get all data for a specific agent.

        Args:
            agent_id: The agent identifier

        Returns:
            Dict[str, Any]: All agent data
        """
        return {
            "state": self.get_state(agent_id),
            "context": self.get_context(agent_id),
            "summaries": self.get_summaries(agent_id),
        }

    def clear_agent(self, agent_id: str) -> None:
        """
        Clear memory for a specific agent.

        Args:
            agent_id: The agent identifier
        """
        if agent_id in self.agent_states:
            del self.agent_states[agent_id]
        
        if agent_id in self.agent_contexts:
            del self.agent_contexts[agent_id]
        
        if agent_id in self.agent_summaries:
            del self.agent_summaries[agent_id]
            
        logger.info(f"Cleared memory for agent {agent_id}")

    def clear_all_agents(self) -> None:
        """Clear memory for all agents."""
        self.agent_states.clear()
        self.agent_contexts.clear()
        self.agent_summaries.clear()
        logger.info("Cleared memory for all agents")

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of agent memory.

        Returns:
            Dict[str, Any]: The current state
        """
        return {
            "agent_states": self.agent_states,
            "agent_contexts": self.agent_contexts,
            "agent_summaries": self.agent_summaries,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the state of agent memory.

        Args:
            state: The state to set
        """
        if "agent_states" in state:
            self.agent_states = state["agent_states"]
        
        if "agent_contexts" in state:
            self.agent_contexts = state["agent_contexts"]
        
        if "agent_summaries" in state:
            self.agent_summaries = state["agent_summaries"]
            
        logger.info("Agent memory state restored")

    def create_hierarchical_summary(self, agent_id: str, level: int = 0) -> str:
        """
        Create a hierarchical summary of agent work.

        This method generates a summary of the agent's work at different levels
        of abstraction, based on the agent's work summaries.

        Args:
            agent_id: The agent identifier
            level: The summary level (0 = most detailed, higher = more abstract)

        Returns:
            str: The hierarchical summary
        """
        # Get agent summaries
        summaries = self.get_summaries(agent_id)
        
        # If no summaries, return empty string
        if not summaries:
            return ""
        
        # If level is out of range, use the highest available level
        if level >= len(summaries):
            level = len(summaries) - 1
        
        # Get summary at the specified level
        summary = summaries[level]
        
        # In a real implementation, this would generate a proper summary
        # based on the agent's work and the requested level of abstraction
        
        # For now, just return a simple string representation
        if isinstance(summary, dict):
            return str(summary)
        else:
            return str(summary)
