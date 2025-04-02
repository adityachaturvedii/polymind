"""
Team Leader Agent module for the AI Agent System.

This module implements the Team Leader agent that is responsible for
coordinating the other agents and making final decisions.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from anthropic import Anthropic

from src.agents.base_agent import BaseAgent
from src.core.config import AgentConfig
from src.memory.memory_manager import MemoryManager
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class TeamLeader(BaseAgent):
    """
    Team Leader Agent for the AI Agent System.

    The TeamLeader is responsible for:
    - Coordinating between all agents
    - Resolving conflicts in approach
    - Ensuring solution coherence
    - Maintaining project scope
    - Synthesizing final outputs
    """

    def __init__(
        self,
        client: Anthropic,
        memory: MemoryManager,
        retriever: Retriever,
        config: AgentConfig,
    ):
        """
        Initialize the TeamLeader agent.

        Args:
            client: Anthropic API client
            memory: Memory manager
            retriever: RAG retriever
            config: Agent configuration
        """
        super().__init__(client, memory, retriever, config, "team_leader")

    def _get_role_specific_prompt(self) -> str:
        """
        Get the role-specific part of the system prompt.

        Returns:
            str: The role-specific prompt
        """
        return """As the Team Leader, your responsibilities include:

1. Strategic oversight and final decision-making
2. Coordinating between all specialized agents
3. Resolving conflicts in approach or recommendations
4. Ensuring solution coherence and alignment with user needs
5. Maintaining project scope and focus
6. Synthesizing inputs from other agents into a cohesive final output

You should:
- Take a high-level, strategic view of the task
- Consider trade-offs between different approaches
- Ensure all aspects of the task are addressed
- Make clear, decisive recommendations when there are conflicting views
- Provide clear direction to other agents
- Synthesize information from multiple sources into a coherent whole

Your output should be well-structured, comprehensive, and actionable.
"""

    def process(
        self, task: str, context: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a task as the Team Leader.

        Args:
            task: The task description
            context: The task context
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The processing result
        """
        # Build context for this task
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for initial task assessment
        assessment_template = """
# Task Assessment

## Task Description
{query}

## Available Context
{rag}

## Your Role
As the Team Leader, provide an initial assessment of this task:
1. What is the core objective?
2. What are the key challenges or constraints?
3. What specialized expertise will be needed?
4. What is the recommended approach at a high level?
5. How should we divide the work among the specialized agents?

Provide a structured assessment that can guide our team's approach.
"""
        
        assessment_prompt = self._format_prompt(full_context, assessment_template)
        
        # Get initial assessment
        assessment = self._call_llm(assessment_prompt)
        
        # Store assessment in memory
        self._store_in_memory(
            {
                "action": "initial_assessment",
                "task": task,
                "assessment": assessment,
                "summary": self._create_summary({"action": "initial_assessment", "result": assessment}, 0),
            },
            task_id,
        )
        
        # Return the assessment
        return {
            "action": "initial_assessment",
            "assessment": assessment,
        }

    def finalize(
        self, task: str, task_context: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Finalize a task by synthesizing all agent outputs.

        Args:
            task: The task description
            task_context: The complete task context including all agent outputs
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The finalized result
        """
        # Build context for finalization
        full_context = self._build_context(task, task_id, **task_context)
        
        # Create prompt for finalization
        finalize_template = """
# Task Finalization

## Task Description
{query}

## Requirements
{requirements}

## Architecture
{sw_architecture}

## Implementation
{sw_implementation}

## Test Results
{test_results}

## Your Role
As the Team Leader, synthesize all the information above into a final, cohesive solution:
1. Summarize the approach taken
2. Highlight key design decisions and their rationales
3. Present the final solution in a clear, structured format
4. Identify any limitations or areas for future improvement
5. Provide a final assessment of how well the solution meets the requirements

Your output should be comprehensive yet concise, focusing on the most important aspects of the solution.
"""
        
        finalize_prompt = self._format_prompt(full_context, finalize_template)
        
        # Get finalized solution
        finalized = self._call_llm(finalize_prompt)
        
        # Store finalized solution in memory
        self._store_in_memory(
            {
                "action": "finalize",
                "task": task,
                "finalized": finalized,
                "summary": self._create_summary({"action": "finalize", "result": finalized}, 0),
            },
            task_id,
        )
        
        # Return the finalized solution
        return {
            "action": "finalize",
            "result": finalized,
        }

    def resolve_conflict(
        self, 
        task: str, 
        conflict: Dict[str, Any], 
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resolve a conflict between agents.

        Args:
            task: The task description
            conflict: The conflict details
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The resolution
        """
        # Build context for conflict resolution
        full_context = self._build_context(task, task_id, **conflict)
        
        # Create prompt for conflict resolution
        resolve_template = """
# Conflict Resolution

## Task Description
{query}

## Conflict Description
{conflict_description}

## Position A: {agent_a}
{position_a}

## Position B: {agent_b}
{position_b}

## Your Role
As the Team Leader, resolve this conflict by:
1. Analyzing the merits of each position
2. Identifying areas of agreement and disagreement
3. Considering the overall project goals and constraints
4. Making a clear decision or finding a compromise
5. Providing a rationale for your decision

Your resolution should be clear, fair, and focused on the best outcome for the project.
"""
        
        resolve_prompt = self._format_prompt(full_context, resolve_template)
        
        # Get resolution
        resolution = self._call_llm(resolve_prompt)
        
        # Store resolution in memory
        self._store_in_memory(
            {
                "action": "resolve_conflict",
                "task": task,
                "conflict": conflict,
                "resolution": resolution,
                "summary": self._create_summary({"action": "resolve_conflict", "result": resolution}, 0),
            },
            task_id,
        )
        
        # Return the resolution
        return {
            "action": "resolve_conflict",
            "resolution": resolution,
        }

    def create_plan(
        self, task: str, requirements: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a plan for executing a task.

        Args:
            task: The task description
            requirements: The task requirements
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The plan
        """
        # Build context for planning
        full_context = self._build_context(task, task_id, **requirements)
        
        # Create prompt for planning
        plan_template = """
# Task Planning

## Task Description
{query}

## Requirements
{requirements}

## Your Role
As the Team Leader, create a comprehensive plan for executing this task:
1. Break down the task into clear, manageable steps
2. Assign responsibilities to the appropriate specialized agents
3. Identify dependencies between steps
4. Establish criteria for success at each step
5. Anticipate potential challenges and how to address them

Your plan should be structured, detailed, and actionable, providing clear guidance for the team.
"""
        
        plan_prompt = self._format_prompt(full_context, plan_template)
        
        # Get plan
        plan = self._call_llm(plan_prompt)
        
        # Store plan in memory
        self._store_in_memory(
            {
                "action": "create_plan",
                "task": task,
                "requirements": requirements,
                "plan": plan,
                "summary": self._create_summary({"action": "create_plan", "result": plan}, 0),
            },
            task_id,
        )
        
        # Return the plan
        return {
            "action": "create_plan",
            "plan": plan,
        }
