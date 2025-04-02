"""
Product Manager Agent module for the AI Agent System.

This module implements the Product Manager agent that is responsible for
analyzing user requirements and managing the product aspects of the task.
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


class ProductManager(BaseAgent):
    """
    Product Manager Agent for the AI Agent System.

    The ProductManager is responsible for:
    - Analyzing user requirements
    - Prioritizing features and capabilities
    - Defining acceptance criteria
    - Ensuring user-centric solutions
    - Managing scope and expectations
    """

    def __init__(
        self,
        client: Anthropic,
        memory: MemoryManager,
        retriever: Retriever,
        config: AgentConfig,
    ):
        """
        Initialize the ProductManager agent.

        Args:
            client: Anthropic API client
            memory: Memory manager
            retriever: RAG retriever
            config: Agent configuration
        """
        super().__init__(client, memory, retriever, config, "product_manager")

    def _get_role_specific_prompt(self) -> str:
        """
        Get the role-specific part of the system prompt.

        Returns:
            str: The role-specific prompt
        """
        return """As the Product Manager, your responsibilities include:

1. Analyzing and interpreting user requirements
2. Prioritizing features and capabilities based on user needs
3. Defining clear acceptance criteria for the solution
4. Ensuring the solution remains user-centric
5. Managing scope and expectations
6. Communicating requirements clearly to the technical team

You should:
- Focus on understanding the core user needs behind the task
- Distinguish between must-have and nice-to-have features
- Create clear, testable acceptance criteria
- Consider usability, accessibility, and user experience
- Balance user needs with technical constraints
- Provide clear guidance on priorities and trade-offs

Your output should be structured, user-focused, and actionable for the technical team.
"""

    def process(
        self, task: str, context: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a task as the Product Manager.

        Args:
            task: The task description
            context: The task context
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The processing result
        """
        # Build context for this task
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for requirements analysis
        requirements_template = """
# Requirements Analysis

## Task Description
{query}

## Available Context
{rag}

## Your Role
As the Product Manager, analyze the requirements for this task:
1. What are the core user needs and goals?
2. What are the key functional requirements?
3. What are the key non-functional requirements (performance, usability, etc.)?
4. What constraints must be considered?
5. What are the priorities among these requirements?
6. What are the acceptance criteria for a successful solution?

Provide a structured analysis that can guide the technical team's work.
"""
        
        requirements_prompt = self._format_prompt(full_context, requirements_template)
        
        # Get requirements analysis
        requirements_analysis = self._call_llm(requirements_prompt)
        
        # Store requirements analysis in memory
        self._store_in_memory(
            {
                "action": "requirements_analysis",
                "task": task,
                "requirements_analysis": requirements_analysis,
                "summary": self._create_summary({"action": "requirements_analysis", "result": requirements_analysis}, 0),
            },
            task_id,
        )
        
        # Return the requirements analysis
        return {
            "action": "requirements_analysis",
            "requirements_analysis": requirements_analysis,
        }

    def analyze_requirements(
        self, task: str, task_info: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze requirements for a task.

        Args:
            task: The task description
            task_info: Information about the task
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The requirements analysis
        """
        # Build context for requirements analysis
        full_context = self._build_context(task, task_id, **task_info)
        
        # Create prompt for detailed requirements analysis
        detailed_template = """
# Detailed Requirements Analysis

## Task Description
{query}

## Task Information
Domain: {domain}
Task Type: {task_type}
Initial Requirements: {requirements}
Constraints: {constraints}

## Your Role
As the Product Manager, provide a detailed requirements analysis:
1. User Stories: Create user stories in the format "As a [user], I want [feature] so that [benefit]"
2. Functional Requirements: List specific functionalities the solution must provide
3. Non-Functional Requirements: Specify performance, usability, reliability, etc. requirements
4. Constraints: Identify technical, time, resource, or other constraints
5. Acceptance Criteria: Define clear, testable criteria for each requirement
6. Priorities: Classify requirements as Must-Have, Should-Have, Could-Have, Won't-Have

Structure your analysis in a clear, organized format that can be easily referenced by the technical team.
"""
        
        detailed_prompt = self._format_prompt(full_context, detailed_template)
        
        # Get detailed requirements analysis
        detailed_analysis = self._call_llm(detailed_prompt)
        
        # Parse the detailed analysis to extract structured requirements
        # In a real implementation, this would use more sophisticated parsing
        structured_requirements = self._parse_requirements(detailed_analysis)
        
        # Store detailed requirements analysis in memory
        self._store_in_memory(
            {
                "action": "detailed_requirements_analysis",
                "task": task,
                "detailed_analysis": detailed_analysis,
                "structured_requirements": structured_requirements,
                "summary": self._create_summary({"action": "detailed_requirements_analysis", "result": structured_requirements}, 0),
            },
            task_id,
        )
        
        # Return the structured requirements
        return structured_requirements

    def _parse_requirements(self, detailed_analysis: str) -> Dict[str, Any]:
        """
        Parse detailed requirements analysis into structured format.

        Args:
            detailed_analysis: The detailed requirements analysis text

        Returns:
            Dict[str, Any]: Structured requirements
        """
        # In a real implementation, this would use more sophisticated parsing
        # For now, just return a simple structured format
        
        # Extract sections based on headings
        sections = {}
        current_section = None
        current_content = []
        
        for line in detailed_analysis.split("\n"):
            if line.startswith("# ") or line.startswith("## "):
                # Save previous section if it exists
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                    current_content = []
                
                # Start new section
                current_section = line.strip("# ").strip()
            else:
                if current_section:
                    current_content.append(line)
        
        # Save the last section
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()
        
        # Extract specific requirement types
        user_stories = sections.get("User Stories", "")
        functional_requirements = sections.get("Functional Requirements", "")
        non_functional_requirements = sections.get("Non-Functional Requirements", "")
        constraints = sections.get("Constraints", "")
        acceptance_criteria = sections.get("Acceptance Criteria", "")
        priorities = sections.get("Priorities", "")
        
        # Create structured requirements
        structured_requirements = {
            "user_stories": user_stories,
            "functional_requirements": functional_requirements,
            "non_functional_requirements": non_functional_requirements,
            "constraints": constraints,
            "acceptance_criteria": acceptance_criteria,
            "priorities": priorities,
            "full_analysis": detailed_analysis,
        }
        
        return structured_requirements

    def validate_solution(
        self, task: str, solution: Dict[str, Any], requirements: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate a solution against requirements.

        Args:
            task: The task description
            solution: The solution to validate
            requirements: The requirements to validate against
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The validation result
        """
        # Build context for validation
        context = {
            "task": task,
            "solution": solution,
            "requirements": requirements,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for validation
        validate_template = """
# Solution Validation

## Task Description
{query}

## Requirements
{requirements}

## Proposed Solution
{solution}

## Your Role
As the Product Manager, validate the proposed solution against the requirements:
1. Does the solution meet all must-have requirements?
2. Does the solution address the core user needs?
3. Does the solution meet the acceptance criteria?
4. Are there any gaps or missing features?
5. Are there any areas where the solution exceeds requirements?
6. What is your overall assessment of the solution?

Provide a structured validation report with specific references to requirements and solution components.
"""
        
        validate_prompt = self._format_prompt(full_context, validate_template)
        
        # Get validation result
        validation = self._call_llm(validate_prompt)
        
        # Store validation in memory
        self._store_in_memory(
            {
                "action": "validate_solution",
                "task": task,
                "solution": solution,
                "requirements": requirements,
                "validation": validation,
                "summary": self._create_summary({"action": "validate_solution", "result": validation}, 0),
            },
            task_id,
        )
        
        # Return the validation result
        return {
            "action": "validate_solution",
            "validation": validation,
        }

    def prioritize_features(
        self, task: str, features: List[Dict[str, Any]], constraints: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prioritize features based on user needs and constraints.

        Args:
            task: The task description
            features: The features to prioritize
            constraints: The constraints to consider
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The prioritized features
        """
        # Build context for prioritization
        context = {
            "task": task,
            "features": features,
            "constraints": constraints,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for prioritization
        prioritize_template = """
# Feature Prioritization

## Task Description
{query}

## Features
{features}

## Constraints
{constraints}

## Your Role
As the Product Manager, prioritize the features based on user needs and constraints:
1. Classify each feature as Must-Have, Should-Have, Could-Have, or Won't-Have (MoSCoW method)
2. Rank features within each priority category
3. Provide a rationale for each prioritization decision
4. Identify any dependencies between features
5. Suggest a phased implementation approach if appropriate

Your prioritization should balance user needs, technical constraints, and project realities.
"""
        
        prioritize_prompt = self._format_prompt(full_context, prioritize_template)
        
        # Get prioritization result
        prioritization = self._call_llm(prioritize_prompt)
        
        # Store prioritization in memory
        self._store_in_memory(
            {
                "action": "prioritize_features",
                "task": task,
                "features": features,
                "constraints": constraints,
                "prioritization": prioritization,
                "summary": self._create_summary({"action": "prioritize_features", "result": prioritization}, 0),
            },
            task_id,
        )
        
        # Return the prioritization result
        return {
            "action": "prioritize_features",
            "prioritization": prioritization,
        }
