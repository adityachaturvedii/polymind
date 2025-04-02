"""
Software Architect Agent module for the AI Agent System.

This module implements the Software Architect agent that is responsible for
designing the software architecture for the system.
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


class SoftwareArchitect(BaseAgent):
    """
    Software Architect Agent for the AI Agent System.

    The SoftwareArchitect is responsible for:
    - Designing overall system architecture
    - Defining interfaces and APIs
    - Ensuring scalability and maintainability
    - Selecting appropriate technologies
    - Establishing coding standards
    """

    def __init__(
        self,
        client: Anthropic,
        memory: MemoryManager,
        retriever: Retriever,
        config: AgentConfig,
    ):
        """
        Initialize the SoftwareArchitect agent.

        Args:
            client: Anthropic API client
            memory: Memory manager
            retriever: RAG retriever
            config: Agent configuration
        """
        super().__init__(client, memory, retriever, config, "sw_architect")

    def _get_role_specific_prompt(self) -> str:
        """
        Get the role-specific part of the system prompt.

        Returns:
            str: The role-specific prompt
        """
        return """As the Software Architect, your responsibilities include:

1. Designing the overall system architecture
2. Defining interfaces, APIs, and data models
3. Ensuring scalability, maintainability, and performance
4. Selecting appropriate technologies and frameworks
5. Establishing coding standards and best practices
6. Making technical decisions about software infrastructure

You should:
- Consider the specific requirements and constraints of the task
- Design clean, modular, and extensible architectures
- Balance technical excellence with practical considerations
- Provide clear rationales for your architectural decisions
- Consider security, reliability, and deployment concerns
- Specify interfaces and communication patterns

Your output should be technically precise, well-structured, and actionable for the Software Engineer.
"""

    def process(
        self, task: str, context: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a task as the Software Architect.

        Args:
            task: The task description
            context: The task context
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The processing result
        """
        # Build context for this task
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for architecture design
        architecture_template = """
# Software Architecture Design

## Task Description
{query}

## Available Context
{rag}

## Your Role
As the Software Architect, design a software architecture for this task:
1. What are the key components and modules needed?
2. What are the interfaces and data flows between components?
3. What technologies and frameworks would be most appropriate?
4. What design patterns should be applied?
5. What are the key technical considerations or challenges?
6. How should the system be structured for maintainability and extensibility?

Provide a structured architecture design that can guide the Software Engineer's implementation.
"""
        
        architecture_prompt = self._format_prompt(full_context, architecture_template)
        
        # Get architecture design
        architecture_design = self._call_llm(architecture_prompt)
        
        # Store architecture design in memory
        self._store_in_memory(
            {
                "action": "architecture_design",
                "task": task,
                "architecture_design": architecture_design,
                "summary": self._create_summary({"action": "architecture_design", "result": architecture_design}, 0),
            },
            task_id,
        )
        
        # Return the architecture design
        return {
            "action": "architecture_design",
            "architecture_design": architecture_design,
        }

    def design_architecture(
        self, 
        task: str, 
        requirements: Dict[str, Any], 
        task_id: Optional[str] = None,
        ml_architecture: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Design software architecture for a task.

        Args:
            task: The task description
            requirements: The task requirements
            task_id: Optional task identifier
            ml_architecture: Optional ML architecture to integrate with

        Returns:
            Dict[str, Any]: The architecture design
        """
        # Build context for architecture design
        context = {
            "task": task,
            "requirements": requirements,
        }
        
        # Add ML architecture if provided
        if ml_architecture:
            context["ml_architecture"] = ml_architecture
        
        full_context = self._build_context(task, task_id, **context)
        
        # Determine if this is an ML-integrated system
        has_ml = ml_architecture is not None
        
        # Choose appropriate design template
        if has_ml:
            design_template = self._get_ml_integrated_design_template()
        else:
            design_template = self._get_standard_design_template()
        
        # Format prompt
        design_prompt = self._format_prompt(full_context, design_template)
        
        # Get detailed architecture design
        detailed_design = self._call_llm(design_prompt)
        
        # Parse the detailed design to extract structured architecture
        # In a real implementation, this would use more sophisticated parsing
        structured_architecture = self._parse_architecture(detailed_design, has_ml)
        
        # Store detailed architecture design in memory
        self._store_in_memory(
            {
                "action": "detailed_architecture_design",
                "task": task,
                "has_ml": has_ml,
                "detailed_design": detailed_design,
                "structured_architecture": structured_architecture,
                "summary": self._create_summary({"action": "detailed_architecture_design", "result": structured_architecture}, 0),
            },
            task_id,
        )
        
        # Return the structured architecture
        return structured_architecture

    def _get_standard_design_template(self) -> str:
        """
        Get the design template for standard software systems.

        Returns:
            str: The design template
        """
        return """
# Software Architecture Design

## Task Description
{query}

## Requirements
{requirements}

## Your Role
As the Software Architect, design a software architecture for this task:

1. **System Overview**:
   - What is the high-level architecture style (monolithic, microservices, etc.)?
   - What are the key components and their responsibilities?
   - How do the components interact with each other?

2. **Component Design**:
   - What are the main modules or services needed?
   - What are the interfaces and contracts between components?
   - What design patterns should be applied?

3. **Data Architecture**:
   - What data models are needed?
   - How should data be stored and accessed?
   - What data flows exist in the system?

4. **Technology Stack**:
   - What programming languages are most appropriate?
   - What frameworks and libraries should be used?
   - What infrastructure components are needed?

5. **Cross-Cutting Concerns**:
   - How should error handling be implemented?
   - What security considerations need to be addressed?
   - How should logging, monitoring, and observability be handled?

6. **Deployment Considerations**:
   - How should the system be deployed?
   - What scalability requirements need to be addressed?
   - What are the performance considerations?

Provide a comprehensive architecture design with clear rationales for your decisions.
"""

    def _get_ml_integrated_design_template(self) -> str:
        """
        Get the design template for ML-integrated software systems.

        Returns:
            str: The design template
        """
        return """
# ML-Integrated Software Architecture Design

## Task Description
{query}

## Requirements
{requirements}

## ML Architecture
{ml_architecture}

## Your Role
As the Software Architect, design a software architecture that integrates with the ML architecture:

1. **System Overview**:
   - What is the high-level architecture style (monolithic, microservices, etc.)?
   - What are the key components and their responsibilities?
   - How do the components interact with each other and the ML components?

2. **Component Design**:
   - What are the main modules or services needed?
   - What are the interfaces between software components and ML components?
   - What design patterns should be applied for ML integration?

3. **Data Architecture**:
   - What data models are needed for both application and ML data?
   - How should data be stored, processed, and accessed?
   - What data pipelines are needed for ML training and inference?

4. **Technology Stack**:
   - What programming languages are most appropriate?
   - What frameworks and libraries should be used?
   - What infrastructure components are needed for both software and ML?

5. **ML Integration Points**:
   - How should ML models be served and accessed?
   - What interfaces are needed for model training and inference?
   - How should model versioning and updates be handled?

6. **Cross-Cutting Concerns**:
   - How should error handling be implemented, especially for ML components?
   - What security considerations need to be addressed?
   - How should logging, monitoring, and observability be handled?

7. **Deployment Considerations**:
   - How should the integrated system be deployed?
   - What scalability requirements need to be addressed?
   - What are the performance considerations for ML inference?

Provide a comprehensive architecture design with clear rationales for your decisions.
"""

    def _parse_architecture(self, detailed_design: str, has_ml: bool) -> Dict[str, Any]:
        """
        Parse detailed architecture design into structured format.

        Args:
            detailed_design: The detailed architecture design text
            has_ml: Whether the architecture includes ML components

        Returns:
            Dict[str, Any]: Structured architecture
        """
        # In a real implementation, this would use more sophisticated parsing
        # For now, just return a simple structured format
        
        # Extract sections based on headings
        sections = {}
        current_section = None
        current_content = []
        
        for line in detailed_design.split("\n"):
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
        
        # Extract specific architecture components
        system_overview = sections.get("System Overview", "")
        component_design = sections.get("Component Design", "")
        data_architecture = sections.get("Data Architecture", "")
        technology_stack = sections.get("Technology Stack", "")
        cross_cutting_concerns = sections.get("Cross-Cutting Concerns", "")
        deployment_considerations = sections.get("Deployment Considerations", "")
        
        # For ML-integrated systems, also extract ML integration points
        ml_integration_points = sections.get("ML Integration Points", "")
        
        # Create structured architecture
        structured_architecture = {
            "has_ml": has_ml,
            "system_overview": system_overview,
            "component_design": component_design,
            "data_architecture": data_architecture,
            "technology_stack": technology_stack,
            "cross_cutting_concerns": cross_cutting_concerns,
            "deployment_considerations": deployment_considerations,
            "full_design": detailed_design,
        }
        
        # Add ML integration points for ML-integrated systems
        if has_ml:
            structured_architecture["ml_integration_points"] = ml_integration_points
        
        return structured_architecture

    def evaluate_implementation(
        self, task: str, implementation: Dict[str, Any], architecture: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an implementation against architectural requirements.

        Args:
            task: The task description
            implementation: The implementation to evaluate
            architecture: The architecture to evaluate against
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The evaluation result
        """
        # Build context for evaluation
        context = {
            "task": task,
            "implementation": implementation,
            "architecture": architecture,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for evaluation
        evaluate_template = """
# Software Architecture Evaluation

## Task Description
{query}

## Architecture Design
{architecture}

## Implementation
{implementation}

## Your Role
As the Software Architect, evaluate the implementation against the architectural design:
1. Does the implementation follow the recommended architecture?
2. Are there any deviations from the design, and if so, are they justified?
3. Does the implementation address the key technical considerations?
4. Are there any architectural improvements that could be made?
5. What is your overall assessment of the implementation?

Provide a structured evaluation with specific references to the architecture and implementation.
"""
        
        evaluate_prompt = self._format_prompt(full_context, evaluate_template)
        
        # Get evaluation result
        evaluation = self._call_llm(evaluate_prompt)
        
        # Store evaluation in memory
        self._store_in_memory(
            {
                "action": "evaluate_implementation",
                "task": task,
                "implementation": implementation,
                "architecture": architecture,
                "evaluation": evaluation,
                "summary": self._create_summary({"action": "evaluate_implementation", "result": evaluation}, 0),
            },
            task_id,
        )
        
        # Return the evaluation result
        return {
            "action": "evaluate_implementation",
            "evaluation": evaluation,
        }

    def design_api(
        self, task: str, requirements: Dict[str, Any], architecture: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Design an API based on requirements and architecture.

        Args:
            task: The task description
            requirements: The requirements for the API
            architecture: The system architecture
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The API design
        """
        # Build context for API design
        context = {
            "task": task,
            "requirements": requirements,
            "architecture": architecture,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for API design
        api_template = """
# API Design

## Task Description
{query}

## Requirements
{requirements}

## Architecture
{architecture}

## Your Role
As the Software Architect, design an API for this system:
1. What API style is most appropriate (REST, GraphQL, gRPC, etc.)?
2. What are the key endpoints or operations?
3. What are the request and response formats?
4. What authentication and authorization mechanisms should be used?
5. What are the error handling and status code conventions?
6. What are the versioning and backward compatibility considerations?

Provide a comprehensive API design with clear rationales for your decisions.
"""
        
        api_prompt = self._format_prompt(full_context, api_template)
        
        # Get API design
        api_design = self._call_llm(api_prompt)
        
        # Store API design in memory
        self._store_in_memory(
            {
                "action": "design_api",
                "task": task,
                "requirements": requirements,
                "architecture": architecture,
                "api_design": api_design,
                "summary": self._create_summary({"action": "design_api", "result": api_design}, 0),
            },
            task_id,
        )
        
        # Return the API design
        return {
            "action": "design_api",
            "api_design": api_design,
        }
