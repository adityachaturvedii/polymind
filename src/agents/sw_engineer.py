"""
Software Engineer Agent module for the AI Agent System.

This module implements the Software Engineer agent that is responsible for
implementing the software components of the system.
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


class SoftwareEngineer(BaseAgent):
    """
    Software Engineer Agent for the AI Agent System.

    The SoftwareEngineer is responsible for:
    - Implementing non-ML components
    - Developing APIs and interfaces
    - Handling system integration
    - Optimizing code performance
    - Implementing testing frameworks
    """

    def __init__(
        self,
        client: Anthropic,
        memory: MemoryManager,
        retriever: Retriever,
        config: AgentConfig,
    ):
        """
        Initialize the SoftwareEngineer agent.

        Args:
            client: Anthropic API client
            memory: Memory manager
            retriever: RAG retriever
            config: Agent configuration
        """
        super().__init__(client, memory, retriever, config, "sw_engineer")

    def _get_role_specific_prompt(self) -> str:
        """
        Get the role-specific part of the system prompt.

        Returns:
            str: The role-specific prompt
        """
        return """As the Software Engineer, your responsibilities include:

1. Implementing non-ML components based on architectural designs
2. Developing APIs, interfaces, and integration points
3. Handling system integration and data flow
4. Optimizing code performance and resource usage
5. Implementing testing frameworks and utilities
6. Ensuring code quality, maintainability, and security

You should:
- Write clean, efficient, and well-documented code
- Follow software engineering best practices and design patterns
- Implement components that are modular, testable, and maintainable
- Provide clear explanations of your implementation choices
- Consider edge cases, error handling, and security concerns
- Ensure compatibility with ML components when needed

Your output should be technically precise, well-structured, and include working code implementations.
"""

    def process(
        self, task: str, context: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a task as the Software Engineer.

        Args:
            task: The task description
            context: The task context
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The processing result
        """
        # Build context for this task
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for software implementation
        implementation_template = """
# Software Implementation

## Task Description
{query}

## Available Context
{rag}

## Your Role
As the Software Engineer, implement a software solution for this task:
1. What specific components or modules are needed?
2. How should the interfaces and data flows be implemented?
3. What design patterns and best practices should be applied?
4. How should error handling and edge cases be managed?
5. How should the solution be tested and validated?
6. What are the key implementation considerations or challenges?

Provide a structured implementation plan with code snippets for key components.
"""
        
        implementation_prompt = self._format_prompt(full_context, implementation_template)
        
        # Get implementation plan
        implementation_plan = self._call_llm(implementation_prompt)
        
        # Store implementation plan in memory
        self._store_in_memory(
            {
                "action": "implementation_plan",
                "task": task,
                "implementation_plan": implementation_plan,
                "summary": self._create_summary({"action": "implementation_plan", "result": implementation_plan}, 0),
            },
            task_id,
        )
        
        # Return the implementation plan
        return {
            "action": "implementation_plan",
            "implementation_plan": implementation_plan,
        }

    def implement(
        self,
        task: str,
        requirements: Dict[str, Any],
        architecture: Dict[str, Any],
        ml_implementation: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Implement a software solution based on requirements and architecture.

        Args:
            task: The task description
            requirements: The task requirements
            architecture: The software architecture
            ml_implementation: Optional ML implementation to integrate with
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The implementation
        """
        # Build context for implementation
        context = {
            "task": task,
            "requirements": requirements,
            "architecture": architecture,
        }
        
        # Add ML implementation if provided
        if ml_implementation:
            context["ml_implementation"] = ml_implementation
        
        full_context = self._build_context(task, task_id, **context)
        
        # Determine if this is an ML-integrated system
        has_ml = ml_implementation is not None and architecture.get("has_ml", False)
        
        # Choose appropriate implementation template
        if has_ml:
            implementation_template = self._get_ml_integrated_implementation_template()
        else:
            implementation_template = self._get_standard_implementation_template()
        
        # Format prompt
        implementation_prompt = self._format_prompt(full_context, implementation_template)
        
        # Get detailed implementation
        detailed_implementation = self._call_llm(implementation_prompt)
        
        # Parse the detailed implementation to extract structured implementation
        # In a real implementation, this would use more sophisticated parsing
        structured_implementation = self._parse_implementation(detailed_implementation, has_ml)
        
        # Store detailed implementation in memory
        self._store_in_memory(
            {
                "action": "detailed_implementation",
                "task": task,
                "has_ml": has_ml,
                "detailed_implementation": detailed_implementation,
                "structured_implementation": structured_implementation,
                "summary": self._create_summary({"action": "detailed_implementation", "result": structured_implementation}, 0),
            },
            task_id,
        )
        
        # Return the structured implementation
        return structured_implementation

    def _get_standard_implementation_template(self) -> str:
        """
        Get the implementation template for standard software systems.

        Returns:
            str: The implementation template
        """
        return """
# Software Implementation

## Task Description
{query}

## Requirements
{requirements}

## Architecture
{architecture}

## Your Role
As the Software Engineer, implement a software solution based on the architecture:

1. **Component Implementation**:
   - How should each component be implemented?
   - What specific classes and functions are needed?
   - How should the interfaces between components be implemented?

2. **Data Model Implementation**:
   - How should the data models be implemented?
   - What validation and serialization is needed?
   - How should data persistence be handled?

3. **API Implementation**:
   - How should the APIs be implemented?
   - What request/response handling is needed?
   - How should authentication and authorization be implemented?

4. **Error Handling Implementation**:
   - How should errors be handled and reported?
   - What exception hierarchy should be used?
   - How should edge cases be managed?

5. **Testing Implementation**:
   - How should unit tests be implemented?
   - What integration tests are needed?
   - How should test fixtures and mocks be set up?

6. **Deployment Implementation**:
   - How should the deployment process be implemented?
   - What configuration management is needed?
   - How should environment variables be handled?

Provide a comprehensive implementation with code snippets for each component.
"""

    def _get_ml_integrated_implementation_template(self) -> str:
        """
        Get the implementation template for ML-integrated software systems.

        Returns:
            str: The implementation template
        """
        return """
# ML-Integrated Software Implementation

## Task Description
{query}

## Requirements
{requirements}

## Architecture
{architecture}

## ML Implementation
{ml_implementation}

## Your Role
As the Software Engineer, implement a software solution that integrates with the ML implementation:

1. **Component Implementation**:
   - How should each component be implemented?
   - What specific classes and functions are needed?
   - How should the interfaces between components be implemented?

2. **Data Model Implementation**:
   - How should the data models be implemented?
   - What validation and serialization is needed?
   - How should data be prepared for ML components?

3. **ML Integration Implementation**:
   - How should the ML models be integrated?
   - What interfaces are needed for model inference?
   - How should model inputs and outputs be handled?

4. **API Implementation**:
   - How should the APIs be implemented?
   - What request/response handling is needed?
   - How should ML results be exposed through APIs?

5. **Error Handling Implementation**:
   - How should errors be handled and reported?
   - What ML-specific error handling is needed?
   - How should edge cases be managed?

6. **Testing Implementation**:
   - How should unit tests be implemented?
   - What integration tests are needed for ML components?
   - How should test fixtures and mocks be set up?

7. **Deployment Implementation**:
   - How should the deployment process be implemented?
   - What ML-specific deployment considerations exist?
   - How should model versioning be handled?

Provide a comprehensive implementation with code snippets for each component.
"""

    def _parse_implementation(self, detailed_implementation: str, has_ml: bool) -> Dict[str, Any]:
        """
        Parse detailed implementation into structured format.

        Args:
            detailed_implementation: The detailed implementation text
            has_ml: Whether the implementation includes ML integration

        Returns:
            Dict[str, Any]: Structured implementation
        """
        # In a real implementation, this would use more sophisticated parsing
        # For now, just return a simple structured format
        
        # Extract sections based on headings
        sections = {}
        current_section = None
        current_content = []
        
        for line in detailed_implementation.split("\n"):
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
        
        # Extract specific implementation components
        component_implementation = sections.get("Component Implementation", "")
        data_model_implementation = sections.get("Data Model Implementation", "")
        api_implementation = sections.get("API Implementation", "")
        error_handling_implementation = sections.get("Error Handling Implementation", "")
        testing_implementation = sections.get("Testing Implementation", "")
        deployment_implementation = sections.get("Deployment Implementation", "")
        
        # For ML-integrated systems, also extract ML integration
        ml_integration_implementation = sections.get("ML Integration Implementation", "")
        
        # Create structured implementation
        structured_implementation = {
            "has_ml": has_ml,
            "component_implementation": component_implementation,
            "data_model_implementation": data_model_implementation,
            "api_implementation": api_implementation,
            "error_handling_implementation": error_handling_implementation,
            "testing_implementation": testing_implementation,
            "deployment_implementation": deployment_implementation,
            "full_implementation": detailed_implementation,
        }
        
        # Add ML integration for ML-integrated systems
        if has_ml:
            structured_implementation["ml_integration_implementation"] = ml_integration_implementation
        
        return structured_implementation

    def implement_component(
        self,
        task: str,
        requirements: Dict[str, Any],
        architecture: Dict[str, Any],
        component: str,
        language: Optional[str] = "python",
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Implement a specific software component.

        Args:
            task: The task description
            requirements: The task requirements
            architecture: The software architecture
            component: The component to implement
            language: The programming language to use
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The component implementation
        """
        # Build context for component implementation
        context = {
            "task": task,
            "requirements": requirements,
            "architecture": architecture,
            "component": component,
            "language": language,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for component implementation
        component_template = """
# Software Component Implementation

## Task Description
{query}

## Requirements
{requirements}

## Architecture
{architecture}

## Component to Implement
{component}

## Programming Language
{language}

## Your Role
As the Software Engineer, implement the {component} component in {language}:
1. What specific functionality does this component need to provide?
2. How does this component fit into the overall architecture?
3. What are the interfaces and data flows for this component?
4. What design patterns and best practices should be applied?
5. How should error handling and edge cases be managed?
6. How should this component be tested?

Provide a detailed implementation with code for the {component} component.
"""
        
        component_prompt = self._format_prompt(full_context, component_template)
        
        # Get component implementation
        component_implementation = self._call_llm(component_prompt)
        
        # Store component implementation in memory
        self._store_in_memory(
            {
                "action": "implement_component",
                "task": task,
                "component": component,
                "language": language,
                "component_implementation": component_implementation,
                "summary": self._create_summary({"action": "implement_component", "result": component_implementation}, 0),
            },
            task_id,
        )
        
        # Return the component implementation
        return {
            "action": "implement_component",
            "component": component,
            "language": language,
            "implementation": component_implementation,
        }

    def implement_api(
        self,
        task: str,
        requirements: Dict[str, Any],
        architecture: Dict[str, Any],
        api_design: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Implement an API based on requirements, architecture, and API design.

        Args:
            task: The task description
            requirements: The task requirements
            architecture: The software architecture
            api_design: The API design
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The API implementation
        """
        # Build context for API implementation
        context = {
            "task": task,
            "requirements": requirements,
            "architecture": architecture,
            "api_design": api_design,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for API implementation
        api_template = """
# API Implementation

## Task Description
{query}

## Requirements
{requirements}

## Architecture
{architecture}

## API Design
{api_design}

## Your Role
As the Software Engineer, implement the API based on the design:
1. How should the API endpoints be implemented?
2. How should request validation and parsing be handled?
3. How should response formatting be implemented?
4. How should authentication and authorization be implemented?
5. How should error handling and status codes be managed?
6. How should the API be tested?

Provide a detailed implementation with code for the API.
"""
        
        api_prompt = self._format_prompt(full_context, api_template)
        
        # Get API implementation
        api_implementation = self._call_llm(api_prompt)
        
        # Store API implementation in memory
        self._store_in_memory(
            {
                "action": "implement_api",
                "task": task,
                "api_design": api_design,
                "api_implementation": api_implementation,
                "summary": self._create_summary({"action": "implement_api", "result": api_implementation}, 0),
            },
            task_id,
        )
        
        # Return the API implementation
        return {
            "action": "implement_api",
            "api_implementation": api_implementation,
        }

    def optimize_code(
        self,
        task: str,
        code: str,
        performance_requirements: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Optimize code based on performance requirements.

        Args:
            task: The task description
            code: The code to optimize
            performance_requirements: The performance requirements
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The optimized code
        """
        # Build context for code optimization
        context = {
            "task": task,
            "code": code,
            "performance_requirements": performance_requirements,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for code optimization
        optimize_template = """
# Code Optimization

## Task Description
{query}

## Current Code
```
{code}
```

## Performance Requirements
{performance_requirements}

## Your Role
As the Software Engineer, optimize the code to meet performance requirements:
1. What are the key performance bottlenecks or issues?
2. What optimization techniques would be most effective?
3. How can the code be refactored for better performance?
4. What data structures or algorithms should be changed?
5. How should the optimized code be tested?

Provide a detailed optimization with the improved code.
"""
        
        optimize_prompt = self._format_prompt(full_context, optimize_template)
        
        # Get optimized code
        optimized_code = self._call_llm(optimize_prompt)
        
        # Store optimized code in memory
        self._store_in_memory(
            {
                "action": "optimize_code",
                "task": task,
                "original_code": code,
                "performance_requirements": performance_requirements,
                "optimized_code": optimized_code,
                "summary": self._create_summary({"action": "optimize_code", "result": optimized_code}, 0),
            },
            task_id,
        )
        
        # Return the optimized code
        return {
            "action": "optimize_code",
            "optimized_code": optimized_code,
        }
