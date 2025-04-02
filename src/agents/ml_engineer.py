"""
ML Engineer Agent module for the AI Agent System.

This module implements the ML Engineer agent that is responsible for
implementing ML models and algorithms.
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


class MLEngineer(BaseAgent):
    """
    ML Engineer Agent for the AI Agent System.

    The MLEngineer is responsible for:
    - Implementing ML models and algorithms
    - Creating training and inference pipelines
    - Optimizing model performance
    - Adapting approaches to specific domains
    - Handling ML-specific debugging
    """

    def __init__(
        self,
        client: Anthropic,
        memory: MemoryManager,
        retriever: Retriever,
        config: AgentConfig,
    ):
        """
        Initialize the MLEngineer agent.

        Args:
            client: Anthropic API client
            memory: Memory manager
            retriever: RAG retriever
            config: Agent configuration
        """
        super().__init__(client, memory, retriever, config, "ml_engineer")

    def _get_role_specific_prompt(self) -> str:
        """
        Get the role-specific part of the system prompt.

        Returns:
            str: The role-specific prompt
        """
        return """As the ML Engineer, your responsibilities include:

1. Implementing ML models and algorithms based on architectural designs
2. Creating training and inference pipelines
3. Optimizing model performance and efficiency
4. Adapting approaches to specific domains (CV, NLP, tabular)
5. Handling ML-specific debugging and troubleshooting
6. Implementing evaluation metrics and validation procedures

You should:
- Write clean, efficient, and well-documented code
- Follow ML best practices and design patterns
- Implement models that balance performance and resource usage
- Provide clear explanations of your implementation choices
- Consider edge cases and error handling
- Ensure reproducibility and maintainability

Your output should be technically precise, well-structured, and include working code implementations.
"""

    def process(
        self, task: str, context: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a task as the ML Engineer.

        Args:
            task: The task description
            context: The task context
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The processing result
        """
        # Build context for this task
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for ML implementation
        implementation_template = """
# ML Implementation

## Task Description
{query}

## Available Context
{rag}

## Your Role
As the ML Engineer, implement an ML solution for this task:
1. What specific ML models or algorithms are needed?
2. How should the data preprocessing pipeline be implemented?
3. How should the training process be structured?
4. How should inference be handled?
5. What evaluation metrics and validation approach should be used?
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
        data_pipeline: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Implement an ML solution based on requirements and architecture.

        Args:
            task: The task description
            requirements: The task requirements
            architecture: The ML architecture
            data_pipeline: Optional data pipeline design
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
        
        # Add data pipeline if provided
        if data_pipeline:
            context["data_pipeline"] = data_pipeline
        
        full_context = self._build_context(task, task_id, **context)
        
        # Determine domain-specific prompt
        domain = architecture.get("domain", "general")
        if domain == "cv":
            implementation_template = self._get_cv_implementation_template()
        elif domain == "nlp":
            implementation_template = self._get_nlp_implementation_template()
        elif domain == "tabular":
            implementation_template = self._get_tabular_implementation_template()
        else:
            implementation_template = self._get_general_implementation_template()
        
        # Format prompt
        implementation_prompt = self._format_prompt(full_context, implementation_template)
        
        # Get detailed implementation
        detailed_implementation = self._call_llm(implementation_prompt)
        
        # Parse the detailed implementation to extract structured implementation
        # In a real implementation, this would use more sophisticated parsing
        structured_implementation = self._parse_implementation(detailed_implementation, domain)
        
        # Store detailed implementation in memory
        self._store_in_memory(
            {
                "action": "detailed_implementation",
                "task": task,
                "domain": domain,
                "detailed_implementation": detailed_implementation,
                "structured_implementation": structured_implementation,
                "summary": self._create_summary({"action": "detailed_implementation", "result": structured_implementation}, 0),
            },
            task_id,
        )
        
        # Return the structured implementation
        return structured_implementation

    def _get_cv_implementation_template(self) -> str:
        """
        Get the implementation template for computer vision tasks.

        Returns:
            str: The implementation template
        """
        return """
# Computer Vision Implementation

## Task Description
{query}

## Requirements
{requirements}

## Architecture
{architecture}

## Data Pipeline
{data_pipeline}

## Your Role
As the ML Engineer, implement a computer vision solution based on the architecture:

1. **Data Preprocessing Implementation**:
   - How should the image preprocessing be implemented?
   - What data augmentation techniques should be used?
   - How should the data loading and batching be handled?

2. **Model Implementation**:
   - How should the model architecture be implemented?
   - What specific layers and components are needed?
   - How should the model be initialized and configured?

3. **Training Implementation**:
   - How should the training loop be structured?
   - How should the loss function and optimizer be implemented?
   - How should checkpointing and early stopping be handled?

4. **Inference Implementation**:
   - How should the inference pipeline be implemented?
   - How should pre and post-processing be handled during inference?
   - How should the model outputs be interpreted?

5. **Evaluation Implementation**:
   - How should the evaluation metrics be implemented?
   - How should the validation process be structured?
   - How should the results be visualized and analyzed?

Provide a comprehensive implementation with code snippets for each component.
"""

    def _get_nlp_implementation_template(self) -> str:
        """
        Get the implementation template for NLP tasks.

        Returns:
            str: The implementation template
        """
        return """
# NLP Implementation

## Task Description
{query}

## Requirements
{requirements}

## Architecture
{architecture}

## Data Pipeline
{data_pipeline}

## Your Role
As the ML Engineer, implement an NLP solution based on the architecture:

1. **Data Preprocessing Implementation**:
   - How should the text preprocessing be implemented?
   - What tokenization approach should be used?
   - How should the data loading and batching be handled?

2. **Model Implementation**:
   - How should the model architecture be implemented?
   - What specific layers and components are needed?
   - How should the model be initialized and configured?

3. **Training Implementation**:
   - How should the training loop be structured?
   - How should the loss function and optimizer be implemented?
   - How should checkpointing and early stopping be handled?

4. **Inference Implementation**:
   - How should the inference pipeline be implemented?
   - How should pre and post-processing be handled during inference?
   - How should the model outputs be interpreted?

5. **Evaluation Implementation**:
   - How should the evaluation metrics be implemented?
   - How should the validation process be structured?
   - How should the results be visualized and analyzed?

Provide a comprehensive implementation with code snippets for each component.
"""

    def _get_tabular_implementation_template(self) -> str:
        """
        Get the implementation template for tabular data tasks.

        Returns:
            str: The implementation template
        """
        return """
# Tabular Data Implementation

## Task Description
{query}

## Requirements
{requirements}

## Architecture
{architecture}

## Data Pipeline
{data_pipeline}

## Your Role
As the ML Engineer, implement a tabular data solution based on the architecture:

1. **Data Preprocessing Implementation**:
   - How should the feature preprocessing be implemented?
   - How should categorical and numerical features be handled?
   - How should the data loading and batching be handled?

2. **Feature Engineering Implementation**:
   - How should the feature transformations be implemented?
   - What feature selection methods should be used?
   - How should feature interactions be handled?

3. **Model Implementation**:
   - How should the model architecture be implemented?
   - What specific algorithms and components are needed?
   - How should the model be initialized and configured?

4. **Training Implementation**:
   - How should the training process be structured?
   - How should hyperparameter tuning be implemented?
   - How should cross-validation be handled?

5. **Inference Implementation**:
   - How should the inference pipeline be implemented?
   - How should pre and post-processing be handled during inference?
   - How should the model outputs be interpreted?

6. **Evaluation Implementation**:
   - How should the evaluation metrics be implemented?
   - How should the validation process be structured?
   - How should the results be visualized and analyzed?

Provide a comprehensive implementation with code snippets for each component.
"""

    def _get_general_implementation_template(self) -> str:
        """
        Get the implementation template for general ML tasks.

        Returns:
            str: The implementation template
        """
        return """
# ML Implementation

## Task Description
{query}

## Requirements
{requirements}

## Architecture
{architecture}

## Data Pipeline
{data_pipeline}

## Your Role
As the ML Engineer, implement an ML solution based on the architecture:

1. **Data Preprocessing Implementation**:
   - How should the data preprocessing be implemented?
   - What transformations should be applied?
   - How should the data loading and batching be handled?

2. **Model Implementation**:
   - How should the model architecture be implemented?
   - What specific algorithms and components are needed?
   - How should the model be initialized and configured?

3. **Training Implementation**:
   - How should the training process be structured?
   - How should the loss function and optimizer be implemented?
   - How should model selection and validation be handled?

4. **Inference Implementation**:
   - How should the inference pipeline be implemented?
   - How should pre and post-processing be handled during inference?
   - How should the model outputs be interpreted?

5. **Evaluation Implementation**:
   - How should the evaluation metrics be implemented?
   - How should the validation process be structured?
   - How should the results be visualized and analyzed?

Provide a comprehensive implementation with code snippets for each component.
"""

    def _parse_implementation(self, detailed_implementation: str, domain: str) -> Dict[str, Any]:
        """
        Parse detailed implementation into structured format.

        Args:
            detailed_implementation: The detailed implementation text
            domain: The domain of the task

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
        data_preprocessing = sections.get("Data Preprocessing Implementation", "")
        model_implementation = sections.get("Model Implementation", "")
        training_implementation = sections.get("Training Implementation", "")
        inference_implementation = sections.get("Inference Implementation", "")
        evaluation_implementation = sections.get("Evaluation Implementation", "")
        
        # For tabular data, also extract feature engineering
        feature_engineering = sections.get("Feature Engineering Implementation", "")
        
        # Create structured implementation
        structured_implementation = {
            "domain": domain,
            "data_preprocessing": data_preprocessing,
            "model_implementation": model_implementation,
            "training_implementation": training_implementation,
            "inference_implementation": inference_implementation,
            "evaluation_implementation": evaluation_implementation,
            "full_implementation": detailed_implementation,
        }
        
        # Add feature engineering for tabular data
        if domain == "tabular":
            structured_implementation["feature_engineering"] = feature_engineering
        
        return structured_implementation

    def implement_component(
        self,
        task: str,
        requirements: Dict[str, Any],
        architecture: Dict[str, Any],
        domain: str,
        component: str,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Implement a specific ML component.

        Args:
            task: The task description
            requirements: The task requirements
            architecture: The ML architecture
            domain: The domain of the task
            component: The component to implement
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The component implementation
        """
        # Build context for component implementation
        context = {
            "task": task,
            "requirements": requirements,
            "architecture": architecture,
            "domain": domain,
            "component": component,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for component implementation
        component_template = """
# ML Component Implementation

## Task Description
{query}

## Requirements
{requirements}

## Architecture
{architecture}

## Component to Implement
{component}

## Your Role
As the ML Engineer, implement the {component} component for this {domain} task:
1. What specific functionality does this component need to provide?
2. How does this component fit into the overall architecture?
3. What are the inputs and outputs of this component?
4. What are the key implementation considerations for this component?
5. How should this component be tested and evaluated?

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
                "domain": domain,
                "component": component,
                "component_implementation": component_implementation,
                "summary": self._create_summary({"action": "implement_component", "result": component_implementation}, 0),
            },
            task_id,
        )
        
        # Return the component implementation
        return {
            "action": "implement_component",
            "component": component,
            "implementation": component_implementation,
        }

    def optimize_model(
        self,
        task: str,
        model: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Optimize an ML model based on performance metrics.

        Args:
            task: The task description
            model: The model to optimize
            performance_metrics: The performance metrics
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The optimized model
        """
        # Build context for model optimization
        context = {
            "task": task,
            "model": model,
            "performance_metrics": performance_metrics,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for model optimization
        optimize_template = """
# Model Optimization

## Task Description
{query}

## Current Model
{model}

## Performance Metrics
{performance_metrics}

## Your Role
As the ML Engineer, optimize the model to improve performance:
1. What are the key performance bottlenecks or issues?
2. What optimization techniques would be most effective?
3. How should hyperparameters be tuned?
4. What architectural changes might improve performance?
5. How should the optimized model be evaluated?

Provide a detailed optimization plan with specific changes to implement.
"""
        
        optimize_prompt = self._format_prompt(full_context, optimize_template)
        
        # Get optimization plan
        optimization_plan = self._call_llm(optimize_prompt)
        
        # Store optimization plan in memory
        self._store_in_memory(
            {
                "action": "optimize_model",
                "task": task,
                "model": model,
                "performance_metrics": performance_metrics,
                "optimization_plan": optimization_plan,
                "summary": self._create_summary({"action": "optimize_model", "result": optimization_plan}, 0),
            },
            task_id,
        )
        
        # Return the optimization plan
        return {
            "action": "optimize_model",
            "optimization_plan": optimization_plan,
        }
