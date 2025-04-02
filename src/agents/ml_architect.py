"""
ML Architect Agent module for the AI Agent System.

This module implements the ML Architect agent that is responsible for
designing ML system architecture across domains.
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


class MLArchitect(BaseAgent):
    """
    ML Architect Agent for the AI Agent System.

    The MLArchitect is responsible for:
    - Designing ML pipelines for CV/NLP/tabular
    - Selecting appropriate models and approaches
    - Defining training strategies
    - Ensuring ML best practices
    - Planning evaluation methodologies
    """

    def __init__(
        self,
        client: Anthropic,
        memory: MemoryManager,
        retriever: Retriever,
        config: AgentConfig,
    ):
        """
        Initialize the MLArchitect agent.

        Args:
            client: Anthropic API client
            memory: Memory manager
            retriever: RAG retriever
            config: Agent configuration
        """
        super().__init__(client, memory, retriever, config, "ml_architect")

    def _get_role_specific_prompt(self) -> str:
        """
        Get the role-specific part of the system prompt.

        Returns:
            str: The role-specific prompt
        """
        return """As the ML Architect, your responsibilities include:

1. Designing ML pipelines for different domains (CV, NLP, tabular data)
2. Selecting appropriate models and approaches for specific problems
3. Defining training strategies and workflows
4. Ensuring ML best practices are followed
5. Planning evaluation methodologies and metrics
6. Making technical decisions about ML infrastructure

You should:
- Consider the specific requirements and constraints of the task
- Recommend state-of-the-art approaches when appropriate
- Balance performance, complexity, and resource requirements
- Provide clear rationales for your architectural decisions
- Consider scalability, maintainability, and deployment concerns
- Specify evaluation metrics and validation strategies

Your output should be technically precise, well-structured, and actionable for the ML Engineer.
"""

    def process(
        self, task: str, context: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a task as the ML Architect.

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
# ML Architecture Design

## Task Description
{query}

## Available Context
{rag}

## Your Role
As the ML Architect, design an ML architecture for this task:
1. What is the core ML problem type (classification, regression, clustering, etc.)?
2. What domain does this fall under (CV, NLP, tabular, etc.)?
3. What are the key components needed in the ML pipeline?
4. What models or approaches would be most appropriate?
5. What training strategy would you recommend?
6. What evaluation metrics and validation approach should be used?
7. What are the key technical considerations or challenges?

Provide a structured architecture design that can guide the ML Engineer's implementation.
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
        self, task: str, requirements: Dict[str, Any], task_info: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Design ML architecture for a task.

        Args:
            task: The task description
            requirements: The task requirements
            task_info: Information about the task
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The architecture design
        """
        # Build context for architecture design
        context = {
            "task": task,
            "requirements": requirements,
            "task_info": task_info,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Determine domain-specific prompt
        domain = task_info.get("domain", "general")
        if domain == "cv":
            design_template = self._get_cv_design_template()
        elif domain == "nlp":
            design_template = self._get_nlp_design_template()
        elif domain == "tabular":
            design_template = self._get_tabular_design_template()
        else:
            design_template = self._get_general_design_template()
        
        # Format prompt
        design_prompt = self._format_prompt(full_context, design_template)
        
        # Get detailed architecture design
        detailed_design = self._call_llm(design_prompt)
        
        # Parse the detailed design to extract structured architecture
        # In a real implementation, this would use more sophisticated parsing
        structured_architecture = self._parse_architecture(detailed_design, domain)
        
        # Store detailed architecture design in memory
        self._store_in_memory(
            {
                "action": "detailed_architecture_design",
                "task": task,
                "domain": domain,
                "detailed_design": detailed_design,
                "structured_architecture": structured_architecture,
                "summary": self._create_summary({"action": "detailed_architecture_design", "result": structured_architecture}, 0),
            },
            task_id,
        )
        
        # Return the structured architecture
        return structured_architecture

    def _get_cv_design_template(self) -> str:
        """
        Get the design template for computer vision tasks.

        Returns:
            str: The design template
        """
        return """
# Computer Vision Architecture Design

## Task Description
{query}

## Requirements
{requirements}

## Your Role
As the ML Architect, design a computer vision architecture for this task:

1. **Problem Formulation**:
   - What specific CV problem type is this (classification, detection, segmentation, etc.)?
   - What are the inputs and expected outputs?
   - What resolution and preprocessing is needed for the images?

2. **Model Architecture**:
   - What base architecture is most appropriate (CNN, Vision Transformer, etc.)?
   - What specific model variant would you recommend (ResNet, EfficientNet, YOLO, etc.)?
   - What modifications or customizations are needed for this specific task?

3. **Training Strategy**:
   - What loss function(s) should be used?
   - What optimizer and learning rate strategy would work best?
   - What data augmentation techniques are appropriate?
   - What regularization methods should be applied?

4. **Evaluation Approach**:
   - What metrics should be used to evaluate performance?
   - How should the validation be structured?
   - What baselines should the model be compared against?

5. **Implementation Considerations**:
   - What frameworks or libraries should be used?
   - What are the computational requirements?
   - What are potential challenges or bottlenecks?

Provide a comprehensive architecture design with clear rationales for your decisions.
"""

    def _get_nlp_design_template(self) -> str:
        """
        Get the design template for NLP tasks.

        Returns:
            str: The design template
        """
        return """
# NLP Architecture Design

## Task Description
{query}

## Requirements
{requirements}

## Your Role
As the ML Architect, design an NLP architecture for this task:

1. **Problem Formulation**:
   - What specific NLP problem type is this (classification, generation, QA, etc.)?
   - What are the inputs and expected outputs?
   - What preprocessing is needed for the text data?

2. **Model Architecture**:
   - What base architecture is most appropriate (Transformer, RNN, etc.)?
   - What specific model variant would you recommend (BERT, GPT, T5, etc.)?
   - What modifications or customizations are needed for this specific task?

3. **Training Strategy**:
   - What loss function(s) should be used?
   - What optimizer and learning rate strategy would work best?
   - What tokenization approach is appropriate?
   - What regularization methods should be applied?

4. **Evaluation Approach**:
   - What metrics should be used to evaluate performance?
   - How should the validation be structured?
   - What baselines should the model be compared against?

5. **Implementation Considerations**:
   - What frameworks or libraries should be used?
   - What are the computational requirements?
   - What are potential challenges or bottlenecks?

Provide a comprehensive architecture design with clear rationales for your decisions.
"""

    def _get_tabular_design_template(self) -> str:
        """
        Get the design template for tabular data tasks.

        Returns:
            str: The design template
        """
        return """
# Tabular Data Architecture Design

## Task Description
{query}

## Requirements
{requirements}

## Your Role
As the ML Architect, design a tabular data architecture for this task:

1. **Problem Formulation**:
   - What specific problem type is this (classification, regression, clustering, etc.)?
   - What are the inputs and expected outputs?
   - What preprocessing is needed for the tabular data?

2. **Model Architecture**:
   - What model types are most appropriate (tree-based, neural network, etc.)?
   - What specific models would you recommend (XGBoost, LightGBM, TabNet, etc.)?
   - What ensemble or stacking approach might be beneficial?

3. **Feature Engineering**:
   - What feature transformations should be applied?
   - How should categorical features be handled?
   - What feature selection methods might be useful?

4. **Training Strategy**:
   - What loss function(s) should be used?
   - What hyperparameter tuning approach would work best?
   - What cross-validation strategy is appropriate?

5. **Evaluation Approach**:
   - What metrics should be used to evaluate performance?
   - How should the validation be structured?
   - What baselines should the model be compared against?

6. **Implementation Considerations**:
   - What frameworks or libraries should be used?
   - What are the computational requirements?
   - What are potential challenges or bottlenecks?

Provide a comprehensive architecture design with clear rationales for your decisions.
"""

    def _get_general_design_template(self) -> str:
        """
        Get the design template for general ML tasks.

        Returns:
            str: The design template
        """
        return """
# ML Architecture Design

## Task Description
{query}

## Requirements
{requirements}

## Your Role
As the ML Architect, design an ML architecture for this task:

1. **Problem Formulation**:
   - What specific ML problem type is this?
   - What are the inputs and expected outputs?
   - What domain does this fall under?

2. **Model Architecture**:
   - What types of models would be most appropriate?
   - What specific model variants would you recommend?
   - What modifications or customizations are needed?

3. **Training Strategy**:
   - What loss function(s) should be used?
   - What optimizer and learning rate strategy would work best?
   - What regularization methods should be applied?

4. **Evaluation Approach**:
   - What metrics should be used to evaluate performance?
   - How should the validation be structured?
   - What baselines should the model be compared against?

5. **Implementation Considerations**:
   - What frameworks or libraries should be used?
   - What are the computational requirements?
   - What are potential challenges or bottlenecks?

Provide a comprehensive architecture design with clear rationales for your decisions.
"""

    def _parse_architecture(self, detailed_design: str, domain: str) -> Dict[str, Any]:
        """
        Parse detailed architecture design into structured format.

        Args:
            detailed_design: The detailed architecture design text
            domain: The domain of the task

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
        problem_formulation = sections.get("Problem Formulation", "")
        model_architecture = sections.get("Model Architecture", "")
        training_strategy = sections.get("Training Strategy", "")
        evaluation_approach = sections.get("Evaluation Approach", "")
        implementation_considerations = sections.get("Implementation Considerations", "")
        
        # For tabular data, also extract feature engineering
        feature_engineering = sections.get("Feature Engineering", "")
        
        # Create structured architecture
        structured_architecture = {
            "domain": domain,
            "problem_formulation": problem_formulation,
            "model_architecture": model_architecture,
            "training_strategy": training_strategy,
            "evaluation_approach": evaluation_approach,
            "implementation_considerations": implementation_considerations,
            "full_design": detailed_design,
        }
        
        # Add feature engineering for tabular data
        if domain == "tabular":
            structured_architecture["feature_engineering"] = feature_engineering
        
        return structured_architecture

    def evaluate_model(
        self, task: str, model: Dict[str, Any], requirements: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a model against architectural requirements.

        Args:
            task: The task description
            model: The model to evaluate
            requirements: The requirements to evaluate against
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The evaluation result
        """
        # Build context for evaluation
        context = {
            "task": task,
            "model": model,
            "requirements": requirements,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for evaluation
        evaluate_template = """
# Model Architecture Evaluation

## Task Description
{query}

## Architecture Requirements
{requirements}

## Implemented Model
{model}

## Your Role
As the ML Architect, evaluate the implemented model against the architectural requirements:
1. Does the model follow the recommended architecture?
2. Are there any deviations from the design, and if so, are they justified?
3. Does the model address the key technical considerations?
4. Are there any architectural improvements that could be made?
5. What is your overall assessment of the implementation?

Provide a structured evaluation with specific references to the requirements and implementation.
"""
        
        evaluate_prompt = self._format_prompt(full_context, evaluate_template)
        
        # Get evaluation result
        evaluation = self._call_llm(evaluate_prompt)
        
        # Store evaluation in memory
        self._store_in_memory(
            {
                "action": "evaluate_model",
                "task": task,
                "model": model,
                "requirements": requirements,
                "evaluation": evaluation,
                "summary": self._create_summary({"action": "evaluate_model", "result": evaluation}, 0),
            },
            task_id,
        )
        
        # Return the evaluation result
        return {
            "action": "evaluate_model",
            "evaluation": evaluation,
        }
