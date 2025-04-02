"""
Data Engineer Agent module for the AI Agent System.

This module implements the Data Engineer agent that is responsible for
data pipeline and feature engineering.
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


class DataEngineer(BaseAgent):
    """
    Data Engineer Agent for the AI Agent System.

    The DataEngineer is responsible for:
    - Designing data pipelines
    - Implementing feature engineering
    - Managing data preprocessing
    - Ensuring data quality
    - Optimizing data flows
    """

    def __init__(
        self,
        client: Anthropic,
        memory: MemoryManager,
        retriever: Retriever,
        config: AgentConfig,
    ):
        """
        Initialize the DataEngineer agent.

        Args:
            client: Anthropic API client
            memory: Memory manager
            retriever: RAG retriever
            config: Agent configuration
        """
        super().__init__(client, memory, retriever, config, "data_engineer")

    def _get_role_specific_prompt(self) -> str:
        """
        Get the role-specific part of the system prompt.

        Returns:
            str: The role-specific prompt
        """
        return """As the Data Engineer, your responsibilities include:

1. Designing and implementing data pipelines
2. Creating feature engineering processes
3. Managing data preprocessing and transformation
4. Ensuring data quality and integrity
5. Optimizing data flows and storage
6. Implementing data validation and monitoring

You should:
- Design efficient and scalable data processing workflows
- Implement robust feature engineering techniques
- Ensure data quality through validation and cleaning
- Consider performance and resource usage in data operations
- Provide clear documentation for data transformations
- Ensure compatibility with ML components

Your output should be technically precise, well-structured, and include working code implementations.
"""

    def process(
        self, task: str, context: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a task as the Data Engineer.

        Args:
            task: The task description
            context: The task context
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The processing result
        """
        # Build context for this task
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for data pipeline design
        pipeline_template = """
# Data Pipeline Design

## Task Description
{query}

## Available Context
{rag}

## Your Role
As the Data Engineer, design a data pipeline for this task:
1. What data sources need to be processed?
2. What data preprocessing steps are required?
3. What feature engineering techniques should be applied?
4. How should data validation and quality checks be implemented?
5. What are the data storage and access patterns?
6. What are the key considerations or challenges for this pipeline?

Provide a structured data pipeline design with code snippets for key components.
"""
        
        pipeline_prompt = self._format_prompt(full_context, pipeline_template)
        
        # Get pipeline design
        pipeline_design = self._call_llm(pipeline_prompt)
        
        # Store pipeline design in memory
        self._store_in_memory(
            {
                "action": "pipeline_design",
                "task": task,
                "pipeline_design": pipeline_design,
                "summary": self._create_summary({"action": "pipeline_design", "result": pipeline_design}, 0),
            },
            task_id,
        )
        
        # Return the pipeline design
        return {
            "action": "pipeline_design",
            "pipeline_design": pipeline_design,
        }

    def design_pipeline(
        self,
        task: str,
        requirements: Dict[str, Any],
        domain: str,
        data_description: Optional[str] = "",
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Design a data pipeline for a specific domain.

        Args:
            task: The task description
            requirements: The task requirements
            domain: The domain (cv, nlp, tabular)
            data_description: Optional description of the data
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The pipeline design
        """
        # Build context for pipeline design
        context = {
            "task": task,
            "requirements": requirements,
            "domain": domain,
            "data_description": data_description,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Determine domain-specific prompt
        if domain == "cv":
            design_template = self._get_cv_pipeline_template()
        elif domain == "nlp":
            design_template = self._get_nlp_pipeline_template()
        elif domain == "tabular":
            design_template = self._get_tabular_pipeline_template()
        else:
            design_template = self._get_general_pipeline_template()
        
        # Format prompt
        design_prompt = self._format_prompt(full_context, design_template)
        
        # Get detailed pipeline design
        detailed_design = self._call_llm(design_prompt)
        
        # Parse the detailed design to extract structured pipeline
        # In a real implementation, this would use more sophisticated parsing
        structured_pipeline = self._parse_pipeline(detailed_design, domain)
        
        # Store detailed pipeline design in memory
        self._store_in_memory(
            {
                "action": "detailed_pipeline_design",
                "task": task,
                "domain": domain,
                "detailed_design": detailed_design,
                "structured_pipeline": structured_pipeline,
                "summary": self._create_summary({"action": "detailed_pipeline_design", "result": structured_pipeline}, 0),
            },
            task_id,
        )
        
        # Return the structured pipeline
        return structured_pipeline

    def _get_cv_pipeline_template(self) -> str:
        """
        Get the pipeline template for computer vision tasks.

        Returns:
            str: The pipeline template
        """
        return """
# Computer Vision Data Pipeline Design

## Task Description
{query}

## Requirements
{requirements}

## Data Description
{data_description}

## Your Role
As the Data Engineer, design a data pipeline for this computer vision task:

1. **Data Collection and Storage**:
   - What data sources will be used?
   - How should the image data be stored and organized?
   - What metadata should be captured and stored?

2. **Data Preprocessing**:
   - How should images be loaded and decoded?
   - What image preprocessing steps are needed (resizing, normalization, etc.)?
   - How should preprocessing be parallelized and optimized?

3. **Data Augmentation**:
   - What data augmentation techniques should be applied?
   - How should augmentation be implemented in the pipeline?
   - How should augmentation parameters be configured?

4. **Feature Engineering**:
   - What feature extraction techniques should be applied?
   - How should features be normalized and transformed?
   - What dimensionality reduction might be needed?

5. **Data Validation and Quality**:
   - How should data quality be assessed and monitored?
   - What validation checks should be implemented?
   - How should data issues be handled?

6. **Pipeline Implementation**:
   - How should the pipeline components be structured?
   - What tools and libraries should be used?
   - How should the pipeline be optimized for performance?

Provide a comprehensive pipeline design with code snippets for each component.
"""

    def _get_nlp_pipeline_template(self) -> str:
        """
        Get the pipeline template for NLP tasks.

        Returns:
            str: The pipeline template
        """
        return """
# NLP Data Pipeline Design

## Task Description
{query}

## Requirements
{requirements}

## Data Description
{data_description}

## Your Role
As the Data Engineer, design a data pipeline for this NLP task:

1. **Data Collection and Storage**:
   - What data sources will be used?
   - How should the text data be stored and organized?
   - What metadata should be captured and stored?

2. **Data Preprocessing**:
   - How should text be cleaned and normalized?
   - What tokenization approach should be used?
   - How should preprocessing be parallelized and optimized?

3. **Feature Engineering**:
   - What text representation techniques should be used?
   - How should features be extracted from text?
   - What embeddings or encodings should be applied?

4. **Data Augmentation**:
   - What text augmentation techniques should be applied?
   - How should augmentation be implemented in the pipeline?
   - How should augmentation parameters be configured?

5. **Data Validation and Quality**:
   - How should data quality be assessed and monitored?
   - What validation checks should be implemented?
   - How should data issues be handled?

6. **Pipeline Implementation**:
   - How should the pipeline components be structured?
   - What tools and libraries should be used?
   - How should the pipeline be optimized for performance?

Provide a comprehensive pipeline design with code snippets for each component.
"""

    def _get_tabular_pipeline_template(self) -> str:
        """
        Get the pipeline template for tabular data tasks.

        Returns:
            str: The pipeline template
        """
        return """
# Tabular Data Pipeline Design

## Task Description
{query}

## Requirements
{requirements}

## Data Description
{data_description}

## Your Role
As the Data Engineer, design a data pipeline for this tabular data task:

1. **Data Collection and Storage**:
   - What data sources will be used?
   - How should the tabular data be stored and organized?
   - What metadata should be captured and stored?

2. **Data Preprocessing**:
   - How should missing values be handled?
   - How should categorical and numerical features be processed?
   - How should outliers be detected and handled?

3. **Feature Engineering**:
   - What feature transformations should be applied?
   - How should feature interactions be created?
   - What feature selection methods should be used?

4. **Data Validation and Quality**:
   - How should data quality be assessed and monitored?
   - What validation checks should be implemented?
   - How should data issues be handled?

5. **Data Splitting and Sampling**:
   - How should data be split for training, validation, and testing?
   - What sampling techniques should be applied?
   - How should class imbalance be addressed?

6. **Pipeline Implementation**:
   - How should the pipeline components be structured?
   - What tools and libraries should be used?
   - How should the pipeline be optimized for performance?

Provide a comprehensive pipeline design with code snippets for each component.
"""

    def _get_general_pipeline_template(self) -> str:
        """
        Get the pipeline template for general data tasks.

        Returns:
            str: The pipeline template
        """
        return """
# Data Pipeline Design

## Task Description
{query}

## Requirements
{requirements}

## Data Description
{data_description}

## Your Role
As the Data Engineer, design a data pipeline for this task:

1. **Data Collection and Storage**:
   - What data sources will be used?
   - How should the data be stored and organized?
   - What metadata should be captured and stored?

2. **Data Preprocessing**:
   - What data cleaning steps are needed?
   - How should different data types be handled?
   - How should preprocessing be parallelized and optimized?

3. **Feature Engineering**:
   - What feature transformations should be applied?
   - How should features be normalized and scaled?
   - What feature selection methods should be used?

4. **Data Validation and Quality**:
   - How should data quality be assessed and monitored?
   - What validation checks should be implemented?
   - How should data issues be handled?

5. **Pipeline Implementation**:
   - How should the pipeline components be structured?
   - What tools and libraries should be used?
   - How should the pipeline be optimized for performance?

Provide a comprehensive pipeline design with code snippets for each component.
"""

    def _parse_pipeline(self, detailed_design: str, domain: str) -> Dict[str, Any]:
        """
        Parse detailed pipeline design into structured format.

        Args:
            detailed_design: The detailed pipeline design text
            domain: The domain of the task

        Returns:
            Dict[str, Any]: Structured pipeline
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
        
        # Extract specific pipeline components
        data_collection = sections.get("Data Collection and Storage", "")
        data_preprocessing = sections.get("Data Preprocessing", "")
        feature_engineering = sections.get("Feature Engineering", "")
        data_validation = sections.get("Data Validation and Quality", "")
        pipeline_implementation = sections.get("Pipeline Implementation", "")
        
        # Domain-specific components
        data_augmentation = sections.get("Data Augmentation", "")
        data_splitting = sections.get("Data Splitting and Sampling", "")
        
        # Create structured pipeline
        structured_pipeline = {
            "domain": domain,
            "data_collection": data_collection,
            "data_preprocessing": data_preprocessing,
            "feature_engineering": feature_engineering,
            "data_validation": data_validation,
            "pipeline_implementation": pipeline_implementation,
            "full_design": detailed_design,
        }
        
        # Add domain-specific components
        if domain in ["cv", "nlp"]:
            structured_pipeline["data_augmentation"] = data_augmentation
        
        if domain == "tabular":
            structured_pipeline["data_splitting"] = data_splitting
        
        return structured_pipeline

    def implement_pipeline(
        self,
        task: str,
        requirements: Dict[str, Any],
        pipeline_design: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Implement a data pipeline based on the design.

        Args:
            task: The task description
            requirements: The task requirements
            pipeline_design: The pipeline design
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The pipeline implementation
        """
        # Build context for pipeline implementation
        context = {
            "task": task,
            "requirements": requirements,
            "pipeline_design": pipeline_design,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for pipeline implementation
        implement_template = """
# Data Pipeline Implementation

## Task Description
{query}

## Requirements
{requirements}

## Pipeline Design
{pipeline_design}

## Your Role
As the Data Engineer, implement the data pipeline based on the design:
1. How should each pipeline component be implemented?
2. What specific functions and classes are needed?
3. How should the pipeline components be connected?
4. How should error handling and logging be implemented?
5. How should the pipeline be tested and validated?
6. How should the pipeline be optimized for performance?

Provide a detailed implementation with code for the data pipeline.
"""
        
        implement_prompt = self._format_prompt(full_context, implement_template)
        
        # Get pipeline implementation
        pipeline_implementation = self._call_llm(implement_prompt)
        
        # Store pipeline implementation in memory
        self._store_in_memory(
            {
                "action": "implement_pipeline",
                "task": task,
                "pipeline_design": pipeline_design,
                "pipeline_implementation": pipeline_implementation,
                "summary": self._create_summary({"action": "implement_pipeline", "result": pipeline_implementation}, 0),
            },
            task_id,
        )
        
        # Return the pipeline implementation
        return {
            "action": "implement_pipeline",
            "pipeline_implementation": pipeline_implementation,
        }

    def implement_feature_engineering(
        self,
        task: str,
        requirements: Dict[str, Any],
        domain: str,
        data_description: str,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Implement feature engineering for a specific domain.

        Args:
            task: The task description
            requirements: The task requirements
            domain: The domain (cv, nlp, tabular)
            data_description: Description of the data
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The feature engineering implementation
        """
        # Build context for feature engineering
        context = {
            "task": task,
            "requirements": requirements,
            "domain": domain,
            "data_description": data_description,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for feature engineering
        feature_template = """
# Feature Engineering Implementation

## Task Description
{query}

## Requirements
{requirements}

## Domain
{domain}

## Data Description
{data_description}

## Your Role
As the Data Engineer, implement feature engineering for this {domain} task:
1. What specific feature transformations are needed?
2. How should raw data be converted into useful features?
3. What normalization or scaling should be applied?
4. How should feature selection be implemented?
5. How should the feature engineering be tested and validated?
6. How should the feature engineering be optimized for performance?

Provide a detailed implementation with code for the feature engineering.
"""
        
        feature_prompt = self._format_prompt(full_context, feature_template)
        
        # Get feature engineering implementation
        feature_implementation = self._call_llm(feature_prompt)
        
        # Store feature engineering implementation in memory
        self._store_in_memory(
            {
                "action": "implement_feature_engineering",
                "task": task,
                "domain": domain,
                "feature_implementation": feature_implementation,
                "summary": self._create_summary({"action": "implement_feature_engineering", "result": feature_implementation}, 0),
            },
            task_id,
        )
        
        # Return the feature engineering implementation
        return {
            "action": "implement_feature_engineering",
            "feature_implementation": feature_implementation,
        }

    def validate_data(
        self,
        task: str,
        data_description: str,
        requirements: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Design and implement data validation.

        Args:
            task: The task description
            data_description: Description of the data
            requirements: The task requirements
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The data validation implementation
        """
        # Build context for data validation
        context = {
            "task": task,
            "data_description": data_description,
            "requirements": requirements,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for data validation
        validate_template = """
# Data Validation Implementation

## Task Description
{query}

## Data Description
{data_description}

## Requirements
{requirements}

## Your Role
As the Data Engineer, implement data validation for this task:
1. What data quality checks should be implemented?
2. How should schema validation be handled?
3. How should missing values and outliers be detected?
4. How should data drift be monitored?
5. How should validation results be reported?
6. How should validation failures be handled?

Provide a detailed implementation with code for the data validation.
"""
        
        validate_prompt = self._format_prompt(full_context, validate_template)
        
        # Get data validation implementation
        validation_implementation = self._call_llm(validate_prompt)
        
        # Store data validation implementation in memory
        self._store_in_memory(
            {
                "action": "implement_data_validation",
                "task": task,
                "data_description": data_description,
                "validation_implementation": validation_implementation,
                "summary": self._create_summary({"action": "implement_data_validation", "result": validation_implementation}, 0),
            },
            task_id,
        )
        
        # Return the data validation implementation
        return {
            "action": "implement_data_validation",
            "validation_implementation": validation_implementation,
        }
