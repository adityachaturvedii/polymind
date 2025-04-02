"""
QA Engineer Agent module for the AI Agent System.

This module implements the QA Engineer agent that is responsible for
testing and quality assurance.
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


class QAEngineer(BaseAgent):
    """
    QA Engineer Agent for the AI Agent System.

    The QAEngineer is responsible for:
    - Designing test strategies
    - Implementing test cases
    - Validating system behavior
    - Identifying bugs and issues
    - Ensuring quality standards
    """

    def __init__(
        self,
        client: Anthropic,
        memory: MemoryManager,
        retriever: Retriever,
        config: AgentConfig,
    ):
        """
        Initialize the QAEngineer agent.

        Args:
            client: Anthropic API client
            memory: Memory manager
            retriever: RAG retriever
            config: Agent configuration
        """
        super().__init__(client, memory, retriever, config, "qa_engineer")

    def _get_role_specific_prompt(self) -> str:
        """
        Get the role-specific part of the system prompt.

        Returns:
            str: The role-specific prompt
        """
        return """As the QA Engineer, your responsibilities include:

1. Designing comprehensive test strategies
2. Implementing test cases and test suites
3. Validating system behavior against requirements
4. Identifying bugs, issues, and edge cases
5. Ensuring quality standards are met
6. Providing detailed test reports and feedback

You should:
- Design tests that cover functional and non-functional requirements
- Implement tests that are thorough, maintainable, and automated where possible
- Validate both software and ML components
- Identify edge cases and potential failure modes
- Provide clear, actionable feedback on issues
- Ensure test coverage across the system

Your output should be technically precise, well-structured, and include working test implementations.
"""

    def process(
        self, task: str, context: Dict[str, Any], task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a task as the QA Engineer.

        Args:
            task: The task description
            context: The task context
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The processing result
        """
        # Build context for this task
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for test strategy
        test_template = """
# Test Strategy

## Task Description
{query}

## Available Context
{rag}

## Your Role
As the QA Engineer, design a test strategy for this task:
1. What are the key components that need to be tested?
2. What types of tests are needed (unit, integration, system, etc.)?
3. What are the critical test scenarios and edge cases?
4. How should test data be generated or acquired?
5. What testing tools and frameworks should be used?
6. What are the key quality metrics to track?

Provide a structured test strategy with example test cases for key components.
"""
        
        test_prompt = self._format_prompt(full_context, test_template)
        
        # Get test strategy
        test_strategy = self._call_llm(test_prompt)
        
        # Store test strategy in memory
        self._store_in_memory(
            {
                "action": "test_strategy",
                "task": task,
                "test_strategy": test_strategy,
                "summary": self._create_summary({"action": "test_strategy", "result": test_strategy}, 0),
            },
            task_id,
        )
        
        # Return the test strategy
        return {
            "action": "test_strategy",
            "test_strategy": test_strategy,
        }

    def test(
        self,
        task: str,
        requirements: Dict[str, Any],
        implementation: Dict[str, Any],
        test_type: Optional[str] = "system",
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Test an implementation against requirements.

        Args:
            task: The task description
            requirements: The task requirements
            implementation: The implementation to test
            test_type: The type of test to perform
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The test results
        """
        # Build context for testing
        context = {
            "task": task,
            "requirements": requirements,
            "implementation": implementation,
            "test_type": test_type,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Determine test template based on test type
        if test_type == "unit":
            test_template = self._get_unit_test_template()
        elif test_type == "integration":
            test_template = self._get_integration_test_template()
        elif test_type == "performance":
            test_template = self._get_performance_test_template()
        else:  # system or any other type
            test_template = self._get_system_test_template()
        
        # Format prompt
        test_prompt = self._format_prompt(full_context, test_template)
        
        # Get test results
        test_results = self._call_llm(test_prompt)
        
        # Parse the test results to extract structured results
        # In a real implementation, this would use more sophisticated parsing
        structured_results = self._parse_test_results(test_results, test_type)
        
        # Store test results in memory
        self._store_in_memory(
            {
                "action": f"{test_type}_test",
                "task": task,
                "test_type": test_type,
                "test_results": test_results,
                "structured_results": structured_results,
                "summary": self._create_summary({"action": f"{test_type}_test", "result": structured_results}, 0),
            },
            task_id,
        )
        
        # Return the structured results
        return structured_results

    def _get_unit_test_template(self) -> str:
        """
        Get the template for unit testing.

        Returns:
            str: The unit test template
        """
        return """
# Unit Testing

## Task Description
{query}

## Requirements
{requirements}

## Implementation
{implementation}

## Your Role
As the QA Engineer, perform unit testing on the implementation:

1. **Test Case Design**:
   - What are the key functions and classes to test?
   - What are the expected behaviors for each unit?
   - What edge cases and error conditions should be tested?

2. **Test Implementation**:
   - How should each test case be implemented?
   - What assertions should be used to verify behavior?
   - How should test fixtures and mocks be set up?

3. **Test Coverage**:
   - What is the test coverage for the implementation?
   - Are there any untested or undertested components?
   - How can test coverage be improved?

4. **Test Results**:
   - What tests pass and what tests fail?
   - What issues or bugs were identified?
   - What are the recommendations for fixing the issues?

5. **Test Quality**:
   - How maintainable and readable are the tests?
   - How reliable and deterministic are the tests?
   - What improvements could be made to the test suite?

Provide comprehensive unit tests with code and detailed results.
"""

    def _get_integration_test_template(self) -> str:
        """
        Get the template for integration testing.

        Returns:
            str: The integration test template
        """
        return """
# Integration Testing

## Task Description
{query}

## Requirements
{requirements}

## Implementation
{implementation}

## Your Role
As the QA Engineer, perform integration testing on the implementation:

1. **Integration Points**:
   - What are the key integration points to test?
   - How do components interact with each other?
   - What are the expected behaviors for each integration?

2. **Test Case Design**:
   - What integration scenarios should be tested?
   - What data flows should be verified?
   - What error handling and edge cases should be tested?

3. **Test Implementation**:
   - How should each integration test be implemented?
   - What test environment setup is needed?
   - How should external dependencies be handled?

4. **Test Results**:
   - What integration tests pass and what tests fail?
   - What issues or bugs were identified?
   - What are the recommendations for fixing the issues?

5. **Integration Quality**:
   - How well do the components work together?
   - Are there any performance or reliability issues?
   - What improvements could be made to the integration?

Provide comprehensive integration tests with code and detailed results.
"""

    def _get_system_test_template(self) -> str:
        """
        Get the template for system testing.

        Returns:
            str: The system test template
        """
        return """
# System Testing

## Task Description
{query}

## Requirements
{requirements}

## Implementation
{implementation}

## Your Role
As the QA Engineer, perform system testing on the implementation:

1. **System Requirements**:
   - What are the key system requirements to verify?
   - What are the expected behaviors of the system?
   - What are the acceptance criteria for the system?

2. **Test Scenario Design**:
   - What end-to-end scenarios should be tested?
   - What user workflows should be verified?
   - What edge cases and error conditions should be tested?

3. **Test Implementation**:
   - How should each system test be implemented?
   - What test data and environment setup is needed?
   - How should test execution be automated?

4. **Test Results**:
   - What system tests pass and what tests fail?
   - What issues or bugs were identified?
   - What are the recommendations for fixing the issues?

5. **System Quality**:
   - How well does the system meet the requirements?
   - Are there any usability, reliability, or performance issues?
   - What improvements could be made to the system?

Provide comprehensive system tests with code and detailed results.
"""

    def _get_performance_test_template(self) -> str:
        """
        Get the template for performance testing.

        Returns:
            str: The performance test template
        """
        return """
# Performance Testing

## Task Description
{query}

## Requirements
{requirements}

## Implementation
{implementation}

## Your Role
As the QA Engineer, perform performance testing on the implementation:

1. **Performance Requirements**:
   - What are the key performance metrics to measure?
   - What are the performance targets and thresholds?
   - What are the critical performance scenarios?

2. **Test Design**:
   - What load and stress tests should be performed?
   - What scalability tests should be conducted?
   - What resource utilization should be monitored?

3. **Test Implementation**:
   - How should each performance test be implemented?
   - What test environment and tools should be used?
   - How should performance data be collected and analyzed?

4. **Test Results**:
   - What are the performance metrics for each test?
   - How does the performance compare to the requirements?
   - What performance bottlenecks or issues were identified?

5. **Performance Optimization**:
   - What are the recommendations for improving performance?
   - What specific optimizations should be implemented?
   - What are the expected performance gains from the optimizations?

Provide comprehensive performance tests with code and detailed results.
"""

    def _parse_test_results(self, test_results: str, test_type: str) -> Dict[str, Any]:
        """
        Parse test results into structured format.

        Args:
            test_results: The test results text
            test_type: The type of test

        Returns:
            Dict[str, Any]: Structured test results
        """
        # In a real implementation, this would use more sophisticated parsing
        # For now, just return a simple structured format
        
        # Extract sections based on headings
        sections = {}
        current_section = None
        current_content = []
        
        for line in test_results.split("\n"):
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
        
        # Extract specific test components
        test_case_design = sections.get("Test Case Design", "")
        test_implementation = sections.get("Test Implementation", "")
        test_results_section = sections.get("Test Results", "")
        
        # Test type specific sections
        test_coverage = sections.get("Test Coverage", "")
        integration_points = sections.get("Integration Points", "")
        system_requirements = sections.get("System Requirements", "")
        performance_requirements = sections.get("Performance Requirements", "")
        
        # Quality sections
        test_quality = sections.get("Test Quality", "")
        integration_quality = sections.get("Integration Quality", "")
        system_quality = sections.get("System Quality", "")
        performance_optimization = sections.get("Performance Optimization", "")
        
        # Create structured test results
        structured_results = {
            "test_type": test_type,
            "test_case_design": test_case_design,
            "test_implementation": test_implementation,
            "test_results": test_results_section,
            "full_results": test_results,
        }
        
        # Add test type specific sections
        if test_type == "unit":
            structured_results["test_coverage"] = test_coverage
            structured_results["test_quality"] = test_quality
        elif test_type == "integration":
            structured_results["integration_points"] = integration_points
            structured_results["integration_quality"] = integration_quality
        elif test_type == "system":
            structured_results["system_requirements"] = system_requirements
            structured_results["system_quality"] = system_quality
        elif test_type == "performance":
            structured_results["performance_requirements"] = performance_requirements
            structured_results["performance_optimization"] = performance_optimization
        
        return structured_results

    def design_test_suite(
        self,
        task: str,
        requirements: Dict[str, Any],
        architecture: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Design a comprehensive test suite.

        Args:
            task: The task description
            requirements: The task requirements
            architecture: The system architecture
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The test suite design
        """
        # Build context for test suite design
        context = {
            "task": task,
            "requirements": requirements,
            "architecture": architecture,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for test suite design
        suite_template = """
# Test Suite Design

## Task Description
{query}

## Requirements
{requirements}

## Architecture
{architecture}

## Your Role
As the QA Engineer, design a comprehensive test suite for this system:
1. What test levels should be included (unit, integration, system, etc.)?
2. What test types should be included (functional, performance, security, etc.)?
3. What are the key test scenarios for each component?
4. How should test data be managed?
5. How should the test suite be organized and maintained?
6. What testing tools and frameworks should be used?

Provide a detailed test suite design with example test cases for key components.
"""
        
        suite_prompt = self._format_prompt(full_context, suite_template)
        
        # Get test suite design
        suite_design = self._call_llm(suite_prompt)
        
        # Store test suite design in memory
        self._store_in_memory(
            {
                "action": "design_test_suite",
                "task": task,
                "suite_design": suite_design,
                "summary": self._create_summary({"action": "design_test_suite", "result": suite_design}, 0),
            },
            task_id,
        )
        
        # Return the test suite design
        return {
            "action": "design_test_suite",
            "suite_design": suite_design,
        }

    def test_ml_model(
        self,
        task: str,
        requirements: Dict[str, Any],
        model: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Test an ML model against requirements.

        Args:
            task: The task description
            requirements: The task requirements
            model: The ML model to test
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The test results
        """
        # Build context for ML model testing
        context = {
            "task": task,
            "requirements": requirements,
            "model": model,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for ML model testing
        ml_test_template = """
# ML Model Testing

## Task Description
{query}

## Requirements
{requirements}

## Model
{model}

## Your Role
As the QA Engineer, test the ML model:
1. What metrics should be used to evaluate the model?
2. How should the model be tested for accuracy and performance?
3. How should the model be tested for robustness and generalization?
4. How should the model be tested for bias and fairness?
5. What are the key edge cases and failure modes to test?
6. How does the model compare to the requirements?

Provide a detailed test plan and results for the ML model.
"""
        
        ml_test_prompt = self._format_prompt(full_context, ml_test_template)
        
        # Get ML model test results
        ml_test_results = self._call_llm(ml_test_prompt)
        
        # Store ML model test results in memory
        self._store_in_memory(
            {
                "action": "test_ml_model",
                "task": task,
                "ml_test_results": ml_test_results,
                "summary": self._create_summary({"action": "test_ml_model", "result": ml_test_results}, 0),
            },
            task_id,
        )
        
        # Return the ML model test results
        return {
            "action": "test_ml_model",
            "ml_test_results": ml_test_results,
        }

    def generate_test_report(
        self,
        task: str,
        test_results: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive test report.

        Args:
            task: The task description
            test_results: The test results
            task_id: Optional task identifier

        Returns:
            Dict[str, Any]: The test report
        """
        # Build context for test report
        context = {
            "task": task,
            "test_results": test_results,
        }
        full_context = self._build_context(task, task_id, **context)
        
        # Create prompt for test report
        report_template = """
# Test Report

## Task Description
{query}

## Test Results
{test_results}

## Your Role
As the QA Engineer, generate a comprehensive test report:
1. What is the overall test status and quality assessment?
2. What are the key test results and metrics?
3. What issues and bugs were identified?
4. What are the recommendations for addressing the issues?
5. What are the next steps for testing and quality assurance?
6. What is the overall assessment of the system quality?

Provide a detailed test report with clear findings and recommendations.
"""
        
        report_prompt = self._format_prompt(full_context, report_template)
        
        # Get test report
        test_report = self._call_llm(report_prompt)
        
        # Store test report in memory
        self._store_in_memory(
            {
                "action": "generate_test_report",
                "task": task,
                "test_report": test_report,
                "summary": self._create_summary({"action": "generate_test_report", "result": test_report}, 0),
            },
            task_id,
        )
        
        # Return the test report
        return {
            "action": "generate_test_report",
            "test_report": test_report,
        }
