"""
MCP Server module for the PolyMind AI Agent System.

This module implements the Model Context Protocol (MCP) server that provides
custom tools for the multi-agent system.

Author: Aditya Chaturvedi
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

from modelcontextprotocol.sdk.server import Server
from modelcontextprotocol.sdk.server.stdio import StdioServerTransport
from modelcontextprotocol.sdk.types import (
    CallToolRequestSchema,
    ErrorCode,
    ListResourcesRequestSchema,
    ListResourceTemplatesRequestSchema,
    ListToolsRequestSchema,
    McpError,
    ReadResourceRequestSchema,
)

from src.core.config import get_config
from src.core.coordinator import Coordinator

logger = logging.getLogger(__name__)


class MCPServer:
    """
    MCP Server for the AI Agent System.

    The MCPServer provides custom tools for the multi-agent system using
    Anthropic's Model Context Protocol (MCP).
    """

    def __init__(self):
        """Initialize the MCP Server."""
        self.config = get_config()
        
        # Initialize the MCP server
        self.server = Server(
            {
                "name": self.config.mcp.server_name,
                "version": self.config.mcp.server_version,
            },
            {
                "capabilities": {
                    "resources": {},
                    "tools": {},
                },
            },
        )
        
        # Initialize the coordinator
        self.coordinator = Coordinator()
        
        # Set up request handlers
        self._setup_tool_handlers()
        self._setup_resource_handlers()
        
        # Set up error handling
        self.server.onerror = self._handle_error
        
        logger.info("MCP Server initialized")

    def _setup_tool_handlers(self):
        """Set up tool request handlers."""
        # List tools handler
        self.server.setRequestHandler(ListToolsRequestSchema, self._handle_list_tools)
        
        # Call tool handler
        self.server.setRequestHandler(CallToolRequestSchema, self._handle_call_tool)

    def _setup_resource_handlers(self):
        """Set up resource request handlers."""
        # List resources handler
        self.server.setRequestHandler(ListResourcesRequestSchema, self._handle_list_resources)
        
        # List resource templates handler
        self.server.setRequestHandler(ListResourceTemplatesRequestSchema, self._handle_list_resource_templates)
        
        # Read resource handler
        self.server.setRequestHandler(ReadResourceRequestSchema, self._handle_read_resource)

    def _handle_error(self, error):
        """
        Handle MCP server errors.

        Args:
            error: The error to handle
        """
        logger.error(f"MCP Server error: {error}")

    async def _handle_list_tools(self, request):
        """
        Handle list tools request.

        Args:
            request: The list tools request

        Returns:
            Dict[str, Any]: The list tools response
        """
        # Define available tools
        tools = [
            {
                "name": "analyze_requirements",
                "description": "Analyze user requirements for a task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task description",
                        },
                        "domain": {
                            "type": "string",
                            "description": "The task domain (cv, nlp, tabular, general)",
                            "enum": ["cv", "nlp", "tabular", "general"],
                        },
                        "task_type": {
                            "type": "string",
                            "description": "The task type (modeling, coding, analysis, deployment)",
                            "enum": ["modeling", "coding", "analysis", "deployment"],
                        },
                    },
                    "required": ["task"],
                },
            },
            {
                "name": "optimize_cv_model",
                "description": "Optimize a computer vision model for performance or size",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "object",
                            "description": "The model to optimize",
                        },
                        "optimization_target": {
                            "type": "string",
                            "description": "Target of optimization",
                            "enum": ["inference_speed", "model_size", "accuracy", "memory_usage"],
                        },
                        "constraints": {
                            "type": "object",
                            "description": "Constraints for optimization",
                        },
                    },
                    "required": ["model", "optimization_target"],
                },
            },
            {
                "name": "fine_tune_language_model",
                "description": "Fine-tune a language model for a specific task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "base_model": {
                            "type": "string",
                            "description": "Base model to fine-tune",
                        },
                        "task_type": {
                            "type": "string",
                            "description": "Type of NLP task",
                            "enum": ["classification", "ner", "qa", "summarization", "generation"],
                        },
                        "training_data": {
                            "type": "object",
                            "description": "Description of training data",
                        },
                        "hyperparameters": {
                            "type": "object",
                            "description": "Fine-tuning hyperparameters",
                        },
                    },
                    "required": ["base_model", "task_type"],
                },
            },
            {
                "name": "create_deployment_pipeline",
                "description": "Create a deployment pipeline for ML models",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "object",
                            "description": "The model to deploy",
                        },
                        "deployment_target": {
                            "type": "string",
                            "description": "Deployment target",
                            "enum": ["cloud", "edge", "mobile", "web", "on-premise"],
                        },
                        "requirements": {
                            "type": "object",
                            "description": "Deployment requirements",
                        },
                    },
                    "required": ["model", "deployment_target"],
                },
            },
            {
                "name": "explain_model_prediction",
                "description": "Generate explanations for model predictions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "object",
                            "description": "The model to explain",
                        },
                        "input_data": {
                            "type": "object",
                            "description": "Input data for explanation",
                        },
                        "explanation_method": {
                            "type": "string",
                            "description": "Method for generating explanations",
                            "enum": ["shap", "lime", "integrated_gradients", "attention", "feature_importance"],
                        },
                    },
                    "required": ["model", "input_data"],
                },
            },
            {
                "name": "analyze_dataset",
                "description": "Analyze a dataset and generate insights",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_description": {
                            "type": "object",
                            "description": "Description of the dataset",
                        },
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of analysis to perform",
                            "enum": ["exploratory", "statistical", "correlation", "distribution", "outlier_detection"],
                        },
                    },
                    "required": ["dataset_description"],
                },
            },
            {
                "name": "design_ml_architecture",
                "description": "Design ML system architecture",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task description",
                        },
                        "requirements": {
                            "type": "object",
                            "description": "The task requirements",
                        },
                        "domain": {
                            "type": "string",
                            "description": "The ML domain (cv, nlp, tabular)",
                            "enum": ["cv", "nlp", "tabular"],
                        },
                    },
                    "required": ["task", "requirements", "domain"],
                },
            },
            {
                "name": "design_software_architecture",
                "description": "Design software system architecture",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task description",
                        },
                        "requirements": {
                            "type": "object",
                            "description": "The task requirements",
                        },
                        "ml_architecture": {
                            "type": "object",
                            "description": "The ML architecture (if applicable)",
                        },
                    },
                    "required": ["task", "requirements"],
                },
            },
            {
                "name": "implement_ml_component",
                "description": "Implement an ML component",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task description",
                        },
                        "requirements": {
                            "type": "object",
                            "description": "The task requirements",
                        },
                        "architecture": {
                            "type": "object",
                            "description": "The ML architecture",
                        },
                        "domain": {
                            "type": "string",
                            "description": "The ML domain (cv, nlp, tabular)",
                            "enum": ["cv", "nlp", "tabular"],
                        },
                        "component": {
                            "type": "string",
                            "description": "The component to implement",
                        },
                    },
                    "required": ["task", "requirements", "architecture", "domain", "component"],
                },
            },
            {
                "name": "implement_code_component",
                "description": "Implement a code component",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task description",
                        },
                        "requirements": {
                            "type": "object",
                            "description": "The task requirements",
                        },
                        "architecture": {
                            "type": "object",
                            "description": "The software architecture",
                        },
                        "component": {
                            "type": "string",
                            "description": "The component to implement",
                        },
                        "language": {
                            "type": "string",
                            "description": "The programming language",
                        },
                    },
                    "required": ["task", "requirements", "architecture", "component"],
                },
            },
            {
                "name": "design_data_pipeline",
                "description": "Design a data pipeline",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task description",
                        },
                        "requirements": {
                            "type": "object",
                            "description": "The task requirements",
                        },
                        "domain": {
                            "type": "string",
                            "description": "The ML domain (cv, nlp, tabular)",
                            "enum": ["cv", "nlp", "tabular"],
                        },
                        "data_description": {
                            "type": "string",
                            "description": "Description of the data",
                        },
                    },
                    "required": ["task", "requirements", "domain"],
                },
            },
            {
                "name": "test_solution",
                "description": "Test a solution",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task description",
                        },
                        "requirements": {
                            "type": "object",
                            "description": "The task requirements",
                        },
                        "implementation": {
                            "type": "object",
                            "description": "The implementation to test",
                        },
                        "test_type": {
                            "type": "string",
                            "description": "The type of test to perform",
                            "enum": ["unit", "integration", "system", "performance"],
                        },
                    },
                    "required": ["task", "requirements", "implementation"],
                },
            },
        ]
        
        # Filter tools based on enabled tools in config
        if self.config.mcp.tools_enabled:
            tools = [tool for tool in tools if tool["name"] in self.config.mcp.tools_enabled]
        
        return {"tools": tools}

    async def _handle_call_tool(self, request):
        """
        Handle call tool request.

        Args:
            request: The call tool request

        Returns:
            Dict[str, Any]: The call tool response
        """
        tool_name = request.params.name
        arguments = request.params.arguments
        
        logger.info(f"Calling tool: {tool_name}")
        
        try:
            # Route to the appropriate tool handler
            if tool_name == "analyze_requirements":
                result = await self._handle_analyze_requirements(arguments)
            elif tool_name == "design_ml_architecture":
                result = await self._handle_design_ml_architecture(arguments)
            elif tool_name == "design_software_architecture":
                result = await self._handle_design_software_architecture(arguments)
            elif tool_name == "implement_ml_component":
                result = await self._handle_implement_ml_component(arguments)
            elif tool_name == "implement_code_component":
                result = await self._handle_implement_code_component(arguments)
            elif tool_name == "design_data_pipeline":
                result = await self._handle_design_data_pipeline(arguments)
            elif tool_name == "test_solution":
                result = await self._handle_test_solution(arguments)
            elif tool_name == "optimize_cv_model":
                result = await self._handle_optimize_cv_model(arguments)
            elif tool_name == "fine_tune_language_model":
                result = await self._handle_fine_tune_language_model(arguments)
            elif tool_name == "create_deployment_pipeline":
                result = await self._handle_create_deployment_pipeline(arguments)
            elif tool_name == "explain_model_prediction":
                result = await self._handle_explain_model_prediction(arguments)
            elif tool_name == "analyze_dataset":
                result = await self._handle_analyze_dataset(arguments)
            else:
                raise McpError(ErrorCode.MethodNotFound, f"Unknown tool: {tool_name}")
            
            # Return the result
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2),
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: {str(e)}",
                    }
                ],
                "isError": True,
            }

    async def _handle_analyze_requirements(self, arguments):
        """
        Handle analyze requirements tool.

        Args:
            arguments: The tool arguments

        Returns:
            Dict[str, Any]: The tool result
        """
        task = arguments.get("task")
        domain = arguments.get("domain", "general")
        task_type = arguments.get("task_type", "coding")
        
        # Create task info
        task_info = {
            "domain": domain,
            "task_type": task_type,
            "requirements": [],
            "constraints": [],
        }
        
        # Get product manager agent
        product_manager = self.coordinator.get_agent("product_manager")
        
        # Analyze requirements
        requirements = product_manager.analyze_requirements(task, task_info)
        
        return requirements

    async def _handle_design_ml_architecture(self, arguments):
        """
        Handle design ML architecture tool.

        Args:
            arguments: The tool arguments

        Returns:
            Dict[str, Any]: The tool result
        """
        task = arguments.get("task")
        requirements = arguments.get("requirements")
        domain = arguments.get("domain")
        
        # Create task info
        task_info = {
            "domain": domain,
            "task_type": "modeling",
        }
        
        # Get ML architect agent
        ml_architect = self.coordinator.get_agent("ml_architect")
        
        # Design architecture
        architecture = ml_architect.design_architecture(task, requirements, task_info)
        
        return architecture

    async def _handle_design_software_architecture(self, arguments):
        """
        Handle design software architecture tool.

        Args:
            arguments: The tool arguments

        Returns:
            Dict[str, Any]: The tool result
        """
        task = arguments.get("task")
        requirements = arguments.get("requirements")
        ml_architecture = arguments.get("ml_architecture")
        
        # Get software architect agent
        sw_architect = self.coordinator.get_agent("sw_architect")
        
        # Design architecture
        architecture = sw_architect.design_architecture(task, requirements, ml_architecture)
        
        return architecture

    async def _handle_implement_ml_component(self, arguments):
        """
        Handle implement ML component tool.

        Args:
            arguments: The tool arguments

        Returns:
            Dict[str, Any]: The tool result
        """
        task = arguments.get("task")
        requirements = arguments.get("requirements")
        architecture = arguments.get("architecture")
        domain = arguments.get("domain")
        component = arguments.get("component")
        
        # Get ML engineer agent
        ml_engineer = self.coordinator.get_agent("ml_engineer")
        
        # Implement component
        implementation = ml_engineer.implement_component(
            task, requirements, architecture, domain, component
        )
        
        return implementation

    async def _handle_implement_code_component(self, arguments):
        """
        Handle implement code component tool.

        Args:
            arguments: The tool arguments

        Returns:
            Dict[str, Any]: The tool result
        """
        task = arguments.get("task")
        requirements = arguments.get("requirements")
        architecture = arguments.get("architecture")
        component = arguments.get("component")
        language = arguments.get("language", "python")
        
        # Get software engineer agent
        sw_engineer = self.coordinator.get_agent("sw_engineer")
        
        # Implement component
        implementation = sw_engineer.implement_component(
            task, requirements, architecture, component, language
        )
        
        return implementation

    async def _handle_design_data_pipeline(self, arguments):
        """
        Handle design data pipeline tool.

        Args:
            arguments: The tool arguments

        Returns:
            Dict[str, Any]: The tool result
        """
        task = arguments.get("task")
        requirements = arguments.get("requirements")
        domain = arguments.get("domain")
        data_description = arguments.get("data_description", "")
        
        # Get data engineer agent
        data_engineer = self.coordinator.get_agent("data_engineer")
        
        # Design pipeline
        pipeline = data_engineer.design_pipeline(
            task, requirements, domain, data_description
        )
        
        return pipeline

    async def _handle_test_solution(self, arguments):
        """
        Handle test solution tool.

        Args:
            arguments: The tool arguments

        Returns:
            Dict[str, Any]: The tool result
        """
        task = arguments.get("task")
        requirements = arguments.get("requirements")
        implementation = arguments.get("implementation")
        test_type = arguments.get("test_type", "system")
        
        # Get QA engineer agent
        qa_engineer = self.coordinator.get_agent("qa_engineer")
        
        # Test solution
        test_results = qa_engineer.test(
            task, requirements, implementation, test_type
        )
        
        return test_results
        
    async def _handle_optimize_cv_model(self, arguments):
        """
        Handle optimize CV model tool.

        Args:
            arguments: The tool arguments

        Returns:
            Dict[str, Any]: The tool result
        """
        model = arguments.get("model")
        optimization_target = arguments.get("optimization_target")
        constraints = arguments.get("constraints", {})
        
        # Get ML engineer agent
        ml_engineer = self.coordinator.get_agent("ml_engineer")
        
        # Optimize model
        optimized_model = ml_engineer.optimize_cv_model(
            model, optimization_target, constraints
        )
        
        return {
            "optimized_model": optimized_model,
            "optimization_target": optimization_target,
            "performance_metrics": ml_engineer.evaluate_model_performance(optimized_model),
            "optimization_summary": f"Model optimized for {optimization_target}"
        }
        
    async def _handle_fine_tune_language_model(self, arguments):
        """
        Handle fine-tune language model tool.

        Args:
            arguments: The tool arguments

        Returns:
            Dict[str, Any]: The tool result
        """
        base_model = arguments.get("base_model")
        task_type = arguments.get("task_type")
        training_data = arguments.get("training_data", {})
        hyperparameters = arguments.get("hyperparameters", {})
        
        # Get ML engineer agent
        ml_engineer = self.coordinator.get_agent("ml_engineer")
        
        # Fine-tune model
        fine_tuned_model = ml_engineer.fine_tune_language_model(
            base_model, task_type, training_data, hyperparameters
        )
        
        return {
            "fine_tuned_model": fine_tuned_model,
            "task_type": task_type,
            "performance_metrics": ml_engineer.evaluate_model_performance(fine_tuned_model),
            "fine_tuning_summary": f"Model fine-tuned for {task_type} task"
        }
        
    async def _handle_create_deployment_pipeline(self, arguments):
        """
        Handle create deployment pipeline tool.

        Args:
            arguments: The tool arguments

        Returns:
            Dict[str, Any]: The tool result
        """
        model = arguments.get("model")
        deployment_target = arguments.get("deployment_target")
        requirements = arguments.get("requirements", {})
        
        # Get ML engineer and software engineer agents
        ml_engineer = self.coordinator.get_agent("ml_engineer")
        sw_engineer = self.coordinator.get_agent("sw_engineer")
        
        # Create deployment pipeline
        deployment_config = ml_engineer.prepare_model_for_deployment(
            model, deployment_target, requirements
        )
        
        deployment_pipeline = sw_engineer.implement_deployment_pipeline(
            deployment_config, deployment_target, requirements
        )
        
        return {
            "deployment_pipeline": deployment_pipeline,
            "deployment_target": deployment_target,
            "deployment_config": deployment_config,
            "deployment_summary": f"Deployment pipeline created for {deployment_target}"
        }
        
    async def _handle_explain_model_prediction(self, arguments):
        """
        Handle explain model prediction tool.

        Args:
            arguments: The tool arguments

        Returns:
            Dict[str, Any]: The tool result
        """
        model = arguments.get("model")
        input_data = arguments.get("input_data")
        explanation_method = arguments.get("explanation_method", "shap")
        
        # Get ML engineer agent
        ml_engineer = self.coordinator.get_agent("ml_engineer")
        
        # Explain model prediction
        explanation = ml_engineer.explain_model_prediction(
            model, input_data, explanation_method
        )
        
        return {
            "explanation": explanation,
            "explanation_method": explanation_method,
            "input_data": input_data,
            "explanation_summary": f"Prediction explained using {explanation_method}"
        }
        
    async def _handle_analyze_dataset(self, arguments):
        """
        Handle analyze dataset tool.

        Args:
            arguments: The tool arguments

        Returns:
            Dict[str, Any]: The tool result
        """
        dataset_description = arguments.get("dataset_description")
        analysis_type = arguments.get("analysis_type", "exploratory")
        
        # Get data engineer agent
        data_engineer = self.coordinator.get_agent("data_engineer")
        
        # Analyze dataset
        analysis_results = data_engineer.analyze_dataset(
            dataset_description, analysis_type
        )
        
        return {
            "analysis_results": analysis_results,
            "analysis_type": analysis_type,
            "dataset_description": dataset_description,
            "analysis_summary": f"Dataset analyzed using {analysis_type} analysis"
        }

    async def _handle_list_resources(self, request):
        """
        Handle list resources request.

        Args:
            request: The list resources request

        Returns:
            Dict[str, Any]: The list resources response
        """
        # For now, we don't provide any static resources
        return {"resources": []}

    async def _handle_list_resource_templates(self, request):
        """
        Handle list resource templates request.

        Args:
            request: The list resource templates request

        Returns:
            Dict[str, Any]: The list resource templates response
        """
        # For now, we don't provide any resource templates
        return {"resourceTemplates": []}

    async def _handle_read_resource(self, request):
        """
        Handle read resource request.

        Args:
            request: The read resource request

        Returns:
            Dict[str, Any]: The read resource response
        """
        # Since we don't provide any resources, this should not be called
        raise McpError(ErrorCode.InvalidRequest, f"Invalid URI: {request.params.uri}")

    async def run(self):
        """Run the MCP server."""
        # Create transport
        transport = StdioServerTransport()
        
        # Connect server to transport
        await self.server.connect(transport)
        
        logger.info("MCP Server running")


def main():
    """Main entry point for the MCP server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    
    # Create and run server
    server = MCPServer()
    
    try:
        import asyncio
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("MCP Server stopped")


if __name__ == "__main__":
    main()
