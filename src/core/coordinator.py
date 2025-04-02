"""
Coordinator module for the PolyMind AI Agent System.

This module implements the main coordinator that orchestrates the agents,
manages the workflow, and handles the overall task processing.

Author: Aditya Chaturvedi
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from anthropic import Anthropic

from src.agents.data_engineer import DataEngineer
from src.agents.ml_architect import MLArchitect
from src.agents.ml_engineer import MLEngineer
from src.agents.product_manager import ProductManager
from src.agents.qa_engineer import QAEngineer
from src.agents.sw_architect import SoftwareArchitect
from src.agents.sw_engineer import SoftwareEngineer
from src.agents.team_leader import TeamLeader
from src.core.config import get_config
from src.core.task_router import TaskRouter
from src.memory.memory_manager import MemoryManager
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class Coordinator:
    """
    Main coordinator for the AI Agent System.

    The Coordinator orchestrates the agents, manages the workflow, and handles
    the overall task processing. It serves as the central hub for communication
    between agents and external interfaces.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Coordinator.

        Args:
            config_path: Optional path to a configuration file.
        """
        # Load configuration
        self.config = get_config()
        if config_path:
            self.config = get_config(config_path)

        # Initialize API client
        self.client = Anthropic(api_key=self.config.api.anthropic_api_key)

        # Initialize memory system
        self.memory = MemoryManager(self.config.memory)

        # Initialize RAG system
        self.retriever = Retriever(self.config.rag)

        # Initialize task router
        self.task_router = TaskRouter()

        # Initialize agents
        self.agents = self._initialize_agents()

        logger.info("Coordinator initialized successfully")

    def _initialize_agents(self) -> Dict[str, Any]:
        """
        Initialize all agent instances.

        Returns:
            Dict[str, Any]: Dictionary of agent instances.
        """
        return {
            "team_leader": TeamLeader(
                self.client,
                self.memory,
                self.retriever,
                self.config.agents,
            ),
            "product_manager": ProductManager(
                self.client,
                self.memory,
                self.retriever,
                self.config.agents,
            ),
            "ml_architect": MLArchitect(
                self.client,
                self.memory,
                self.retriever,
                self.config.agents,
            ),
            "sw_architect": SoftwareArchitect(
                self.client,
                self.memory,
                self.retriever,
                self.config.agents,
            ),
            "ml_engineer": MLEngineer(
                self.client,
                self.memory,
                self.retriever,
                self.config.agents,
            ),
            "sw_engineer": SoftwareEngineer(
                self.client,
                self.memory,
                self.retriever,
                self.config.agents,
            ),
            "data_engineer": DataEngineer(
                self.client,
                self.memory,
                self.retriever,
                self.config.agents,
            ),
            "qa_engineer": QAEngineer(
                self.client,
                self.memory,
                self.retriever,
                self.config.agents,
            ),
        }

    def process_task(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Process a user task through the agent system.

        This method orchestrates the entire workflow:
        1. Route the task to determine domain and type
        2. Analyze requirements with Product Manager
        3. Design architecture with Architects
        4. Implement solution with Engineers
        5. Test solution with QA Engineer
        6. Deliver final result with Team Leader

        Args:
            task: The user task description
            **kwargs: Additional task parameters

        Returns:
            Dict[str, Any]: The task result
        """
        # Generate task ID
        task_id = str(uuid.uuid4())
        logger.info(f"Processing task {task_id}: {task}")

        # Initialize task context
        task_context = {
            "task_id": task_id,
            "task": task,
            "parameters": kwargs,
            "status": "started",
            "result": None,
        }

        # Store task in memory
        self.memory.short_term.update_task_context(task_id, task_context)

        try:
            # Step 1: Route the task
            task_info = self.task_router.route_task(task)
            task_context["task_info"] = task_info
            logger.info(f"Task routed: {task_info}")

            # Step 2: Analyze requirements with Product Manager
            requirements = self.agents["product_manager"].analyze_requirements(
                task, task_info, task_id
            )
            task_context["requirements"] = requirements
            logger.info("Requirements analysis completed")

            # Step 3: Design architecture
            # ML architecture if needed
            ml_architecture = None
            if task_info.get("domain") in ["cv", "nlp", "tabular"]:
                ml_architecture = self.agents["ml_architect"].design_architecture(
                    task, requirements, task_info, task_id
                )
                task_context["ml_architecture"] = ml_architecture
                logger.info("ML architecture design completed")

            # Software architecture
            sw_architecture = self.agents["sw_architect"].design_architecture(
                task, requirements, task_info, task_id, ml_architecture
            )
            task_context["sw_architecture"] = sw_architecture
            logger.info("Software architecture design completed")

            # Step 4: Implementation
            # Data pipeline if needed
            data_pipeline = None
            if task_info.get("domain") in ["cv", "nlp", "tabular"]:
                data_pipeline = self.agents["data_engineer"].design_pipeline(
                    task, requirements, task_info, ml_architecture, task_id
                )
                task_context["data_pipeline"] = data_pipeline
                logger.info("Data pipeline design completed")

            # ML implementation if needed
            ml_implementation = None
            if task_info.get("domain") in ["cv", "nlp", "tabular"]:
                ml_implementation = self.agents["ml_engineer"].implement(
                    task,
                    requirements,
                    ml_architecture,
                    data_pipeline,
                    task_id,
                )
                task_context["ml_implementation"] = ml_implementation
                logger.info("ML implementation completed")

            # Software implementation
            sw_implementation = self.agents["sw_engineer"].implement(
                task,
                requirements,
                sw_architecture,
                ml_implementation,
                task_id,
            )
            task_context["sw_implementation"] = sw_implementation
            logger.info("Software implementation completed")

            # Step 5: Testing
            test_results = self.agents["qa_engineer"].test(
                task,
                requirements,
                sw_implementation,
                ml_implementation,
                task_id,
            )
            task_context["test_results"] = test_results
            logger.info("Testing completed")

            # Step 6: Final review and delivery
            result = self.agents["team_leader"].finalize(
                task,
                task_context,
                task_id,
            )
            task_context["result"] = result
            task_context["status"] = "completed"
            logger.info("Task completed successfully")

            # Update task in memory
            self.memory.short_term.update_task_context(task_id, task_context)
            self.memory.long_term.store_task(task_id, task_context)

            return result

        except Exception as e:
            logger.error(f"Error processing task: {str(e)}", exc_info=True)
            task_context["status"] = "failed"
            task_context["error"] = str(e)
            self.memory.short_term.update_task_context(task_id, task_context)
            raise

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.

        Args:
            task_id: The task ID

        Returns:
            Dict[str, Any]: The task status and context
        """
        return self.memory.short_term.get_task_context(task_id)

    def get_agent(self, agent_name: str) -> Any:
        """
        Get an agent instance by name.

        Args:
            agent_name: The name of the agent

        Returns:
            Any: The agent instance
        """
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        return self.agents[agent_name]
