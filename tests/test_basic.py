"""
Basic tests for the AI Agent System.

This module contains basic tests to verify that the system is working correctly.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.core.config import get_config
from src.core.coordinator import Coordinator
from src.agents.team_leader import TeamLeader
from src.agents.product_manager import ProductManager
from src.memory.memory_manager import MemoryManager
from src.rag.retriever import Retriever


class TestBasic(unittest.TestCase):
    """Basic tests for the AI Agent System."""

    def setUp(self):
        """Set up test environment."""
        # Mock Anthropic API key
        os.environ["ANTHROPIC_API_KEY"] = "mock_api_key"
        
        # Mock configuration
        self.config = get_config()
        
        # Mock Anthropic client
        self.anthropic_mock = MagicMock()
        self.anthropic_mock.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Mock response")]
        )

    @patch("anthropic.Anthropic")
    def test_coordinator_initialization(self, anthropic_mock):
        """Test that the coordinator can be initialized."""
        anthropic_mock.return_value = self.anthropic_mock
        
        # Initialize coordinator
        coordinator = Coordinator()
        
        # Check that coordinator was initialized
        self.assertIsNotNone(coordinator)
        self.assertIsNotNone(coordinator.memory)
        self.assertIsNotNone(coordinator.retriever)
        self.assertIsNotNone(coordinator.task_router)
        self.assertIsNotNone(coordinator.agents)

    @patch("anthropic.Anthropic")
    def test_team_leader_initialization(self, anthropic_mock):
        """Test that the team leader agent can be initialized."""
        anthropic_mock.return_value = self.anthropic_mock
        
        # Mock memory and retriever
        memory = MagicMock(spec=MemoryManager)
        retriever = MagicMock(spec=Retriever)
        
        # Initialize team leader
        team_leader = TeamLeader(
            self.anthropic_mock,
            memory,
            retriever,
            self.config.agents,
        )
        
        # Check that team leader was initialized
        self.assertIsNotNone(team_leader)
        self.assertEqual(team_leader.role, "team_leader")
        self.assertEqual(team_leader.model, "claude-3-5-sonnet")

    @patch("anthropic.Anthropic")
    def test_product_manager_initialization(self, anthropic_mock):
        """Test that the product manager agent can be initialized."""
        anthropic_mock.return_value = self.anthropic_mock
        
        # Mock memory and retriever
        memory = MagicMock(spec=MemoryManager)
        retriever = MagicMock(spec=Retriever)
        
        # Initialize product manager
        product_manager = ProductManager(
            self.anthropic_mock,
            memory,
            retriever,
            self.config.agents,
        )
        
        # Check that product manager was initialized
        self.assertIsNotNone(product_manager)
        self.assertEqual(product_manager.role, "product_manager")
        self.assertEqual(product_manager.model, "claude-3-5-sonnet")

    @patch("anthropic.Anthropic")
    @patch("src.core.coordinator.Coordinator._initialize_agents")
    def test_process_task(self, init_agents_mock, anthropic_mock):
        """Test that a task can be processed."""
        anthropic_mock.return_value = self.anthropic_mock
        
        # Mock agents
        agents_mock = {
            "team_leader": MagicMock(),
            "product_manager": MagicMock(),
            "ml_architect": MagicMock(),
            "sw_architect": MagicMock(),
            "ml_engineer": MagicMock(),
            "sw_engineer": MagicMock(),
            "data_engineer": MagicMock(),
            "qa_engineer": MagicMock(),
        }
        init_agents_mock.return_value = agents_mock
        
        # Mock task router
        task_router_mock = MagicMock()
        task_router_mock.route_task.return_value = {
            "domain": "cv",
            "task_type": "modeling",
            "requirements": ["Test requirement"],
            "constraints": [],
        }
        
        # Mock product manager
        agents_mock["product_manager"].analyze_requirements.return_value = {
            "user_stories": "Test user story",
            "functional_requirements": "Test functional requirement",
            "non_functional_requirements": "Test non-functional requirement",
            "constraints": "Test constraint",
            "acceptance_criteria": "Test acceptance criteria",
            "priorities": "Test priorities",
        }
        
        # Mock ML architect
        agents_mock["ml_architect"].design_architecture.return_value = {
            "architecture": "Test ML architecture",
        }
        
        # Mock software architect
        agents_mock["sw_architect"].design_architecture.return_value = {
            "architecture": "Test software architecture",
        }
        
        # Mock data engineer
        agents_mock["data_engineer"].design_pipeline.return_value = {
            "pipeline": "Test data pipeline",
        }
        
        # Mock ML engineer
        agents_mock["ml_engineer"].implement.return_value = {
            "implementation": "Test ML implementation",
        }
        
        # Mock software engineer
        agents_mock["sw_engineer"].implement.return_value = {
            "implementation": "Test software implementation",
        }
        
        # Mock QA engineer
        agents_mock["qa_engineer"].test.return_value = {
            "test_results": "Test results",
        }
        
        # Mock team leader
        agents_mock["team_leader"].finalize.return_value = {
            "result": "Test result",
        }
        
        # Initialize coordinator with mocks
        coordinator = Coordinator()
        coordinator.task_router = task_router_mock
        
        # Process task
        result = coordinator.process_task("Test task")
        
        # Check that task was processed
        self.assertIsNotNone(result)
        self.assertEqual(result, {"result": "Test result"})
        
        # Check that agents were called
        agents_mock["product_manager"].analyze_requirements.assert_called_once()
        agents_mock["ml_architect"].design_architecture.assert_called_once()
        agents_mock["sw_architect"].design_architecture.assert_called_once()
        agents_mock["data_engineer"].design_pipeline.assert_called_once()
        agents_mock["ml_engineer"].implement.assert_called_once()
        agents_mock["sw_engineer"].implement.assert_called_once()
        agents_mock["qa_engineer"].test.assert_called_once()
        agents_mock["team_leader"].finalize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
