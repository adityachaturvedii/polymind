"""
Configuration module for the AI Agent System.

This module handles loading and managing configuration settings for the entire system,
including agent settings, memory configuration, and API credentials.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pydantic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class MemoryConfig(pydantic.BaseModel):
    """Configuration for the memory system."""

    # Short-term memory settings
    conversation_buffer_size: int = 20
    working_memory_capacity: int = 50
    decay_rate: float = 0.95

    # Long-term memory settings
    episodic_importance_threshold: float = 0.7
    semantic_update_frequency: str = "daily"
    procedural_versioning: bool = True

    # Vector database settings
    vector_db: str = "chroma"
    embedding_model: str = "all-mpnet-base-v2"
    vector_db_path: Path = Path("data/embeddings")

    # Knowledge graph settings
    knowledge_graph_enabled: bool = True
    knowledge_graph_uri: Optional[str] = None


class AgentConfig(pydantic.BaseModel):
    """Configuration for individual agents."""

    # Context window sizes for each agent role
    context_window_sizes: Dict[str, int] = {
        "team_leader": 8000,
        "product_manager": 6000,
        "ml_architect": 7000,
        "sw_architect": 7000,
        "ml_engineer": 6000,
        "sw_engineer": 6000,
        "data_engineer": 5000,
        "qa_engineer": 5000,
    }

    # Summary levels for hierarchical summarization
    summary_levels: int = 3

    # Agent-specific settings
    temperature: Dict[str, float] = {
        "team_leader": 0.7,
        "product_manager": 0.7,
        "ml_architect": 0.5,
        "sw_architect": 0.5,
        "ml_engineer": 0.3,
        "sw_engineer": 0.3,
        "data_engineer": 0.4,
        "qa_engineer": 0.2,
    }

    # Model versions for each agent - using Claude 3.5 Sonnet
    model_versions: Dict[str, str] = {
        "team_leader": "claude-3-5-sonnet",
        "product_manager": "claude-3-5-sonnet",
        "ml_architect": "claude-3-5-sonnet",
        "sw_architect": "claude-3-5-sonnet",
        "ml_engineer": "claude-3-5-sonnet",
        "sw_engineer": "claude-3-5-sonnet",
        "data_engineer": "claude-3-5-sonnet",
        "qa_engineer": "claude-3-5-sonnet",
    }


class RAGConfig(pydantic.BaseModel):
    """Configuration for the RAG system."""

    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.7
    reranking_enabled: bool = True

    # Knowledge sources
    knowledge_base_path: Path = Path("data/knowledge")
    external_sources_enabled: bool = False
    external_sources: List[str] = []


class DomainConfig(pydantic.BaseModel):
    """Configuration for domain-specific components."""

    # Computer Vision settings
    cv_enabled: bool = True
    cv_models_path: Optional[Path] = None
    cv_default_image_size: tuple = (224, 224)

    # NLP settings
    nlp_enabled: bool = True
    nlp_models_path: Optional[Path] = None
    nlp_max_sequence_length: int = 512

    # Tabular data settings
    tabular_enabled: bool = True
    tabular_models_path: Optional[Path] = None
    tabular_default_metrics: List[str] = ["accuracy", "f1", "roc_auc"]


class APIConfig(pydantic.BaseModel):
    """Configuration for API connections."""

    # Anthropic API settings
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_api_url: str = "https://api.anthropic.com"

    # Other potential API settings
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")


class MCPConfig(pydantic.BaseModel):
    """Configuration for MCP server."""

    enabled: bool = True
    server_name: str = "polymind_mcp"
    server_version: str = "0.1.0"
    tools_enabled: List[str] = [
        "analyze_requirements",
        "design_ml_architecture",
        "design_software_architecture",
        "implement_ml_component",
        "implement_code_component",
        "design_data_pipeline",
        "test_solution",
    ]


class SystemConfig(pydantic.BaseModel):
    """Main system configuration."""

    # General settings
    project_name: str = "polymind"
    debug_mode: bool = False
    log_level: str = "INFO"
    data_dir: Path = Path("data")

    # Component configurations
    memory: MemoryConfig = MemoryConfig()
    agents: AgentConfig = AgentConfig()
    rag: RAGConfig = RAGConfig()
    domains: DomainConfig = DomainConfig()
    api: APIConfig = APIConfig()
    mcp: MCPConfig = MCPConfig()

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "SystemConfig":
        """Load configuration from a file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = pydantic.parse_file_as(Dict[str, Any], config_path)
            return cls(**config_data)

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            f.write(self.model_dump_json(indent=2))


# Default configuration instance
config = SystemConfig()


def load_config(config_path: Optional[Union[str, Path]] = None) -> SystemConfig:
    """
    Load the system configuration.

    Args:
        config_path: Path to the configuration file. If None, uses default settings.

    Returns:
        SystemConfig: The loaded configuration.
    """
    if config_path is None:
        return config

    return SystemConfig.from_file(config_path)


def get_config() -> SystemConfig:
    """
    Get the current system configuration.

    Returns:
        SystemConfig: The current configuration.
    """
    return config
