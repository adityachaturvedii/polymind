"""
Main entry point for the PolyMind AI Agent System.

This module provides a command-line interface for interacting with the
multi-agent system.

Author: Aditya Chaturvedi
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import typer
from dotenv import load_dotenv

from src.core.config import get_config, SystemConfig
from src.core.coordinator import Coordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create Typer app
app = typer.Typer(
    name="polymind",
    help="PolyMind: AI Agent System for ML Modeling and Code",
    add_completion=False,
)


@app.command()
def process(
    task: str = typer.Argument(..., help="The task to process"),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    output_path: Optional[str] = typer.Option(
        None, "--output", "-o", help="Path to save output"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """
    Process a task using the multi-agent system.
    """
    # Set log level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = get_config()
    if config_path:
        config = SystemConfig.from_file(config_path)
    
    # Check for API key
    if not config.api.anthropic_api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            typer.echo("Error: ANTHROPIC_API_KEY environment variable not set")
            raise typer.Exit(code=1)
        config.api.anthropic_api_key = api_key
    
    # Initialize coordinator
    coordinator = Coordinator(config_path)
    
    try:
        # Process task
        typer.echo(f"Processing task: {task}")
        result = coordinator.process_task(task)
        
        # Print result
        typer.echo("\nResult:")
        typer.echo(json.dumps(result, indent=2))
        
        # Save output if requested
        if output_path:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            typer.echo(f"\nOutput saved to: {output_path}")
        
        return result
    except Exception as e:
        typer.echo(f"Error: {str(e)}")
        if verbose:
            import traceback
            typer.echo(traceback.format_exc())
        raise typer.Exit(code=1)


@app.command()
def run_mcp_server():
    """
    Run the MCP server.
    """
    from src.mcp.server import main as mcp_main
    
    typer.echo("Starting MCP server...")
    mcp_main()


@app.command()
def init_config(
    output_path: str = typer.Option(
        "config/config.json", "--output", "-o", help="Path to save configuration"
    ),
):
    """
    Initialize a default configuration file.
    """
    # Get default configuration
    config = get_config()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save configuration
    config.save_to_file(output_path)
    
    typer.echo(f"Default configuration saved to: {output_path}")


@app.command()
def init_knowledge_base(
    domain: str = typer.Argument(..., help="Domain to initialize (cv, nlp, tabular, general)"),
    source_dir: Optional[str] = typer.Option(
        None, "--source", "-s", help="Source directory with documents"
    ),
):
    """
    Initialize the knowledge base for a specific domain.
    """
    from src.rag.retriever import Retriever
    from src.core.config import RAGConfig
    
    # Validate domain
    valid_domains = ["cv", "nlp", "tabular", "general"]
    if domain not in valid_domains:
        typer.echo(f"Error: Invalid domain. Must be one of: {', '.join(valid_domains)}")
        raise typer.Exit(code=1)
    
    # Get configuration
    config = get_config()
    
    # Initialize retriever
    retriever = Retriever(config.rag)
    
    # If source directory is provided, add documents
    if source_dir:
        if not os.path.isdir(source_dir):
            typer.echo(f"Error: Source directory not found: {source_dir}")
            raise typer.Exit(code=1)
        
        # Add documents from source directory
        typer.echo(f"Adding documents from {source_dir} to {domain} domain...")
        
        # Walk through directory
        for root, _, files in os.walk(source_dir):
            for file in files:
                # Skip hidden files
                if file.startswith("."):
                    continue
                
                # Get file path
                file_path = os.path.join(root, file)
                
                # Get file extension
                _, ext = os.path.splitext(file)
                
                # Skip binary files
                if ext.lower() in [".jpg", ".jpeg", ".png", ".gif", ".pdf", ".zip"]:
                    continue
                
                try:
                    # Read file
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # Create metadata
                    metadata = {
                        "title": file,
                        "path": os.path.relpath(file_path, source_dir),
                        "type": "code" if ext.lower() in [".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".cs"] else "text",
                        "language": ext.lower()[1:] if ext else "",
                    }
                    
                    # Add document
                    doc_id = retriever.add_document(content, metadata, domain)
                    
                    typer.echo(f"Added document: {metadata['path']} (ID: {doc_id})")
                except Exception as e:
                    typer.echo(f"Error adding document {file_path}: {str(e)}")
    
    typer.echo(f"Knowledge base initialized for domain: {domain}")


@app.command()
def test_agents():
    """
    Run a simple test of the agent system.
    """
    # Get configuration
    config = get_config()
    
    # Initialize coordinator
    coordinator = Coordinator()
    
    # Test task
    task = "Create a simple neural network for image classification"
    
    # Process task
    typer.echo(f"Processing test task: {task}")
    result = coordinator.process_task(task)
    
    # Print result
    typer.echo("\nResult:")
    typer.echo(json.dumps(result, indent=2))
    
    typer.echo("\nAgent test completed successfully")


if __name__ == "__main__":
    app()
