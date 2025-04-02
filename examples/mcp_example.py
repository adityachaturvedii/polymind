"""
MCP example of using the AI Agent System.

This example demonstrates how to use the MCP server to extend the capabilities
of the multi-agent system.
"""

import os
import json
import logging
import subprocess
import time
from dotenv import load_dotenv

from src.core.coordinator import Coordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Load environment variables
load_dotenv()


def start_mcp_server():
    """Start the MCP server in a separate process."""
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return None
    
    # Start MCP server
    print("Starting MCP server...")
    process = subprocess.Popen(
        ["python", "-m", "src.mcp.server"],
        env={**os.environ, "ANTHROPIC_API_KEY": api_key},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for server to start
    time.sleep(2)
    
    return process


def stop_mcp_server(process):
    """Stop the MCP server."""
    if process:
        print("Stopping MCP server...")
        process.terminate()
        process.wait()


def main():
    """Run an example of using the AI Agent System with MCP."""
    # Start MCP server
    mcp_process = start_mcp_server()
    if not mcp_process:
        return
    
    try:
        # Initialize coordinator
        coordinator = Coordinator()
        
        # Define a task
        task = "Design a convolutional neural network architecture for image classification"
        
        # Process task
        print(f"Processing task: {task}")
        result = coordinator.process_task(task)
        
        # Print result
        print("\nResult:")
        print(json.dumps(result, indent=2))
    finally:
        # Stop MCP server
        stop_mcp_server(mcp_process)


if __name__ == "__main__":
    main()
