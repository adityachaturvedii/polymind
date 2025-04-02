"""
Simple example of using the AI Agent System.

This example demonstrates how to use the multi-agent system to process a task.
"""

import os
import json
import logging
from dotenv import load_dotenv

from src.core.coordinator import Coordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Load environment variables
load_dotenv()


def main():
    """Run a simple example of the AI Agent System."""
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return
    
    # Initialize coordinator
    coordinator = Coordinator()
    
    # Define a task
    task = "Create a simple neural network for image classification using PyTorch"
    
    # Process task
    print(f"Processing task: {task}")
    result = coordinator.process_task(task)
    
    # Print result
    print("\nResult:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
