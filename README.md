# PolyMind: AI Agent System for ML Modeling and Code

A multi-agent architecture using Claude Sonnet 3.5 and Anthropic's MCP for machine learning modeling and code generation tasks.

**Author:** Aditya Chaturvedi

## Overview

This project implements a sophisticated multi-agent system designed to tackle complex machine learning and software development tasks. The system leverages Claude Sonnet 3.5 as the base model and extends its capabilities through Anthropic's Model Context Protocol (MCP).

Key features include:

- **Hybrid Agent Architecture**: Specialized agents with distinct roles and expertise
- **Advanced Memory System**: Short-term, long-term, and agent-specific memory components
- **Retrieval-Augmented Generation (RAG)**: Domain-specific knowledge retrieval
- **Multi-Domain Support**: Computer Vision, NLP, and Tabular data
- **MCP Integration**: Custom tools for extended capabilities

## Architecture

The system is built around a team of specialized agents, each with a specific role:

1. **Team Leader**: Coordinates between agents and makes final decisions
2. **Product Manager**: Analyzes requirements and manages product aspects
3. **ML Architect**: Designs ML system architecture across domains
4. **Software Architect**: Designs software architecture and interfaces
5. **ML Engineer**: Implements ML models and algorithms
6. **Software Engineer**: Implements non-ML components and integration
7. **Data Engineer**: Designs data pipelines and feature engineering
8. **QA Engineer**: Tests and validates system behavior

These agents collaborate through a central coordinator, with memory systems ensuring context preservation and knowledge sharing.

## Memory System

The memory system consists of three main components:

- **Short-Term Memory**: Manages recent conversations and working memory
- **Long-Term Memory**: Stores persistent knowledge using vector embeddings
- **Agent-Specific Memory**: Maintains context for each agent role

## RAG System

The Retrieval-Augmented Generation system provides domain-specific knowledge for:

- Computer Vision
- Natural Language Processing
- Tabular Data

## MCP Integration

The system integrates with Anthropic's Model Context Protocol to provide custom tools for:

- Requirements analysis
- Architecture design
- Implementation assistance
- Testing and validation

## Installation

### Prerequisites

- Python 3.9+
- Anthropic API key

### Setup

1. Clone the repository:

```bash
git clone https://github.com/adityachaturvedii/polymind.git
cd polymind
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -e .
```

4. Set up your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_api_key_here  # On Windows: set ANTHROPIC_API_KEY=your_api_key_here
```

Alternatively, create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

### Command Line Interface

```bash
python -m src.main process "Create a neural network for image classification"
```

### Programmatic Usage

```python
from src.core.coordinator import Coordinator

coordinator = Coordinator()
result = coordinator.process_task("Create a neural network for image classification")
```

### MCP Server

To run the MCP server:

```bash
python -m src.main run-mcp-server
```

## Examples

See the `examples` directory for usage examples:

- `simple_example.py`: Basic usage without MCP
- `mcp_example.py`: Advanced usage with MCP integration

## Testing

Run the test suite:

```bash
python run_tests.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Anthropic for Claude and the Model Context Protocol
- The open-source community for various libraries and tools used in this project
