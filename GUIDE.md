# AI Agent System for ML Modeling and Code

This guide explains how to install, configure, and use the AI Agent System for machine learning modeling and code generation tasks.

## Overview

The AI Agent System is a multi-agent architecture using Claude Sonnet 3.5 and Anthropic's MCP for machine learning modeling and code generation tasks. It features:

- A hybrid agent architecture with specialized roles
- Advanced memory management system
- Retrieval-augmented generation (RAG) for domain-specific knowledge
- Support for multiple ML domains (Computer Vision, NLP, Tabular Data)
- MCP integration for extended capabilities

## Installation

### Prerequisites

- Python 3.9+
- Anthropic API key

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/project_agent.git
cd project_agent
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies using uv:

```bash
pip install uv
uv pip install -e .
```

4. Set up your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_api_key_here  # On Windows: set ANTHROPIC_API_KEY=your_api_key_here
```

Alternatively, create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_api_key_here
```

## Configuration

The system uses a configuration file to customize its behavior. You can generate a default configuration file using:

```bash
python -m src.main init-config
```

This will create a `config/config.json` file that you can edit to customize the system.

## Usage

### Command Line Interface

The system provides a command-line interface for processing tasks:

```bash
python -m src.main process "Create a neural network for image classification"
```

Additional options:

```bash
python -m src.main process --help
```

### Running the MCP Server

To use the MCP server for extended capabilities:

```bash
python -m src.main run-mcp-server
```

### Initializing the Knowledge Base

To initialize the knowledge base for a specific domain:

```bash
python -m src.main init-knowledge-base cv --source /path/to/documents
```

Replace `cv` with `nlp`, `tabular`, or `general` for other domains.

### Programmatic Usage

You can also use the system programmatically in your Python code:

```python
from src.core.coordinator import Coordinator

# Initialize coordinator
coordinator = Coordinator()

# Process a task
result = coordinator.process_task("Create a neural network for image classification")

# Print result
print(result)
```

See the `examples` directory for more detailed examples.

## Agent Roles

The system uses a hybrid agent architecture with the following specialized roles:

1. **Team Leader**: Strategic oversight and final decision-making
2. **Product Manager**: User need interpretation and requirement management
3. **ML Architect**: ML system design across domains
4. **Software Architect**: Code and system architecture design
5. **ML Engineer**: ML implementation and optimization
6. **Software Engineer**: General code implementation
7. **Data Engineer**: Data pipeline and feature engineering
8. **QA Engineer**: Testing and quality assurance

## Memory System

The system uses a sophisticated memory system with:

- Short-term memory for recent conversations and working memory
- Long-term memory for persistent knowledge using vector embeddings
- Agent-specific memory for context management
- Hierarchical summarization for efficient context usage

## RAG System

The Retrieval-Augmented Generation (RAG) system provides domain-specific knowledge for:

- Computer Vision
- Natural Language Processing
- Tabular Data
- General coding and software development

## MCP Integration

The system integrates with Anthropic's Model Context Protocol (MCP) to provide custom tools for:

- Analyzing requirements
- Designing ML and software architectures
- Implementing ML and code components
- Designing data pipelines
- Testing solutions

## Examples

### Simple Example

```python
from src.core.coordinator import Coordinator

# Initialize coordinator
coordinator = Coordinator()

# Process a task
result = coordinator.process_task("Create a simple neural network for image classification")

# Print result
print(result)
```

### MCP Example

```python
import subprocess
import time

# Start MCP server
process = subprocess.Popen(
    ["python", "-m", "src.mcp.server"],
    env={"ANTHROPIC_API_KEY": "your_api_key_here"},
)

# Wait for server to start
time.sleep(2)

# Use the system with MCP
from src.core.coordinator import Coordinator
coordinator = Coordinator()
result = coordinator.process_task("Design a CNN architecture")

# Stop MCP server
process.terminate()
```

## Troubleshooting

### API Key Issues

If you encounter API key errors, ensure your Anthropic API key is correctly set in the environment or `.env` file.

### Memory Usage

The system can use significant memory for vector embeddings. If you encounter memory issues, consider:

- Reducing the size of the knowledge base
- Using a smaller embedding model
- Limiting the number of concurrent tasks

### MCP Server Issues

If the MCP server fails to start, check:

- Your Anthropic API key is correctly set
- The MCP settings in `config/mcp_settings.json` are correct
- You have the required dependencies installed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
