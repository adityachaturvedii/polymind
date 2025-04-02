User Guide
=========

This guide explains how to install, configure, and use the PolyMind AI Agent System for machine learning modeling and code generation tasks.

Installation
-----------

Prerequisites
~~~~~~~~~~~~

- Python 3.9+
- Anthropic API key

Setup
~~~~~

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/adityachaturvedii/polymind.git
   cd polymind

2. Create and activate a virtual environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:

.. code-block:: bash

   pip install -e .

4. Set up your Anthropic API key:

.. code-block:: bash

   export ANTHROPIC_API_KEY=your_api_key_here  # On Windows: set ANTHROPIC_API_KEY=your_api_key_here

Alternatively, create a `.env` file in the project root:

.. code-block:: text

   ANTHROPIC_API_KEY=your_api_key_here

Configuration
------------

The system uses a configuration file to customize its behavior. You can generate a default configuration file using:

.. code-block:: bash

   python -m src.main init-config

This will create a `config/config.json` file that you can edit to customize the system.

Usage
-----

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~

The system provides a command-line interface for processing tasks:

.. code-block:: bash

   python -m src.main process "Create a neural network for image classification"

Additional options:

.. code-block:: bash

   python -m src.main process --help

Running the MCP Server
~~~~~~~~~~~~~~~~~~~~~

To use the MCP server for extended capabilities:

.. code-block:: bash

   python -m src.main run-mcp-server

Initializing the Knowledge Base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To initialize the knowledge base for a specific domain:

.. code-block:: bash

   python -m src.main init-knowledge-base cv --source /path/to/documents

Replace `cv` with `nlp`, `tabular`, or `general` for other domains.

Programmatic Usage
~~~~~~~~~~~~~~~~

You can also use the system programmatically in your Python code:

.. code-block:: python

   from src.core.coordinator import Coordinator

   # Initialize coordinator
   coordinator = Coordinator()

   # Process a task
   result = coordinator.process_task("Create a neural network for image classification")

   # Print result
   print(result)

Examples
-------

Simple Example
~~~~~~~~~~~~~

.. code-block:: python

   from src.core.coordinator import Coordinator

   # Initialize coordinator
   coordinator = Coordinator()

   # Process a task
   result = coordinator.process_task("Create a simple neural network for image classification")

   # Print result
   print(result)

MCP Example
~~~~~~~~~~

.. code-block:: python

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

Troubleshooting
--------------

API Key Issues
~~~~~~~~~~~~~

If you encounter API key errors, ensure your Anthropic API key is correctly set in the environment or `.env` file.

Memory Usage
~~~~~~~~~~~

The system can use significant memory for vector embeddings. If you encounter memory issues, consider:

- Reducing the size of the knowledge base
- Using a smaller embedding model
- Limiting the number of concurrent tasks

MCP Server Issues
~~~~~~~~~~~~~~~

If the MCP server fails to start, check:

- Your Anthropic API key is correctly set
- The MCP settings in `config/mcp_settings.json` are correct
- You have the required dependencies installed
