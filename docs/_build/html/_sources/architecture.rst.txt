Architecture
===========

This section describes the architecture of the PolyMind system.

System Overview
--------------

PolyMind is a sophisticated multi-agent system designed to tackle complex machine learning and software development tasks. The system leverages Claude Sonnet 3.5 as the base model and extends its capabilities through Anthropic's Model Context Protocol (MCP).

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

Agent Architecture
-----------------

.. automodule:: src.agents.base_agent
   :members:
   :undoc-members:
   :show-inheritance:

Memory System
------------

The memory system consists of three main components:

- **Short-Term Memory**: Manages recent conversations and working memory
- **Long-Term Memory**: Stores persistent knowledge using vector embeddings
- **Agent-Specific Memory**: Maintains context for each agent role

.. automodule:: src.memory.memory_manager
   :members:
   :undoc-members:
   :show-inheritance:

RAG System
---------

The Retrieval-Augmented Generation system provides domain-specific knowledge for:

- Computer Vision
- Natural Language Processing
- Tabular Data

.. automodule:: src.rag.retriever
   :members:
   :undoc-members:
   :show-inheritance:

MCP Integration
--------------

The system integrates with Anthropic's Model Context Protocol to provide custom tools for:

- Requirements analysis
- Architecture design
- Implementation assistance
- Testing and validation

.. automodule:: src.mcp.server
   :members:
   :undoc-members:
   :show-inheritance:

System Workflow
--------------

The typical workflow of the system is as follows:

1. **Task Routing**: The system analyzes the user task and routes it to the appropriate agents.
2. **Requirements Analysis**: The Product Manager agent analyzes the requirements.
3. **Architecture Design**: The ML Architect and Software Architect design the system architecture.
4. **Implementation**: The ML Engineer, Software Engineer, and Data Engineer implement the solution.
5. **Testing**: The QA Engineer tests the solution.
6. **Delivery**: The Team Leader finalizes and delivers the solution.

This workflow is orchestrated by the Coordinator, which manages the communication between agents and ensures that each step is completed successfully.

.. automodule:: src.core.coordinator
   :members:
   :undoc-members:
   :show-inheritance:
