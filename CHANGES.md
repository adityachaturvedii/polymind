# PolyMind Project Changes


### Directory Structure
- Renamed the project directory from "project_agent" to "polymind"


### Project Name Updates
- Changed project name in pyproject.toml from "project_agent" to "polymind"
- Updated project_name in SystemConfig from "project_agent" to "polymind"
- Updated MCP server name from "project_agent_mcp" to "polymind_mcp"
- Updated MCP settings in config/mcp_settings.json
- Updated CLI app name in src/main.py from "project_agent" to "polymind"

## System Overview

PolyMind is a sophisticated multi-agent system designed to tackle complex machine learning and software development tasks. The system leverages Claude Sonnet 3.5 as the base model and extends its capabilities through Anthropic's Model Context Protocol (MCP).

### Key Components

1. **Hybrid Agent Architecture**
   - Base agent framework with specialized roles
   - Team Leader agent for coordination and decision-making
   - Product Manager agent for requirements analysis
   - ML Architect agent for ML system design across domains
   - Software Architect agent for code and system architecture
   - ML Engineer agent for ML implementation and optimization
   - Software Engineer agent for general code implementation
   - Data Engineer agent for data pipeline and feature engineering
   - QA Engineer agent for testing and quality assurance

2. **Advanced Memory System**
   - Short-term memory for conversations and working memory
   - Long-term memory with vector embeddings for persistent knowledge
   - Agent-specific memory for role-based context management
   - Hierarchical summarization for efficient context usage

3. **RAG System**
   - Vector database integration for knowledge retrieval
   - Domain-specific knowledge organization (CV, NLP, Tabular)
   - Context building for different task types

4. **MCP Integration**
   - Custom MCP server implementation
   - Tools for requirements analysis, architecture design, implementation, and testing
   - Configuration for Claude Sonnet 3.5 integration
   - Extended specialized tools for model optimization, fine-tuning, deployment, explainability, and data analysis

5. **Domain-Specific Components**
   - Computer Vision (CV) models and utilities
