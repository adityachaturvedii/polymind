"""
Task Router module for the AI Agent System.

This module is responsible for analyzing and routing tasks to the appropriate
domain and pipeline based on the task description and requirements.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from anthropic import Anthropic

from src.core.config import get_config

logger = logging.getLogger(__name__)


class TaskRouter:
    """
    Task Router for the AI Agent System.

    The TaskRouter analyzes task descriptions and routes them to the appropriate
    domain (CV, NLP, tabular, general) and determines the task type (modeling,
    coding, analysis, etc.).
    """

    def __init__(self):
        """Initialize the TaskRouter."""
        self.config = get_config()
        self.client = Anthropic(api_key=self.config.api.anthropic_api_key)

        # Domain keywords for basic pattern matching
        self.domain_keywords = {
            "cv": {
                "image", "vision", "camera", "photo", "picture", "object detection",
                "segmentation", "classification", "cnn", "convolutional", "opencv",
                "visual", "face", "recognition", "yolo", "rcnn", "resnet",
            },
            "nlp": {
                "text", "language", "nlp", "sentiment", "translation", "summarization",
                "chatbot", "bert", "gpt", "transformer", "token", "embedding", "word",
                "sentence", "document", "corpus", "linguistic", "speech", "intent",
                "named entity", "ner", "pos tagging", "parsing",
            },
            "tabular": {
                "table", "csv", "excel", "row", "column", "dataframe", "pandas",
                "regression", "classification", "clustering", "feature", "xgboost",
                "lightgbm", "random forest", "decision tree", "linear", "logistic",
                "categorical", "numerical", "time series", "forecast",
            },
        }

        # Task type keywords for basic pattern matching
        self.task_type_keywords = {
            "modeling": {
                "train", "model", "predict", "classify", "regression", "cluster",
                "neural network", "deep learning", "machine learning", "hyperparameter",
                "accuracy", "precision", "recall", "f1", "auc", "loss", "gradient",
                "backpropagation", "epoch", "batch", "validation", "test",
            },
            "coding": {
                "code", "implement", "function", "class", "method", "api", "endpoint",
                "interface", "library", "framework", "module", "package", "algorithm",
                "data structure", "optimization", "refactor", "debug",
            },
            "analysis": {
                "analyze", "explore", "visualize", "plot", "graph", "chart", "insight",
                "statistic", "correlation", "distribution", "hypothesis", "test",
                "significance", "p-value", "confidence", "interval", "mean", "median",
                "mode", "variance", "standard deviation", "outlier",
            },
            "deployment": {
                "deploy", "production", "serve", "api", "endpoint", "docker", "container",
                "kubernetes", "cloud", "aws", "azure", "gcp", "scale", "performance",
                "latency", "throughput", "monitoring", "logging", "alert",
            },
        }

    def _keyword_match(self, text: str, keywords: Set[str]) -> int:
        """
        Count the number of keyword matches in the text.

        Args:
            text: The text to analyze
            keywords: Set of keywords to match

        Returns:
            int: Number of matches
        """
        text = text.lower()
        count = 0
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text):
                count += 1
        return count

    def _analyze_with_claude(self, task: str) -> Dict[str, Any]:
        """
        Use Claude to analyze the task and determine domain and type.

        Args:
            task: The task description

        Returns:
            Dict[str, Any]: Analysis results
        """
        prompt = f"""
        Analyze the following task and determine:
        1. The primary domain (computer vision, NLP, tabular data, or general coding)
        2. The task type (modeling, coding, analysis, deployment)
        3. Key requirements and constraints
        
        Task: {task}
        
        Respond in JSON format with the following structure:
        {{
            "domain": "cv" | "nlp" | "tabular" | "general",
            "task_type": "modeling" | "coding" | "analysis" | "deployment",
            "requirements": [list of key requirements],
            "constraints": [list of constraints if any]
        }}
        """

        response = self.client.messages.create(
            model=self.config.agents.model_versions["team_leader"],
            max_tokens=1000,
            temperature=0.2,
            system="You are an AI assistant that analyzes tasks and determines their domain and type.",
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract JSON from response
        content = response.content[0].text
        # Find JSON block in the response
        import json
        import re

        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = re.search(r'{.*}', content, re.DOTALL).group(0)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from Claude response: {content}")
            # Return a default analysis
            return {
                "domain": "general",
                "task_type": "coding",
                "requirements": ["Implement the requested functionality"],
                "constraints": []
            }

    def route_task(self, task: str) -> Dict[str, Any]:
        """
        Analyze and route a task to the appropriate domain and pipeline.

        This method uses a combination of keyword matching and Claude analysis
        to determine the task domain and type.

        Args:
            task: The task description

        Returns:
            Dict[str, Any]: Task routing information
        """
        logger.info(f"Routing task: {task}")

        # Step 1: Basic keyword matching for initial assessment
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            domain_scores[domain] = self._keyword_match(task, keywords)

        task_type_scores = {}
        for task_type, keywords in self.task_type_keywords.items():
            task_type_scores[task_type] = self._keyword_match(task, keywords)

        # Step 2: Use Claude for more sophisticated analysis
        claude_analysis = self._analyze_with_claude(task)

        # Step 3: Combine results
        # If Claude is confident, use its domain and task type
        # Otherwise, use the highest scoring from keyword matching
        if claude_analysis:
            domain = claude_analysis.get("domain")
            task_type = claude_analysis.get("task_type")
            requirements = claude_analysis.get("requirements", [])
            constraints = claude_analysis.get("constraints", [])
        else:
            # Fallback to keyword matching
            domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            if domain_scores[domain] == 0:
                domain = "general"  # Default to general if no matches

            task_type = max(task_type_scores.items(), key=lambda x: x[1])[0]
            if task_type_scores[task_type] == 0:
                task_type = "coding"  # Default to coding if no matches

            requirements = ["Implement the requested functionality"]
            constraints = []

        # Step 4: Prepare routing information
        routing_info = {
            "domain": domain,
            "task_type": task_type,
            "requirements": requirements,
            "constraints": constraints,
            "domain_scores": domain_scores,
            "task_type_scores": task_type_scores,
        }

        logger.info(f"Task routed: {routing_info}")
        return routing_info
