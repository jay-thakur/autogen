"""Azure tools for AutoGen.

This package provides tools for integrating with Azure services.
"""

from ._ai_search import AzureAISearchTool, SearchQuery, SearchResult
from ._config import AzureAISearchConfig

__all__ = ["AzureAISearchTool", "AzureAISearchConfig", "SearchQuery", "SearchResult"]
