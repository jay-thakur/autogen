"""Test fixtures for Azure AI Search tool tests."""

import pytest
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from autogen_core import ComponentModel


class MockAzureKeyCredential:
    def __init__(self, key="test-key"):
        self.key = key

class MockResourceNotFoundError(Exception):
    def __init__(self, message="Resource not found", **kwargs):
        self.message = message
        super().__init__(message)

class MockHttpResponseError(Exception):
    def __init__(self, message="Http error", **kwargs):
        self.message = message
        super().__init__(message)


sys.modules['azure'] = MagicMock()
sys.modules['azure.core'] = MagicMock()
sys.modules['azure.core.credentials'] = MagicMock()
sys.modules['azure.core.exceptions'] = MagicMock()
sys.modules['azure.search'] = MagicMock()
sys.modules['azure.search.documents'] = MagicMock()
sys.modules['azure.search.documents.aio'] = MagicMock()


sys.modules['azure.core.credentials'].AzureKeyCredential = MockAzureKeyCredential
sys.modules['azure.core.exceptions'].ResourceNotFoundError = MockResourceNotFoundError
sys.modules['azure.core.exceptions'].HttpResponseError = MockHttpResponseError


@pytest.fixture
def test_config():
    """Create a test configuration for the Azure AI Search tool."""
    return ComponentModel(
        provider="autogen_ext.tools.azure.AzureAISearchTool",
        config={
            "name": "TestAzureSearch",
            "description": "Test Azure AI Search Tool",
            "endpoint": "https://test-search-service.search.windows.net",
            "index_name": "test-index",
            "api_version": "2023-10-01-Preview",
            "credential": {"api_key": "test-key"},
            "query_type": "simple",
            "search_fields": ["content", "title"],
            "select_fields": ["id", "content", "title", "source"],
            "top": 5
        }
    )

@pytest.fixture
def semantic_config():
    """Create a test configuration for semantic search."""
    return ComponentModel(
        provider="autogen_ext.tools.azure.AzureAISearchTool",
        config={
            "name": "TestAzureSearch",
            "description": "Test Azure AI Search Tool",
            "endpoint": "https://test-search-service.search.windows.net",
            "index_name": "test-index",
            "api_version": "2023-10-01-Preview",
            "credential": {"api_key": "test-key"},
            "query_type": "semantic",
            "semantic_config_name": "test-semantic-config",
            "search_fields": ["content", "title"],
            "select_fields": ["id", "content", "title", "source"],
            "top": 5
        }
    )

@pytest.fixture
def vector_config():
    """Create a test configuration for vector search."""
    return ComponentModel(
        provider="autogen_ext.tools.azure.AzureAISearchTool",
        config={
            "name": "TestAzureSearch",
            "description": "Test Azure AI Search Tool",
            "endpoint": "https://test-search-service.search.windows.net",
            "index_name": "test-index",
            "api_version": "2023-10-01-Preview",
            "credential": {"api_key": "test-key"},
            "query_type": "vector",
            "vector_fields": ["embedding"],
            "select_fields": ["id", "content", "title", "source"],
            "top": 5
        }
    )

@pytest.fixture
def mock_search_response():
    """Create a mock search response."""
    return [
        {
            "@search.score": 0.95,
            "id": "doc1",
            "content": "This is the first document content",
            "title": "Document 1",
            "source": "test-source-1"
        },
        {
            "@search.score": 0.85,
            "id": "doc2",
            "content": "This is the second document content",
            "title": "Document 2",
            "source": "test-source-2"
        }
    ]

# Create a proper async iterator class for testing
class AsyncIterator:
    def __init__(self, items):
        self.items = items.copy()  # Work with a copy
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)

@pytest.fixture
def mock_search_client(mock_search_response):
    """Create a mock search client for testing."""
    # Create a mock client
    mock_client = MagicMock()
    
    # Set up the client to be used as an async context manager
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    # Create a search method that returns an async iterator
    mock_client.search = MagicMock()  # Not AsyncMock to avoid the attribute error
    
    # Make the search method return an async iterator of the response items
    search_results = AsyncIterator(mock_search_response)
    mock_client.search.return_value = search_results
    
    # Set up count method (needed for some tests)
    mock_client.search.return_value.get_count = MagicMock(return_value=len(mock_search_response))
    
    # Create a patcher for the SearchClient constructor
    patcher = patch('azure.search.documents.aio.SearchClient', return_value=mock_client)
    
    return mock_client, patcher
