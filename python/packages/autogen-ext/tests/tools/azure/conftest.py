"""Test fixtures for Azure AI Search tool tests."""

import sys
import warnings
from types import ModuleType
from typing import Any, Dict, List, TypeVar, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogen_core import ComponentModel

warnings.filterwarnings(
    "ignore",
    message="Type google.*uses PyType_Spec with a metaclass that has custom tp_new",
    category=DeprecationWarning,
)


class MockAzureKeyCredential:
    """Mock implementation of AzureKeyCredential."""

    def __init__(self, key: str = "test-key") -> None:
        self.key = key


class MockTokenCredential:
    """Mock implementation of TokenCredential."""

    def get_token(self, *scopes: str, **kwargs: Any) -> MagicMock:
        return MagicMock(token="mock-token")


class MockResourceNotFoundError(Exception):
    """Mock implementation of ResourceNotFoundError."""

    def __init__(self, message: str = "Resource not found", **kwargs: Any) -> None:
        self.message = message
        super().__init__(message)


class MockHttpResponseError(Exception):
    """Mock implementation of HttpResponseError."""

    def __init__(self, message: str = "Http error", **kwargs: Any) -> None:
        self.message = message
        super().__init__(message)


T = TypeVar("T", bound="SimpleMock")


class SimpleMock:
    """A simple module mock that doesn't suffer from recursion issues."""

    def __init__(self) -> None:
        self._attributes: Dict[str, "SimpleMock"] = {}

    def __getattr__(self, name: str) -> "SimpleMock":
        """Return a new SimpleMock for any attribute access."""
        if name not in self._attributes:
            self._attributes[name] = SimpleMock()
        return self._attributes[name]

    def __call__(self, *args: Any, **kwargs: Any) -> "SimpleMock":
        """Make the mock callable."""
        return SimpleMock()

    def __iter__(self) -> Any:
        """Make the mock iterable to support module imports."""
        return iter(())

    def __dir__(self) -> List[str]:
        """Return list of attributes for dir() and attribute completions."""
        return list(self._attributes.keys())


azure_mock = SimpleMock()

sys.modules["azure"] = cast(ModuleType, azure_mock)  # type: ignore
sys.modules["azure.core"] = cast(ModuleType, azure_mock.core)  # type: ignore
sys.modules["azure.core.credentials"] = cast(ModuleType, azure_mock.core.credentials)  # type: ignore
sys.modules["azure.core.exceptions"] = cast(ModuleType, azure_mock.core.exceptions)  # type: ignore
sys.modules["azure.search"] = cast(ModuleType, azure_mock.search)  # type: ignore
sys.modules["azure.search.documents"] = cast(ModuleType, azure_mock.search.documents)  # type: ignore
sys.modules["azure.search.documents.aio"] = cast(ModuleType, azure_mock.search.documents.aio)  # type: ignore
sys.modules["azure.search.documents.indexes"] = cast(ModuleType, azure_mock.search.documents.indexes)  # type: ignore

mock_credentials: Any = sys.modules["azure.core.credentials"]
mock_credentials.AzureKeyCredential = MockAzureKeyCredential
mock_credentials.TokenCredential = MockTokenCredential

mock_exceptions: Any = sys.modules["azure.core.exceptions"]
mock_exceptions.ResourceNotFoundError = MockResourceNotFoundError
mock_exceptions.HttpResponseError = MockHttpResponseError


@pytest.fixture
def test_config() -> ComponentModel:
    """Create a test configuration for the Azure AI Search tool."""
    return ComponentModel(
        provider="autogen_ext.tools.azure.test_ai_search_tool.MockAzureAISearchTool",
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
            "top": 5,
        },
    )


@pytest.fixture
def semantic_config() -> ComponentModel:
    """Create a test configuration for semantic search."""
    return ComponentModel(
        provider="autogen_ext.tools.azure.test_ai_search_tool.MockAzureAISearchTool",
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
            "top": 5,
            "openai_client": MagicMock(),
            "embedding_model": "mock-embedding-model",
        },
    )


@pytest.fixture
def vector_config() -> ComponentModel:
    """Create a test configuration for vector search."""
    return ComponentModel(
        provider="autogen_ext.tools.azure.test_ai_search_tool.MockAzureAISearchTool",
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
            "top": 5,
            "openai_client": MagicMock(),
            "embedding_model": "mock-embedding-model",
        },
    )


@pytest.fixture
def mock_search_response() -> List[Dict[str, Any]]:
    """Create a mock search response."""
    return [
        {
            "@search.score": 0.95,
            "id": "doc1",
            "content": "This is the first document content",
            "title": "Document 1",
            "source": "test-source-1",
        },
        {
            "@search.score": 0.85,
            "id": "doc2",
            "content": "This is the second document content",
            "title": "Document 2",
            "source": "test-source-2",
        },
    ]


class AsyncIterator:
    """Async iterator for testing."""

    def __init__(self, items: List[Dict[str, Any]]) -> None:
        self.items = items.copy()

    def __aiter__(self) -> "AsyncIterator":
        return self

    async def __anext__(self) -> Dict[str, Any]:
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)

    async def get_count(self) -> int:
        """Return count of items."""
        return len(self.items)


@pytest.fixture
def mock_search_client(mock_search_response: List[Dict[str, Any]]) -> tuple[MagicMock, Any]:
    """Create a mock search client for testing."""
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    search_results = AsyncIterator(mock_search_response)
    mock_client.search = MagicMock(return_value=search_results)

    patcher = patch("azure.search.documents.aio.SearchClient", return_value=mock_client)

    return mock_client, patcher
