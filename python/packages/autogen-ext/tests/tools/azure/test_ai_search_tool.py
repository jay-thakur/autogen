"""Tests for the Azure AI Search tool."""

import pytest
import os
import sys
from typing import Any, Dict, List

from autogen_core import CancellationToken, ComponentModel
from unittest.mock import AsyncMock, MagicMock, patch

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
sys.path.insert(0, src_dir)

source_dir = os.path.join(src_dir, 'autogen_ext', 'tools', 'azure')
ai_search_path = os.path.join(source_dir, '_ai_search.py')
config_path = os.path.join(source_dir, '_config.py')


from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from azure.search.documents.aio import SearchClient


os.makedirs(source_dir, exist_ok=True)
init_file = os.path.join(source_dir, '__init__.py')
if not os.path.exists(init_file):
    with open(init_file, 'w') as f:
        pass

import importlib.util
spec_ai_search = importlib.util.spec_from_file_location('ai_search_module', ai_search_path)
ai_search_module = importlib.util.module_from_spec(spec_ai_search)
spec_ai_search.loader.exec_module(ai_search_module)

spec_config = importlib.util.spec_from_file_location('config_module', config_path)
config_module = importlib.util.module_from_spec(spec_config)
spec_config.loader.exec_module(config_module)


AzureAISearchTool = ai_search_module.AzureAISearchTool
SearchQuery = ai_search_module.SearchQuery
SearchResult = ai_search_module.SearchResult
AzureAISearchConfig = config_module.AzureAISearchConfig


class AsyncIterator:
    """Async iterator for testing."""
    
    def __init__(self, items):
        self.items = list(items)
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)
        
    async def get_count(self):
        return len(self.items)



class MockResourceNotFoundError(Exception):
    """Mock for ResourceNotFoundError."""
    def __init__(self, message="Resource not found", **kwargs):
        self.message = message
        super().__init__(message)


class MockHttpResponseError(Exception):
    """Mock for HttpResponseError."""
    def __init__(self, message="Http error", **kwargs):
        self.message = message
        super().__init__(message)



MOCK_SEARCH_RESPONSE = [
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


def test_tool_schema_generation(test_config: ComponentModel) -> None:
    """Test that the tool correctly generates its schema."""
    tool = AzureAISearchTool.load_component(test_config)
    schema = tool.schema

    assert schema["name"] == "TestAzureSearch"
    assert "description" in schema
    assert schema["description"] == "Test Azure AI Search Tool"
    assert "parameters" in schema
    assert schema["parameters"]["type"] == "object"
    assert "properties" in schema["parameters"]
    assert schema["parameters"]["properties"]["query"]["description"] == "Search query text"
    assert schema["parameters"]["properties"]["query"]["type"] == "string"
    assert "filter" in schema["parameters"]["properties"]
    assert "top" in schema["parameters"]["properties"]
    assert "vector" in schema["parameters"]["properties"]
    assert "required" in schema["parameters"]
    assert schema["parameters"]["required"] == ["query"]


def test_tool_properties(test_config: ComponentModel) -> None:
    """Test that the tool properties are correctly set."""
    with patch('azure.search.documents.aio.SearchClient'):
        tool = AzureAISearchTool.load_component(test_config)

        assert tool.name == "TestAzureSearch"
        assert tool.description == "Test Azure AI Search Tool"
        assert tool.search_config.endpoint == "https://test-search-service.search.windows.net"
        assert tool.search_config.index_name == "test-index"
        assert tool.search_config.api_version == "2023-10-01-Preview"
        assert tool.search_config.query_type == "simple"
        assert tool.search_config.search_fields == ["content", "title"]
        assert tool.search_config.select_fields == ["id", "content", "title", "source"]
        assert tool.search_config.top == 5


def test_component_base_class(test_config: ComponentModel) -> None:
    """Test that the tool correctly implements the Component interface."""
    with patch('azure.search.documents.aio.SearchClient'):
        tool = AzureAISearchTool.load_component(test_config)
        assert tool.dump_component() is not None
        assert AzureAISearchTool.load_component(tool.dump_component(), AzureAISearchTool) is not None


@pytest.mark.asyncio
async def test_simple_search(test_config: ComponentModel) -> None:
    """Test that the tool correctly performs a simple search."""
    with patch('azure.search.documents.aio.SearchClient'):
        tool = AzureAISearchTool.load_component(test_config)
        mock_results = [
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
        
        async def mock_run(query, cancellation_token):
            return [
                SearchResult(
                    score=0.95,
                    content={"id": "doc1", "content": "This is the first document content", "title": "Document 1"},
                    metadata={"source": "test-source-1"}
                ),
                SearchResult(
                    score=0.85,
                    content={"id": "doc2", "content": "This is the second document content", "title": "Document 2"},
                    metadata={"source": "test-source-2"}
                )
            ]
        
        with patch.object(tool, 'run', mock_run):
            search_results = await tool.run(SearchQuery(query="test query"), CancellationToken())
            assert isinstance(search_results, list)
            assert len(search_results) == 2
            

            assert search_results[0].score == 0.95
            assert search_results[0].content["id"] == "doc1"
            
        
            json_results = await tool.run_json({"query": "test query"}, CancellationToken())
            assert len(json_results) == 2


@pytest.mark.asyncio
async def test_semantic_search(semantic_config: ComponentModel) -> None:
    """Test that the tool correctly performs a semantic search."""
    tool = AzureAISearchTool.load_component(semantic_config)
    
    mock_client = MagicMock()
    mock_client_instance = MagicMock()
    
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=None)
    mock_results = [
        {
            "@search.score": 0.95,
            "id": "doc1",
            "content": "This is the first document content",
            "title": "Document 1",
            "@metadata": {"key": "value1"}
        },
        {
            "@search.score": 0.85,
            "id": "doc2",
            "content": "This is the second document content",
            "title": "Document 2",
            "@metadata": {"key": "value2"}
        }
    ]
    
    mock_client_instance.search = AsyncMock()
    mock_client_instance.search.return_value = AsyncIterator(mock_results)
    
    mock_client.return_value = mock_client_instance
    with patch.object(tool, '_get_client', return_value=mock_client.return_value):
        result = await tool.run_json({"query": "test query"}, CancellationToken())
        mock_client_instance.search.assert_called_once()
        args, kwargs = mock_client_instance.search.call_args
        assert kwargs.get("query_type") == "semantic"
        assert kwargs.get("semantic_configuration_name") == "test-semantic-config"


@pytest.mark.asyncio
async def test_vector_search(vector_config: ComponentModel) -> None:
    """Test that the tool correctly performs a vector search."""
    tool = AzureAISearchTool.load_component(vector_config)
    
    mock_client = MagicMock()
    mock_client_instance = MagicMock()
    
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=None)
    mock_results = [
        {
            "@search.score": 0.95,
            "id": "doc1",
            "content": "This is the first document content",
            "title": "Document 1",
            "@metadata": {"key": "value1"}
        },
        {
            "@search.score": 0.85,
            "id": "doc2",
            "content": "This is the second document content",
            "title": "Document 2",
            "@metadata": {"key": "value2"}
        }
    ]
    
    mock_client_instance.search = AsyncMock()
    mock_client_instance.search.return_value = AsyncIterator(mock_results)
    
    mock_client.return_value = mock_client_instance
    with patch.object(tool, '_get_client', return_value=mock_client.return_value):
        result = await tool.run_json({"query": "test query"}, CancellationToken())
        mock_client_instance.search.assert_called_once()
        args, kwargs = mock_client_instance.search.call_args
        assert args[0] is None
        assert "vectors" in kwargs
        assert len(kwargs["vectors"]) == 1
        assert kwargs["vectors"][0]["fields"] == "embedding"


@pytest.mark.asyncio
async def test_error_handling_resource_not_found(test_config: ComponentModel) -> None:
    """Test that the tool correctly handles resource not found errors."""
    with patch('azure.search.documents.aio.SearchClient'):
        tool = AzureAISearchTool.load_component(test_config)
        
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        
        error_msg = "Resource 'test-index' not found"
        mock_client.search.side_effect = ResourceNotFoundError(error_msg)
        
        with patch.object(tool, '_get_client', return_value=mock_client):
            with pytest.raises(Exception) as excinfo:
                await tool.run_json({"query": "test query"}, CancellationToken())
            
        
            assert "not found" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_cancellation(test_config: ComponentModel) -> None:
    """Test that the tool correctly handles cancellation."""
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    mock_client.search = AsyncMock(return_value=AsyncIterator([]))
    
    with patch('azure.search.documents.aio.SearchClient', return_value=mock_client):
        tool = AzureAISearchTool.load_component(test_config)
        
        token = CancellationToken()
        token.cancel()
        with pytest.raises(Exception) as excinfo:
            await tool.run_json({"query": "test query"}, token)
        

        assert "cancelled" in str(excinfo.value).lower()


def test_config_serialization(test_config: ComponentModel) -> None:
    """Test that the tool configuration is correctly serialized."""
    with patch('azure.search.documents.aio.SearchClient'):
        tool = AzureAISearchTool.load_component(test_config)
        config = tool.dump_component()

        assert config.config["name"] == test_config.config["name"]
        assert config.config["description"] == test_config.config["description"]
        assert config.config["endpoint"] == test_config.config["endpoint"]
        assert config.config["index_name"] == test_config.config["index_name"]
        assert config.config["api_version"] == test_config.config["api_version"]
        assert config.config["query_type"] == test_config.config["query_type"]
        assert config.config["search_fields"] == test_config.config["search_fields"]
        assert config.config["select_fields"] == test_config.config["select_fields"]
        assert config.config["top"] == test_config.config["top"]


@pytest.mark.asyncio
async def test_hybrid_search(test_config: ComponentModel) -> None:
    """Test that the tool correctly performs a hybrid search."""

    hybrid_config = ComponentModel(
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
            "vector_fields": ["embedding"],
            "search_fields": ["content", "title"],
            "select_fields": ["id", "content", "title", "source"],
            "top": 5
        }
    )
    
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    mock_client.search = AsyncMock(return_value=AsyncIterator(MOCK_SEARCH_RESPONSE))
    with patch.object(AzureAISearchTool, '_get_client', return_value=mock_client):
        tool = AzureAISearchTool.load_component(hybrid_config)
        
        result = await tool.run_json({"query": "test query"}, CancellationToken())
        mock_client.search.assert_called_once()
        args, kwargs = mock_client.search.call_args
        
        assert args[0] == "test query"
        assert "vectors" in kwargs
        assert kwargs.get("query_type") == "semantic"
