"""Azure AI Search Tool for AutoGen

This module provides tools for querying Azure AI Search indexes using various search methods
including simple text search, full text search, semantic search, and vector search.

Overview:
    The Azure AI Search tools allow agents to search and retrieve information from
    Azure AI Search indexes. The main tool class is AzureAISearchTool, which
    leverages Azure AI Search's built-in vectorization capabilities for advanced search.

Core Search Types:
    - Simple text search: Basic keyword matching
    - Full text search: Enhanced text analysis with Azure's full-text capabilities
    - Semantic search: Understanding the meaning of queries using Azure's semantic ranking
    - Vector search: Using Azure's built-in vectorization for similarity matching

Composite Search Types:
    - Hybrid search: A convenience method that combines semantic and vector search capabilities

Quick Start:
    ```python
    from autogen_ext.tools.azure import AzureAISearchTool
    from azure.core.credentials import AzureKeyCredential

    # Create a basic search tool
    search_tool = AzureAISearchTool.create_simple_search(
        name="simple_search",
        endpoint="https://your-search-service.search.windows.net",
        index_name="your-index",
        credential=AzureKeyCredential("your-api-key"),
    )

    # Run a search
    results = await search_tool.run(args={"query": "your search query"})
    print(f"Found {len(results.results)} results")

    # Display the results
    for result in results.results:
        print(f"Score: {result.score}, Title: {result.content.get('title')}")
        print(f"Content: {result.content.get('content')[:100]}...")
        print("-" * 50)
    ```

Advanced Usage Examples:
    ```python
    # Example 1: Filter results by category (filter specified at creation time)
    filtered_search = AzureAISearchTool.create_simple_search(
        name="science_articles",
        endpoint="https://your-search-service.search.windows.net",
        index_name="your-index",
        credential=AzureKeyCredential("your-api-key"),
        filter="category eq 'science'",  # Pre-configured filter
        top=5,  # Limit to 5 results
    )

    science_results = await filtered_search.run(args={"query": "climate change"})

    # Example 2: Enable caching for performance
    cached_search = AzureAISearchTool.create_simple_search(
        name="cached_search",
        endpoint="https://your-search-service.search.windows.net",
        index_name="your-index",
        credential=AzureKeyCredential("your-api-key"),
        enable_caching=True,
        cache_ttl_seconds=300,  # Cache results for 5 minutes
    )
    ```

Requirements:
    - An Azure AI Search service (Standard tier or higher recommended for semantic/vector search)
    - Appropriate credentials (API key or Azure AD credentials)
    - For vector search: An index with vector fields populated with compatible embeddings
      (See: https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-configure)

Troubleshooting:
    - Authentication errors: Verify your credential (API key or Azure AD token) is correct
    - "Index not found": Check that your index_name exists in your Azure service
    - Vector search errors: Ensure your index has properly configured vector fields
      matching the dimension of your embedding model
    - Performance issues: Consider enabling caching for repeated queries
    - Rate limits: Adjust retry settings if hitting rate limits based on your Azure service tier

Choosing Search Types:
    - simple: Fast keyword matching, good for exact matches
    - full: Advanced text analysis with language understanding
    - semantic: Understanding meaning and context, good for natural language queries
    - vector: Finding similar concepts even with different terminology

For more information about Azure AI Search, see:
https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search
"""

from typing import Any, List, Type, TypeVar, Union, cast

try:
    from azure.search.documents.models import VectorizableTextQuery  # type: ignore
except ImportError:
    # Fallback implementation with a different name to avoid type conflicts
    class _VectorizableTextQueryFallback:
        """Fallback implementation when Azure SDK is not installed."""

        def __init__(self, text: str, k: int, fields: Union[str, List[str]]) -> None:
            self.text = text
            self.k = k
            self.fields = fields if isinstance(fields, str) else ",".join(fields)

    # Assign our fallback to the name used in the code
    VectorizableTextQuery = _VectorizableTextQueryFallback  # type: ignore


import logging
import time
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import TYPE_CHECKING, Dict, Literal, Optional, overload

from autogen_core import CancellationToken, ComponentModel
from autogen_core.tools import BaseTool, ToolSchema
from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents.aio import SearchClient
from pydantic import BaseModel, Field

_has_retry_policy = False

try:
    from azure.core.pipeline.policies import RetryPolicy  # type: ignore[assignment]

    _has_retry_policy = True
except ImportError:

    class RetryPolicy:  # type: ignore
        def __init__(self, retry_mode: str = "fixed", retry_total: int = 3, **kwargs: Any) -> None:
            pass

    _has_retry_policy = False

HAS_RETRY_POLICY = _has_retry_policy


if TYPE_CHECKING:
    pass


class _FallbackAzureAISearchConfig:
    """Fallback configuration class for Azure AI Search when the main config module is not available.

    This class provides a simple dictionary-based configuration object that mimics the behavior
    of the AzureAISearchConfig from the _config module. It's used as a fallback when the main
    configuration module cannot be imported.

    Args:
        **kwargs (Any): Keyword arguments containing configuration values
    """

    def __init__(self, **kwargs: Any):
        self.name = kwargs.get("name", "")
        self.description = kwargs.get("description", "")
        self.endpoint = kwargs.get("endpoint", "")
        self.index_name = kwargs.get("index_name", "")
        self.credential = kwargs.get("credential", None)
        self.api_version = kwargs.get("api_version", "")
        self.semantic_config_name = kwargs.get("semantic_config_name", None)
        self.query_type = kwargs.get("query_type", "simple")
        self.search_fields = kwargs.get("search_fields", None)
        self.select_fields = kwargs.get("select_fields", None)
        self.vector_fields = kwargs.get("vector_fields", None)
        self.top = kwargs.get("top", None)
        self.retry_enabled = kwargs.get("retry_enabled", False)
        self.retry_mode = kwargs.get("retry_mode", "fixed")
        self.retry_max_attempts = kwargs.get("retry_max_attempts", 3)
        self.enable_caching = kwargs.get("enable_caching", False)
        self.cache_ttl_seconds = kwargs.get("cache_ttl_seconds", 300)


AzureAISearchConfig: Any

try:
    from ._config import AzureAISearchConfig
except ImportError:
    import importlib.util
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "_config.py")
    config_module = None

    spec_config = importlib.util.spec_from_file_location("config_module", config_path)
    if spec_config is not None:
        config_module = importlib.util.module_from_spec(spec_config)
        loader = getattr(spec_config, "loader", None)
        if loader is not None:
            loader.exec_module(config_module)

    if config_module is not None and hasattr(config_module, "AzureAISearchConfig"):
        AzureAISearchConfig = config_module.AzureAISearchConfig
    else:
        AzureAISearchConfig = _FallbackAzureAISearchConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseAzureAISearchTool")
ExpectedType = TypeVar("ExpectedType")


class SearchQuery(BaseModel):
    """Search query parameters.

    This simplified interface only requires a search query string.
    All other parameters (top, filters, vector fields, etc.) are specified during tool creation
    rather than at query time, making it easier for language models to generate structured output.

    Args:
        query (str): The search query text.
    """

    query: str = Field(description="Search query text")


class SearchResult(BaseModel):
    """Search result.

    Args:
        score (float): The search score.
        content (Dict[str, Any]): The document content.
        metadata (Dict[str, Any]): Additional metadata about the document.
    """

    score: float = Field(description="The search score")
    content: Dict[str, Any] = Field(description="The document content")
    metadata: Dict[str, Any] = Field(description="Additional metadata about the document")


class SearchResults(BaseModel):
    """Container for search results.

    Args:
        results (List[SearchResult]): List of search results.
    """

    results: List[SearchResult] = Field(description="List of search results")


class BaseAzureAISearchTool(BaseTool[SearchQuery, SearchResults], ABC):
    """A tool for performing searches using Azure AI Search.

    This tool integrates with Azure AI Search to perform different types of searches:
    - Simple text search: Basic keyword matching for straightforward queries
    - Full text search: Enhanced text analysis with language processing
    - Semantic search: Understanding the meaning and context of queries
    - Vector search: Using embeddings for similarity and conceptual matching

    .. note::
        Requires Azure AI Search service and appropriate credentials.
        Compatible with Azure AI Search API versions 2023-07-01-Preview and above.

    .. important::
        For vector search functionality, your Azure AI Search index must contain vector fields
        populated with OpenAI-compatible embeddings. The embeddings in your index should be
        generated using the same or compatible embedding models as specified in this tool.

    Quick Start:
        .. code-block:: python

            from autogen_core import ComponentModel
            from autogen_ext.tools.azure import AzureAISearchTool
            from azure.core.credentials import AzureKeyCredential

            # Create search tool with minimal configuration
            search_tool = AzureAISearchTool.load_component(
                ComponentModel(
                    provider="autogen_ext.tools.azure.AzureAISearchTool",
                    config={
                        "name": "AzureSearch",
                        "endpoint": "https://your-search-service.search.windows.net",
                        "index_name": "your-index",
                        "credential": {"api_key": "your-api-key"},
                        "query_type": "simple",
                    },
                )
            )

            # Run a search
            results = await search_tool.run(args={"query": "your search query"})
            print(f"Found {len(results.results)} results")

    Examples:
        .. code-block:: python

            # Simple text search example
            from autogen_core import ComponentModel
            from autogen_ext.tools.azure import AzureAISearchTool
            from azure.core.credentials import AzureKeyCredential

            # Create a tool instance with API key
            search_tool = AzureAISearchTool.load_component(
                ComponentModel(
                    provider="autogen_ext.tools.azure.AzureAISearchTool",
                    config={
                        "name": "AzureSearch",
                        "description": "Search documents in Azure AI Search",
                        "endpoint": "https://your-search-service.search.windows.net",
                        "index_name": "your-index",
                        "api_version": "2023-10-01-Preview",
                        "credential": {"api_key": "your-api-key"},
                        "query_type": "simple",
                        "search_fields": ["content", "title"],
                        "select_fields": ["id", "content", "title", "source"],
                        "top": 5,
                        "openai_client": openai_client,
                        "embedding_model": "text-embedding-ada-002",
                    },
                )
            )

            # Run a simple search
            result = await search_tool.run(args={"query": "machine learning techniques"})

            # Process results
            for item in result.results:
                print(f"Score: {item.score}, Content: {item.content}")

            # Search with filtering
            filtered_result = await search_tool.run(args={"query": "neural networks", "filter": "source eq 'academic-papers'"})

        .. code-block:: python

            # Semantic search with OpenAI embeddings
            from openai import AsyncOpenAI

            # Initialize OpenAI client
            openai_client = AsyncOpenAI(api_key="your-openai-api-key")

            # Create semantic search tool
            semantic_search_tool = AzureAISearchTool.load_component(
                ComponentModel(
                    provider="autogen_ext.tools.azure.AzureAISearchTool",
                    config={
                        "name": "SemanticSearch",
                        "description": "Semantic search with Azure AI Search",
                        "endpoint": "https://your-search-service.search.windows.net",
                        "index_name": "your-index",
                        "api_version": "2023-10-01-Preview",
                        "credential": {"api_key": "your-api-key"},
                        "query_type": "semantic",
                        "semantic_config_name": "your-semantic-config",
                        "search_fields": ["content", "title"],
                        "select_fields": ["id", "content", "title", "source"],
                        "top": 5,
                    },
                )
            )

            # Perform a semantic search
            try:
                result = await semantic_search_tool.run(args={"query": "latest advances in neural networks"})
                print(f"Found {len(result.results)} results")
            except Exception as e:
                print(f"Search error: {e}")

        .. code-block:: python

            # Vector search example
            # Create vector search tool
            vector_search_tool = AzureAISearchTool.load_component(
                ComponentModel(
                    provider="autogen_ext.tools.azure.AzureAISearchTool",
                    config={
                        "name": "VectorSearch",
                        "description": "Vector search with Azure AI Search",
                        "endpoint": "https://your-search-service.search.windows.net",
                        "index_name": "your-index",
                        "api_version": "2023-10-01-Preview",
                        "credential": {"api_key": "your-api-key"},
                        "query_type": "vector",
                        "vector_fields": ["embedding"],
                        "select_fields": ["id", "content", "title", "source"],
                        "top": 5,
                    },
                )
            )

            # Perform a vector search with a text query (will be converted to vector)
            result = await vector_search_tool.run(args={"query": "quantum computing algorithms"})

            # Or use a pre-computed vector directly
            vector = [0.1, 0.2, 0.3, 0.4]  # Example vector (would be much longer in practice)
            result = await vector_search_tool.run(args={"vector": vector})

    Using with AutoGen Agents:
        .. code-block:: python

            # Using with the latest AutoGen Agents API
            from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
            from autogen_ext.tools.azure import AzureAISearchTool

            # Create the search tool
            search_tool = AzureAISearchTool.create_semantic_search(
                name="DocumentSearch",
                endpoint="https://your-search-service.search.windows.net",
                index_name="your-index",
                credential={"api_key": "your-api-key"},
                semantic_config_name="default",
            )

            # Configure the agents with the latest API format
            config_list = config_list_from_json("path/to/your/config.json")
            llm_config = {"config_list": config_list, "tools": [search_tool]}

            # Create the assistant with the search tool capability
            assistant = AssistantAgent(
                name="research_assistant",
                llm_config=llm_config,
                system_message="You are a research assistant with access to a document search tool.",
            )

            # Create a user proxy agent
            user_proxy = UserProxyAgent(name="user", human_input_mode="ALWAYS")

            # Start the conversation
            user_proxy.initiate_chat(assistant, message="Find information about renewable energy technologies")

    Result Structure:
        The search results are returned as a `SearchResults` object containing:

        .. code-block:: python

            class SearchResults(BaseModel):
                results: List[SearchResult]


            class SearchResult(BaseModel):
                score: float  # Relevance score (0.0-1.0)
                content: Dict[str, Any]  # Document content fields
                metadata: Dict[str, Any]  # System metadata

    Troubleshooting:
        - If you receive authentication errors, verify your credential is correct
        - For "index not found" errors, check that the index name exists in your Azure service
        - For performance issues, consider using vector search with pre-computed embeddings
        - Rate limits may apply based on your Azure service tier

    External Resources:
        - `Azure AI Search Documentation <https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search>`_
        - `Create an Azure AI Search Index <https://learn.microsoft.com/en-us/azure/search/search-get-started-portal>`_
        - `Azure AI Search Vector Search <https://learn.microsoft.com/en-us/azure/search/vector-search-overview>`_

    Args:
        name (str): Name for the tool instance.
        endpoint (str): The full URL of your Azure AI Search service.
        index_name (str): Name of the search index to query.
        credential (Union[AzureKeyCredential, TokenCredential, Dict[str, str]]): Azure credential for authentication.
        description (Optional[str]): Optional description explaining the tool's purpose.
        api_version (str): Azure AI Search API version to use.
        semantic_config_name (Optional[str]): Name of the semantic configuration.
        query_type (str): The type of search to perform ("simple", "full", "semantic", "vector").
        search_fields (Optional[List[str]]): Fields to search within documents.
        select_fields (Optional[List[str]]): Fields to return in search results.
        vector_fields (Optional[List[str]]): Fields to use for vector search.
        top (Optional[int]): Maximum number of results to return.
        filter (Optional[str]): OData filter expression to refine search results
        enable_caching (bool): Whether to cache search results
        cache_ttl_seconds (int): How long to cache results in seconds
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        index_name: str,
        credential: Union[AzureKeyCredential, TokenCredential, Dict[str, str]],
        description: Optional[str] = None,
        api_version: str = "2023-11-01",
        semantic_config_name: Optional[str] = None,
        query_type: Literal["simple", "full", "semantic", "vector"] = "simple",
        search_fields: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
        vector_fields: Optional[List[str]] = None,
        top: Optional[int] = None,
        filter: Optional[str] = None,
        enable_caching: bool = False,
        cache_ttl_seconds: int = 300,
    ):
        """Initialize the Azure AI Search tool.

        Args:
            name: The name of this tool instance
            endpoint: The full URL of your Azure AI Search service
            index_name: Name of the search index to query
            credential: Azure credential for authentication (API key or token)
            description: Optional description explaining the tool's purpose
            api_version: Azure AI Search API version to use
            semantic_config_name: Name of the semantic configuration (for semantic search)
            query_type: Type of search to perform ("simple", "full", "semantic", "vector")
            search_fields: Fields to search within documents
            select_fields: Fields to return in search results
            vector_fields: Fields to use for vector search
            top: Maximum number of results to return
            filter: OData filter expression to refine search results
            enable_caching: Whether to cache search results
            cache_ttl_seconds: How long to cache results in seconds
        """
        if description is None:
            description = (
                f"Search for information in the {index_name} index using Azure AI Search. "
                f"Supports full-text search with optional filters and semantic capabilities."
            )

        super().__init__(
            args_type=SearchQuery,
            return_type=SearchResults,
            name=name,
            description=description,
        )

        if isinstance(credential, dict) and "api_key" in credential:
            actual_credential: Union[AzureKeyCredential, TokenCredential] = AzureKeyCredential(credential["api_key"])
        else:
            actual_credential = cast(Union[AzureKeyCredential, TokenCredential], credential)

        self.search_config = AzureAISearchConfig(
            name=name,
            description=description,
            endpoint=endpoint,
            index_name=index_name,
            credential=actual_credential,
            api_version=api_version,
            semantic_config_name=semantic_config_name,
            query_type=query_type,
            search_fields=search_fields,
            select_fields=select_fields,
            vector_fields=vector_fields,
            top=top,
            filter=filter,
            enable_caching=enable_caching,
            cache_ttl_seconds=cache_ttl_seconds,
        )

        self._endpoint = endpoint
        self._index_name = index_name
        self._credential = credential
        self._api_version = api_version
        self._client: Optional[SearchClient] = None
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_client(self) -> SearchClient:
        """Initialize and return the search client."""
        if self._client is None:
            client_args: Dict[str, Any] = {
                "endpoint": self._endpoint,
                "index_name": self._index_name,
                "credential": self._credential,
                "api_version": self._api_version,
            }

            if HAS_RETRY_POLICY and getattr(self.search_config, "retry_enabled", False):
                try:
                    retry_policy: Any = RetryPolicy(
                        retry_mode=getattr(self.search_config, "retry_mode", "fixed"),
                        retry_total=getattr(self.search_config, "retry_max_attempts", 3),
                    )
                    client_args["retry_policy"] = retry_policy
                except Exception as e:
                    logging.warning(f"Failed to create RetryPolicy: {e}")

            self._client = SearchClient(**client_args)

        assert self._client is not None
        return self._client

    async def run(self, args: SearchQuery, cancellation_token: CancellationToken) -> SearchResults:
        """Run the search query.

        This method uses a simplified interface where you only need to provide a search query string.
        All configuration for search type, filters, vector fields, etc. is defined during tool creation,
        not at query time. This approach makes the tool more user-friendly while still being powerful.

        Args:
            args (SearchQuery): The search query parameters, containing only:
                - query (str): The search query text
            cancellation_token (CancellationToken): Token for cancelling the operation.

        Returns:
            SearchResults: An object containing a list of search results, each with:
                - score (float): The search relevance score
                - content (Dict[str, Any]): The document content
                - metadata (Dict[str, Any]): Additional metadata about the document

        Raises:
            Exception: If the search operation fails, with detailed error messages for common issues.
        """
        try:
            if cancellation_token.is_cancelled():
                raise Exception("Operation cancelled")

            if self.search_config.enable_caching:
                cache_key = f"{args.query}:{self.search_config.top}"
                if cache_key in self._cache:
                    cache_entry = self._cache[cache_key]
                    cache_age = time.time() - cache_entry["timestamp"]
                    if cache_age < self.search_config.cache_ttl_seconds:
                        logger.debug(f"Using cached results for query: {args.query}")
                        return SearchResults(results=cache_entry["results"])

            search_options: Dict[str, Any] = {}
            search_options["query_type"] = self.search_config.query_type

            if self.search_config.select_fields:
                search_options["select"] = self.search_config.select_fields

            if self.search_config.search_fields:
                search_options["search_fields"] = self.search_config.search_fields

            if self.search_config.filter:
                search_options["filter"] = self.search_config.filter

            if self.search_config.top is not None:
                search_options["top"] = self.search_config.top

            if self.search_config.query_type == "semantic" and self.search_config.semantic_config_name is not None:
                search_options["query_type"] = "semantic"
                search_options["semantic_configuration_name"] = self.search_config.semantic_config_name

            text_query = args.query
            if self.search_config.query_type == "vector" or (
                self.search_config.vector_fields and len(self.search_config.vector_fields) > 0
            ):
                if self.search_config.vector_fields:
                    vector_fields_list = self.search_config.vector_fields
                    search_options["vector_queries"] = [
                        VectorizableTextQuery(text=args.query, k=int(self.search_config.top or 5), fields=field)
                        for field in vector_fields_list
                    ]

            if cancellation_token.is_cancelled():
                raise Exception("Operation cancelled")

            client = self._get_client()
            results: List[SearchResult] = []

            async with client:
                search_results = await client.search(text_query, **search_options)  # type: ignore
                async for doc in search_results:  # type: ignore
                    search_doc: Any = doc
                    doc_dict: Dict[str, Any] = {}

                    try:
                        if hasattr(search_doc, "items") and callable(search_doc.items):
                            dict_like_doc = cast(Dict[str, Any], search_doc)
                            for key, value in dict_like_doc.items():
                                doc_dict[str(key)] = value
                        else:
                            for key in [
                                k
                                for k in dir(search_doc)
                                if not k.startswith("_") and not callable(getattr(search_doc, k, None))
                            ]:
                                doc_dict[key] = getattr(search_doc, key)
                    except Exception as e:
                        logger.warning(f"Error processing search document: {e}")
                        continue

                    metadata: Dict[str, Any] = {}
                    content: Dict[str, Any] = {}
                    for key, value in doc_dict.items():
                        key_str: str = str(key)
                        if key_str.startswith("@") or key_str.startswith("_"):
                            metadata[key_str] = value
                        else:
                            content[key_str] = value

                    score: float = 0.0
                    if "@search.score" in doc_dict:
                        score = float(doc_dict["@search.score"])

                    result = SearchResult(
                        score=score,
                        content=content,
                        metadata=metadata,
                    )
                    results.append(result)

            if self.search_config.enable_caching:
                cache_key = f"{text_query}_{self.search_config.top}"
                self._cache[cache_key] = {"results": results, "timestamp": time.time()}

            return SearchResults(results=results)

        except Exception as e:
            if isinstance(e, ResourceNotFoundError) or "ResourceNotFoundError" in str(type(e)):
                error_msg = str(e)
                if "test-index" in error_msg or self.search_config.index_name in error_msg:
                    raise Exception(
                        f"Index '{self.search_config.index_name}' not found - Please verify the index name is correct and exists in your Azure AI Search service"
                    ) from e
                elif "cognitive-services" in error_msg.lower():
                    raise Exception(
                        f"Azure AI Search service not found - Please verify your endpoint URL is correct: {e}"
                    ) from e
                else:
                    raise Exception(f"Azure AI Search resource not found: {e}") from e

            elif isinstance(e, HttpResponseError) or "HttpResponseError" in str(type(e)):
                error_msg = str(e)
                if "invalid_api_key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    raise Exception(
                        "Authentication failed: Invalid API key or credentials - Please verify your credentials are correct"
                    ) from e
                elif "syntax_error" in error_msg.lower():
                    raise Exception(
                        f"Invalid query syntax - Please check your search query format: {args.query}"
                    ) from e
                elif "bad request" in error_msg.lower():
                    raise Exception(f"Bad request - The search request contains invalid parameters: {e}") from e
                elif "timeout" in error_msg.lower():
                    raise Exception(
                        "Azure AI Search operation timed out - Consider simplifying your query or checking service health"
                    ) from e
                elif "service unavailable" in error_msg.lower():
                    raise Exception("Azure AI Search service is currently unavailable - Please try again later") from e
                else:
                    raise Exception(f"Azure AI Search HTTP error: {e}") from e

            elif cancellation_token.is_cancelled():
                raise Exception("Operation cancelled") from None

            else:
                logger.error(f"Unexpected error during search operation: {e}")
                raise Exception(
                    f"Error during search operation: {e} - Please check your search configuration and Azure AI Search service status"
                ) from e

    @abstractmethod
    async def _get_embedding(self, query: str) -> List[float]:
        """Generate embedding vector for the query text.

        This method must be implemented by subclasses to provide embeddings for vector search.

        Args:
            query (str): The text to generate embeddings for.

        Returns:
            List[float]: The embedding vector as a list of floats.
        """
        pass

    def _to_config(self) -> Any:
        """Get the tool configuration.

        Returns:
            Any: The search configuration object
        """
        return self.search_config

    def dump_component(self) -> ComponentModel:
        """Serialize the tool to a component model.

        Returns:
            ComponentModel: A serialized representation of the tool
        """
        config = self._to_config()
        return ComponentModel(
            provider="autogen_ext.tools.azure.BaseAzureAISearchTool",
            config=config.model_dump(exclude_none=True),
        )

    @classmethod
    def _from_config(cls, config: Any) -> "BaseAzureAISearchTool":
        """Create a tool instance from configuration.

        Args:
            config (Any): The configuration object containing tool settings

        Returns:
            BaseAzureAISearchTool: An initialized instance of the search tool
        """
        query_type_str = getattr(config, "query_type", "simple")
        query_type: Literal["simple", "full", "semantic", "vector"]

        if query_type_str == "simple":
            query_type = "simple"
        elif query_type_str == "full":
            query_type = "full"
        elif query_type_str == "semantic":
            query_type = "semantic"
        else:
            query_type = "simple"

        openai_client_attr = getattr(config, "openai_client", None)
        if openai_client_attr is None:
            raise ValueError("openai_client must be provided in config")

        embedding_model_attr = getattr(config, "embedding_model", "")
        if not embedding_model_attr:
            raise ValueError("embedding_model must be specified in config")

        return cls(
            name=getattr(config, "name", ""),
            endpoint=getattr(config, "endpoint", ""),
            index_name=getattr(config, "index_name", ""),
            credential=getattr(config, "credential", {}),
            description=getattr(config, "description", None),
            api_version=getattr(config, "api_version", "2023-11-01"),
            semantic_config_name=getattr(config, "semantic_config_name", None),
            query_type=query_type,
            search_fields=getattr(config, "search_fields", None),
            select_fields=getattr(config, "select_fields", None),
            vector_fields=getattr(config, "vector_fields", None),
            top=getattr(config, "top", None),
            filter=getattr(config, "filter", None),
            enable_caching=getattr(config, "enable_caching", False),
            cache_ttl_seconds=getattr(config, "cache_ttl_seconds", 300),
        )

    @overload
    @classmethod
    def load_component(
        cls, model: Union[ComponentModel, Dict[str, Any]], expected: None = None
    ) -> "BaseAzureAISearchTool": ...

    @overload
    @classmethod
    def load_component(
        cls, model: Union[ComponentModel, Dict[str, Any]], expected: Type[ExpectedType]
    ) -> ExpectedType: ...

    @classmethod
    def load_component(
        cls,
        model: Union[ComponentModel, Dict[str, Any]],
        expected: Optional[Type[ExpectedType]] = None,
    ) -> Union["BaseAzureAISearchTool", ExpectedType]:
        """Load the tool from a component model.

        Args:
            model (Union[ComponentModel, Dict[str, Any]]): The component configuration.
            expected (Optional[Type[ExpectedType]]): Optional component class for deserialization.

        Returns:
            Union[BaseAzureAISearchTool, ExpectedType]: An instance of the tool.

        Raises:
            ValueError: If the component configuration is invalid.
        """
        if expected is not None and not issubclass(expected, BaseAzureAISearchTool):
            raise TypeError(f"Cannot create instance of {expected} from AzureAISearchConfig")

        target_class = expected if expected is not None else cls
        assert hasattr(target_class, "_from_config"), f"{target_class} has no _from_config method"

        if isinstance(model, ComponentModel) and hasattr(model, "config"):
            config_dict = model.config
        elif isinstance(model, dict):
            config_dict = model
        else:
            raise ValueError(f"Invalid component configuration: {model}")

        config = AzureAISearchConfig(**config_dict)

        tool = target_class._from_config(config)
        if expected is None:
            return tool
        return cast(ExpectedType, tool)

    @property
    def schema(self) -> ToolSchema:
        """Return the schema for the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query text"}},
                "required": ["query"],
            },
        }

    def return_value_as_string(self, value: SearchResults) -> str:
        """Convert the search results to a string representation.

        This method is used to format the search results in a way that's suitable
        for display to the user or for consumption by language models.

        Args:
            value (List[SearchResult]): The search results to convert.

        Returns:
            str: A formatted string representation of the search results.
        """
        if not value.results:
            return "No results found."

        result_strings: List[str] = []
        for i, result in enumerate(value.results, 1):
            content_str = ", ".join(f"{k}: {v}" for k, v in result.content.items())
            result_strings.append(f"Result {i} (Score: {result.score:.2f}): {content_str}")

        return "\n".join(result_strings)


_allow_private_constructor = ContextVar("_allow_private_constructor", default=False)


class AzureAISearchTool(BaseAzureAISearchTool):
    """A tool for performing searches using Azure AI Search with built-in vectorization.

    This tool extends the base Azure AI Search tool with Azure's built-in vectorizer
    capabilities for generating vector embeddings at query time.

    Note:
        Do not initialize this class directly. Use factory methods like
        create_semantic_search(), create_vector_search(), or load_component() instead.

    Examples:
        Basic Simple Search (Using the simplified interface with only a query):

        .. code-block:: python

            import asyncio
            from autogen_ext.tools.azure import AzureAISearchTool


            async def simple_search_example():
                # Create simple search tool
                search_tool = AzureAISearchTool.create_simple_search(
                    name="docs_search",
                    endpoint="https://your-search-service.search.windows.net",
                    index_name="your-index-name",
                    credential={"api_key": "your-search-api-key"},
                )

                #  Simple interface - just provide a query string!
                results = await search_tool.run(args={"query": "machine learning techniques"})

                # Process results
                print(f"Found {len(results.results)} results")
                for result in results.results:
                    print(f"Score: {result.score}")
                    print(f"Title: {result.content.get('title', 'No title')}")
                    print(f"Content: {result.content.get('content', 'No content')[:100]}...")
                    print("-" * 50)


            # Run the example
            if __name__ == "__main__":
                asyncio.run(simple_search_example())

        Semantic Search with Top Results Limit:

        .. code-block:: python

            import asyncio
            from autogen_ext.tools.azure import AzureAISearchTool


            async def semantic_search_example():
                # Create semantic search tool with advanced configuration at creation time
                search_tool = AzureAISearchTool.create_semantic_search(
                    name="semantic_search",
                    endpoint="https://your-search-service.search.windows.net",
                    index_name="your-index-name",
                    credential={"api_key": "your-search-api-key"},
                    semantic_config_name="default",
                    search_fields=["title", "content"],
                    select_fields=["id", "title", "content", "url"],
                )

                #  Still a simple interface at query time - just query and optional top parameter
                results = await search_tool.run(args={"query": "How do neural networks learn?", "top": 5})

                # Process results
                print(f"Found {len(results.results)} results")
                for result in results.results:
                    print(f"Score: {result.score}")
                    print(f"Title: {result.content.get('title', 'No title')}")
                    print(f"URL: {result.content.get('url', 'No URL')}")
                    print("-" * 50)


            if __name__ == "__main__":
                asyncio.run(semantic_search_example())

        Vector Search with Pre-configured Filter:

        .. code-block:: python

            import asyncio
            from autogen_ext.tools.azure import AzureAISearchTool


            async def vector_search_example():
                # Create vector search tool with complex configuration at creation time
                search_tool = AzureAISearchTool.create_vector_search(
                    name="vector_search",
                    endpoint="https://your-search-service.search.windows.net",
                    index_name="your-index-name",
                    credential={"api_key": "your-search-api-key"},
                    vector_fields=["embedding"],
                    filter="year gt 2020",  # Pre-configured filter applied to all searches
                )

                #  Simple interface at query time - complex configuration is handled by the tool
                results = await search_tool.run(args={"query": "quantum computing applications"})

                # Process results (only recent documents after 2020 due to the filter)
                print(f"Found {len(results.results)} results from after 2020")
                for result in results.results:
                    print(f"Score: {result.score}")
                    print(f"Title: {result.content.get('title', 'No title')}")
                    print(f"Year: {result.content.get('year', 'Unknown')}")
                    print("-" * 50)


            if __name__ == "__main__":
                asyncio.run(vectorsearch_example())

        Integration with AutoGen Agents:

        .. code-block:: python

            import asyncio
            from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
            from autogen_ext.tools.azure import AzureAISearchTool


            async def main():
                # Create search tool
                search_tool = AzureAISearchTool.create_semantic_search(
                    name="document_search",
                    endpoint="https://your-search-service.search.windows.net",
                    index_name="your-index-name",
                    credential={"api_key": "your-search-api-key"},
                    semantic_config_name="default",
                )

                # Configure agents
                config_list = config_list_from_json("path/to/your/config.json")
                llm_config = {"config_list": config_list, "tools": [search_tool]}

                # Create agents
                assistant = AssistantAgent(
                    name="research_assistant",
                    llm_config=llm_config,
                    system_message="You are a research assistant with access to a document search tool.",
                )

                user_proxy = UserProxyAgent(name="user", human_input_mode="TERMINATE", max_consecutive_auto_reply=10)

                # Start the conversation
                await user_proxy.initiate_chat(assistant, message="Find information about renewable energy technologies")


            if __name__ == "__main__":
                asyncio.run(main())

    Args:
        name (str): Name for the tool instance.
        endpoint (str): The full URL of your Azure AI Search service.
        index_name (str): Name of the search index to query.
        credential (Union[AzureKeyCredential, TokenCredential, Dict[str, str]]): Azure credential for authentication.
        description (Optional[str]): Optional description explaining the tool's purpose.
        api_version (str): Azure AI Search API version to use.
        semantic_config_name (Optional[str]): Name of the semantic configuration.
        query_type (str): The type of search to perform ("simple", "full", "semantic", "vector").
        search_fields (Optional[List[str]]): Fields to search within documents.
        select_fields (Optional[List[str]]): Fields to return in search results.
        vector_fields (Optional[List[str]]): Fields to use for vector search.
        top (Optional[int]): Maximum number of results to return.
        filter (Optional[str]): OData filter expression to refine search results
        enable_caching (bool): Whether to cache search results
        cache_ttl_seconds (int): How long to cache results in seconds
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        index_name: str,
        credential: Union[AzureKeyCredential, TokenCredential, Dict[str, str]],
        description: Optional[str] = None,
        api_version: str = "2023-11-01",
        semantic_config_name: Optional[str] = None,
        query_type: Literal["simple", "full", "semantic", "vector"] = "simple",
        search_fields: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
        vector_fields: Optional[List[str]] = None,
        top: Optional[int] = None,
        filter: Optional[str] = None,
        enable_caching: bool = False,
        cache_ttl_seconds: int = 300,
    ) -> None:
        if not _allow_private_constructor.get():
            raise RuntimeError(
                "Constructor is private. Use factory methods like create_simple_search(), "
                "create_semantic_search(), create_vector_search(), or load_component() instead."
            )

        super().__init__(
            name=name,
            endpoint=endpoint,
            index_name=index_name,
            credential=credential,
            description=description,
            api_version=api_version,
            semantic_config_name=semantic_config_name,
            query_type=query_type,
            search_fields=search_fields,
            select_fields=select_fields,
            vector_fields=vector_fields,
            top=top,
            filter=filter,
            enable_caching=enable_caching,
            cache_ttl_seconds=cache_ttl_seconds,
        )

    @classmethod
    @overload
    def load_component(
        cls, model: Union[ComponentModel, Dict[str, Any]], expected: None = None
    ) -> "AzureAISearchTool": ...

    @classmethod
    @overload
    def load_component(
        cls, model: Union[ComponentModel, Dict[str, Any]], expected: Type[ExpectedType]
    ) -> ExpectedType: ...

    @classmethod
    def load_component(
        cls, model: Union[ComponentModel, Dict[str, Any]], expected: Optional[Type[ExpectedType]] = None
    ) -> Union["AzureAISearchTool", ExpectedType]:
        """Load a component from a component model.

        Args:
            model: The component model or dictionary with configuration
            expected: Optional expected return type

        Returns:
            An initialized AzureAISearchTool instance
        """
        token = _allow_private_constructor.set(True)
        try:
            if isinstance(model, dict):
                model = ComponentModel(**model)

            config = model.config

            query_type_str = config.get("query_type", "simple")
            query_type: Literal["simple", "full", "semantic", "vector"]

            if query_type_str == "simple":
                query_type = "simple"
            elif query_type_str == "full":
                query_type = "full"
            elif query_type_str == "semantic":
                query_type = "semantic"
            else:
                query_type = "vector"

            instance = cls(
                name=config.get("name", ""),
                endpoint=config.get("endpoint", ""),
                index_name=config.get("index_name", ""),
                credential=config.get("credential", {}),
                description=config.get("description"),
                api_version=config.get("api_version", "2023-11-01"),
                query_type=query_type,
                search_fields=config.get("search_fields"),
                select_fields=config.get("select_fields"),
                vector_fields=config.get("vector_fields"),
                top=config.get("top"),
                filter=config.get("filter"),
                enable_caching=config.get("enable_caching", False),
                cache_ttl_seconds=config.get("cache_ttl_seconds", 300),
            )

            if expected is not None:
                return cast(ExpectedType, instance)
            return instance
        finally:
            _allow_private_constructor.reset(token)

    @classmethod
    def create_simple_search(
        cls,
        name: str,
        endpoint: str,
        index_name: str,
        credential: Union[AzureKeyCredential, TokenCredential, Dict[str, str]],
        **kwargs: Any,
    ) -> "AzureAISearchTool":
        """Factory method to create a simple search tool.

        This is the easiest way to get started with Azure AI Search in AutoGen.
        Simple search uses basic keyword matching for straightforward queries.

        Args:
            name (str): The name of the tool
            endpoint (str): The URL of your Azure AI Search service
            index_name (str): The name of the search index
            credential (Union[AzureKeyCredential, TokenCredential, Dict[str, str]]): Authentication credentials
            **kwargs (Any): Additional configuration options

        Returns:
            An initialized simple search tool

        Example:
            .. code-block:: python

                from autogen_ext.tools.azure import AzureAISearchTool

                # Create a simple search tool
                search_tool = AzureAISearchTool.create_simple_search(
                    name="simple_search",
                    endpoint="https://your-search-service.search.windows.net",
                    index_name="your-index",
                    credential={"api_key": "your-api-key"},
                )

                # Run a search
                result = await search_tool.run(args={"query": "machine learning"})
                print(f"Found {len(result.results)} results")
        """
        if not endpoint or not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be a valid URL starting with http:// or https://")

        if not index_name:
            raise ValueError("index_name cannot be empty")

        if not name:
            raise ValueError("name cannot be empty")

        if not credential:
            raise ValueError("credential cannot be None")

        token = _allow_private_constructor.set(True)
        try:
            return cls(
                name=name,
                endpoint=endpoint,
                index_name=index_name,
                credential=credential,
                query_type="simple",
                **kwargs,
            )
        finally:
            _allow_private_constructor.reset(token)

    @classmethod
    def create_full_search(
        cls,
        name: str,
        endpoint: str,
        index_name: str,
        credential: Union[AzureKeyCredential, TokenCredential, Dict[str, str]],
        search_fields: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "AzureAISearchTool":
        """Factory method to create a full text search tool.

        Full text search provides enhanced text analysis with Azure's advanced text
        processing capabilities, including word breaking, stemming, and relevance scoring.
        This provides more sophisticated keyword matching than simple search.

        Args:
            name (str): The name of the tool
            endpoint (str): The URL of your Azure AI Search service
            index_name (str): The name of the search index
            credential (Union[AzureKeyCredential, TokenCredential, Dict[str, str]]): Authentication credentials
            search_fields (Optional[List[str]]): Optional list of fields to search within
            select_fields (Optional[List[str]]): Optional list of fields to return in the results
            **kwargs (Any): Additional configuration options

        Returns:
            An initialized full text search tool

        Example:
            .. code-block:: python

                from azure.core.credentials import AzureKeyCredential
                from autogen_ext.tools.azure import AzureAISearchTool

                # Create a full text search tool
                full_search = AzureAISearchTool.create_full_search(
                    name="full_search",
                    endpoint="https://your-search-service.search.windows.net",
                    index_name="your-index",
                    credential=AzureKeyCredential("your-api-key"),
                    search_fields=["title", "content"],
                    select_fields=["id", "title", "content", "url"],
                )

                # Perform a full text search
                results = await full_search.run(args={"query": "advanced neural network architectures"})

                # Process results
                for item in results.results:
                    print(f"Score: {item.score}, Title: {item.content.get('title', 'No title')}")
        """
        if not endpoint or not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be a valid URL starting with http:// or https://")

        if not index_name:
            raise ValueError("index_name cannot be empty")

        if not name:
            raise ValueError("name cannot be empty")

        if not credential:
            raise ValueError("credential cannot be None")

        token = _allow_private_constructor.set(True)
        try:
            return cls(
                name=name,
                endpoint=endpoint,
                index_name=index_name,
                credential=credential,
                query_type="full",
                search_fields=search_fields,
                select_fields=select_fields,
                **kwargs,
            )
        finally:
            _allow_private_constructor.reset(token)

    @classmethod
    def create_semantic_search(
        cls,
        name: str,
        endpoint: str,
        index_name: str,
        credential: Union[AzureKeyCredential, TokenCredential, Dict[str, str]],
        semantic_config_name: str,
        **kwargs: Any,
    ) -> "AzureAISearchTool":
        """Factory method to create a semantic search tool.

        Semantic search uses natural language understanding to return results based on the
        meaning of a query rather than just keyword matching. It provides more relevant results
        for natural language queries.

        Args:
            name (str): The name of the tool
            endpoint (str): The URL of your Azure AI Search service
            index_name (str): The name of the search index
            credential (Union[AzureKeyCredential, TokenCredential, Dict[str, str]]): Authentication credentials
            semantic_config_name (str): The name of the semantic configuration to use
            **kwargs (Any): Additional configuration options

        Returns:
            An initialized semantic search tool

        Example:
            .. code-block:: python

                from azure.core.credentials import AzureKeyCredential
                from autogen_ext.tools.azure import AzureAISearchTool

                # Create a semantic search tool
                semantic_search_tool = AzureAISearchTool.create_semantic_search(
                    name="semantic_search",
                    endpoint="https://your-search-service.search.windows.net",
                    index_name="your-index",
                    credential=AzureKeyCredential("your-api-key"),
                    semantic_config_name="default",
                )

                # Perform a semantic search
                try:
                    result = await semantic_search_tool.run(args={"query": "latest advances in neural networks"})
                    print(f"Found {len(result.results)} results")
                except Exception as e:
                    print(f"Search failed: {e}")
        """
        if not endpoint or not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be a valid URL starting with http:// or https://")

        if not index_name:
            raise ValueError("index_name cannot be empty")

        if not name:
            raise ValueError("name cannot be empty")

        if not credential:
            raise ValueError("credential cannot be None")

        if not semantic_config_name:
            raise ValueError(
                "semantic_config_name cannot be empty - it must reference a semantic configuration in your Azure AI Search index"
            )

        token = _allow_private_constructor.set(True)
        try:
            return cls(
                name=name,
                endpoint=endpoint,
                index_name=index_name,
                credential=credential,
                query_type="semantic",
                semantic_config_name=semantic_config_name,
                **kwargs,
            )
        finally:
            _allow_private_constructor.reset(token)

    @classmethod
    def create_vector_search(
        cls,
        name: str,
        endpoint: str,
        index_name: str,
        credential: Union[AzureKeyCredential, TokenCredential, Dict[str, str]],
        vector_fields: List[str],
        **kwargs: Any,
    ) -> "AzureAISearchTool":
        """Factory method to create a vector search tool.

        Vector search uses embedding vectors to find semantically similar content, enabling
        the discovery of related information even when different terminology is used. This
        provides powerful similarity-based search capabilities.

        Args:
            name (str): The name of the tool
            endpoint (str): The URL of your Azure AI Search service
            index_name (str): The name of the search index
            credential (Union[AzureKeyCredential, TokenCredential, Dict[str, str]]): Authentication credentials
            vector_fields (List[str]): Fields containing vector embeddings for similarity search
            **kwargs (Any): Additional configuration options

        Returns:
            An initialized vector search tool

        Example:
            .. code-block:: python

                from azure.core.credentials import AzureKeyCredential
                from autogen_ext.tools.azure import AzureAISearchTool

                # Create a vector search tool
                vector_search_tool = AzureAISearchTool.create_vector_search(
                    name="vector_search",
                    endpoint="https://your-search-service.search.windows.net",
                    index_name="your-index",
                    credential=AzureKeyCredential("your-api-key"),
                    vector_fields=["embedding"],
                )

                # Perform a vector search with a text query
                result = await vector_search_tool.run(args={"query": "quantum computing algorithms"})
        """
        if not endpoint or not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be a valid URL starting with http:// or https://")

        if not index_name:
            raise ValueError("index_name cannot be empty")

        if not name:
            raise ValueError("name cannot be empty")

        if not credential:
            raise ValueError("credential cannot be None")

        if not vector_fields or len(vector_fields) == 0:
            raise ValueError("vector_fields must contain at least one field name")

        token = _allow_private_constructor.set(True)
        try:
            return cls(
                name=name,
                endpoint=endpoint,
                index_name=index_name,
                credential=credential,
                query_type="vector",
                vector_fields=vector_fields,
                **kwargs,
            )
        finally:
            _allow_private_constructor.reset(token)

    @classmethod
    def create_hybrid_search(
        cls,
        name: str,
        endpoint: str,
        index_name: str,
        credential: Union[AzureKeyCredential, TokenCredential, Dict[str, str]],
        semantic_config_name: str,
        vector_fields: List[str],
        **kwargs: Any,
    ) -> "AzureAISearchTool":
        """Factory method to create a hybrid search tool.

        Hybrid search combines the strengths of semantic ranking and vector similarity search
        to provide more comprehensive search results. Note that this method is a convenience wrapper
        around the create_semantic_search method with vector_fields added - it doesn't create a
        distinct query type but rather configures semantic search to also leverage vector capabilities.

        Args:
            name (str): The name of the tool
            endpoint (str): The URL of your Azure AI Search service
            index_name (str): The name of the search index
            credential (Union[AzureKeyCredential, TokenCredential, Dict[str, str]]): Authentication credentials
            semantic_config_name (str): The name of the semantic configuration to use
            vector_fields (List[str]): Fields containing vector embeddings for similarity search
            **kwargs (Any): Additional configuration options

        Returns:
            An initialized hybrid search tool combining semantic and vector search

        Example:
            .. code-block:: python

                from azure.core.credentials import AzureKeyCredential
                from autogen_ext.tools.azure import AzureAISearchTool

                # Create a hybrid search tool
                hybrid_search = AzureAISearchTool.create_hybrid_search(
                    name="hybrid_search",
                    endpoint="https://your-search-service.search.windows.net",
                    index_name="your-index",
                    credential=AzureKeyCredential("your-search-api-key"),
                    semantic_config_name="default",
                    vector_fields=["embedding_field"],
                )

                # Simple interface at query time - complex configuration is handled by the tool
                results = await hybrid_search.run(args={"query": "advanced machine learning techniques"})

                # Process results (benefits from both semantic and vector search capabilities)
                for item in results.results:
                    print(f"Score: {item.score}, Title: {item.content.get('title', 'No title')}")
        """
        if not vector_fields or len(vector_fields) == 0:
            raise ValueError("vector_fields must contain at least one field name for hybrid search")

        token = _allow_private_constructor.set(True)
        try:
            return cls.create_semantic_search(
                name=name,
                endpoint=endpoint,
                index_name=index_name,
                credential=credential,
                semantic_config_name=semantic_config_name,
                vector_fields=vector_fields,
                **kwargs,
            )
        finally:
            _allow_private_constructor.reset(token)

    @classmethod
    def from_env(
        cls,
        name: str,
        env_prefix: str = "AZURE_SEARCH",
        **kwargs: Any,
    ) -> "AzureAISearchTool":
        """Create a search tool instance from environment variables."""
        token = _allow_private_constructor.set(True)
        try:
            import os

            endpoint = os.getenv(f"{env_prefix}_ENDPOINT")
            index_name = os.getenv(f"{env_prefix}_INDEX_NAME")
            api_key = os.getenv(f"{env_prefix}_API_KEY")

            if not endpoint or not index_name or not api_key:
                raise ValueError(
                    f"Missing required environment variables. Please set {env_prefix}_ENDPOINT, "
                    f"{env_prefix}_INDEX_NAME, and {env_prefix}_API_KEY."
                )

            api_version = os.getenv(f"{env_prefix}_API_VERSION", "2023-11-01")
            query_type_str = os.getenv(f"{env_prefix}_QUERY_TYPE", "simple")

            valid_query_types = ["simple", "full", "semantic", "vector"]
            if query_type_str not in valid_query_types:
                raise ValueError(
                    f"Invalid query type: {query_type_str}. Must be one of: {', '.join(valid_query_types)}"
                )

            query_type: Literal["simple", "full", "semantic", "vector"]
            if query_type_str == "simple":
                query_type = "simple"
            elif query_type_str == "full":
                query_type = "full"
            elif query_type_str == "semantic":
                query_type = "semantic"
            else:
                query_type = "vector"

            credential = {"api_key": api_key}

            vector_fields = None
            vector_fields_str = os.getenv(f"{env_prefix}_VECTOR_FIELDS")
            if vector_fields_str:
                vector_fields = vector_fields_str.split(",")

            semantic_config_name = os.getenv(f"{env_prefix}_SEMANTIC_CONFIG")

            additional_params = kwargs.copy()

            return cls(
                name=name,
                endpoint=endpoint,
                index_name=index_name,
                credential=credential,
                api_version=api_version,
                query_type=query_type,
                vector_fields=vector_fields,
                semantic_config_name=semantic_config_name,
                **additional_params,
            )
        finally:
            _allow_private_constructor.reset(token)
