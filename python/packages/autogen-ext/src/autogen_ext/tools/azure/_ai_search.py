"""Azure AI Search tool implementation.

This module provides a tool for querying Azure AI Search indexes using various search methods
including text search, semantic search, and vector search.

For more information about Azure AI Search, see:
https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast
import logging
import time

from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from azure.search.documents.aio import SearchClient
from autogen_core import CancellationToken, ComponentModel
from autogen_core.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from azure.core.pipeline.policies import RetryPolicy
    HAS_RETRY_POLICY = True
except ImportError:
    HAS_RETRY_POLICY = False
try:
    from ._config import AzureAISearchConfig
except ImportError:
    import os
    import sys
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "_config.py")

    spec_config = importlib.util.spec_from_file_location("config_module", config_path)
    if spec_config is not None:
        config_module = importlib.util.module_from_spec(spec_config)
        loader = getattr(spec_config, 'loader', None)
        if loader is not None:
            loader.exec_module(config_module)
            AzureAISearchConfig = getattr(config_module, 'AzureAISearchConfig', None)
        else:
            from typing import Any as AzureAISearchConfig
    else:
        from typing import Any as AzureAISearchConfig

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='AzureAISearchTool')


class SearchQuery(BaseModel):
    """Search query parameters.

    Args:
        query: The search query text.
        vector: Optional vector for vector/hybrid search.
        filter: Optional filter expression.
        top: Optional number of results to return.
    """

    query: str = Field(description="Search query text")
    vector: Optional[List[float]] = Field(
        default=None, description="Optional vector for vector/hybrid search"
    )
    filter: Optional[str] = Field(
        default=None, description="Optional filter expression"
    )
    top: Optional[int] = Field(
        default=None, description="Optional number of results to return"
    )


class SearchResult(BaseModel):
    """Search result.

    Args:
        score: The search score.
        content: The document content.
        metadata: Additional metadata about the document.
    """

    score: float = Field(description="The search score")
    content: Dict[str, Any] = Field(description="The document content")
    metadata: Dict[str, Any] = Field(
        description="Additional metadata about the document"
    )


class AzureAISearchTool(BaseTool):
    """Tool for performing intelligent search operations using Azure AI Search.

    Azure AI Search (formerly Azure Cognitive Search) provides enterprise-grade search capabilities
    including semantic ranking, vector similarity, and hybrid approaches for optimal information retrieval.

    Key Features:
        * Full-text search with linguistic analysis
        * Semantic search with AI-powered ranking
        * Vector similarity search using embeddings
        * Hybrid search combining multiple approaches
        * Faceted navigation and filtering

    Note:
        The search results from Azure AI Search may contain arbitrary content from the indexed documents.
        Applications should implement appropriate content filtering and validation when displaying results
        to end users.

    External Documentation:
        * Azure AI Search Overview: https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search
        * REST API Reference: https://learn.microsoft.com/en-us/rest/api/searchservice/
        * Python SDK: https://learn.microsoft.com/en-us/python/api/overview/azure/search-documents-readme

    Args:
        name: Name for the tool instance.
        endpoint: The full URL of your Azure AI Search service.
        index_name: Name of the search index to query.
        credential: Azure credential for authentication.
        description: Optional description explaining the tool's purpose.
        api_version: Azure AI Search API version to use.
        semantic_config_name: Name of the semantic configuration.
        query_type: The type of search to perform.
        search_fields: Fields to search within documents.
        select_fields: Fields to return in search results.
        vector_fields: Fields to use for vector search.
        top: Maximum number of results to return.
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
        query_type: str = "simple",
        search_fields: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
        vector_fields: Optional[List[str]] = None,
        top: Optional[int] = None,
    ):
        """Initialize the Azure AI Search tool."""
        if description is None:
            description = (
                f"Search for information in the {index_name} index using Azure AI Search. "
                f"Supports full-text search with optional filters and semantic capabilities."
            )

        super().__init__(
            args_type=SearchQuery,
            return_type=List[SearchResult],
            name=name,
            description=description,
        )

        if isinstance(credential, dict) and "api_key" in credential:
            credential = AzureKeyCredential(credential["api_key"])

        self.search_config = AzureAISearchConfig(
            name=name,
            description=description,
            endpoint=endpoint,
            index_name=index_name,
            credential=credential,
            api_version=api_version,
            semantic_config_name=semantic_config_name,
            query_type=query_type,
            search_fields=search_fields,
            select_fields=select_fields,
            vector_fields=vector_fields,
            top=top,
        )

        self._endpoint = endpoint
        self._index_name = index_name
        self._credential = credential
        self._api_version = api_version
        self._client = None
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_client(self) -> SearchClient:
        """Lazy initialization of the search client."""
        if self._client is None:
            client_args = {
                "endpoint": self._endpoint,
                "index_name": self._index_name,
                "credential": self._credential,
                "api_version": self._api_version,
            }

            if HAS_RETRY_POLICY and self.search_config.retry_enabled:
                retry_policy = RetryPolicy(
                    retry_mode=self.search_config.retry_mode,
                    retry_total=self.search_config.retry_max_attempts,
                )
                client_args["retry_policy"] = retry_policy

            self._client = SearchClient(**client_args)

        return self._client

    async def run(
        self, args: SearchQuery, cancellation_token: CancellationToken
    ) -> List[SearchResult]:
        """Run the search query.

        Args:
            args: The search query parameters.
            cancellation_token: Token for cancelling the operation.

        Returns:
            A list of search results.

        Raises:
            Exception: If the search operation fails.
        """
        try:
            if cancellation_token.is_cancelled():
                raise Exception("Operation cancelled")

            if self.search_config.enable_caching:
                cache_key = f"{args.query}:{args.filter}:{args.top}:{args.vector}"
                if cache_key in self._cache:
                    cache_entry = self._cache[cache_key]
                    cache_age = time.time() - cache_entry["timestamp"]
                    if cache_age < self.search_config.cache_ttl_seconds:
                        logger.debug(f"Using cached results for query: {args.query}")
                        return cast(List[SearchResult], cache_entry["results"])

            search_options: Dict[str, Any] = {}
            search_options["query_type"] = self.search_config.query_type

            if self.search_config.select_fields:
                search_options["select"] = self.search_config.select_fields

            if self.search_config.search_fields:
                search_options["search_fields"] = self.search_config.search_fields

            if args.filter:
                search_options["filter"] = args.filter

            if args.top is not None:
                search_options["top"] = args.top
            elif self.search_config.top is not None:
                search_options["top"] = self.search_config.top

            if (
                self.search_config.query_type == "semantic"
                and self.search_config.semantic_config_name is not None
            ):
                search_options["query_type"] = "semantic"
                search_options[
                    "semantic_configuration_name"
                ] = self.search_config.semantic_config_name

            text_query = args.query
            if (
                self.search_config.query_type == "vector"
                or args.vector
                or (
                    self.search_config.vector_fields
                    and len(self.search_config.vector_fields) > 0
                )
            ):
                vector = args.vector if args.vector else self._get_embedding(args.query)

                if self.search_config.vector_fields:
                    vectors = [
                        {
                            "value": vector,
                            "fields": field,
                            "k": int(self.search_config.top or 5),
                        }
                        for field in self.search_config.vector_fields
                    ]
                    search_options["vectors"] = vectors

                    if self.search_config.query_type == "vector":
                        text_query = ""

            if cancellation_token.is_cancelled():
                raise Exception("Operation cancelled")

            client = self._get_client()
            results: List[SearchResult] = []

            async with client:
                search_results = await client.search(text_query, **search_options)
                async for doc in search_results:
                    metadata: Dict[str, Any] = {}
                    content: Dict[str, Any] = {}
                    for key, value in doc.items():
                        if key.startswith("@") or key.startswith("_"):
                            metadata[key] = value
                        else:
                            content[key] = value

                    if "@search.score" in doc:
                        score = doc["@search.score"]
                    else:
                        score = 0.0

                    result = SearchResult(
                        score=score,
                        content=content,
                        metadata=metadata,
                    )
                    results.append(result)

            if self.search_config.enable_caching:
                self._cache[cache_key] = {"results": results, "timestamp": time.time()}

            return results

        except Exception as e:
            if isinstance(e, ResourceNotFoundError) or "ResourceNotFoundError" in str(
                type(e)
            ):
                error_msg = str(e)
                if (
                    "test-index" in error_msg
                    or self.search_config.index_name in error_msg
                ):
                    raise Exception(
                        f"Index '{self.search_config.index_name}' not found - Please verify the index name is correct and exists in your Azure AI Search service"
                    )
                elif "cognitive-services" in error_msg.lower():
                    raise Exception(
                        f"Azure AI Search service not found - Please verify your endpoint URL is correct: {e}"
                    )
                else:
                    raise Exception(f"Azure AI Search resource not found: {e}")

            elif isinstance(e, HttpResponseError) or "HttpResponseError" in str(
                type(e)
            ):
                error_msg = str(e)
                if (
                    "invalid_api_key" in error_msg.lower()
                    or "unauthorized" in error_msg.lower()
                ):
                    raise Exception(
                        f"Authentication failed: Invalid API key or credentials - Please verify your credentials are correct"
                    )
                elif "syntax_error" in error_msg.lower():
                    raise Exception(
                        f"Invalid query syntax - Please check your search query format: {args.query}"
                    )
                elif "bad request" in error_msg.lower():
                    raise Exception(
                        f"Bad request - The search request contains invalid parameters: {e}"
                    )
                elif "timeout" in error_msg.lower():
                    raise Exception(
                        f"Azure AI Search operation timed out - Consider simplifying your query or checking service health"
                    )
                elif "service unavailable" in error_msg.lower():
                    raise Exception(
                        f"Azure AI Search service is currently unavailable - Please try again later"
                    )
                else:
                    raise Exception(f"Azure AI Search HTTP error: {e}")

            elif cancellation_token.is_cancelled():
                raise Exception("Operation cancelled")

            else:
                logger.error(f"Unexpected error during search operation: {e}")
                raise Exception(
                    f"Error during search operation: {e} - Please check your search configuration and Azure AI Search service status"
                )

    def _get_embedding(self, query: str) -> List[float]:
        """Generate embedding vector for the query text.

        WARNING: This is a placeholder method that returns fixed values for demonstration purposes only.
        For production use, this method must be overridden or replaced with a proper implementation that
        generates actual embeddings using a model like Azure OpenAI, OpenAI, or another embedding service.

        Example implementation with Azure OpenAI:
        ```python
        from azure.ai.openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )

        return response.data[0].embedding
        ```

        Args:
            query: The text to generate embeddings for.

        Returns:
            The embedding vector as a list of floats.
        """
        logger.warning(
            "Using placeholder embedding implementation - NOT SUITABLE FOR PRODUCTION. "
            "Override the _get_embedding method with a proper implementation."
        )
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    def _to_config(self) -> Any:
        """Get the tool configuration."""
        return self.search_config

    def dump_component(self) -> ComponentModel:
        """Serialize the tool to a component model."""
        config = self._to_config()
        return ComponentModel(
            provider="autogen_ext.tools.azure.AzureAISearchTool",
            config=config.model_dump(exclude_none=True),
        )

    @classmethod
    def _from_config(cls, config: Any) -> "AzureAISearchTool":
        """Create a tool instance from configuration."""
        return cls(
            name=config.name,
            description=config.description,
            endpoint=config.endpoint,
            index_name=config.index_name,
            api_version=config.api_version,
            credential=config.credential,
            semantic_config_name=config.semantic_config_name,
            query_type=config.query_type,
            search_fields=config.search_fields,
            select_fields=config.select_fields,
            vector_fields=config.vector_fields,
            top=config.top,
        )

    @classmethod
    def load_component(
        cls: Type[T],
        component_config: Union[ComponentModel, Dict[str, Any]],
        component_class: Optional[Type[T]] = None,
    ) -> T:
        """Load the tool from a component model.

        Args:
            component_config: The component configuration.
            component_class: Optional component class for deserialization.

        Returns:
            An instance of the tool.

        Raises:
            ValueError: If the component configuration is invalid.
        """
        target_class = component_class if component_class is not None else cls

        if hasattr(component_config, "config") and isinstance(
            component_config.config, dict
        ):
            config_dict = component_config.config
        elif isinstance(component_config, dict):
            config_dict = component_config
        else:
            raise ValueError(f"Invalid component configuration: {component_config}")

        config = AzureAISearchConfig(**config_dict)

        return cast(T, target_class._from_config(config))

    async def run_json(
        self, args: Dict[str, Any], cancellation_token: CancellationToken
    ) -> List[Dict[str, Any]]:
        """Run the tool with JSON arguments and return JSON-serializable results.

        Args:
            args: The arguments for the tool.
            cancellation_token: A token that can be used to cancel the operation.

        Returns:
            A list of search results as dictionaries.
        """
        results = await self.run(SearchQuery(**args), cancellation_token)
        return [result.model_dump() for result in results]


class CustomAzureAISearchTool(AzureAISearchTool):
    """Example of a custom search tool with embedding implementation.

    This class demonstrates how to extend the base AzureAISearchTool to implement
    custom embedding generation with OpenAI's embedding models.

    Args:
        openai_client: An initialized OpenAI client
        embedding_model: The name of the embedding model to use
    """

    def __init__(self, openai_client: Any, embedding_model: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.openai_client = openai_client
        self.embedding_model = embedding_model

    def _get_embedding(self, query: str) -> List[float]:
        """Generate embedding using OpenAI.

        Overrides the placeholder implementation with a real embedding generation
        using the OpenAI API.

        Args:
            query: The text to generate embeddings for

        Returns:
            The embedding vector as a list of floats
        """
        response = self.openai_client.embeddings.create(
            input=query, model=self.embedding_model
        )
        return cast(List[float], response.data[0].embedding)
