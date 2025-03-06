"""Configuration for Azure AI Search tool.

This module provides configuration classes for the Azure AI Search tool, including
settings for authentication, search behavior, retry policies, and caching.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Union
import logging

from azure.core.credentials import AzureKeyCredential, TokenCredential
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class AzureAISearchConfig(BaseModel):
    """Configuration for Azure AI Search tool.

    This class defines the configuration parameters for :class:`AzureAISearchTool`.
    It provides options for customizing search behavior including query types,
    field selection, authentication, retry policies, and caching strategies.

    .. note::

        This class requires the :code:`azure` extra for the :code:`autogen-ext` package.

        .. code-block:: bash

            pip install -U "autogen-ext[azure]"

    Example:
        .. code-block:: python

            from azure.core.credentials import AzureKeyCredential
            from autogen_ext.tools.azure import AzureAISearchConfig

            config = AzureAISearchConfig(
                name="doc_search",
                endpoint="https://my-search.search.windows.net",
                index_name="my-index",
                credential=AzureKeyCredential("<your-key>"),
                query_type="semantic",
                semantic_config_name="default"
            )

    For more details, see:
        * `Azure AI Search Overview <https://learn.microsoft.com/azure/search/search-what-is-azure-search>`_
        * `Vector Search <https://learn.microsoft.com/azure/search/vector-search-overview>`_
        * `Semantic Search <https://learn.microsoft.com/azure/search/semantic-search-overview>`_

    Args:
        name: Name for the tool instance, used to identify it in the agent's toolkit.
        description: Human-readable description of what this tool does and how to use it.
        endpoint: The full URL of your Azure AI Search service, in the format
            'https://<service-name>.search.windows.net'.
        index_name: Name of the target search index in your Azure AI Search service.
            The index must be pre-created and properly configured.
        api_version: Azure AI Search REST API version to use. Defaults to '2023-11-01'.
            Only change if you need specific features from a different API version.
        credential: Azure authentication credential:
            - AzureKeyCredential: For API key authentication (admin/query key)
            - TokenCredential: For Azure AD authentication (e.g., DefaultAzureCredential)
        semantic_config_name: Name of a semantic configuration defined in your search service.
            Required only when using semantic/hybrid search capabilities.
        query_type: The search query mode to use:
            - 'simple': Basic keyword search (default)
            - 'full': Full Lucene query syntax
            - 'semantic': Semantic ranking with ML models
            - 'vector': Vector similarity search
        search_fields: List of index fields to search within. If not specified,
            searches all searchable fields. Example: ['title', 'content'].
        select_fields: Fields to return in search results. If not specified,
            returns all fields. Use to optimize response size.
        vector_fields: Vector field names for vector search. Must be configured
            in your search index as vector fields. Required for vector search.
        top: Maximum number of documents to return in search results.
            Helps control response size and processing time.
    """

    name: str = Field(description="The name of the tool")
    description: Optional[str] = Field(default=None, description="A description of the tool")
    endpoint: str = Field(description="The endpoint URL for your Azure AI Search service")
    index_name: str = Field(description="The name of the search index to query")
    api_version: str = Field(
        default="2023-11-01",
        description="API version to use"
    )
    credential: Union[AzureKeyCredential, TokenCredential] = Field(
        description="The credential to use for authentication"
    )
    semantic_config_name: Optional[str] = Field(
        default=None,
        description="Optional name of semantic configuration to use"
    )
    query_type: Literal["simple", "full", "semantic", "vector"] = Field(
        default="simple",
        description="Type of query to perform"
    )
    search_fields: Optional[list[str]] = Field(
        default=None,
        description="Optional list of fields to search in"
    )
    select_fields: Optional[list[str]] = Field(
        default=None,
        description="Optional list of fields to return in results"
    )
    vector_fields: Optional[list[str]] = Field(
        default=None,
        description="Optional list of vector fields for vector search"
    )
    top: Optional[int] = Field(
        default=None,
        description="Optional number of results to return"
    )
    # Retry policy settings
    retry_enabled: bool = Field(
        default=True,
        description="Whether to enable retry policy for transient errors"
    )
    retry_max_attempts: Optional[int] = Field(
        default=3,
        description="Maximum number of retry attempts for failed requests"
    )
    retry_mode: Literal["fixed", "exponential"] = Field(
        default="exponential",
        description="Retry backoff strategy: fixed or exponential"
    )
    # Caching settings
    enable_caching: bool = Field(
        default=False,
        description="Whether to enable client-side caching of search results"
    )
    cache_ttl_seconds: int = Field(
        default=300,  # 5 minutes
        description="Time-to-live for cached search results in seconds"
    )
    # Embedding settings
    embedding_provider: Optional[str] = Field(
        default=None,
        description="Name of embedding provider to use (e.g., 'azure_openai', 'openai')"
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Model name to use for generating embeddings"
    )
    embedding_dimension: Optional[int] = Field(
        default=None,
        description="Dimension of embedding vectors produced by the model"  
    )
    
    model_config = {"arbitrary_types_allowed": True}
    
    @model_validator(mode='before')
    @classmethod
    def validate_credentials(cls, data: Any) -> Any:
        """Validate and convert credential data.
        
        This validator converts a dictionary with an 'api_key' into an AzureKeyCredential.
        It also supports passing in an existing AzureKeyCredential or TokenCredential directly.
        """
        if isinstance(data, dict) and 'credential' in data:
            credential = data['credential']
            
            if isinstance(credential, dict) and 'api_key' in credential:
                data['credential'] = AzureKeyCredential(credential['api_key'])
                
        return data
    
    def model_dump(self, **kwargs):
        """Custom model_dump to handle credentials."""
        data = super().model_dump(**kwargs)
        
        if isinstance(self.credential, AzureKeyCredential):
            data['credential'] = {"type": "AzureKeyCredential"}
        elif isinstance(self.credential, TokenCredential):
            data['credential'] = {"type": "TokenCredential"}
            
        return data
