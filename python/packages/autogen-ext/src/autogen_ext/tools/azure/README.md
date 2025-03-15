# Azure Tools for AutoGen

This module provides tools for integrating Azure services with AutoGen.

## Documentation

Detailed documentation, including usage examples, API reference, and configuration options, is now available in the module and class docstrings.

Please refer to:

- Module docstring in `_ai_search.py` for overview and quick start guide
- Class docstrings in `OpenAIAzureAISearchTool` for comprehensive API documentation
- Factory methods documentation for specialized search tool creation

## Installation

```bash
pip install "autogen-ext[azure]"
```

## Quick Start

```python
from autogen_core import ComponentModel
from autogen_ext.tools.azure import AzureAISearchTool

# Create a search tool with minimal configuration
search_tool = AzureAISearchTool.load_component(
    ComponentModel(
        provider="autogen_ext.tools.azure.AzureAISearchTool",
        config={
            "name": "AzureSearch",
            "endpoint": "https://your-search-service.search.windows.net",
            "index_name": "your-index",
            "credential": {"api_key": "your-api-key"},
            "query_type": "simple"
        }
    )
)
```

Set up your environment variables:

```bash
export AZURE_SEARCH_ENDPOINT="https://your-search-service.search.windows.net"
export AZURE_SEARCH_INDEX_NAME="your-index"
export AZURE_SEARCH_API_KEY="your-api-key"
export AZURE_SEARCH_QUERY_TYPE="semantic"
export AZURE_SEARCH_SEMANTIC_CONFIG="default"
export AZURE_SEARCH_VECTOR_FIELDS="embedding"
```

Then create the search tool:

```python
# Load configuration from environment variables
search_tool = OpenAIAzureAISearchTool.from_env(
    openai_client=openai_client,
    embedding_model="text-embedding-ada-002",
    name="env_search"
)
```

### Error Handling

The tool includes built-in retry logic for common transient errors:

- Rate limiting
- Network connectivity issues
- Server errors

It uses exponential backoff with jitter to efficiently recover from transient failures.

### Security Best Practices

1. **Use environment variables** instead of hardcoding credentials
2. **Use Azure Key Vault** for storing sensitive credentials
3. **Use managed identities** when running in Azure
4. **Apply least privilege** to your search service API keys
5. **Regularly rotate** your API keys

### Performance Optimization

For bulk operations, use the batched embedding functionality:

```python
async def process_multiple_queries(queries):
    # Process multiple queries in batch for efficiency
    embeddings = await search_tool._get_embeddings_batch(queries)
    # Use embeddings for search or other operations
```

### Debugging

Enable more verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("autogen_ext.tools.azure").setLevel(logging.DEBUG)
```

### Troubleshooting

If you encounter issues:

1. Verify your Azure AI Search service is running
2. Check that your index exists and is properly configured
3. Ensure your API key has appropriate permissions
4. For vector search, confirm your index has vector fields configured 