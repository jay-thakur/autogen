# Azure Tools for AutoGen

This module provides tools for integrating Azure services with AutoGen.

## Azure AI Search Tool

The Azure AI Search tool enables agents to search through documents and data stored in Azure AI Search indexes.

### Installation

```bash
pip install "autogen-ext[azure]"
```

### Features

- **Simple Search**: Basic keyword-based search
- **Semantic Search**: AI-powered search with semantic ranking
- **Vector Search**: Similarity search using embeddings
- **Hybrid Search**: Combines text and vector search for optimal results

### Advanced Usage

#### Custom Embedding Provider

You can customize the embedding generation by extending the base class:

```python
from autogen_ext.tools.azure import AzureAISearchTool
from azure.core.credentials import AzureKeyCredential
import openai

class CustomSearchTool(AzureAISearchTool):
    def __init__(self, openai_client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openai_client = openai_client
        
    def _get_embedding(self, query):
        response = self.openai_client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

# Usage
openai_client = openai.OpenAI(api_key="your-openai-key")
search_tool = CustomSearchTool(
    openai_client=openai_client,
    name="document_search",
    endpoint="https://your-search-service.search.windows.net",
    index_name="your-index",
    credential=AzureKeyCredential("your-api-key"),
    query_type="vector",
    vector_fields=["embedding"]
)
```

#### Filtering Results

You can apply filters to narrow down search results:

```python
results = await search_tool.run_json(
    {
        "query": "financial reports",
        "filter": "year eq 2023 and department eq 'Finance'"
    }, 
    CancellationToken()
)
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `endpoint` | Azure AI Search service URL | Required |
| `index_name` | Name of the search index | Required |
| `credential` | Azure credential for authentication | Required |
| `query_type` | Type of search: "simple", "full", "semantic", "vector" | "simple" |
| `semantic_config_name` | Name of semantic configuration | None |
| `vector_fields` | Fields to use for vector search | None |
| `search_fields` | Fields to search within | All searchable fields |
| `select_fields` | Fields to return in results | All fields |
| `top` | Maximum number of results to return | 50 |

### Error Handling

The tool provides detailed error messages for common issues:

- Index not found
- Authentication failures
- Invalid query syntax
- Service unavailability

### Troubleshooting

If you encounter issues:

1. Verify your Azure AI Search service is running
2. Check that your index exists and is properly configured
3. Ensure your API key has appropriate permissions
4. For vector search, confirm your index has vector fields configured 