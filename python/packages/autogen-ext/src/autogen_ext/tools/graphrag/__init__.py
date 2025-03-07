from ._config import (
    GlobalContextConfig,
    GlobalDataConfig,
    LocalContextConfig,
    LocalDataConfig,
    MapReduceConfig,
    SearchConfig,
)

# Conditionally import GraphRAG tools to handle missing azure.search.documents.indexes dependency
try:
    from ._global_search import GlobalSearchTool, GlobalSearchToolArgs, GlobalSearchToolReturn
    from ._local_search import LocalSearchTool, LocalSearchToolArgs, LocalSearchToolReturn
    _HAS_REQUIRED_DEPENDENCIES = True
except ImportError:
    # Create placeholder classes if dependencies are missing
    _HAS_REQUIRED_DEPENDENCIES = False
    
    class GlobalSearchToolArgs:
        pass
        
    class GlobalSearchToolReturn:
        pass
        
    class GlobalSearchTool:
        pass
        
    class LocalSearchToolArgs:
        pass
        
    class LocalSearchToolReturn:
        pass
        
    class LocalSearchTool:
        pass

__all__ = [
    "GlobalSearchTool",
    "LocalSearchTool",
    "GlobalDataConfig",
    "LocalDataConfig",
    "GlobalContextConfig",
    "GlobalSearchToolArgs",
    "GlobalSearchToolReturn",
    "LocalContextConfig",
    "LocalSearchToolArgs",
    "LocalSearchToolReturn",
    "MapReduceConfig",
    "SearchConfig",
]
