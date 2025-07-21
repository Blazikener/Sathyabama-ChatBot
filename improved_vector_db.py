import os
from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Settings
import logging

logger = logging.getLogger(__name__)

def get_optimal_embedding_model():
    """Get the best available embedding model"""
    try:
        # Try to use a more powerful embedding model
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        logger.info("✓ Using all-mpnet-base-v2 embedding model")
        return embed_model
    except Exception as e:
        logger.warning(f"Failed to load all-mpnet-base-v2, falling back to MiniLM: {e}")
        try:
            embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("✓ Using all-MiniLM-L6-v2 embedding model")
            return embed_model
        except Exception as e2:
            logger.error(f"Failed to load any embedding model: {e2}")
            raise

def get_optimal_chunking_strategy(content_type="general"):
    """Get optimal chunking strategy based on content type"""
    strategies = {
        "syllabus": {"chunk_size": 2048, "chunk_overlap": 400},
        "admission": {"chunk_size": 1536, "chunk_overlap": 300}, 
        "general": {"chunk_size": 1024, "chunk_overlap": 200}
    }
    
    config = strategies.get(content_type, strategies["general"])
    return SimpleNodeParser.from_defaults(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )

def validate_documents(documents):
    """Validate and filter documents"""
    if not documents:
        raise ValueError("No documents found to index")
    
    valid_docs = []
    for doc in documents:
        if doc.text and len(doc.text.strip()) > 50:  # Minimum content length
            valid_docs.append(doc)
        else:
            logger.warning(f"Skipping document with insufficient content: {doc.metadata}")
    
    if not valid_docs:
        raise ValueError("No valid documents found after filtering")
    
    logger.info(f"✓ Validated {len(valid_docs)} documents for indexing")
    return valid_docs

def get_pdf_engine(path: str, storage_name: str, content_type="general"):
    """Enhanced PDF/text engine with better error handling and optimization"""
    
    # Create storage directory if needed
    Path(storage_name).mkdir(parents=True, exist_ok=True)
    
    # Initialize embedding model
    embed_model = get_optimal_embedding_model()
    
    # Set global embedding model
    Settings.embed_model = embed_model
    
    # Check if storage exists and is valid
    storage_path = Path(storage_name)
    if storage_path.exists() and any(storage_path.iterdir()):
        try:
            logger.info(f"Loading existing index from {storage_name}")
            storage_context = StorageContext.from_defaults(persist_dir=storage_name)
            index = VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)
            
            # Test the index with a simple query
            query_engine = index.as_query_engine(
                similarity_top_k=5,
                response_mode="tree_summarize"
            )
            
            logger.info(f"✓ Successfully loaded existing index: {storage_name}")
            return query_engine
            
        except Exception as e:
            logger.warning(f"Failed to load existing index {storage_name}: {e}")
            logger.info(f"Rebuilding index...")
    
    # Build new index
    logger.info(f"Building new vector index: {storage_name}")
    
    try:
        # Load documents
        if os.path.isdir(path):
            # For directories, load all supported files
            documents = SimpleDirectoryReader(
                input_dir=path,
                recursive=True,
                required_exts=[".pdf", ".txt", ".md", ".docx"]
            ).load_data()
        else:
            # For single files
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            documents = SimpleDirectoryReader(input_files=[path]).load_data()
        
        # Validate documents
        documents = validate_documents(documents)
        
        # Get optimal chunking strategy
        parser = get_optimal_chunking_strategy(content_type)
        nodes = parser.get_nodes_from_documents(documents)
        
        if not nodes:
            raise ValueError("No nodes generated from documents")
        
        logger.info(f"✓ Generated {len(nodes)} nodes for indexing")
        
        # Create index
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        
        # Persist index
        index.storage_context.persist(persist_dir=storage_name)
        logger.info(f"✓ Index persisted to {storage_name}")
        
        # Create enhanced query engine
        query_engine = index.as_query_engine(
            similarity_top_k=5,  # Retrieve more context
            response_mode="tree_summarize",  # Better response synthesis
            verbose=True
        )
        
        logger.info(f"✓ Query engine created for {storage_name}")
        return query_engine
        
    except Exception as e:
        logger.error(f"Failed to create index for {storage_name}: {str(e)}")
        raise

def test_query_engine(query_engine, test_query="What information is available?"):
    """Test query engine functionality"""
    try:
        response = query_engine.query(test_query)
        logger.info(f"✓ Query engine test successful: {len(str(response))} characters returned")
        return True
    except Exception as e:
        logger.error(f"✗ Query engine test failed: {e}")
        return False

