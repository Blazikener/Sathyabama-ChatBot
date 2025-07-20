import os
from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser

def get_pdf_engine(path: str, storage_name: str):
    # Create storage directory if needed
    Path(storage_name).mkdir(parents=True, exist_ok=True)
    
    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Check if storage exists
    if os.path.exists(storage_name) and any(Path(storage_name).iterdir()):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=storage_name)
            index = VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)
            return index.as_query_engine(similarity_top_k=3)
        except:
            print(f"Rebuilding {storage_name} index due to loading error")
    
    # Build new index
    print(f"Building new vector index: {storage_name}")
    
    if os.path.isdir(path):
        documents = SimpleDirectoryReader(path).load_data()
    else:
        documents = SimpleDirectoryReader(input_files=[path]).load_data()
    
    # Split documents into smaller chunks
    parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=200)
    nodes = parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=storage_name)
    return index.as_query_engine(similarity_top_k=3)