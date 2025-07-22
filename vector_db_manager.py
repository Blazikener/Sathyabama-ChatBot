import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, Settings
from llama_index.readers.file import PDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pandas as pd
from llama_index.core import Document

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.embed_model = embed_model

class VectorDBManager:
    def __init__(self):
        self.embed_model = embed_model
        self.indices = {}
    
    def get_index(self, data, index_name):
        """Create or load vector index for given data"""
        index = None
        if not os.path.exists(index_name):
            print(f"Building index: {index_name}")
            index = VectorStoreIndex.from_documents(data, embed_model=self.embed_model, show_progress=True)
            index.storage_context.persist(persist_dir=index_name)
        else:
            print(f"Loading existing index: {index_name}")
            # Set the embedding model in storage context when loading
            storage_context = StorageContext.from_defaults(persist_dir=index_name)
            index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        return index
    
    def create_syllabus_index(self, syllabus_data_path):
        """Create vector index for syllabus data"""
        if os.path.exists(syllabus_data_path):
            if syllabus_data_path.endswith(".pdf"):
                documents = PDFReader().load_data(file=syllabus_data_path)
            elif syllabus_data_path.endswith(".csv"):
                df = pd.read_csv(syllabus_data_path)
                documents = [Document(text=row.to_string()) for _, row in df.iterrows()]
            else:
                with open(syllabus_data_path, "r") as f:
                    content = f.read()
                documents = [Document(text=content)]
            
            index = self.get_index(documents, "syllabus_index")
            self.indices["syllabus"] = index
            return index.as_query_engine()
        return None
    
    def create_admission_index(self, admission_data_path):
        """Create vector index for admission details"""
        if os.path.exists(admission_data_path):
            if admission_data_path.endswith(".pdf"):
                documents = PDFReader().load_data(file=admission_data_path)
            else:
                with open(admission_data_path, "r") as f:
                    content = f.read()
                documents = [Document(text=content)]
            
            index = self.get_index(documents, "admission_index")
            self.indices["admission"] = index
            return index.as_query_engine()
        return None
    
    def create_food_menu_index(self, food_menu_data_path):
        """Create vector index for food menu"""
        if os.path.exists(food_menu_data_path):
            if food_menu_data_path.endswith(".csv"):
                df = pd.read_csv(food_menu_data_path)
                documents = [Document(text=row.to_string()) for _, row in df.iterrows()]
            else:
                with open(food_menu_data_path, "r") as f:
                    content = f.read()
                documents = [Document(text=content)]
            
            index = self.get_index(documents, "food_menu_index")
            self.indices["food_menu"] = index
            return index.as_query_engine()
        return None
    
    def create_bus_details_index(self, bus_data_path):
        """Create vector index for bus details"""
        if os.path.exists(bus_data_path):
            if bus_data_path.endswith(".csv"):
                df = pd.read_csv(bus_data_path)
                documents = [Document(text=row.to_string()) for _, row in df.iterrows()]
            else:
                with open(bus_data_path, "r") as f:
                    content = f.read()
                documents = [Document(text=content)]
            
            index = self.get_index(documents, "bus_details_index")
            self.indices["bus_details"] = index
            return index.as_query_engine()
        return None
    
    def get_all_query_engines(self):
        """Return all available query engines"""
        engines = {}
                
        engines["syllabus"] = self.create_syllabus_index(os.path.join("data", "syllabus.txt"))
        engines["admission"] = self.create_admission_index(os.path.join("data", "admission_details.txt"))
        engines["food_menu"] = self.create_food_menu_index(os.path.join("data", "food_menu.csv"))
        engines["bus_details"] = self.create_bus_details_index(os.path.join("data", "bus_details.csv"))
        
        return {k: v for k, v in engines.items() if v is not None}
    
