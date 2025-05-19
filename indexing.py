from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from langchain_openai import OpenAIEmbeddings
from config import EMBEDDING_MODEL, OUTPUT_DIR
import os
import chromadb
import uuid
import json

class ResearchIndex:
    def __init__(self, persist_dir=None):
        """Initialize the research index
        
        Args:
            persist_dir: Directory to persist the index. If None, the index will be in-memory only.
        """
        # Set up the embedding model
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=1024)
        
        self.documents = []
        self.index = None
        self.persist_dir = persist_dir if persist_dir else os.path.join(OUTPUT_DIR, "vector_index")
        
        # Create persistent storage if specified
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._setup_persistent_index()
        else:
            # Create an empty index
            self.index = None
        
    def _setup_persistent_index(self):
        """Set up a persistent index using ChromaDB"""
        try:
            # Create client and collection
            chroma_client = chromadb.PersistentClient(self.persist_dir)
            chroma_collection = chroma_client.get_or_create_collection("research_data")
            
            # Create vector store and storage context
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Load or create index
            if chroma_collection.count() > 0:
                self.index = VectorStoreIndex.from_vector_store(vector_store)
            else:
                self.index = None
                
        except Exception as e:
            print(f"Error setting up persistent index: {str(e)}")
            print("Falling back to in-memory index")
            self.index = None
            self.persist_dir = None
    
    def add_document(self, content, metadata=None):
        """Add a document to the index"""
        if metadata is None:
            metadata = {}
        
        # Add timestamp and unique ID if not provided
        if "timestamp" not in metadata:
            from datetime import datetime
            metadata["timestamp"] = datetime.now().isoformat()
        
        if "id" not in metadata:
            metadata["id"] = str(uuid.uuid4())
        
        # Create document
        doc = Document(text=content, metadata=metadata)
        self.documents.append(doc)
        
        # Rebuild index when new documents are added
        self._build_index()
        
        # Save metadata separately for easy access
        if self.persist_dir:
            metadata_path = os.path.join(self.persist_dir, "metadata.json")
            try:
                # Load existing metadata if available
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        all_metadata = json.load(f)
                else:
                    all_metadata = []
                
                # Add new metadata
                all_metadata.append(metadata)
                
                # Save updated metadata
                with open(metadata_path, 'w') as f:
                    json.dump(all_metadata, f, indent=2)
            except Exception as e:
                print(f"Error saving metadata: {str(e)}")
        
    def _build_index(self):
        """Build the vector index from added documents"""
        if not self.documents:
            return
        
        try:
            if self.persist_dir:
                # Get the storage context from the existing setup
                chroma_client = chromadb.PersistentClient(self.persist_dir)
                chroma_collection = chroma_client.get_or_create_collection("research_data")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Create or update the index
                if self.index is None:
                    self.index = VectorStoreIndex.from_documents(
                        self.documents,
                        storage_context=storage_context
                    )
                else:
                    # Add only new documents
                    for doc in self.documents:
                        self.index.insert(doc)
            else:
                # In-memory index
                self.index = VectorStoreIndex.from_documents(self.documents)
        except Exception as e:
            print(f"Error building index: {str(e)}")
    
    def query(self, query_text, similarity_top_k=3):
        """Query the index for relevant information"""
        if not self.index:
            return "No documents have been indexed yet."
        
        try:    
            query_engine = self.index.as_query_engine(similarity_top_k=similarity_top_k)
            response = query_engine.query(query_text)
            return str(response)
        except Exception as e:
            return f"Error querying index: {str(e)}"
    
    def get_all_documents(self):
        """Get all documents with their metadata"""
        docs_with_metadata = []
        for doc in self.documents:
            docs_with_metadata.append({
                "content": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
                "metadata": doc.metadata
            })
        return docs_with_metadata
    