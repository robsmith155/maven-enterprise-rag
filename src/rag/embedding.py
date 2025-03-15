# src/rag/embedding.py

from typing import List, Dict, Any, Optional, Union
import numpy as np
from openai import OpenAI
from .chunking import Chunk

# Add imports for SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class OpenAIEmbeddings:
    """Class for generating embeddings using OpenAI's API."""
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        dimensions: Optional[int] = None
    ):
        """
        Initialize the OpenAI embeddings generator.
        
        Args:
            model: OpenAI embedding model to use
            batch_size: Number of texts to embed in a single API call
            dimensions: Dimensionality of embeddings (if None, use model default)
        """
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.client = OpenAI()
        
        # Set embedding dimension based on model or dimensions parameter
        if dimensions is not None:
            self.embedding_dim = dimensions
        elif model == "text-embedding-3-small":
            self.embedding_dim = 1536
        elif model == "text-embedding-3-large":
            self.embedding_dim = 3072
        else:
            # Default for other models
            self.embedding_dim = 1536
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Call OpenAI API
            response = self.client.embeddings.create(
                model=self.model,
                input=batch_texts,
                dimensions=self.dimensions
            )
            
            # Extract embeddings
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_chunks(self, chunks: List[Chunk]) -> Dict[str, List[float]]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            Dictionary with chunk IDs as keys and embeddings as values
        """
        # Extract texts and IDs
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Create mapping from chunk ID to embedding
        return {chunk_id: embedding for chunk_id, embedding in zip(chunk_ids, embeddings)}
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[query],
            dimensions=self.dimensions
        )
        
        return response.data[0].embedding


class SentenceTransformerEmbeddings:
    """Class for generating embeddings using SentenceTransformers."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        device: str = "cpu"
    ):
        """
        Initialize the SentenceTransformer embeddings generator.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            batch_size: Number of texts to embed in a single batch
            device: Device to run the model on ('cpu' or 'cuda')
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "SentenceTransformers is not installed. "
                "Please install it with `pip install sentence-transformers`."
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        
        # Load model
        self.model = SentenceTransformer(model_name, device=device)
        
        # Set embedding dimension based on model
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Generate embeddings
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_tensor=False,  # Return numpy array
                normalize_embeddings=True  # L2 normalize embeddings
            )
            
            # Convert to list of lists
            batch_embeddings_list = [emb.tolist() for emb in batch_embeddings]
            all_embeddings.extend(batch_embeddings_list)
        
        return all_embeddings
    
    def embed_chunks(self, chunks: List[Chunk]) -> Dict[str, List[float]]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            Dictionary with chunk IDs as keys and embeddings as values
        """
        # Extract texts and IDs
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Create mapping from chunk ID to embedding
        return {chunk_id: embedding for chunk_id, embedding in zip(chunk_ids, embeddings)}
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        # For BGE models, prepend instruction for better retrieval performance
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for retrieval: {query}"
            
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        
        return embedding.tolist()
    
    def cleanup(self):
        """
        Release GPU and CPU memory by moving the model to CPU, deleting it, and clearing caches.
        This should be called when the model is no longer needed to free resources.
        """
        if hasattr(self, 'model'):
            # First move model to CPU if it's on GPU
            if self.device == 'cuda':
                self.model.to('cpu')
                
                # Clear CUDA cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            
            # Delete model attributes to help with garbage collection
            if hasattr(self.model, '_modules'):
                for module in list(self.model._modules.values()):
                    if hasattr(module, '_parameters'):
                        module._parameters.clear()
                    if hasattr(module, '_buffers'):
                        module._buffers.clear()
                    if hasattr(module, '_modules'):
                        module._modules.clear()
            
            # Delete the model
            del self.model
            self.model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Update device attribute
            self.device = 'cpu'


def create_embeddings_model(
    provider: str = "openai",
    model_name: str = None,
    batch_size: int = None,
    dimensions: Optional[int] = None,
    device: str = "cpu"
) -> Union[OpenAIEmbeddings, SentenceTransformerEmbeddings]:
    """
    Create an embeddings model based on the provider.
    
    Args:
        provider: Provider of the embeddings model ('openai' or 'sentence_transformers')
        model_name: Name of the model to use
        batch_size: Number of texts to embed in a single batch
        dimensions: Dimensionality of embeddings (for OpenAI only)
        device: Device to run the model on (for SentenceTransformers only)
        
    Returns:
        An instance of OpenAIEmbeddings or SentenceTransformerEmbeddings
    """
    if provider.lower() == "openai":
        # Default model for OpenAI
        if model_name is None:
            model_name = "text-embedding-3-small"
        
        # Default batch size for OpenAI
        if batch_size is None:
            batch_size = 100
            
        return OpenAIEmbeddings(
            model=model_name,
            batch_size=batch_size,
            dimensions=dimensions
        )
    
    elif provider.lower() == "sentence_transformers":
        # Default model for SentenceTransformers
        if model_name is None:
            model_name = "BAAI/bge-small-en-v1.5"
        
        # Default batch size for SentenceTransformers
        if batch_size is None:
            batch_size = 32
            
        return SentenceTransformerEmbeddings(
            model_name=model_name,
            batch_size=batch_size,
            device=device
        )
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Supported providers are 'openai' and 'sentence_transformers'."
        )