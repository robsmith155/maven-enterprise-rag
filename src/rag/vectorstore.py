
import os
import lancedb
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .chunking import Chunk
from .metadata import extract_metadata_filters


class LanceDBStore:
    """Vector store implementation using LanceDB."""
    
    def __init__(
        self,
        db_path: str,
        table_name: str = "sec_filings",
        embedding_dim: int = 1536  # Default for text-embedding-3-small
    ):
        """
        Initialize the LanceDB vector store.
        
        Args:
            db_path: Path to the LanceDB database
            table_name: Name of the table to store vectors
            embedding_dim: Dimensionality of the embeddings
        """
        self.db_path = db_path
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Connect to database
        self.db = lancedb.connect(db_path)
        self.table = None
    
    def create_table(self, chunks: List[Chunk], embeddings: Dict[str, List[float]]):
        """
        Create a table with chunks and embeddings.
        
        Args:
            chunks: List of Chunk objects
            embeddings: Dictionary mapping chunk IDs to embedding vectors
        """
        # Create DataFrame from chunks and embeddings
        data = []
        
        for chunk in chunks:
            if chunk.chunk_id not in embeddings:
                continue
            
            # Create row
            row = {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "vector": embeddings[chunk.chunk_id],
                **{f"meta_{k}": v for k, v in chunk.metadata.items()}
            }
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Create or replace table
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
        
        self.table = self.db.create_table(
            self.table_name,
            df,
            mode="create"
        )
        
        # Create vector index
        self.table.create_index(
            "cosine",
            num_partitions=16,
            num_sub_vectors=32
        )
    
    def add_chunks(self, chunks: List[Chunk], embeddings: Dict[str, List[float]]):
        """
        Add chunks and embeddings to the table.
        
        Args:
            chunks: List of Chunk objects
            embeddings: Dictionary mapping chunk IDs to embedding vectors
        """
        if self.table is None:
            self.create_table(chunks, embeddings)
            return
        
        # Create DataFrame from chunks and embeddings
        data = []
        
        for chunk in chunks:
            if chunk.chunk_id not in embeddings:
                continue
            
            # Create row
            row = {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "vector": embeddings[chunk.chunk_id],
                **{f"meta_{k}": v for k, v in chunk.metadata.items()}
            }
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Add to table
        self.table.add(df)
    
    def search(
        self,
        query_embedding: List[float],
        filter_expr: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            filter_expr: Filter expression for metadata filtering
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        if self.table is None:
            raise ValueError("Table not created yet")
        
        # Create query
        query = self.table.search(query_embedding).limit(limit)
        
        # Add filter if provided
        if filter_expr:
            query = query.where(filter_expr)
        
        # Execute query
        results = query.to_pandas()
        
        # Convert to list of dictionaries
        return results.to_dict(orient="records")
    
    def search_hybrid(
        self,
        query_embedding: List[float],
        metadata_filters: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with vector similarity and metadata filtering.
        
        Args:
            query_embedding: Query embedding vector
            metadata_filters: Dictionary of metadata filters
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        # Convert metadata filters to filter expression
        filter_conditions = []
        
        for key, value in metadata_filters.items():
            meta_key = f"meta_{key}"
            
            if isinstance(value, list):
                # Handle list of values (OR condition)
                if value:  # Only add if the list is not empty
                    if all(isinstance(v, str) for v in value):
                        # Format string values with quotes
                        values_str = "'" + "', '".join(value) + "'"
                        filter_conditions.append(f"{meta_key} IN ({values_str})")
                    else:
                        # Format numeric values without quotes
                        values_str = ", ".join(str(v) for v in value)
                        filter_conditions.append(f"{meta_key} IN ({values_str})")
            else:
                # Handle single value
                if isinstance(value, str):
                    filter_conditions.append(f"{meta_key} = '{value}'")
                else:
                    filter_conditions.append(f"{meta_key} = {value}")
        
        filter_expr = " AND ".join(filter_conditions) if filter_conditions else None
        
        # Perform search
        return self.search(query_embedding, filter_expr, limit)
    
    def extract_and_search_hybrid(
        self,
        query: str,
        query_embedding: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extract metadata filters from query and perform hybrid search.
        
        Args:
            query: Text query to extract metadata filters from
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        # Extract metadata filters from query
        metadata_filters = extract_metadata_filters(query)
        
        # Log extracted filters
        print(f"Extracted metadata filters: {metadata_filters}")
        
        # Perform hybrid search
        return self.search_hybrid(query_embedding, metadata_filters, limit)