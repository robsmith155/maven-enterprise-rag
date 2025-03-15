import gc
import json
import os
import time
import logging
import torch
import traceback
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
from typing import Dict, List, Any, Optional, Tuple, Iterator


# For reranking
try:
    from sentence_transformers import CrossEncoder
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

try:
    from tqdm.auto import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False
    tqdm = lambda x, **kwargs: x  # Fallback for when tqdm is not available

from .data_loader import DataLoader
from .chunking import create_chunker, create_chunks_from_text
from .embedding import create_embeddings_model
from .metadata import extract_metadata_filters
from .vectorstore import LanceDBStore
from .table_handler import process_tables, remove_html_tables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline."""
    # Data paths
    data_dir: str = "."
    reports_dir: str = "sec_processing_results_20250305_081032/reports"
    output_dir: str = "rag_output"
    
    # Chunking configuration
    chunker_type: str = "token"  # token, word, sentence, semantic
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Embedding configuration
    embedding_provider: str = "openai"  # openai, sentence_transformers
    embedding_model: str = "text-embedding-3-small"  # For OpenAI: text-embedding-3-small, text-embedding-3-large
                                                    # For SentenceTransformers: BAAI/bge-small-en-v1.5, BAAI/bge-large-en-v1.5
    embedding_batch_size: int = 100
    embedding_device: str = "cpu"  # cpu, cuda (for SentenceTransformers only)
    embedding_dimensions: Optional[int] = None  # Only for OpenAI
    
    # Vector store configuration
    vector_db_path: str = "vector_db"
    table_name: str = "sec_filings"
    
    # Processing options
    process_tables: bool = False
    remove_html_tables: bool = False  # Whether to remove HTML tables from content before embedding
    max_documents: Optional[int] = None  # None for all documents

    # Search options
    use_hybrid_search: bool = False  # Whether to use hybrid search with company name extraction
    use_reranker: bool = False  # Whether to use reranking
    reranker_model: Optional[str] = None  # Model to use for reranking
    reranker_provider: str = "sentence_transformers"  # Provider for reranking model
    reranker_top_k: Optional[int] = None  # Maximum number of documents to rerank, None = use default (limit * factor)
    rerank_all_results: bool = False  # If True, always rerank all results regardless of reranker_top_k

    @classmethod
    def from_json(cls, json_path: str) -> 'PipelineConfig':
        """Load configuration from a JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to a JSON file."""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for SEC filings.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Pipeline configuration
        """    
        self.config = config
        self.data_loader = DataLoader(config.data_dir)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(config.output_dir, "pipeline_config.json")
        config.to_json(config_path)
        
        # Initialize components
        self.chunker = create_chunker(
            config.chunker_type,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Create embeddings model based on provider
        self.embedder = create_embeddings_model(
            provider=config.embedding_provider,
            model_name=config.embedding_model,
            batch_size=config.embedding_batch_size,
            dimensions=config.embedding_dimensions,
            device=config.embedding_device
        )
        
        vector_db_path = os.path.join(config.output_dir, config.vector_db_path)
        self.vector_store = LanceDBStore(
            db_path=vector_db_path,
            table_name=config.table_name,
            embedding_dim=self.embedder.embedding_dim
        )
        
        # Initialize reranker if configured
        self.reranker = None
        if config.use_reranker and config.reranker_model and HAVE_SENTENCE_TRANSFORMERS:
            logger.info(f"Initializing reranker model: {config.reranker_model}")
            try:
                self.reranker = CrossEncoder(config.reranker_model, device=config.embedding_device)
                logger.info("Reranker initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize reranker: {e}")
                self.reranker = None

    def process_documents(self, verbose: bool = True) -> None:
        """
        Process all documents in the dataset.
        
        Args:
            verbose: Whether to output detailed logging information
        """
        if verbose:
            logger.info("Loading SEC filings...")
        sec_filings = self.data_loader.load_preprocessed_reports(self.config.reports_dir)
        
        if verbose:
            logger.info(f"Loaded {len(sec_filings)} companies")
        
        # Process documents
        all_chunks = []
        document_count = 0
        
        # Create progress bar for companies
        companies = list(sec_filings.items())
        company_iter = tqdm(companies, desc="Processing companies", disable=not verbose) if HAVE_TQDM else companies
        
        for ticker, years in company_iter:
            if verbose:
                logger.info(f"Processing company {ticker}")
            
            for year, sections in years.items():
                if verbose:
                    logger.info(f"  Processing year {year}")
                
                # Create progress bar for sections if there are many
                if HAVE_TQDM and len(sections) > 5 and verbose:
                    section_items = tqdm(sections.items(), desc=f"  Sections for {ticker} {year}", leave=False)
                else:
                    section_items = sections.items()
                
                # Try to get additional info from the report file
                report_file = os.path.join(self.config.reports_dir, f"report_{ticker}_{year}.json")
                accession_number = None
                original_file_path = None
                
                if os.path.exists(report_file):
                    try:
                        with open(report_file, 'r') as f:
                            report_data = json.load(f)
                            accession_number = report_data.get('accession_number')
                            original_file_path = report_data.get('file_path')
                    except Exception as e:
                        logger.warning(f"Error loading report file {report_file}: {e}")
                
                # Generate document ID and title
                document_id = f"{ticker}_{year}"
                if accession_number:
                    document_id = f"{document_id}_{accession_number}"
                
                document_title = f"{ticker} {year} SEC Filing"
                
                for section, content in section_items:
                    if verbose:
                        logger.info(f"    Processing section {section}")
                    
                    # Process tables if enabled
                    if self.config.process_tables:
                        content = process_tables(content)
                    
                    # Remove HTML tables if configured
                    if self.config.remove_html_tables:
                        content = remove_html_tables(content)
                    
                    # Create metadata with document identification
                    metadata = {
                        "ticker": ticker,
                        "year": year,
                        "section": section,
                        "document_id": document_id,
                        "document_title": document_title,
                        "processed_file_path": report_file
                    }
                    
                    # Add additional metadata if available
                    if accession_number:
                        metadata["accession_number"] = accession_number
                    if original_file_path:
                        metadata["original_file_path"] = original_file_path
                    
                    # Create chunks
                    chunks = create_chunks_from_text(content, metadata, self.chunker)
                    if verbose:
                        logger.info(f"    Created {len(chunks)} chunks")
                    
                    all_chunks.extend(chunks)
                    
                    document_count += 1
                    if self.config.max_documents and document_count >= self.config.max_documents:
                        logger.info(f"Reached maximum document count ({self.config.max_documents})")
                        break
                
                if self.config.max_documents and document_count >= self.config.max_documents:
                    break
            
            if self.config.max_documents and document_count >= self.config.max_documents:
                break
        
        if verbose:
            logger.info(f"Created a total of {len(all_chunks)} chunks")
        
        # Generate embeddings in batches
        if verbose:
            logger.info("Generating embeddings...")
        
        chunk_embeddings = {}
        total_batches = (len(all_chunks) - 1) // self.config.embedding_batch_size + 1
        
        # Create progress bar for embedding batches
        batch_range = range(0, len(all_chunks), self.config.embedding_batch_size)
        if HAVE_TQDM and verbose:
            batch_range = tqdm(batch_range, total=total_batches, desc="Generating embeddings")
        
        for i in batch_range:
            batch_chunks = all_chunks[i:i + self.config.embedding_batch_size]
            if verbose and not HAVE_TQDM:
                logger.info(f"Processing batch {i // self.config.embedding_batch_size + 1}/{total_batches}")
            
            batch_embeddings = self.embedder.embed_chunks(batch_chunks)
            chunk_embeddings.update(batch_embeddings)
        
        if verbose:
            logger.info(f"Generated embeddings for {len(chunk_embeddings)} chunks")
        
        # Store in vector database
        if verbose:
            logger.info("Storing in vector database...")
        self.vector_store.create_table(all_chunks, chunk_embeddings)
        
        if verbose:
            logger.info("Processing complete!")

    def update_reranker(self):
        """
        Update the reranker based on the current configuration.
        This method should be called whenever the reranking configuration changes.
        """
        # Reset the reranker
        self.reranker = None
    
        # Initialize reranker if configured
        if self.config.use_reranker and self.config.reranker_model and HAVE_SENTENCE_TRANSFORMERS:
            logger.info(f"Updating reranker model: {self.config.reranker_model}")
            try:
                self.reranker = CrossEncoder(self.config.reranker_model, device=self.config.embedding_device)
                logger.info("Reranker updated successfully")
            except Exception as e:
                logger.error(f"Failed to update reranker: {e}")
                self.reranker = None

    def cleanup(self):
        """
        Release both GPU and CPU memory by cleaning up embedding and reranking models.
        This performs a thorough cleanup to free all resources when the pipeline is no longer needed.
        """
        # Clean up embedder if it has a cleanup method
        if hasattr(self.embedder, 'cleanup') and callable(self.embedder.cleanup):
            logger.info("Cleaning up embedder model")
            self.embedder.cleanup()
        
        # Clean up reranker
        if self.reranker is not None:
            logger.info("Cleaning up reranker model")
            try:
                # Move reranker to CPU if it's on GPU
                if self.config.embedding_device == 'cuda' and hasattr(self.reranker, 'to'):
                    self.reranker.to('cpu')
                    
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("CUDA cache cleared")
                
                # Clean up CrossEncoder internal modules
                if hasattr(self.reranker, 'model') and hasattr(self.reranker.model, '_modules'):
                    for module in list(self.reranker.model._modules.values()):
                        if hasattr(module, '_parameters'):
                            module._parameters.clear()
                        if hasattr(module, '_buffers'):
                            module._buffers.clear()
                        if hasattr(module, '_modules'):
                            module._modules.clear()
                
                # Delete the reranker
                del self.reranker
                self.reranker = None
                
                # Force garbage collection
                gc.collect()
                
                logger.info("Reranker cleanup completed")
            except Exception as e:
                logger.error(f"Error cleaning up reranker: {e}")
        
        # Additional cleanup for other components if needed
        # This helps ensure thorough memory cleanup
        gc.collect()

    def search(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for documents related to a query.
        
        Args:
            query: Query string
            metadata_filters: Optional metadata filters
            limit: Maximum number of results to return
            
        Returns:
            List of search results
        """
        logger.info(f"Searching for: {query}")
        
        # Generate query embedding
        logger.info("Generating query embedding...")
        query_embedding = self.embedder.embed_query(query)
        
        # Retrieve more results if reranking will be applied
        search_factor = 3  # Factor to multiply limit by when reranking
        search_limit = limit * search_factor if self.config.use_reranker and self.reranker else limit
        
        # Perform search with detailed logging
        if self.config.use_hybrid_search:
            # Extract metadata filters from query if none provided
            if metadata_filters is None:
                metadata_filters = extract_metadata_filters(query)
                logger.info(f"Extracted metadata filters: {metadata_filters}")
            
            logger.info(f"Performing hybrid search with metadata filters: {metadata_filters}")
            results = self.vector_store.search_hybrid(query_embedding, metadata_filters, search_limit)
            logger.info(f"Found {len(results)} results with hybrid search")
        elif metadata_filters:
            logger.info(f"Performing search with metadata filters: {metadata_filters}")
            results = self.vector_store.search_hybrid(query_embedding, metadata_filters, search_limit)
            logger.info(f"Found {len(results)} results with metadata filtering")
        else:
            logger.info("Performing standard vector search")
            results = self.vector_store.search(query_embedding, limit=search_limit)
            logger.info(f"Found {len(results)} results")
        
        # Apply reranking with detailed logging
        if self.config.use_reranker and self.reranker and results:
            logger.info(f"Applying reranking with model: {self.config.reranker_model}")
            
            # Determine how many documents to rerank
            if self.config.reranker_top_k is not None:
                # Use specified reranker_top_k if provided
                rerank_limit = min(self.config.reranker_top_k, len(results))
                logger.info(f"Reranking top {rerank_limit} of {len(results)} results (reranker_top_k={self.config.reranker_top_k})")
            else:
                # Otherwise use all retrieved results
                rerank_limit = len(results)
                logger.info(f"Reranking all {rerank_limit} results (using default)")
            
            if rerank_limit < len(results):
                # Only rerank the top K documents
                to_rerank = results[:rerank_limit]
                remaining = results[rerank_limit:]
                
                # Prepare pairs for reranking
                pairs = [(query, result["text"]) for result in to_rerank]
                
                # Get reranking scores
                rerank_scores = self.reranker.predict(pairs)
                
                # Add scores to results
                for i, score in enumerate(rerank_scores):
                    to_rerank[i]["rerank_score"] = float(score)
                
                # Sort reranked results
                reranked_results = sorted(to_rerank, key=lambda x: x["rerank_score"], reverse=True)
                
                # Combine reranked results with remaining results
                # For remaining results, set a lower rerank score to ensure they come after reranked results
                if remaining:
                    min_rerank_score = min(float(score) for score in rerank_scores) if rerank_scores.size > 0 else 0
                    for result in remaining:
                        result["rerank_score"] = min_rerank_score - 0.1  # Ensure they come after reranked results
                
                # Combine and limit results
                results = reranked_results + remaining
            else:
                # Rerank all documents
                logger.info(f"Reranking all {len(results)} results...")
                
                # Prepare pairs for reranking
                pairs = [(query, result["text"]) for result in results]
                
                # Get reranking scores
                rerank_scores = self.reranker.predict(pairs)
                
                # Add scores to results
                for i, score in enumerate(rerank_scores):
                    results[i]["rerank_score"] = float(score)
                
                # Sort by reranking score
                results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            
            # Limit results
            results = results[:limit]
            
            logger.info(f"Reranked results, returning top {len(results)}")
        
        return results


def run_pipeline(config_path: Optional[str] = None) -> RAGPipeline:
    """
    Run the RAG pipeline.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Initialized RAG pipeline
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = PipelineConfig.from_json(config_path)
    else:
        config = PipelineConfig()
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    # Process documents
    pipeline.process_documents()
    
    return pipeline


def load_existing_pipeline(run_dir, config_name, debug_mode=True):
    """
    Load an existing pipeline from a previous evaluation run.
    
    Args:
        run_dir: Base run directory (e.g., "rag_evaluation_hybrid_rerank_v7/run_20250311_192132")
        config_name: Path to specific configuration (e.g., "late_chunk200_bge-large-en-v1.5_no_tables/no_reranking_hybrid")
        debug_mode: If True, print detailed debugging information
        
    Returns:
        Loaded RAGPipeline instance
    """
    # Find the pipeline directory for the specific configuration
    pipeline_dir = os.path.join(run_dir, config_name)
    if debug_mode:
        print(f"Looking for pipeline directory: {pipeline_dir}")
    
    if not os.path.exists(pipeline_dir):
        raise ValueError(f"Pipeline directory does not exist: {pipeline_dir}")
    
    # Extract the base chunker configuration from the config_name
    base_chunker_config = config_name.split('/')[0]  # Get the first part before any slash
    
    # Check for output directory where config is stored
    output_dir = os.path.join(pipeline_dir, "output")
    if debug_mode:
        print(f"Checking output directory: {output_dir}")
    
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    # Vector DB is in the "vectorstore" directory under the base chunker config directory
    vector_db_path = os.path.join(run_dir, base_chunker_config, "vectorstore")
    if debug_mode:
        print(f"Checking vectorstore path: {vector_db_path}")
    
    if not os.path.exists(vector_db_path):
        raise ValueError(f"Vectorstore directory does not exist: {vector_db_path}")
    
    # List contents of vectorstore directory
    if debug_mode:
        print(f"Vectorstore directory contents: {os.listdir(vector_db_path)}")
    
    # Load configuration from output directory
    config_path = os.path.join(output_dir, "pipeline_config.json")
    if debug_mode:
        print(f"Looking for config file: {config_path}")
    
    if not os.path.exists(config_path):
        raise ValueError(f"Config file does not exist: {config_path}")
    
    # Create pipeline configuration
    pipeline_config = PipelineConfig()
    
    # Load configuration data
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    if debug_mode:
        print(f"Loaded configuration data: {config_data}")
    
    # Manually set configuration attributes
    for key, value in config_data.items():
        if hasattr(pipeline_config, key):
            setattr(pipeline_config, key, value)
            if debug_mode:
                print(f"  Set {key} = {value}")
        else:
            if debug_mode:
                print(f"  Warning: PipelineConfig has no attribute '{key}'")
    
    # Update the configuration based on the config_name
    if "with_reranking" in config_name or "reranking" in config_name:
        pipeline_config.use_reranker = True
    else:
        pipeline_config.use_reranker = False
        
    if "hybrid" in config_name:
        pipeline_config.use_hybrid_search = True
    else:
        pipeline_config.use_hybrid_search = False
    
    # Ensure the vector_db path is set correctly to point to the shared vector store
    # Use relative path from output_dir to vector_db_path
    pipeline_config.vector_db_path = os.path.relpath(vector_db_path, output_dir)
    pipeline_config.output_dir = output_dir
    
    if debug_mode:
        print(f"Final configuration:")
        print(f"  vector_db_path: {pipeline_config.vector_db_path}")
        print(f"  output_dir: {pipeline_config.output_dir}")
        print(f"  table_name: {pipeline_config.table_name if hasattr(pipeline_config, 'table_name') else 'documents'}")
    
    # Create pipeline with the loaded configuration
    if debug_mode:
        print("Creating RAGPipeline with configuration...")
    pipeline = RAGPipeline(pipeline_config)
    
    # Manually initialize the table in the vector store
    try:
        # Connect to the database
        import lancedb
        if debug_mode:
            print(f"Connecting to LanceDB at {vector_db_path}")
        db = lancedb.connect(vector_db_path)
        
        # List all tables in the database
        table_names = db.table_names()
        if debug_mode:
            print(f"Available tables in the database: {table_names}")
        
        if not table_names:
            raise ValueError(f"No tables found in vectorstore at {vector_db_path}")
        
        # Check if the table exists
        table_name = getattr(pipeline_config, 'table_name', 'documents')
        if debug_mode:
            print(f"Looking for table: {table_name}")
            
        if table_name in table_names:
            # Set the table in the vector store
            if debug_mode:
                print(f"Opening table: {table_name}")
            table = db.open_table(table_name)
            
            # Check table contents
            if debug_mode:
                try:
                    row_count = table.count()
                    print(f"Table {table_name} contains {row_count} rows")
                    
                    if row_count > 0:
                        # Get schema
                        schema = table.schema()
                        print(f"Table schema: {schema}")
                        
                        # Get a sample row
                        sample = table.to_pandas().head(1)
                        print(f"Sample row columns: {sample.columns.tolist()}")
                except Exception as e:
                    print(f"Error checking table contents: {str(e)}")
            
            # Set the table in the vector store
            pipeline.vector_store.table = table
            if debug_mode:
                print(f"Successfully loaded existing table: {table_name}")
        else:
            # Try to find any table if the specified one doesn't exist
            alternative_table = table_names[0]
            if debug_mode:
                print(f"Table {table_name} not found, using alternative: {alternative_table}")
            table = db.open_table(alternative_table)
            pipeline.vector_store.table = table
            
            # Update the table name in the config if it has that attribute
            if hasattr(pipeline_config, 'table_name'):
                pipeline_config.table_name = alternative_table
    except Exception as e:
        if debug_mode:
            print(f"Error loading table: {str(e)}")
            traceback.print_exc()
        raise
    
    # Update reranker if needed
    if hasattr(pipeline_config, 'use_reranker') and pipeline_config.use_reranker:
        if debug_mode:
            print("Updating reranker...")
        if hasattr(pipeline, 'reranker') and pipeline.reranker is None and hasattr(pipeline_config, 'reranker_model') and pipeline_config.reranker_model:
            try:
                
                pipeline.reranker = CrossEncoder(pipeline_config.reranker_model, device=pipeline_config.embedding_device)
                if debug_mode:
                    print(f"Initialized reranker model: {pipeline_config.reranker_model}")
            except Exception as e:
                if debug_mode:
                    print(f"Failed to initialize reranker: {str(e)}")
    
    if debug_mode:
        print(f"Loaded pipeline from {pipeline_dir} with configuration from {config_name}")
    return pipeline