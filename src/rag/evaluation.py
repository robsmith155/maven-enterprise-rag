import os
import json
import re
import time
import traceback
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import glob
import logging
import lancedb
from typing import Any, Dict, List, Optional

from .chunking import create_chunks_from_text
from .metrics import calculate_consecutive_recall, calculate_mrr_at_k
from .pipeline import PipelineConfig, RAGPipeline
from .plotting import generate_comparison_plots, plot_base_configuration_comparison, plot_consecutive_recall_results
from ..utils import remove_html_tables


def evaluate_rag_configurations(
    benchmark_path: str,
    reports_dir: str,
    output_base_dir: str = "rag_evaluation",
    chunking_methods: List[str] = ["token", "semantic", "late"],
    chunk_sizes: List[int] = [100, 200],
    embedding_models: List[str] = ["BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5"],
    reranking_options: List[bool] = [False, True],
    hybrid_search_options: List[bool] = [False, True],
    remove_tables_options: List[bool] = [False, True],
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_top_k: int = 50,  # New parameter: limit for reranking
    k_values: List[int] = [1, 3, 5, 10, 20, 50, 100, 200, 500],
    coverage_threshold: float = 0.5,
    max_questions: Optional[int] = None,
    skip_existing: bool = True,
    restart_from_existing: bool = False,
    verbose: bool = True
):
    """
    Evaluation of RAG pipeline configurations with different chunking methods,
    embedding models, chunk sizes, reranking options, and hybrid search options.
    
    Args:
        benchmark_path: Path to benchmark questions JSON file
        reports_dir: Directory containing the processed reports
        output_base_dir: Base directory to save evaluation results
        chunking_methods: List of chunking methods to test
        chunk_sizes: List of chunk sizes to test
        embedding_models: List of embedding models to test
        reranking_options: List of reranking options (True/False)
        hybrid_search_options: List of hybrid search options (True/False)
        reranker_model: Model to use for reranking when enabled
        reranker_top_k: Maximum number of documents to rerank (for efficiency)
        k_values: List of k values for Recall@k and MRR@k
        coverage_threshold: Minimum percentage of reference span that must be covered
        max_questions: Maximum number of questions to evaluate
        skip_existing: Skip configurations that have already been evaluated
        restart_from_existing: If True, try to restart from an existing run directory
        verbose: If True, show detailed progress information
        
    Returns:
        Dictionary with all evaluation results
    """
    
    # Temporarily reduce logging level for the RAG pipeline
    original_log_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create or find a timestamp for this evaluation run
    timestamp = None
    run_dir = None
    
    if restart_from_existing:
        # Look for existing run directories
        run_dirs = sorted(glob.glob(os.path.join(output_base_dir, "run_*")))
        if run_dirs:
            run_dir = run_dirs[-1]  # Use the most recent run
            timestamp = os.path.basename(run_dir).replace("run_", "")
            print(f"Restarting from existing run directory: {run_dir}")
    
    if run_dir is None:
        # Create a new run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_base_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"Created new run directory: {run_dir}")
    
    # Dictionary to store all results
    all_results = {}
    failed_configs = {}
    
    # Generate all configurations
    configs = []
    
    # First, identify all unique base configurations (chunking + embedding + tables)
    base_configs = {}
    
    for chunking_method in chunking_methods:
        for chunk_size in chunk_sizes:
            for embedding_model in embedding_models:
                for remove_tables in remove_tables_options:
                    # Create a unique key for this base configuration
                    embedding_model_name = embedding_model.split('/')[-1]
                    tables_str = "no_tables" if remove_tables else "with_tables"
                    base_key = f"{chunking_method}_chunk{chunk_size}_{embedding_model_name}_{tables_str}"
                    
                    # Create a directory name for this base configuration
                    base_dir = os.path.join(run_dir, base_key)
                    
                    # Store the base configuration
                    base_configs[base_key] = {
                        "key": base_key,
                        "dir": base_dir,
                        "chunking_method": chunking_method,
                        "chunk_size": chunk_size,
                        "embedding_model": embedding_model,
                        "remove_html_tables": remove_tables,
                        "vectorstore_dir": os.path.join(base_dir, "vectorstore")
                    }
                    
                    # Create the standard retrieval configuration (no reranking, no hybrid)
                    standard_config = {
                        "base_key": base_key,
                        "name": f"{base_key}_standard",
                        "chunking_method": chunking_method,
                        "chunk_size": chunk_size,
                        "embedding_model": embedding_model,
                        "use_reranking": False,
                        "use_hybrid_search": False,
                        "remove_html_tables": remove_tables,
                        "reranker_model": None,
                        "reranker_top_k": None,
                        "rerank_all_results": False
                    }
                    configs.append(standard_config)
                    
                    # Create all variant configurations
                    for use_reranking in reranking_options:
                        for use_hybrid_search in hybrid_search_options:
                            # Skip the standard configuration (already added)
                            if not use_reranking and not use_hybrid_search:
                                continue
                            
                            # Create a unique name for this configuration
                            variant_name = f"{base_key}_{'with' if use_reranking else 'no'}_reranking_{'hybrid' if use_hybrid_search else 'standard'}"
                            
                            # Add to configs list
                            configs.append({
                                "base_key": base_key,
                                "name": variant_name,
                                "chunking_method": chunking_method,
                                "chunk_size": chunk_size,
                                "embedding_model": embedding_model,
                                "use_reranking": use_reranking,
                                "use_hybrid_search": use_hybrid_search,
                                "remove_html_tables": remove_tables,
                                "reranker_model": reranker_model if use_reranking else None,
                                "reranker_top_k": reranker_top_k if use_reranking else None,
                                "rerank_all_results": use_reranking
                            })
    
    # Reorder configurations to process base configurations first
    # This ensures vectorstores are created before they're needed by other configs
    def is_standard_config(config):
        return not config["use_reranking"] and not config["use_hybrid_search"]
    
    # Sort configs so standard configurations come first
    configs.sort(key=lambda x: 0 if is_standard_config(x) else 1)
    
    # Log configuration summary
    if verbose:
        print(f"Evaluating {len(configs)} RAG pipeline configurations:")
        print(f"Base configurations: {len(base_configs)}")
        print(f"Total configurations: {len(configs)}")
        print("\nBase configurations:")
        for key, base_config in base_configs.items():
            print(f"- {key}")
        print("\nAll configurations:")
        for i, config in enumerate(configs):
            print(f"{i+1}. {config['name']}")
    
    # Save configuration details
    config_file = os.path.join(run_dir, "configurations.json")
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            json.dump({
                "chunking_methods": chunking_methods,
                "chunk_sizes": chunk_sizes,
                "embedding_models": embedding_models,
                "reranking_options": reranking_options,
                "hybrid_search_options": hybrid_search_options,
                "reranker_model": reranker_model,
                "reranker_top_k": reranker_top_k,
                "k_values": k_values,
                "coverage_threshold": coverage_threshold,
                "max_questions": max_questions,
                "timestamp": timestamp,
                "base_configurations": list(base_configs.keys())
            }, f, indent=2)
    
    # Create progress log file if it doesn't exist
    progress_log_path = os.path.join(run_dir, "progress_log.txt")
    if not os.path.exists(progress_log_path):
        with open(progress_log_path, "w") as f:
            f.write(f"Evaluation started at {timestamp}\n")
            f.write(f"Base configurations: {len(base_configs)}\n")
            f.write(f"Total configurations to evaluate: {len(configs)}\n\n")
    
    # Load existing results if restarting
    if restart_from_existing:
        # First check for results in the base directories
        for base_key, base_config in base_configs.items():
            for config_dir in glob.glob(os.path.join(base_config["dir"], "*")):
                if os.path.isdir(config_dir):
                    config_name = os.path.basename(config_dir)
                    results_path = os.path.join(config_dir, "results.json")
                    if os.path.exists(results_path):
                        try:
                            with open(results_path, "r") as f:
                                results = json.load(f)
                            all_results[config_name] = results
                            if verbose:
                                print(f"Loaded existing results for {config_name}")
                        except Exception as e:
                            if verbose:
                                print(f"Error loading existing results for {config_name}: {str(e)}")
    
    # Dictionary to track created vector stores
    vector_stores = {}
    
    # Evaluate each configuration
    config_iterator = tqdm(configs, desc="Evaluating configurations") if verbose else configs
    for config_idx, config in enumerate(config_iterator):
        config_start_time = time.time()
        
        # Get the base configuration
        base_key = config["base_key"]
        base_config = base_configs[base_key]
        
        # Create directory for this configuration
        base_dir = base_config["dir"]
        os.makedirs(base_dir, exist_ok=True)
        
        # Ensure vectorstore directory exists
        vectorstore_dir = base_config["vectorstore_dir"]
        os.makedirs(vectorstore_dir, exist_ok=True)
        
        # Determine the configuration directory
        if is_standard_config(config):
            # For standard configs, use the "standard" subdirectory
            config_dir = os.path.join(base_dir, "standard")
        else:
            # For variant configs, use a descriptive name
            variant_name = config["name"].replace(base_key + "_", "")
            config_dir = os.path.join(base_dir, variant_name)
        
        os.makedirs(config_dir, exist_ok=True)
        
        # Check if this configuration has already been evaluated
        results_path = os.path.join(config_dir, "results.json")
        if skip_existing and os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
                
                if verbose:
                    print(f"\nSkipping configuration {config['name']} (already evaluated)")
                all_results[config["name"]] = results
                # Get the base key for this configuration
                base_key = config["base_key"]
                base_dir = base_configs[base_key]["dir"]

                # Collect all results for this base configuration
                base_config_results = {}
                for name, res in all_results.items():
                    if name.startswith(base_key):
                        base_config_results[name] = res

                # If we have at least two configurations for this base, generate comparison plots
                if len(base_config_results) >= 2:
                    if verbose:
                        print(f"Generating comparison plots for base configuration: {base_key}")
                    
                    # Generate comparison plots for this base configuration
                    plot_base_configuration_comparison(
                        base_config_results=base_config_results,
                        k_values=k_values,
                        output_dir=base_dir,
                        base_key=base_key
                    )
                
                # Log progress
                with open(progress_log_path, "a") as f:
                    f.write(f"[{config_idx+1}/{len(configs)}] Skipped {config['name']} (already evaluated)\n")
                
                continue
            except Exception as e:
                if verbose:
                    print(f"Error loading existing results for {config['name']}, will re-evaluate: {str(e)}")
        
        if verbose:
            print(f"\nEvaluating configuration: {config['name']} ({config_idx+1}/{len(configs)})")
            print(f"Base configuration: {base_key}")
            print(f"Chunking method: {config['chunking_method']}")
            print(f"Chunk size: {config['chunk_size']}")
            print(f"Embedding model: {config['embedding_model']}")
            print(f"Reranking: {'Enabled' if config['use_reranking'] else 'Disabled'}")
            if config['use_reranking']:
                print(f"Reranker top-k: {config['reranker_top_k']}")
            print(f"Hybrid search: {'Enabled' if config['use_hybrid_search'] else 'Disabled'}")
        
        # Log progress
        with open(progress_log_path, "a") as f:
            f.write(f"[{config_idx+1}/{len(configs)}] Started {config['name']}\n")
        
        try:
            # Create output directory for this pipeline
            pipeline_output_dir = os.path.join(config_dir, "output")
            os.makedirs(pipeline_output_dir, exist_ok=True)
            
            # Create pipeline configuration
            pipeline_config = PipelineConfig(
                # Data paths
                data_dir=".",  # Not used directly in this context
                reports_dir=reports_dir,
                output_dir=pipeline_output_dir,
                
                # Chunking configuration
                chunker_type=config["chunking_method"],
                chunk_size=config["chunk_size"],
                chunk_overlap=50,  # Fixed overlap for all configurations
                
                # Embedding configuration
                embedding_model=config["embedding_model"],
                embedding_provider="sentence_transformers",
                embedding_batch_size=200,
                embedding_device="cuda",  # Adjust based on your hardware

                # Remove tables
                remove_html_tables=config["remove_html_tables"],
                
                # Reranking configuration
                use_reranker=config["use_reranking"],
                reranker_model=config["reranker_model"] if config["use_reranking"] else None,
                reranker_provider="sentence_transformers",
                reranker_top_k=config["reranker_top_k"] if config["use_reranking"] else None,
                rerank_all_results=config.get("rerank_all_results", False),
                
                # Hybrid search configuration
                use_hybrid_search=config["use_hybrid_search"]
            )
            
            # Save the configuration to the pipeline directory
            config_path = os.path.join(pipeline_output_dir, "pipeline_config.json")
            with open(config_path, "w") as f:
                # Use __dict__ to get all attributes, filter out private ones
                config_dict = {k: v for k, v in pipeline_config.__dict__.items() if not k.startswith('_')}
                json.dump(config_dict, f, indent=2)
            
            # Check if this is a standard configuration or if we need to reuse a vector store
            is_standard = is_standard_config(config)
            
            # Create pipeline with the configuration
            pipeline = RAGPipeline(pipeline_config)
            
            # Check if we've already created this vector store
            if base_key in vector_stores and not is_standard:
                # Reuse existing vector store
                if verbose:
                    print(f"Reusing existing vector store for {base_key}")
                
                # IMPORTANT FIX: Directly modify the vector store to use the correct path and database
                
                # Update the vector store path
                pipeline.vector_store.db_path = vectorstore_dir
                
                # Connect to the database at the correct location
                pipeline.vector_store.db = lancedb.connect(vectorstore_dir)
                
                # Load the table if it exists
                try:
                    if pipeline.vector_store.table_name in pipeline.vector_store.db.table_names():
                        pipeline.vector_store.table = pipeline.vector_store.db.open_table(pipeline.vector_store.table_name)
                        if verbose:
                            print(f"Successfully loaded existing table: {pipeline.vector_store.table_name}")
                    else:
                        raise ValueError(f"Table {pipeline.vector_store.table_name} not found in vectorstore")
                except Exception as e:
                    if verbose:
                        print(f"Error loading table: {e}")
                    # If we can't load the table, create a new vectorstore
                    is_standard = True
            
            if is_standard or base_key not in vector_stores:
                # Create new vector store
                if verbose:
                    print(f"Creating new vector store for {base_key}")
                
                # IMPORTANT FIX: Directly modify the vector store to use the correct path
                pipeline.vector_store.db_path = vectorstore_dir
                pipeline.vector_store.db = lancedb.connect(vectorstore_dir)
                
                # Process documents with minimal verbosity
                if verbose:
                    print("Processing documents...")
                    # Use a custom process_documents function with reduced verbosity
                    process_documents_with_progress(pipeline, verbose=True)
                else:
                    process_documents_with_progress(pipeline, verbose=False)
                
                # Store pipeline for reuse
                vector_stores[base_key] = pipeline
            
            # Update reranker and hybrid search settings if needed
            if not is_standard:
                # Update reranker if needed
                if config["use_reranking"] and hasattr(pipeline, 'update_reranker'):
                    pipeline.update_reranker()
                
                # Update hybrid search if needed
                if config["use_hybrid_search"] and hasattr(pipeline, 'update_hybrid_search'):
                    pipeline.update_hybrid_search()
            
            # Evaluate the pipeline
            if verbose:
                print("Evaluating pipeline...")
            results = evaluate_pipeline_with_consecutive_recall(
                pipeline=pipeline,
                benchmark_path=benchmark_path,
                k_values=k_values,
                max_questions=max_questions,
                coverage_threshold=coverage_threshold
            )
            
            # Save results
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            
            # Generate individual plots
            plot_consecutive_recall_results(
                results=results,
                output_path=os.path.join(config_dir, "evaluation_plots.png")
            )
            
            # Store results for comparison
            all_results[config["name"]] = results
            
            # Perform thorough memory cleanup after evaluation
            if verbose:
                print("Performing memory cleanup...")
                
            # Clean up pipeline resources if cleanup method exists
            if hasattr(pipeline, 'cleanup') and callable(pipeline.cleanup):
                if verbose:
                    print("Cleaning up pipeline resources...")
                pipeline.cleanup()
            
            # Manually clean up pipeline object
            if 'pipeline' in locals():
                # Set major components to None to help garbage collection
                if hasattr(pipeline, 'embedder'):
                    pipeline.embedder = None
                if hasattr(pipeline, 'reranker'):
                    pipeline.reranker = None
                if hasattr(pipeline, 'chunker'):
                    pipeline.chunker = None
                
                # Delete the pipeline object
                del pipeline
            
            # Force garbage collection multiple times to ensure thorough cleanup
            import gc
            gc.collect()
            gc.collect()  # Second collection can help with circular references
            
            # Clear CUDA cache again to be sure
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if verbose:
                        print("CUDA cache cleared")
                    
                    # Print memory stats if verbose
                    if verbose:
                        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                        print(f"GPU memory after cleanup: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")
            except ImportError:
                pass
            
            config_end_time = time.time()
            elapsed_time = config_end_time - config_start_time
            if verbose:
                print(f"Configuration {config['name']} evaluated in {elapsed_time:.2f} seconds")
            
            # Log progress
            with open(progress_log_path, "a") as f:
                f.write(f"[{config_idx+1}/{len(configs)}] Completed {config['name']} in {elapsed_time:.2f} seconds\n")
            
        except Exception as e:
            error_message = f"Error evaluating {config['name']}: {str(e)}"
            traceback_str = traceback.format_exc()
            if verbose:
                print(f"\n{error_message}")
                print(traceback_str)
            
            # Save error information
            failed_configs[config["name"]] = {
                "error": str(e),
                "traceback": traceback_str
            }
            
            # Save error to file
            with open(os.path.join(config_dir, "error.txt"), "w") as f:
                f.write(f"{error_message}\n\n")
                f.write(traceback_str)
            
            # Log progress
            with open(progress_log_path, "a") as f:
                f.write(f"[{config_idx+1}/{len(configs)}] FAILED {config['name']}: {str(e)}\n")
    
    # Save failed configurations
    if failed_configs:
        with open(os.path.join(run_dir, "failed_configurations.json"), "w") as f:
            json.dump(failed_configs, f, indent=2)
    
    # Generate comparison plots for successful configurations
    if all_results:
        generate_comparison_plots(
            all_results=all_results,
            k_values=k_values,
            output_dir=run_dir
        )
        
        # Generate summary CSV
        generate_summary_csv(
            all_results=all_results,
            k_values=k_values,
            output_path=os.path.join(run_dir, "summary.csv")
        )
    
    # Generate completion report
    completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(run_dir, "completion_report.txt"), "w") as f:
        f.write(f"Evaluation completed at: {completion_time}\n")
        f.write(f"Total configurations: {len(configs)}\n")
        f.write(f"Successful configurations: {len(all_results)}\n")
        f.write(f"Failed configurations: {len(failed_configs)}\n\n")
        
        if failed_configs:
            f.write("Failed configurations:\n")
            for config_name, error_info in failed_configs.items():
                f.write(f"- {config_name}: {error_info['error']}\n")
    
    if verbose:
        print(f"\nEvaluation complete. Results saved to {run_dir}")
        print(f"Successful configurations: {len(all_results)}/{len(configs)}")
        print(f"Failed configurations: {len(failed_configs)}/{len(configs)}")
    
    # Restore original logging level
    logging.getLogger().setLevel(original_log_level)
    
    return {
        "results": all_results,
        "failed": failed_configs,
        "output_dir": run_dir
    }

def process_documents_with_progress(pipeline, verbose=True):
    """
    Process documents with a simplified progress display.
    
    Args:
        pipeline: RAGPipeline instance
        verbose: Whether to show progress information
    """
    
    # Temporarily reduce logging level
    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(logging.WARNING)
    
    try:
        # Load documents
        if verbose:
            print("Loading documents...")
        sec_filings = pipeline.data_loader.load_preprocessed_reports(pipeline.config.reports_dir)
        
        if verbose:
            print(f"Loaded {len(sec_filings)} documents")
        
        # Process in batches with a single progress bar
        all_chunks = []
        
        if verbose:
            print("Chunking documents...")
        
        # Process documents with a single progress bar
        companies = list(sec_filings.items())
        for ticker, years in tqdm(companies, desc="Processing companies") if verbose else companies:
            for year, sections in years.items():
                # Generate document ID and title
                document_id = f"{ticker}_{year}"
                document_title = f"{ticker} {year} SEC Filing"
                
                for section, content in sections.items():
                    # Process tables if enabled
                    if hasattr(pipeline.config, 'process_tables') and pipeline.config.process_tables:
                      
                        content = process_tables(content)
                    
                    # Remove HTML tables if configured
                    if pipeline.config.remove_html_tables:
                      
                        content = remove_html_tables(content)
                    
                    # Create metadata with document identification
                    metadata = {
                        "ticker": ticker,
                        "year": year,
                        "section": section,
                        "document_id": document_id,
                        "document_title": document_title
                    }
                    
                    # Create chunks using the helper function
                    chunks = create_chunks_from_text(content, metadata, pipeline.chunker)
                    all_chunks.extend(chunks)
        
        if verbose:
            print(f"Created {len(all_chunks)} chunks")
            print("Generating embeddings...")
        
        # Generate embeddings with progress bar
        chunk_ids = [chunk.chunk_id for chunk in all_chunks]
        texts = [chunk.text for chunk in all_chunks]
        
        # Generate embeddings with a single progress bar
        embeddings = {}
        batch_size = pipeline.embedder.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_ids = chunk_ids[i:i+batch_size]
            
            if verbose:
                desc = f"Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
                print(f"\r{desc}", end="")
            
            # Use the correct method name: embed_texts instead of embed_documents
            batch_embeddings = pipeline.embedder.embed_texts(batch_texts)
            
            for chunk_id, embedding in zip(batch_ids, batch_embeddings):
                embeddings[chunk_id] = embedding
        
        if verbose:
            print("\nCreating vector store...")
        
        # Create vector store
        pipeline.vector_store.create_table(all_chunks, embeddings)
        
        if verbose:
            print("Vector store created successfully")
    
    finally:
        # Restore original logging level
        logger.setLevel(original_level)


def evaluate_pipeline_with_consecutive_recall(
    pipeline,
    benchmark_path: str,
    k_values: List[int] = [1, 3, 5, 10],
    max_questions: Optional[int] = None,
    coverage_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate the RAG pipeline using consecutive text coverage recall.
    """
    # Load benchmark questions
    with open(benchmark_path, 'r') as f:
        benchmark_data = json.load(f)
    
    # Limit number of questions if specified
    if max_questions is not None:
        benchmark_data = benchmark_data[:max_questions]
    
    # Results dictionary - use string keys for consistency
    results = {
        "consecutive_recall_at_k": {str(k): 0.0 for k in k_values},
        "mrr_at_k": {str(k): 0.0 for k in k_values},
        "question_results": {}
    }
    
    # Sort k_values to ensure we evaluate in ascending order
    sorted_k_values = sorted(k_values)
    max_k = max(sorted_k_values)
    
    # Process each question
    for question_data in tqdm(benchmark_data, desc="Evaluating questions"):
        question_id = question_data["id"]
        question_text = question_data["question"]
        
        # Get relevant span texts from source information
        relevant_spans = []
        for source in question_data["source_information"]:
            span_text = source["span_text"]
            if span_text:
                relevant_spans.append(span_text)
        
        # Search using your existing pipeline - retrieve maximum number of results once
        retrieved_docs = pipeline.search(question_text, limit=max_k)
        
        # Extract text from retrieved documents
        retrieved_texts = [doc.get("text", "") for doc in retrieved_docs]
        
        # Calculate metrics for each k value using subsets of the same result list
        question_results = {
            "question": question_text,
            "consecutive_recall_at_k": {},
            "mrr_at_k": {}
        }
        
        for k in sorted_k_values:
            # Get the top-k retrieved texts (subset of the same results)
            top_k_texts = retrieved_texts[:min(k, len(retrieved_texts))]
            
            # Calculate consecutive recall
            consecutive_recall = calculate_consecutive_recall(relevant_spans, top_k_texts, coverage_threshold)
            
            # Calculate MRR at k
            mrr_at_k = calculate_mrr_at_k(relevant_spans, retrieved_texts, coverage_threshold, k)
            
            # Store with string keys for consistency
            results["consecutive_recall_at_k"][str(k)] += consecutive_recall
            results["mrr_at_k"][str(k)] += mrr_at_k
            
            question_results["consecutive_recall_at_k"][str(k)] = consecutive_recall
            question_results["mrr_at_k"][str(k)] = mrr_at_k
        
        # Store results for this question
        results["question_results"][question_id] = question_results
    
    # Calculate averages
    num_questions = len(benchmark_data)
    if num_questions > 0:
        # Calculate averages for each k value
        for k in k_values:
            k_str = str(k)
            results["consecutive_recall_at_k"][k_str] /= num_questions
            results["mrr_at_k"][k_str] /= num_questions
        
        # Calculate overall average metrics
        results["average_consecutive_recall"] = sum(results["consecutive_recall_at_k"].values()) / len(k_values)
        results["average_mrr"] = sum(results["mrr_at_k"].values()) / len(k_values)
    else:
        # Handle the case of no questions
        results["average_consecutive_recall"] = 0.0
        results["average_mrr"] = 0.0
    
    return results


# def evaluate_pipeline_with_consecutive_recall(
#     pipeline,
#     benchmark_path: str,
#     k_values: List[int] = [1, 3, 5, 10],
#     max_questions: Optional[int] = None,
#     coverage_threshold: float = 0.5
# ) -> Dict[str, Any]:
#     """
#     Evaluate the RAG pipeline using consecutive text coverage recall.
    
#     Args:
#         pipeline: RAG pipeline to evaluate
#         benchmark_path: Path to benchmark questions JSON file
#         k_values: List of k values for Recall@k and MRR@k
#         max_questions: Maximum number of questions to evaluate
#         coverage_threshold: Minimum percentage of reference span that must be covered
        
#     Returns:
#         Evaluation results
#     """
    
#     # Load benchmark questions
#     with open(benchmark_path, 'r') as f:
#         benchmark_data = json.load(f)
    
#     # Limit number of questions if specified
#     if max_questions is not None:
#         benchmark_data = benchmark_data[:max_questions]
    
#     # Results dictionary
#     results = {
#         "consecutive_recall_at_k": {k: 0.0 for k in k_values},
#         "mrr_at_k": {k: 0.0 for k in k_values},
#         "question_results": {}
#     }
    
#     # Sort k_values to ensure we evaluate in ascending order
#     sorted_k_values = sorted(k_values)
#     max_k = max(sorted_k_values)
    
#     # Process each question
#     for question_data in tqdm(benchmark_data, desc="Evaluating questions"):
#         question_id = question_data["id"]
#         question_text = question_data["question"]
        
#         # Get relevant span texts from source information
#         relevant_spans = []
#         for source in question_data["source_information"]:
#             span_text = source["span_text"]
#             if span_text:
#                 relevant_spans.append(span_text)
        
#         # Search using your existing pipeline - retrieve maximum number of results once
#         retrieved_docs = pipeline.search(question_text, limit=max_k)
        
#         # Extract text from retrieved documents
#         retrieved_texts = [doc.get("text", "") for doc in retrieved_docs]
        
#         # Calculate metrics for each k value using subsets of the same result list
#         question_results = {
#             "question": question_text,
#             "consecutive_recall_at_k": {},
#             "mrr_at_k": {}
#         }
        
#         for k in sorted_k_values:
#             # Get the top-k retrieved texts (subset of the same results)
#             top_k_texts = retrieved_texts[:min(k, len(retrieved_texts))]
            
#             # Calculate consecutive recall
#             consecutive_recall = calculate_consecutive_recall(relevant_spans, top_k_texts, coverage_threshold)
            
#             # Calculate MRR at k
#             mrr_at_k = calculate_mrr_at_k(relevant_spans, retrieved_texts, coverage_threshold, k)
            
#             results["consecutive_recall_at_k"][k] += consecutive_recall
#             results["mrr_at_k"][k] += mrr_at_k
            
#             question_results["consecutive_recall_at_k"][k] = consecutive_recall
#             question_results["mrr_at_k"][k] = mrr_at_k
        
#         # Store results for this question
#         results["question_results"][question_id] = question_results
    
#     # Calculate averages
#     num_questions = len(benchmark_data)
#     if num_questions > 0:
#         for k in k_values:
#             results["consecutive_recall_at_k"][k] /= num_questions
#             results["mrr_at_k"][k] /= num_questions
    
#     return results

# def evaluate_pipeline_with_consecutive_recall(
#     pipeline,
#     benchmark_path: str,
#     k_values: List[int] = [1, 3, 5, 10],
#     max_questions: Optional[int] = None,
#     coverage_threshold: float = 0.5
# ) -> Dict[str, Any]:
#     """
#     Evaluate the RAG pipeline using consecutive text coverage recall.
    
#     Args:
#         pipeline: RAG pipeline to evaluate
#         benchmark_path: Path to benchmark questions JSON file
#         k_values: List of k values for Recall@k and MRR@k
#         max_questions: Maximum number of questions to evaluate
#         coverage_threshold: Minimum percentage of reference span that must be covered
        
#     Returns:
#         Evaluation results
#     """
    
#     # Load benchmark questions
#     with open(benchmark_path, 'r') as f:
#         benchmark_data = json.load(f)
    
#     # Limit number of questions if specified
#     if max_questions is not None:
#         benchmark_data = benchmark_data[:max_questions]
    
#     # Results dictionary
#     results = {
#         "consecutive_recall_at_k": {k: 0.0 for k in k_values},
#         "mrr_at_k": {k: 0.0 for k in k_values},
#         "question_results": {}
#     }
    
#     # Process each question
#     for question_data in tqdm(benchmark_data, desc="Evaluating questions"):
#         question_id = question_data["id"]
#         question_text = question_data["question"]
        
#         # Get relevant span texts from source information
#         relevant_spans = []
#         for source in question_data["source_information"]:
#             span_text = source["span_text"]
#             if span_text:
#                 relevant_spans.append(span_text)
        
#         # Search using your existing pipeline
#         max_k = max(k_values)
#         retrieved_docs = pipeline.search(question_text, limit=max_k)
        
#         # Extract text from retrieved documents
#         retrieved_texts = [doc.get("text", "") for doc in retrieved_docs]
        
#         # Calculate metrics for each k value
#         question_results = {
#             "question": question_text,
#             "consecutive_recall_at_k": {},
#             "mrr_at_k": {}
#         }
        
#         for k in k_values:
#             if k <= len(retrieved_docs):
#                 # Get the top-k retrieved texts
#                 top_k_texts = retrieved_texts[:k]
                
#                 # Calculate consecutive recall
#                 consecutive_recall = calculate_consecutive_recall(relevant_spans, top_k_texts, coverage_threshold)
                
#                 # Calculate MRR at k
#                 mrr_at_k = calculate_mrr_at_k(relevant_spans, retrieved_texts, coverage_threshold, k)
                
#                 results["consecutive_recall_at_k"][k] += consecutive_recall
#                 results["mrr_at_k"][k] += mrr_at_k
                
#                 question_results["consecutive_recall_at_k"][k] = consecutive_recall
#                 question_results["mrr_at_k"][k] = mrr_at_k
#             else:
#                 results["consecutive_recall_at_k"][k] += 0.0
#                 results["mrr_at_k"][k] += 0.0
                
#                 question_results["consecutive_recall_at_k"][k] = 0.0
#                 question_results["mrr_at_k"][k] = 0.0
        
#         # Store results for this question
#         results["question_results"][question_id] = question_results
    
#     # Calculate averages
#     num_questions = len(benchmark_data)
#     if num_questions > 0:
#         for k in k_values:
#             results["consecutive_recall_at_k"][k] /= num_questions
#             results["mrr_at_k"][k] /= num_questions
    
#     return results


def generate_summary_csv(all_results, k_values, output_path):
    """
    Generate a summary CSV with all evaluation results.
    
    Args:
        all_results: Dictionary with results for all configurations
        k_values: List of k values for Recall@k and MRR@k
        output_path: Path to save the CSV file
    """
    
    # Create a list to store rows
    rows = []
    
    for config_name, results in all_results.items():
        # Parse configuration name to extract components
        components = {}
        
        # Extract chunking method
        chunking_match = re.search(r'(token|semantic|late)', config_name)
        if chunking_match:
            components['chunking_method'] = chunking_match.group(1)
        else:
            components['chunking_method'] = "unknown"
        
        # Extract chunk size
        size_match = re.search(r'chunk(\d+)', config_name)
        if size_match:
            components['chunk_size'] = int(size_match.group(1))
        else:
            components['chunk_size'] = 0
        
        # Extract embedding model
        embedding_match = re.search(r'(bge-small-en-v1\.5|bge-large-en-v1\.5|bge-base-en-v1\.5|e5-small-v2)', config_name)
        if embedding_match:
            components['embedding_model'] = embedding_match.group(1)
        else:
            components['embedding_model'] = "unknown"
        
        # Extract tables option
        components['with_tables'] = 'with_tables' in config_name
        
        # Extract reranking
        components['reranking'] = 'with_reranking' in config_name
        
        # Extract hybrid search
        components['hybrid_search'] = 'hybrid' in config_name
        
        # Create row with configuration details
        row = {
            'config_name': config_name,
            'chunking_method': components['chunking_method'],
            'chunk_size': components['chunk_size'],
            'embedding_model': components['embedding_model'],
            'with_tables': components['with_tables'],
            'reranking': components['reranking'],
            'hybrid_search': components['hybrid_search']
        }
        
        # Add metrics for each k value
        for k in k_values:
            k_str = str(k)
            
            # Add consecutive recall@k
            try:
                if "consecutive_recall_at_k" in results and k_str in results["consecutive_recall_at_k"]:
                    row[f'recall@{k}'] = results["consecutive_recall_at_k"][k_str]
                else:
                    row[f'recall@{k}'] = 0
            except (KeyError, TypeError):
                row[f'recall@{k}'] = 0
            
            # Add MRR@k if available
            try:
                if "mrr_at_k" in results and k_str in results["mrr_at_k"]:
                    row[f'mrr@{k}'] = results["mrr_at_k"][k_str]
                else:
                    row[f'mrr@{k}'] = 0
            except (KeyError, TypeError):
                row[f'mrr@{k}'] = 0
        
        # Add average metrics if available
        try:
            if "average_consecutive_recall" in results:
                row['avg_recall'] = results["average_consecutive_recall"]
            else:
                row['avg_recall'] = 0
        except (KeyError, TypeError):
            row['avg_recall'] = 0
            
        try:
            if "average_mrr" in results:
                row['avg_mrr'] = results["average_mrr"]
            else:
                row['avg_mrr'] = 0
        except (KeyError, TypeError):
            row['avg_mrr'] = 0
        
        # Add to rows
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by average recall (descending)
    df = df.sort_values('avg_recall', ascending=False)
    
    # Save to CSV
    df.to_csv(output_path, index=False)