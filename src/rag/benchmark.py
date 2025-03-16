import os
import json
import time
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from datetime import datetime
from IPython.display import display, HTML

# Import project modules
from .chatbot import rag_with_openai
from .pipeline import load_existing_pipeline


def run_rag_benchmark(
    pipeline,
    benchmark_path: str,
    output_dir: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1000,
    temperature: float = 0.0,  # Use 0 for deterministic outputs
    top_k: int = 5,
    initial_retrieval_multiplier: int = 3,
    max_questions: Optional[int] = None
):
    """
    Run the RAG pipeline on benchmark questions and store results for later analysis.
    
    Args:
        pipeline: The RAG pipeline to evaluate
        benchmark_path: Path to the benchmark questions file
        output_dir: Directory to save results
        model: OpenAI model to use
        max_tokens: Maximum tokens for completion
        temperature: Temperature for sampling
        top_k: Number of chunks to retrieve
        initial_retrieval_multiplier: Multiplier for initial retrieval in hybrid search
        max_questions: Maximum number of questions to evaluate (None for all)
        
    Returns:
        List of result dictionaries
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load benchmark questions
    with open(benchmark_path, 'r') as f:
        benchmark_questions = json.load(f)
    
    # Limit the number of questions if specified
    if max_questions is not None:
        benchmark_questions = benchmark_questions[:max_questions]
    
    # Initialize results storage
    all_results = []
    
    # Process each question
    for i, question_item in enumerate(tqdm(benchmark_questions, desc="Processing questions")):
        question_id = question_item.get('id', f"q{i}")
        question = question_item.get('question', '')
        reference_answer = question_item.get('answer', '')
        
        # Skip if question or reference answer is missing
        if not question or not reference_answer:
            print(f"Skipping question {question_id}: Missing question or reference answer")
            continue
        
        try:
            # Generate answer using RAG pipeline
            start_time = time.time()
            rag_result = rag_with_openai(
                pipeline=pipeline,
                question=question,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                initial_retrieval_multiplier=initial_retrieval_multiplier,
                debug_mode=False
            )
            end_time = time.time()
            
            # Extract the model's answer
            model_answer = rag_result.get('answer', '')
            
            # Compile result
            result = {
                'question_id': question_id,
                'question': question,
                'reference_answer': reference_answer,
                'model_answer': model_answer,
                'processing_time': end_time - start_time,
                'token_usage': {
                    'prompt_tokens': rag_result.get('prompt_tokens', 0),
                    'completion_tokens': rag_result.get('completion_tokens', 0),
                    'total_tokens': rag_result.get('total_tokens', 0)
                },
                'sources': rag_result.get('sources', []),
                'context': rag_result.get('context', '')
            }
            
            all_results.append(result)
            
            # Save individual result
            with open(os.path.join(output_dir, f"result_{question_id}.json"), 'w') as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"Error processing question {question_id}: {str(e)}")
            continue
    
    # Save summary information
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': model,
        'top_k': top_k,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'num_questions': len(all_results),
        'total_tokens_used': sum(r.get('token_usage', {}).get('total_tokens', 0) for r in all_results),
        'avg_processing_time': sum(r.get('processing_time', 0) for r in all_results) / len(all_results) if all_results else 0
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return all_results

def display_benchmark_results(results_dir: str, max_display: Optional[int] = None):
    """
    Display benchmark results in a nice format.
    
    Args:
        results_dir: Directory containing the results
        max_display: Maximum number of results to display (None for all)
    """
    # Load all result files
    result_files = [f for f in os.listdir(results_dir) if f.startswith('result_') and f.endswith('.json')]
    
    if max_display is not None:
        result_files = result_files[:max_display]
    
    all_results = []
    for file in result_files:
        with open(os.path.join(results_dir, file), 'r') as f:
            result = json.load(f)
            all_results.append(result)
    
    # Create a DataFrame for easier display
    df = pd.DataFrame([
        {
            'Question ID': r.get('question_id', ''),
            'Question': r.get('question', ''),
            'Reference Answer': r.get('reference_answer', ''),
            'Model Answer': r.get('model_answer', ''),
            'Processing Time (s)': round(r.get('processing_time', 0), 2),
            'Total Tokens': r.get('token_usage', {}).get('total_tokens', 0)
        }
        for r in all_results
    ])
    
    # Display summary
    with open(os.path.join(results_dir, "summary.json"), 'r') as f:
        summary = json.load(f)
    
    print(f"Benchmark Results Summary:")
    print(f"Model: {summary.get('model', '')}")
    print(f"Top-k: {summary.get('top_k', '')}")
    print(f"Number of questions: {summary.get('num_questions', 0)}")
    print(f"Total tokens used: {summary.get('total_tokens_used', 0)}")
    print(f"Average processing time: {summary.get('avg_processing_time', 0):.2f} seconds")
    print("\n")
    
    # Display each result in a nice format
    for i, result in enumerate(all_results):
        question = result.get('question', '')
        reference_answer = result.get('reference_answer', '').replace('\n', '<br>')
        model_answer = result.get('model_answer', '').replace('\n', '<br>')
        question_id = result.get('question_id', '')
        processing_time = result.get('processing_time', 0)
        total_tokens = result.get('token_usage', {}).get('total_tokens', 0)
        
        html = f"""
        <div style="margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
            <h3>Question {i+1}: {question_id}</h3>
            <div style="margin-bottom: 10px;"><strong>Question:</strong> {question}</div>
            
            <div style="display: flex; margin-bottom: 15px;">
                <div style="flex: 1; padding-right: 10px;">
                    <h4>Reference Answer:</h4>
                    <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
                        {reference_answer}
                    </div>
                </div>
                <div style="flex: 1; padding-left: 10px;">
                    <h4>Model Answer:</h4>
                    <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
                        {model_answer}
                    </div>
                </div>
            </div>
            
            <div><strong>Processing Time:</strong> {processing_time:.2f} seconds</div>
            <div><strong>Total Tokens:</strong> {total_tokens}</div>
        </div>
        """
        display(HTML(html))


def evaluate_best_pipeline(
    run_dir: str,
    config_name: str,
    benchmark_path: str,
    output_dir: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1000,
    temperature: float = 0.0,
    top_k: int = 5,
    initial_retrieval_multiplier: int = 3,
    max_questions: Optional[int] = None,
    print_example_prompt: bool = False
):
    """
    Evaluate the best RAG pipeline configuration on benchmark questions.
    
    Args:
        run_dir: Directory containing the pipeline
        config_name: Name of the pipeline configuration
        benchmark_path: Path to the benchmark questions file
        output_dir: Directory to save results
        model: OpenAI model to use
        max_tokens: Maximum tokens for completion
        temperature: Temperature for sampling
        top_k: Number of chunks to retrieve
        initial_retrieval_multiplier: Multiplier for initial retrieval in hybrid search
        max_questions: Maximum number of questions to evaluate (None for all)
        print_example_prompt: Whether to print an example prompt for the first question
    """
    # Load the pipeline
    print(f"Loading pipeline from {run_dir}, config: {config_name}")
    pipeline = load_existing_pipeline(run_dir, config_name)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"llm_eval_{config_name}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Running benchmark evaluation. Results will be saved to {output_path}")
    
    # If print_example_prompt is True, print an example prompt for the first question
    if print_example_prompt:
        # Load benchmark questions
        with open(benchmark_path, 'r') as f:
            benchmark_questions = json.load(f)
        
        if benchmark_questions:
            # Get the first question
            first_question = benchmark_questions[0].get('question', '')
            
            if first_question:
                # Retrieve chunks for the first question
                is_hybrid_search = hasattr(pipeline, 'config') and hasattr(pipeline.config, 'use_hybrid_search') and pipeline.config.use_hybrid_search
                initial_limit = top_k * initial_retrieval_multiplier if is_hybrid_search else top_k
                retrieved_chunks = pipeline.search(query=first_question, limit=initial_limit)
                
                # For hybrid search, sort and filter
                if is_hybrid_search and len(retrieved_chunks) > top_k:
                    retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x.get('score', 0.0), reverse=True)
                    retrieved_chunks = retrieved_chunks[:top_k]
                
                # Format context and prepare sources
                context_texts = []
                
                for i, chunk in enumerate(retrieved_chunks):
                    # Extract text and metadata
                    text = chunk.get("text", "")
                    source_id = chunk.get("meta_document_id", f"doc_{i}")
                    source_title = chunk.get("meta_document_title", "Unknown")
                    chunk_id = chunk.get("id", "")
                    
                    # Format the chunk with sequential reference numbers
                    formatted_chunk = f"[{i+1}] Document: {source_title} (ID: {source_id})\nChunk ID: {chunk_id}\n{text}\n"
                    context_texts.append(formatted_chunk)
                
                # If no chunks were retrieved, provide a message
                if not retrieved_chunks:
                    context_texts = ["No relevant documents were found in the database."]
                
                # Combine all context
                combined_context = "\n\n".join(context_texts)
                
                # Create system prompt
                system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Your task is to provide accurate, concise answers based solely on the information in the context.

Guidelines:
1. Only use information present in the provided context
2. If the answer cannot be determined from the context, say "I don't have enough information to answer this question" rather than making up information
3. Use numbered references to cite your sources, starting from [1] and continuing sequentially (e.g., [1], [2])
4. Only cite sources that you actually use in your answer - do not include unused references
5. If you cannot answer the question, do not include any references
6. Present financial data clearly, using tables where appropriate
7. Be concise and direct in your answer

Format your answer as follows:
[Your detailed answer with numbered citations like [1], [2], etc.]

References:
[1] Document title and ID
[2] Document title and ID
..."""
                
                # Create user message
                user_message = f"""Question: {first_question}

Context:
{combined_context}

Please provide a comprehensive answer based solely on the information in the context above. 
Use numbered references like [1], [2] to cite your sources, and include a "References:" section at the end listing all cited sources."""
                
                # Print the example prompt
                print("\n" + "="*80)
                print("EXAMPLE PROMPT FOR FIRST QUESTION")
                print("="*80)
                print("\nSystem Prompt:")
                print("-"*80)
                print(system_prompt)
                print("\nUser Message:")
                print("-"*80)
                print(user_message)
                print("="*80 + "\n")
    
    # Run evaluation
    results = run_rag_benchmark(
        pipeline=pipeline,
        benchmark_path=benchmark_path,
        output_dir=output_path,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        initial_retrieval_multiplier=initial_retrieval_multiplier,
        max_questions=max_questions
    )
    
    print(f"Evaluation completed. Results saved to {output_path}")
    print(f"Total questions processed: {len(results)}")
    
    # Return the output path for later display
    return output_path