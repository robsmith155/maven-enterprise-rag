# metrics_debug.py
import json
import os
import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional
import pandas as pd
from colorama import Fore, Style, init

from .metrics import normalize_text, get_consecutive_coverage, calculate_consecutive_recall

# Initialize colorama for colored terminal output
init()

def debug_text_normalization(text1, text2):
    """
    Debug text normalization by showing before and after normalization for two texts.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Dictionary with normalization details
    """
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    # Calculate similarity before and after normalization
    raw_similarity = SequenceMatcher(None, text1, text2).ratio()
    norm_similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    return {
        "original_text1": text1,
        "original_text2": text2,
        "normalized_text1": norm1,
        "normalized_text2": norm2,
        "original_similarity": raw_similarity,
        "normalized_similarity": norm_similarity,
        "improvement": norm_similarity - raw_similarity
    }

def visualize_text_differences(text1, text2, normalized=False):
    """
    Visualize differences between two texts with color highlighting.
    
    Args:
        text1: First text
        text2: Second text
        normalized: Whether to show normalized texts
    """
    if normalized:
        text1 = normalize_text(text1)
        text2 = normalize_text(text2)
    
    matcher = SequenceMatcher(None, text1, text2)
    
    # Build output for text1
    output1 = []
    output2 = []
    
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            output1.append(text1[i1:i2])
            output2.append(text2[j1:j2])
        elif op == 'delete':
            output1.append(Fore.RED + text1[i1:i2] + Style.RESET_ALL)
            output2.append('')
        elif op == 'insert':
            output1.append('')
            output2.append(Fore.GREEN + text2[j1:j2] + Style.RESET_ALL)
        elif op == 'replace':
            output1.append(Fore.RED + text1[i1:i2] + Style.RESET_ALL)
            output2.append(Fore.GREEN + text2[j1:j2] + Style.RESET_ALL)
    
    print("Text 1:", ''.join(output1))
    print("Text 2:", ''.join(output2))


def debug_consecutive_coverage(reference, retrieved):
    """
    Debug consecutive coverage calculation with detailed steps.
    
    Args:
        reference: Reference text
        retrieved: Retrieved text
        
    Returns:
        Dictionary with coverage details
    """
    # Original texts
    orig_ref = reference
    orig_ret = retrieved
    
    # Normalized texts
    norm_ref = normalize_text(reference)
    norm_ret = normalize_text(retrieved)
    
    # Calculate coverage using the main function
    coverage = get_consecutive_coverage(reference, retrieved)
    
    # Check for direct containment
    contained = norm_ref in norm_ret
    
    # Extract the matched text
    if contained:
        # If directly contained, the entire reference is the matched text
        matched_text = norm_ref
        match_size = len(norm_ref)
    else:
        # Otherwise use SequenceMatcher to find the longest common substring
        matcher = SequenceMatcher(None, norm_ref, norm_ret)
        match = matcher.find_longest_match(0, len(norm_ref), 0, len(norm_ret))
        
        if match.size > 0:
            matched_text = norm_ref[match.a:match.a + match.size]
            match_size = match.size
        else:
            matched_text = ""
            match_size = 0
    
    return {
        "original_reference": orig_ref,
        "original_retrieved": orig_ret,
        "normalized_reference": norm_ref,
        "normalized_retrieved": norm_ret,
        "coverage": coverage,
        "direct_containment": contained,
        "match_size": match_size,
        "reference_length": len(norm_ref),
        "matched_text": matched_text
    }


def debug_consecutive_recall(reference_spans, retrieved_texts, threshold=0.5):
    """
    Debug consecutive recall calculation with detailed information for each span.
    
    Args:
        reference_spans: List of reference text spans
        retrieved_texts: List of retrieved text chunks
        threshold: Coverage threshold
        
    Returns:
        Dictionary with detailed recall information
    """
    # Skip empty references
    valid_refs = [ref for ref in reference_spans if ref and ref.strip()]
    if not valid_refs:
        return {"recall": 0.0, "spans_covered": 0, "total_spans": 0, "threshold": threshold, "span_details": []}
    
    # Use the main function to calculate recall
    recall = calculate_consecutive_recall(reference_spans, retrieved_texts, threshold)
    
    # Count how many reference spans are sufficiently covered
    covered_spans = 0
    span_details = []
    
    for i, ref in enumerate(valid_refs):
        # Find the best coverage among all retrieved texts
        best_coverage = 0.0
        best_match_index = -1
        coverage_details = None
        
        for j, ret in enumerate(retrieved_texts):
            # Use the main function to calculate coverage
            coverage = get_consecutive_coverage(ref, ret)
            coverage_info = debug_consecutive_coverage(ref, ret)
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_match_index = j
                coverage_details = coverage_info
        
        # Check if this span is covered
        is_covered = best_coverage >= threshold
        if is_covered:
            covered_spans += 1
        
        # Add span details
        span_details.append({
            "span_index": i,
            "best_coverage": best_coverage,
            "best_match_index": best_match_index,
            "is_covered": is_covered,
            "coverage_details": coverage_details
        })
    
    return {
        "overall_recall": recall,
        "spans_covered": covered_spans,
        "total_spans": len(valid_refs),
        "threshold": threshold,
        "span_details": span_details
    }


def create_test_dataset(output_path="data/benchmark/test_consecutive_recall.json", 
                       source_benchmark="data/benchmark/benchmark_questions_custom2.json",
                       sample_size=None):
    """
    Create a test dataset for evaluating consecutive recall with real examples from the benchmark.
    
    Args:
        output_path: Path to save the test dataset
        source_benchmark: Path to the source benchmark file
        sample_size: Optional, number of questions to sample (None = use all)
        
    Returns:
        Test dataset
    """
    # Load the source benchmark file
    try:
        with open(source_benchmark, 'r') as f:
            benchmark_data = json.load(f)
        
        print(f"Loaded {len(benchmark_data)} questions from {source_benchmark}")
    except Exception as e:
        print(f"Error loading benchmark file: {e}")
        print("Creating synthetic test data instead")
        return _create_synthetic_test_dataset(output_path)
    
    # Sample questions if requested
    if sample_size is not None and sample_size < len(benchmark_data):
        import random
        random.shuffle(benchmark_data)
        benchmark_data = benchmark_data[:sample_size]
        print(f"Sampled {sample_size} questions from benchmark")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save test dataset
    with open(output_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"Test dataset created at {output_path}")
    return benchmark_data

def _create_synthetic_test_dataset(output_path="data/benchmark/test_consecutive_recall.json"):
    """
    Create a synthetic test dataset for evaluating consecutive recall with known examples.
    Each question includes multiple reference spans to better match the real benchmark structure.
    
    Args:
        output_path: Path to save the test dataset
        
    Returns:
        Test dataset
    """
    test_data = [
        {
            "id": "test_1",
            "question": "What was Apple's share repurchase program in 2020 and how many shares did they repurchase?",
            "answer": "As of September 26, 2020, Apple was authorized to purchase up to $225 billion of common stock, of which $168.6 billion had been utilized. During 2020, they repurchased 917 million shares for $72.5 billion, including 141 million shares under a $10.0 billion November 2019 ASR and 64 million shares under a $6.0 billion May 2020 ASR.",
            "source_information": [
                {
                    "company": "AAPL",
                    "ticker": "AAPL",
                    "year": "2020",
                    "section": "Item 7",
                    "subsection": None,
                    "span_text": "As of September 26,2020, the Company was authorized to purchase up to $225 billion of the Company\u2019s common stock under a share repurchase program, of which $168.6 billion had been utilized.",
                    "span_location": {
                        "document_id": "SEC_10-K_2020_AAPL",
                        "start_offset": 0,
                        "end_offset": 175
                    },
                    "contains_table": False,
                    "processed_path": "./data/processed/reports/report_AAPL_2020.json"
                },
                {
                    "company": "AAPL",
                    "ticker": "AAPL",
                    "year": "2020",
                    "section": "Item 7",
                    "subsection": None,
                    "span_text": "During 2020, the Company repurchased 917 million shares of its common stock for $72.5 billion, including 141 million shares delivered under a $10.0 billion November 2019 ASR and 64 million shares delivered under a $6.0 billion May 2020 ASR",
                    "span_location": {
                        "document_id": "SEC_10-K_2020_AAPL",
                        "start_offset": 175,
                        "end_offset": 400
                    },
                    "contains_table": False,
                    "processed_path": "./data/processed/reports/report_AAPL_2020.json"
                }
            ],
            "question_type": "single_company_single_year",
            "difficulty": "medium"
        },
        {
            "id": "test_2",
            "question": "What were Microsoft's revenue and operating income in 2022?",
            "answer": "Microsoft reported revenue of $198.3 billion for the fiscal year 2022, an increase of 18% compared to 2021. Their operating income was $83.4 billion, an increase of 19% compared to the prior year.",
            "source_information": [
                {
                    "company": "MSFT",
                    "ticker": "MSFT",
                    "year": "2022",
                    "section": "Item 7",
                    "subsection": None,
                    "span_text": "Revenue was $198.3 billion for fiscal year 2022, an increase of 18% compared to fiscal year 2021.",
                    "span_location": {
                        "document_id": "SEC_10-K_2022_MSFT",
                        "start_offset": 0,
                        "end_offset": 85
                    },
                    "contains_table": False,
                    "processed_path": "./data/processed/reports/report_MSFT_2022.json"
                },
                {
                    "company": "MSFT",
                    "ticker": "MSFT",
                    "year": "2022",
                    "section": "Item 7",
                    "subsection": None,
                    "span_text": "Operating income was $83.4 billion for fiscal year 2022, an increase of 19% compared to fiscal year 2021.",
                    "span_location": {
                        "document_id": "SEC_10-K_2022_MSFT",
                        "start_offset": 85,
                        "end_offset": 170
                    },
                    "contains_table": False,
                    "processed_path": "./data/processed/reports/report_MSFT_2022.json"
                }
            ],
            "question_type": "single_company_single_year",
            "difficulty": "easy"
        },
        {
            "id": "test_3",
            "question": "What were Amazon's operating expenses and net income in 2021?",
            "answer": "Amazon's operating expenses were $445.8 billion in 2021, including $42.7 billion for fulfillment, $33.7 billion for technology and content, $32.6 billion for sales and marketing, and $8.8 billion for general and administrative expenses. Their net income was $33.4 billion in 2021, an increase from $21.3 billion in 2020.",
            "source_information": [
                {
                    "company": "AMZN",
                    "ticker": "AMZN",
                    "year": "2021",
                    "section": "Item 7",
                    "subsection": None,
                    "span_text": "Operating expenses were $445.8 billion in 2021, including $42.7 billion for fulfillment, $33.7 billion for technology and content, $32.6 billion for sales and marketing, and $8.8 billion for general and administrative expenses.",
                    "span_location": {
                        "document_id": "SEC_10-K_2021_AMZN",
                        "start_offset": 0,
                        "end_offset": 200
                    },
                    "contains_table": False,
                    "processed_path": "./data/processed/reports/report_AMZN_2021.json"
                },
                {
                    "company": "AMZN",
                    "ticker": "AMZN",
                    "year": "2021",
                    "section": "Item 7",
                    "subsection": None,
                    "span_text": "Net income increased to $33.4 billion in 2021, compared to $21.3 billion in 2020.",
                    "span_location": {
                        "document_id": "SEC_10-K_2021_AMZN",
                        "start_offset": 200,
                        "end_offset": 275
                    },
                    "contains_table": False,
                    "processed_path": "./data/processed/reports/report_AMZN_2021.json"
                }
            ],
            "question_type": "single_company_single_year",
            "difficulty": "medium"
        },
        {
            "id": "test_4",
            "question": "How did Google's revenue and costs change from 2020 to 2021?",
            "answer": "Google's revenue increased by 41% from 2020 to 2021, reaching $257.6 billion. Their costs and expenses increased by 27%, with R&D expenses growing by 28% to $31.6 billion, sales and marketing expenses increasing by 22% to $22.9 billion, and general and administrative expenses increasing by 33% to $13.0 billion.",
            "source_information": [
                {
                    "company": "GOOGL",
                    "ticker": "GOOGL",
                    "year": "2021",
                    "section": "Item 7",
                    "subsection": None,
                    "span_text": "For the fiscal year ended December 31, 2021, our revenues increased $75.3 billion, or 41%, from 2020 to $257.6 billion.",
                    "span_location": {
                        "document_id": "SEC_10-K_2021_GOOGL",
                        "start_offset": 0,
                        "end_offset": 100
                    },
                    "contains_table": False,
                    "processed_path": "./data/processed/reports/report_GOOGL_2021.json"
                },
                {
                    "company": "GOOGL",
                    "ticker": "GOOGL",
                    "year": "2021",
                    "section": "Item 7",
                    "subsection": None,
                    "span_text": "For the fiscal year ended December 31, 2021, our costs and expenses increased $41.4 billion, or 27%, from 2020.",
                    "span_location": {
                        "document_id": "SEC_10-K_2021_GOOGL",
                        "start_offset": 100,
                        "end_offset": 200
                    },
                    "contains_table": False,
                    "processed_path": "./data/processed/reports/report_GOOGL_2021.json"
                },
                {
                    "company": "GOOGL",
                    "ticker": "GOOGL",
                    "year": "2021",
                    "section": "Item 7",
                    "subsection": None,
                    "span_text": "Research and development (R&D) expenses increased $6.9 billion, or 28%, from 2020 to $31.6 billion, primarily due to an increase in compensation expenses of $5.4 billion, largely resulting from a 20% increase in headcount.",
                    "span_location": {
                        "document_id": "SEC_10-K_2021_GOOGL",
                        "start_offset": 200,
                        "end_offset": 300
                    },
                    "contains_table": False,
                    "processed_path": "./data/processed/reports/report_GOOGL_2021.json"
                },
                {
                    "company": "GOOGL",
                    "ticker": "GOOGL",
                    "year": "2021",
                    "section": "Item 7",
                    "subsection": None,
                    "span_text": "Sales and marketing expenses increased $4.1 billion, or 22%, from 2020 to $22.9 billion, primarily due to an increase in compensation expenses of $2.4 billion, largely resulting from a 17% increase in headcount.",
                    "span_location": {
                        "document_id": "SEC_10-K_2021_GOOGL",
                        "start_offset": 300,
                        "end_offset": 400
                    },
                    "contains_table": False,
                    "processed_path": "./data/processed/reports/report_GOOGL_2021.json"
                },
                {
                    "company": "GOOGL",
                    "ticker": "GOOGL",
                    "year": "2021",
                    "section": "Item 7",
                    "subsection": None,
                    "span_text": "General and administrative expenses increased $3.2 billion, or 33%, from 2020 to $13.0 billion, primarily due to an increase in compensation expenses of $1.6 billion, largely resulting from a 17% increase in headcount.",
                    "span_location": {
                        "document_id": "SEC_10-K_2021_GOOGL",
                        "start_offset": 400,
                        "end_offset": 500
                    },
                    "contains_table": False,
                    "processed_path": "./data/processed/reports/report_GOOGL_2021.json"
                }
            ],
            "question_type": "single_company_multi_year",
            "difficulty": "hard"
        }
    ]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save test dataset
    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Synthetic test dataset created at {output_path}")
    return test_data


def run_debug_evaluation(pipeline, benchmark_path, k_values=[1, 3, 5, 10], max_questions=None, 
                         coverage_threshold=0.5, output_path="debug_recall_results.json"):
    """
    Run a debug evaluation of the RAG pipeline with detailed recall information.
    
    Args:
        pipeline: RAG pipeline
        benchmark_path: Path to benchmark questions
        k_values: List of k values for recall@k
        max_questions: Maximum number of questions to evaluate
        coverage_threshold: Coverage threshold
        output_path: Path to save debug results
        
    Returns:
        Detailed evaluation results
    """
    # Load benchmark questions
    with open(benchmark_path, 'r') as f:
        benchmark_data = json.load(f)
    
    # Limit number of questions if specified
    if max_questions is not None:
        benchmark_data = benchmark_data[:max_questions]
    
    # Sort k values
    sorted_k_values = sorted(k_values)
    max_k = max(sorted_k_values)
    
    # Results
    debug_results = []
    
    # Process each question
    for question_data in benchmark_data:
        question_id = question_data["id"]
        question_text = question_data["question"]
        
        # Get relevant span texts
        relevant_spans = []
        for source in question_data["source_information"]:
            span_text = source["span_text"]
            if span_text:
                relevant_spans.append(span_text)
        
        # Search using pipeline
        retrieved_docs = pipeline.search(question_text, limit=max_k)
        retrieved_texts = [doc.get("text", "") for doc in retrieved_docs]
        
        # Debug results for this question
        question_debug = {
            "question_id": question_id,
            "question": question_text,
            "relevant_spans": relevant_spans,
            "retrieved_texts": retrieved_texts,
            "recall_at_k": {}
        }
        
        # Calculate recall for each k
        for k in sorted_k_values:
            top_k_texts = retrieved_texts[:min(k, len(retrieved_texts))]
            
            # Debug consecutive recall
            recall_debug = debug_consecutive_recall(relevant_spans, top_k_texts, coverage_threshold)
            question_debug["recall_at_k"][k] = recall_debug
        
        debug_results.append(question_debug)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(debug_results, f, indent=2)
    
    print(f"Debug evaluation results saved to {output_path}")
    return debug_results


def analyze_debug_results(debug_results_path):
    """
    Analyze debug results and provide insights for improvement.
    
    Args:
        debug_results_path: Path to debug results JSON
        
    Returns:
        Analysis report
    """
    # Load debug results
    with open(debug_results_path, 'r') as f:
        debug_results = json.load(f)
    
    # Overall statistics
    total_questions = len(debug_results)
    total_spans = sum(len(q["relevant_spans"]) for q in debug_results)
    
    # Analyze recall at different k values
    k_values = sorted([int(k) for k in debug_results[0]["recall_at_k"].keys()])
    recall_by_k = {k: [] for k in k_values}
    
    for question in debug_results:
        for k in k_values:
            recall = question["recall_at_k"][str(k)]["overall_recall"]
            recall_by_k[k].append(recall)
    
    avg_recall_by_k = {k: sum(recalls) / len(recalls) for k, recalls in recall_by_k.items()}
    
    # Analyze span coverage
    span_coverage = []
    for question in debug_results:
        max_k = max(k_values)
        for span_detail in question["recall_at_k"][str(max_k)]["span_details"]:
            span_coverage.append({
                "question_id": question["question_id"],
                "span_index": span_detail["span_index"],
                "best_coverage": span_detail["best_coverage"],
                "is_covered": span_detail["is_covered"]
            })
    
    # Convert to DataFrame for analysis
    df_coverage = pd.DataFrame(span_coverage)
    
    # Identify problematic spans (low coverage)
    problematic_spans = df_coverage[df_coverage["best_coverage"] < 0.3]
    
    # Generate report
    report = {
        "total_questions": total_questions,
        "total_spans": total_spans,
        "average_recall_by_k": avg_recall_by_k,
        "spans_covered_percentage": (df_coverage["is_covered"].sum() / len(df_coverage)) * 100,
        "problematic_spans_count": len(problematic_spans),
        "problematic_spans_percentage": (len(problematic_spans) / len(df_coverage)) * 100,
        "coverage_distribution": {
            "0-0.1": len(df_coverage[df_coverage["best_coverage"] < 0.1]),
            "0.1-0.3": len(df_coverage[(df_coverage["best_coverage"] >= 0.1) & (df_coverage["best_coverage"] < 0.3)]),
            "0.3-0.5": len(df_coverage[(df_coverage["best_coverage"] >= 0.3) & (df_coverage["best_coverage"] < 0.5)]),
            "0.5-0.7": len(df_coverage[(df_coverage["best_coverage"] >= 0.5) & (df_coverage["best_coverage"] < 0.7)]),
            "0.7-0.9": len(df_coverage[(df_coverage["best_coverage"] >= 0.7) & (df_coverage["best_coverage"] < 0.9)]),
            "0.9-1.0": len(df_coverage[df_coverage["best_coverage"] >= 0.9])
        }
    }
    
    # Print summary
    print("\n=== RECALL ANALYSIS SUMMARY ===")
    print(f"Total questions: {total_questions}")
    print(f"Total reference spans: {total_spans}")
    print("\nAverage Recall by k:")
    for k, recall in avg_recall_by_k.items():
        print(f"  Recall@{k}: {recall:.4f}")
    
    print(f"\nSpans covered: {df_coverage['is_covered'].sum()} / {len(df_coverage)} ({report['spans_covered_percentage']:.2f}%)")
    print(f"Problematic spans: {len(problematic_spans)} / {len(df_coverage)} ({report['problematic_spans_percentage']:.2f}%)")
    
    print("\nCoverage distribution:")
    for range_name, count in report["coverage_distribution"].items():
        print(f"  {range_name}: {count} spans ({count/len(df_coverage)*100:.2f}%)")
    
    return report


def print_detailed_span_analysis(debug_results_path, question_id=None, k=None, only_failures=False):
    """
    Print detailed analysis of span coverage for specific questions.
    
    Args:
        debug_results_path: Path to debug results JSON
        question_id: Optional question ID to filter
        k: Optional k value to analyze
        only_failures: If True, only show spans that failed to meet threshold
    """
    # Load debug results
    with open(debug_results_path, 'r') as f:
        debug_results = json.load(f)
    
    # Filter by question ID if specified
    if question_id:
        debug_results = [q for q in debug_results if q["question_id"] == question_id]
    
    # Use maximum k if not specified
    if k is None:
        k = max(int(k) for k in debug_results[0]["recall_at_k"].keys())
    
    print(f"\n=== DETAILED SPAN ANALYSIS (k={k}) ===")
    
    for question in debug_results:
        q_id = question["question_id"]
        q_text = question["question"]
        
        print(f"\nQuestion {q_id}: {q_text}")
        
        recall_data = question["recall_at_k"][str(k)]
        overall_recall = recall_data["overall_recall"]
        spans_covered = recall_data["spans_covered"]
        total_spans = recall_data["total_spans"]
        
        print(f"Overall recall: {overall_recall:.4f} ({spans_covered}/{total_spans} spans covered)")
        
        for i, span_detail in enumerate(recall_data["span_details"]):
            is_covered = span_detail["is_covered"]
            
            # Skip covered spans if only showing failures
            if only_failures and is_covered:
                continue
            
            best_coverage = span_detail["best_coverage"]
            best_match_idx = span_detail["best_match_index"]
            
            print(f"\nSpan {i+1} - Coverage: {best_coverage:.4f} - {'COVERED' if is_covered else 'NOT COVERED'}")
            
            coverage_details = span_detail["coverage_details"]
            orig_ref = coverage_details["original_reference"]
            norm_ref = coverage_details["normalized_reference"]
            
            if best_match_idx >= 0:
                orig_ret = coverage_details["original_retrieved"]
                norm_ret = coverage_details["normalized_retrieved"]
                matched_text = coverage_details["matched_text"]
                
                print("\nOriginal reference:")
                print(orig_ref[:100] + "..." if len(orig_ref) > 100 else orig_ref)
                
                print("\nNormalized reference:")
                print(norm_ref[:100] + "..." if len(norm_ref) > 100 else norm_ref)
                
                print("\nBest matching retrieved text:")
                print(orig_ret[:100] + "..." if len(orig_ret) > 100 else orig_ret)
                
                print("\nNormalized retrieved text:")
                print(norm_ret[:100] + "..." if len(norm_ret) > 100 else norm_ret)
                
                print("\nMatched substring:")
                print(matched_text)
            else:
                print("No matching retrieved text found")
            
            print("-" * 80)

def compare_normalization_methods(text1, text2):
    """
    Compare different normalization methods to find the best approach.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Comparison of different normalization methods
    """
    # Current normalization
    current_norm1 = normalize_text(text1)
    current_norm2 = normalize_text(text2)
    current_similarity = SequenceMatcher(None, current_norm1, current_norm2).ratio()
    
    # Basic normalization (just lowercase and whitespace)
    basic_norm1 = re.sub(r'\s+', ' ', text1.lower()).strip()
    basic_norm2 = re.sub(r'\s+', ' ', text2.lower()).strip()
    basic_similarity = SequenceMatcher(None, basic_norm1, basic_norm2).ratio()
    
    # Aggressive normalization (remove all non-alphanumeric)
    aggressive_norm1 = re.sub(r'[^a-z0-9\s]', '', text1.lower())
    aggressive_norm1 = re.sub(r'\s+', ' ', aggressive_norm1).strip()
    
    aggressive_norm2 = re.sub(r'[^a-z0-9\s]', '', text2.lower())
    aggressive_norm2 = re.sub(r'\s+', ' ', aggressive_norm2).strip()
    
    aggressive_similarity = SequenceMatcher(None, aggressive_norm1, aggressive_norm2).ratio()
    
    # Word-based comparison (compare sets of words)
    words1 = set(re.findall(r'\b[a-z0-9]+\b', text1.lower()))
    words2 = set(re.findall(r'\b[a-z0-9]+\b', text2.lower()))
    
    if words1 and words2:
        word_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
    else:
        word_similarity = 0.0
    
    return {
        "original_texts": {
            "text1": text1,
            "text2": text2,
            "raw_similarity": SequenceMatcher(None, text1, text2).ratio()
        },
        "current_normalization": {
            "text1": current_norm1,
            "text2": current_norm2,
            "similarity": current_similarity
        },
        "basic_normalization": {
            "text1": basic_norm1,
            "text2": basic_norm2,
            "similarity": basic_similarity
        },
        "aggressive_normalization": {
            "text1": aggressive_norm1,
            "text2": aggressive_norm2,
            "similarity": aggressive_similarity
        },
        "word_based_comparison": {
            "words1": list(words1)[:10],  # Show first 10 words
            "words2": list(words2)[:10],  # Show first 10 words
            "similarity": word_similarity
        },
        "best_method": max(
            [
                ("current", current_similarity),
                ("basic", basic_similarity),
                ("aggressive", aggressive_similarity),
                ("word_based", word_similarity)
            ],
            key=lambda x: x[1]
        )[0]
    }

def suggest_normalization_improvements(debug_results_path, k=None, min_samples=5):
    """
    Analyze normalization issues and suggest improvements based on debug results.
    
    Args:
        debug_results_path: Path to debug results JSON
        k: Optional k value to analyze
        min_samples: Minimum number of samples to analyze
        
    Returns:
        Suggestions for normalization improvements
    """
    # Load debug results
    with open(debug_results_path, 'r') as f:
        debug_results = json.load(f)
    
    # Use maximum k if not specified
    if k is None:
        k = max(int(k) for k in debug_results[0]["recall_at_k"].keys())
    
    # Collect problematic span pairs (reference and retrieved)
    problematic_pairs = []
    
    for question in debug_results:
        recall_data = question["recall_at_k"][str(k)]
        
        for span_detail in recall_data["span_details"]:
            best_coverage = span_detail["best_coverage"]
            
            # Focus on spans with coverage between 0.3 and 0.5 (close to threshold)
            if 0.3 <= best_coverage < 0.5:
                coverage_details = span_detail["coverage_details"]
                orig_ref = coverage_details["original_reference"]
                best_match_idx = span_detail["best_match_index"]
                
                if best_match_idx >= 0:
                    orig_ret = coverage_details["original_retrieved"]
                    problematic_pairs.append((orig_ref, orig_ret))
    
    # Limit number of samples
    if len(problematic_pairs) > min_samples:
        import random
        random.shuffle(problematic_pairs)
        problematic_pairs = problematic_pairs[:min_samples]
    
    # Analyze normalization methods
    normalization_results = []
    
    for ref, ret in problematic_pairs:
        comparison = compare_normalization_methods(ref, ret)
        normalization_results.append(comparison)
    
    # Count best methods
    method_counts = {"current": 0, "basic": 0, "aggressive": 0, "word_based": 0}
    
    for result in normalization_results:
        best_method = result["best_method"]
        method_counts[best_method] += 1
    
    # Generate suggestions
    suggestions = []
    
    if method_counts["aggressive"] > method_counts["current"]:
        suggestions.append("Consider using more aggressive normalization by removing punctuation and special characters")
    
    if method_counts["word_based"] > method_counts["current"]:
        suggestions.append("Consider using word-based similarity instead of character-based similarity")
    
    # Look for common patterns in problematic pairs
    common_issues = []
    
    # Check for number formatting issues
    number_issues = 0
    for result in normalization_results:
        text1 = result["original_texts"]["text1"]
        text2 = result["original_texts"]["text2"]
        
        # Check for numbers with different formatting
        if re.search(r'\d+,\d+', text1) or re.search(r'\d+,\d+', text2):
            number_issues += 1
    
    if number_issues >= len(normalization_results) * 0.3:
        common_issues.append("Number formatting (commas in numbers)")
        suggestions.append("Improve number normalization by standardizing comma and decimal formatting")
    
    # Check for currency symbol issues
    currency_issues = 0
    for result in normalization_results:
        text1 = result["original_texts"]["text1"]
        text2 = result["original_texts"]["text2"]
        
        # Check for currency symbols
        if re.search(r'[$€£¥]', text1) or re.search(r'[$€£¥]', text2):
            currency_issues += 1
    
    if currency_issues >= len(normalization_results) * 0.3:
        common_issues.append("Currency symbols")
        suggestions.append("Improve currency normalization by standardizing how currency symbols are handled")
    
    # Return analysis
    return {
        "analyzed_pairs": len(problematic_pairs),
        "best_method_counts": method_counts,
        "common_issues": common_issues,
        "suggestions": suggestions,
        "sample_comparisons": normalization_results[:3]  # Include a few examples
    }