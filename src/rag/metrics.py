import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

# Helper function to normalize text by removing markdown and special formatting
def normalize_text(text):
    """
    Normalize text by removing markdown formatting, extra whitespace, and special characters.
    
    Args:
        text: The text to normalize
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    # Normalize Unicode characters (convert special quotes, apostrophes, etc. to ASCII equivalents)
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[\u2018\u2019\u201c\u201d]', "'", text)  # Replace curly quotes with straight quotes
    
    # Remove markdown formatting
    # Remove bold and italic markers
    text = re.sub(r'\*\*|__|\*|_', '', text)
    
    # Remove code blocks and inline code
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove headers
    text = re.sub(r'#{1,6}\s+', '', text)
    
    # Remove bullet points and numbered lists
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize dates (ensure space after commas in dates)
    text = re.sub(r'(\d+),(\d+)', r'\1, \2', text)
    
    # Normalize whitespace (convert all whitespace to single spaces)
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize numbers and currency
    # Keep the digits but standardize format
    text = re.sub(r'\$\s*(\d+(?:\.\d+)?)', r'dollar \1', text)
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    
    # Remove punctuation that doesn't affect meaning
    text = re.sub(r'[,.;:!?()[\]{}]', ' ', text)
    
    # Normalize to lowercase and strip
    return text.lower().strip()


# Helper function to calculate the longest common substring and its coverage
def get_consecutive_coverage(reference, retrieved):
    """
    Calculate the coverage of the reference text by the longest common substring in the retrieved text.
    
    Args:
        reference: The reference text
        retrieved: The retrieved text
        
    Returns:
        Coverage as a proportion of the reference text covered by the longest common substring
    """
    # Normalize texts with enhanced normalization
    norm_reference = normalize_text(reference)
    norm_retrieved = normalize_text(retrieved)
    
    # Skip empty strings
    if not norm_reference or not norm_retrieved:
        return 0.0
    
    # Direct containment check - if the normalized reference is fully contained in the retrieved text
    if norm_reference in norm_retrieved:
        return 1.0
    
    # If the reference is very long, try checking for substantial overlap
    # This helps with cases where the reference might be split across multiple retrieved chunks
    ref_words = norm_reference.split()
    ret_words = norm_retrieved.split()
    
    # Create sets of word trigrams for more robust matching
    if len(ref_words) >= 3 and len(ret_words) >= 3:
        ref_trigrams = set(' '.join(ref_words[i:i+3]) for i in range(len(ref_words)-2))
        ret_trigrams = set(' '.join(ret_words[i:i+3]) for i in range(len(ret_words)-2))
        
        if ref_trigrams and ret_trigrams:
            # Calculate trigram overlap
            common_trigrams = ref_trigrams.intersection(ret_trigrams)
            if len(common_trigrams) / len(ref_trigrams) > 0.8:  # 80% of trigrams match
                return len(common_trigrams) / len(ref_trigrams)
    
    # Fall back to SequenceMatcher for more complex cases
    matcher = SequenceMatcher(None, norm_reference, norm_retrieved)
    match = matcher.find_longest_match(0, len(norm_reference), 0, len(norm_retrieved))
    
    if match.size > 0:
        # Calculate coverage as proportion of reference covered
        coverage = match.size / len(norm_reference)
        return coverage
    
    return 0.0


# Helper function to calculate consecutive recall
def calculate_consecutive_recall(reference_spans, retrieved_texts, threshold):
    """
    Calculate the consecutive recall metric for a set of reference spans and retrieved texts.
    
    Args:
        reference_spans: List of reference text spans
        retrieved_texts: List of retrieved text chunks
        threshold: Minimum coverage threshold to consider a span as covered
        
    Returns:
        Consecutive recall score (proportion of reference spans that are covered)
    """
    # Skip empty references
    valid_refs = [ref for ref in reference_spans if ref and ref.strip()]
    if not valid_refs:
        return 0.0
    
    # Count how many reference spans are sufficiently covered
    covered_spans = 0
    
    for ref in valid_refs:
        # Find the best coverage among all retrieved texts
        best_coverage = 0.0
        
        for ret in retrieved_texts:
            coverage = get_consecutive_coverage(ref, ret)
            best_coverage = max(best_coverage, coverage)
        
        # If coverage exceeds threshold, consider this span covered
        if best_coverage >= threshold:
            covered_spans += 1
    
    # Calculate recall as proportion of reference spans that are covered
    return covered_spans / len(valid_refs)


# Helper function to calculate MRR at k
def calculate_mrr_at_k(reference_spans, retrieved_texts, threshold, k):
    # Limit to top-k retrieved texts
    top_k_texts = retrieved_texts[:k]
    
    # Check each retrieved chunk in order
    for i, text in enumerate(top_k_texts):
        # Check if any reference span is covered by this text
        for ref in reference_spans:
            if get_consecutive_coverage(ref, text) >= threshold:
                return 1.0 / (i + 1)  # MRR formula
    
    # No chunk met the threshold within top-k
    return 0.0