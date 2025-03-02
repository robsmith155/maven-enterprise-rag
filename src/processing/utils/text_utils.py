"""
Text Utilities for SEC Filing Parser.

This module provides utility functions for processing text content from SEC filings,
with a focus on cleaning, normalizing, and converting between formats.
"""

import re
import unicodedata
from typing import List, Union
from bs4 import BeautifulSoup, Tag


def normalize_text(text: str) -> str:
    """Normalize unicode characters and convert to lowercase.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text in lowercase
    """
    return unicodedata.normalize('NFKD', text).lower()


def process_text(text: str) -> str:
    """Clean and normalize text while preserving meaningful spaces.
    
    Args:
        text: Text to process
        
    Returns:
        Cleaned and normalized text
    """
    if not text.strip():
        return ''
        
    # Normalize unicode spaces
    text = unicodedata.normalize('NFKC', text)
    
    # Preserve certain spaces but collapse others
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Fix spacing around numbers and special characters
    text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
    text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'\$\s+', '$', text)  # No space after dollar
    text = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', text)  # Keep numbers with commas together
    
    return text.strip()


def clean_markdown_lines(lines: List[str]) -> List[str]:
    """Clean up markdown lines by removing extra whitespace and empty lines.
    
    Args:
        lines: List of markdown lines
        
    Returns:
        Cleaned list of markdown lines
    """
    cleaned_lines = []
    last_line_empty = True
    
    for line in lines:
        line = line.rstrip()
        is_empty = not line.strip()
        
        # Collapse multiple empty lines
        if is_empty and last_line_empty:
            continue
            
        cleaned_lines.append(line)
        last_line_empty = is_empty
        
    return cleaned_lines
