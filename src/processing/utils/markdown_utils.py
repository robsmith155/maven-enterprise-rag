"""
Markdown Utilities for SEC Filing Parser.

This module provides utility functions for converting and processing markdown content,
with a focus on HTML-to-markdown conversion and markdown formatting.
"""

from typing import List, Union
from bs4 import BeautifulSoup, Tag

from .html_utils import is_bold, get_header_level
from .text_utils import process_text, clean_markdown_lines


def process_tag(tag: Union[Tag, str], depth: int = 0) -> str:
    """Process a BeautifulSoup tag and convert it to markdown.
    
    Args:
        tag: BeautifulSoup tag or string to process
        depth: Current nesting depth for lists
        
    Returns:
        Markdown formatted text
    """
    if isinstance(tag, str):
        return process_text(tag)
        
    if not tag.name:  # NavigableString
        return process_text(str(tag))
        
    # Get text content
    parts = []
    for child in tag.children:
        processed = process_tag(child, depth + 1)
        if processed:
            parts.append(processed)
            
    content = ' '.join(filter(None, parts))
    content = process_text(content)
    
    if not content:
        return ''
        
    # Handle headers and bold text
    header_level = get_header_level(tag)
    if header_level is not None:
        # Clean content and format as header
        clean_content = content.strip().strip('#').strip()
        hashes = '#' * header_level
        return f"\n\n{hashes} {clean_content}\n\n"
    elif is_bold(tag):
        # Format bold text
        return f"**{content}**"
    
    # Handle other HTML elements
    elif tag.name == 'p':
        return f"\n\n{content}\n\n"
        
    elif tag.name == 'br':
        return '\n'
        
    elif tag.name == 'div':
        # Only add extra newlines if it's a block-level div
        if len(content) > 50 or '\n' in content:
            return f"\n\n{content}\n\n"
        return content
        
    elif tag.name in ['em', 'i']:
        return f"*{content}*"
        
    elif tag.name == 'u':
        return f"_{content}_"
        
    elif tag.name == 'li':
        prefix = '  ' * (depth - 1)  # Indent nested lists
        return f"\n{prefix}* {content}"
        
    elif tag.name in ['ul', 'ol']:
        return f"\n{content}\n"
        
    elif tag.name in ['code', 'pre']:
        return f"`{content}`"
        
    elif tag.name == 'a' and tag.get('href'):
        return f"[{content}]({tag['href']})"
        
    elif tag.name == 'table':
        return f"\n\n{content}\n\n"
        
    else:
        return content


def html_to_markdown(html_content: str) -> str:
    """Convert HTML content to markdown while preserving formatting.
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Markdown formatted text
    """
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Remove script and style elements
    for tag in soup(['script', 'style']):
        tag.decompose()
        
    # Process the document
    markdown = process_tag(soup.body if soup.body else soup)
    
    # Clean up whitespace while preserving structure
    lines = markdown.split('\n')
    cleaned_lines = clean_markdown_lines(lines)
    
    return '\n'.join(cleaned_lines).strip()
