"""
Utility modules for SEC Filing Parser.

This package contains utility functions for processing SEC filings,
including HTML cleaning, text normalization, and markdown conversion.
"""

from .html_utils import (
    clean_table_html,
    is_bold,
    get_header_level,
    clean_html,
    remove_style_attributes
)

from .text_utils import (
    normalize_text,
    process_text,
    clean_markdown_lines
)

from .markdown_utils import (
    process_tag,
    html_to_markdown
)

__all__ = [
    'clean_table_html',
    'is_bold',
    'get_header_level',
    'clean_html',
    'remove_style_attributes',
    'normalize_text',
    'process_text',
    'clean_markdown_lines',
    'process_tag',
    'html_to_markdown'
]
