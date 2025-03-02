"""
HTML Utilities for SEC Filing Parser.

This module provides utility functions for processing HTML content from SEC filings,
with a focus on cleaning and normalizing HTML tables and other elements.
"""

import re
from typing import Optional
from bs4 import BeautifulSoup, Tag


def clean_table_html(table_soup: Tag) -> str:
    """Clean HTML tables with advanced processing for financial data.
    
    This function implements a multi-phase approach to clean financial tables:
    1. Extract all meaningful data with position information
    2. Process data to combine related cells (like currency symbols with values)
    3. Rebuild clean HTML table with optimized structure
    
    Args:
        table_soup: BeautifulSoup table element
        
    Returns:
        Cleaned HTML table string with optimized financial data presentation
    """
    if not table_soup or not isinstance(table_soup, Tag):
        return ""
    
    # Make a copy to avoid modifying the original
    table = BeautifulSoup(str(table_soup), 'lxml').find('table')
    if not table:
        return ""
    
    # Phase 1: Extract all cell data with position information
    table_data = []
    for row_idx, row in enumerate(table.find_all('tr')):
        row_data = []
        for col_idx, cell in enumerate(row.find_all(['td', 'th'])):
            # Extract cell attributes
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            
            # Get cell text and clean it
            cell_text = cell.get_text(strip=True)
            
            # Store cell data
            cell_info = {
                'text': cell_text,
                'rowspan': rowspan,
                'colspan': colspan,
                'is_header': cell.name == 'th',
                'row': row_idx,
                'col': col_idx,
                'original_html': str(cell)
            }
            row_data.append(cell_info)
        table_data.append(row_data)
    
    # Phase 2: Process data to combine related cells
    for row_idx, row in enumerate(table_data):
        for col_idx, cell in enumerate(row):
            # Process currency symbols and numbers
            if cell['text']:
                # Handle currency symbols with numbers
                cell['text'] = _process_financial_cell(cell['text'])
    
    # Phase 3: Rebuild clean HTML table
    new_table = BeautifulSoup('<table class="cleaned-financial-table"></table>', 'lxml').table
    
    for row_data in table_data:
        tr = BeautifulSoup('<tr></tr>', 'lxml').tr
        for cell in row_data:
            tag_name = 'th' if cell['is_header'] else 'td'
            cell_tag = BeautifulSoup(f'<{tag_name}></{tag_name}>', 'lxml').find(tag_name)
            
            # Add content
            cell_tag.string = cell['text']
            
            # Preserve rowspan and colspan if needed
            if cell['rowspan'] > 1:
                cell_tag['rowspan'] = str(cell['rowspan'])
            if cell['colspan'] > 1:
                cell_tag['colspan'] = str(cell['colspan'])
                
            tr.append(cell_tag)
        
        # Only add non-empty rows
        if tr.find_all(['td', 'th']):
            new_table.append(tr)
    
    return str(new_table)


def _process_financial_cell(text: str) -> str:
    """Process financial cell content to handle currency symbols and number formats.
    
    Args:
        text: Cell text content
        
    Returns:
        Processed text with standardized financial notation
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle currency symbols
    currency_symbols = ['$', '€', '£', '¥']
    for symbol in currency_symbols:
        # Ensure currency symbol is attached to the number
        if symbol in text and not text.startswith(symbol):
            text = text.replace(symbol, f' {symbol}').strip()
            parts = text.split()
            for i, part in enumerate(parts):
                if part == symbol and i < len(parts) - 1 and re.match(r'^[\d,.()\-]+$', parts[i+1]):
                    parts[i] = f'{symbol}{parts[i+1]}'
                    parts[i+1] = ''
            text = ' '.join(filter(None, parts))
    
    # Handle parentheses notation for negative numbers
    if re.match(r'^\([\d,.]+\)$', text):
        # Convert (1,234) to -1,234
        text = '-' + text[1:-1]
    
    # Handle currency with parentheses: $(1,234) to -$1,234
    for symbol in currency_symbols:
        if text.startswith(f'{symbol}(') and text.endswith(')'):
            text = f'-{symbol}{text[2:-1]}'
    
    return text


def is_bold(tag: Tag) -> bool:
    """Check if a tag represents bold text.
    
    Args:
        tag: BeautifulSoup tag to check
        
    Returns:
        True if tag represents bold text, False otherwise
    """
    if tag.name in ['strong', 'b']:
        return True
        
    style = tag.get('style', '').lower()
    if 'font-weight:' in style.replace(' ', ''):
        weight = style.split('font-weight:')[1].split(';')[0].strip()
        return weight in ['bold', '700', '800', '900']
        
    return False


def get_header_level(tag: Tag) -> Optional[int]:
    """Determine header level based on HTML structure.
    
    Args:
        tag: BeautifulSoup tag to check
        
    Returns:
        Header level (1-6) if tag is a header, None otherwise
    """
    if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        return int(tag.name[1])
    return None


def clean_html(html_content: str) -> str:
    """
    Clean HTML content while preserving important formatting.
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Cleaned text with preserved formatting
    """
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Remove script and style elements
    for element in soup(['script', 'style']):
        element.decompose()
    
    # Get text while preserving some formatting
    text = ''
    for element in soup.descendants:
        if element.name == 'p':
            text += str(element.get_text(strip=True)) + '\n\n'
        elif element.name == 'br':
            text += '\n'
        elif element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            text += str(element.get_text(strip=True)) + '\n\n'
        elif element.name == 'li':
            text += '* ' + str(element.get_text(strip=True)) + '\n'
        elif element.string and element.string.strip():
            text += str(element.string) + ' '
    
    # Clean up the text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove empty lines
    text = re.sub(r'&#160;|&nbsp;', ' ', text)  # Replace HTML spaces
    text = text.strip()
    
    return text


def remove_style_attributes(element):
    """Recursively remove style attributes from an HTML element and its children.
    
    Args:
        element: BeautifulSoup element to process
        
    Returns:
        The processed element with style attributes removed
    """
    if hasattr(element, 'attrs'):
        # Remove style-related attributes
        style_attrs = ['style', 'bgcolor', 'background', 'color', 'width', 'height', 
                      'align', 'valign', 'border', 'cellspacing', 'cellpadding']
        
        for attr in style_attrs:
            if attr in element.attrs:
                del element[attr]
    
    # Process children recursively
    if hasattr(element, 'children'):
        for child in element.children:
            if hasattr(child, 'name'):
                remove_style_attributes(child)
    
    return element
