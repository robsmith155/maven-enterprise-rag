"""
SEC Filing Parser for 10-K Reports.

This module provides functionality to parse SEC 10-K filing reports, with a focus on
extracting specific sections (like Item 1, Item 1A) in a format suitable for RAG workflows.
"""

import re
import unicodedata
from typing import Dict, Optional, Tuple, List, Union
from bs4 import BeautifulSoup, Tag
from IPython.display import HTML, display, clear_output
from textwrap import fill
from ipywidgets import widgets, Layout
import lxml.etree as ET

try:
    from xbrl import XbrlParser
    XBRL_AVAILABLE = True
except ImportError:
    XBRL_AVAILABLE = False

# Import utility functions
from .utils.html_utils import (
    clean_table_html,
    is_bold,
    get_header_level,
    clean_html,
    remove_style_attributes
)
from .utils.text_utils import (
    normalize_text,
    process_text,
    clean_markdown_lines
)
from .utils.markdown_utils import (
    process_tag,
    html_to_markdown
)

# CSS styles for consistent display across methods
_BASE_STYLE = '''
    <style>
        /* Base document styles */
        .sec-document, .sec-wrapper {
            font-family: Arial, sans-serif;
            line-height: 1.5;
            color: #333;
            max-width: 100%;
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Table styles */
        .table-wrapper, .sec-table-wrapper {
            font-family: Arial, sans-serif;
            margin: 20px 0;
            max-width: 100%;
            overflow-x: auto;
        }
        .table-wrapper table, .sec-table-wrapper table {
            border-collapse: collapse;
            width: 100%;
            background: white;
        }
        .table-wrapper th, .table-wrapper td,
        .sec-table-wrapper th, .sec-table-wrapper td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .table-wrapper th, .sec-table-wrapper th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .table-wrapper tr:nth-child(even),
        .sec-table-wrapper tr:nth-child(even) {
            background-color: #fcfcfc;
        }
        
        /* Section styles */
        .section-content, .sec-content {
            background: white;
            padding: 20px;
            border-radius: 4px;
            border: 1px solid #eee;
            color: #333;
            width: 100%;
        }
        .section-header {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        
        /* Code and pre styles */
        pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            border: 1px solid #eee;
            white-space: pre-wrap;
        }
        
        /* Markdown table styles */
        .markdown-table {
            font-family: monospace;
            white-space: pre;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            margin-top: 10px;
            border: 1px solid #eee;
            overflow-x: auto;
        }
        
        /* Paragraph styles */
        .sec-content p {
            margin: 10px 0;
            color: #333;
        }
        
        /* Table styles */
        .sec-content table {
            border-collapse: collapse;
            margin: 15px 0;
            width: 100%;
            background-color: white !important;
        }
        .sec-content th, .sec-content td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
            color: #333 !important;
            background-color: white !important;
        }
        .sec-content th {
            background-color: #f8f9fa !important;
            font-weight: 600;
        }
        .sec-content tr:nth-child(even) td {
            background-color: #fcfcfc !important;
        }
        
        /* Additional styles */
        .text-content {
            margin: 15px 0;
            white-space: pre-wrap;
        }
        .pagination-info {
            text-align: center;
            margin: 10px 0;
            padding: 10px;
            color: #444;
            font-weight: 500;
            border-top: 1px solid #eee;
        }
    </style>
'''

class SECFilingParser:
    """Parser for SEC 10-K filing reports."""

    # Patterns for matching section headers
    ITEM_PATTERNS = [
        r'Item\s+(\d+[A-Z]?)',
        r'Item&#160;(\d+[A-Z]?)',
        r'Item&nbsp;(\d+[A-Z]?)',
        r'ITEM\s+(\d+[A-Z]?)'
    ]
    
    def __init__(self):
        """Initialize the SEC filing parser."""
        self.raw_content: Optional[str] = None
        self.soup: Optional[BeautifulSoup] = None
        self.metadata: Dict = {}
        self.main_document: Optional[BeautifulSoup] = None


    def read_file(self, file_path: str) -> None:
        """
        Read and load the SEC filing file.
        
        Args:
            file_path: Path to the SEC filing file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.raw_content = file.read()
            self.soup = BeautifulSoup(self.raw_content, 'lxml')
            self._find_document_section()
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
    
    def _find_document_section(self) -> None:
        """Find and extract the main 10-K document section."""
        if not self.raw_content:
            raise ValueError("No content loaded. Call read_file() first.")
        
        # Find the start of the 10-K document
        doc_start = re.search(r'<DOCUMENT>\s*<TYPE>10-K', self.raw_content)
        if not doc_start:
            raise ValueError("Could not find 10-K document section")
        
        # Find the corresponding </DOCUMENT> tag
        doc_end = self.raw_content.find('</DOCUMENT>', doc_start.start())
        if doc_end == -1:
            raise ValueError("Could not find end of 10-K document section")
        
        # Extract the document content and parse it
        doc_content = self.raw_content[doc_start.start():doc_end + 11]  # +11 for </DOCUMENT>
        self.main_document = BeautifulSoup(doc_content, 'lxml')
    
    def extract_metadata(self) -> Dict:
        """
        Extract header information and format into metadata.
        
        Returns:
            Dictionary containing metadata fields
        """
        if not self.raw_content:
            raise ValueError("No content loaded. Call read_file() first.")
        
        metadata = {}
        
        # Regular expressions for metadata extraction
        patterns = {
            'company_name': r'COMPANY CONFORMED NAME:\s*(.+?)\n',
            'cik': r'CENTRAL INDEX KEY:\s*(.+?)\n',
            'filing_date': r'FILED AS OF DATE:\s*(.+?)\n',
            'period_end_date': r'CONFORMED PERIOD OF REPORT:\s*(.+?)\n',
            'fiscal_year_end': r'FISCAL YEAR END:\s*(.+?)\n',
            'industry': r'STANDARD INDUSTRIAL CLASSIFICATION:\s*(.+?)\n'
        }
        
        # Extract metadata using regex
        for field, pattern in patterns.items():
            match = re.search(pattern, self.raw_content)
            if match:
                metadata[field] = match.group(1).strip()
        
            
        self.metadata = metadata
        return metadata
    
    def _find_section_tag(self, section_text: str) -> Optional[Tag]:
        """Find the second occurrence of a section tag (first is usually TOC)."""
        tags = [
            tag for tag in self.main_document.find_all(
                text=lambda text: text and section_text.lower() in normalize_text(str(text))
            )
        ]
        return tags[1].parent if len(tags) >= 2 else None

    def _extract_content_between_tags(self, from_tag: Tag, to_tag: Optional[Tag]) -> str:
        """Extract content between two tags, excluding the end tag."""
        content = ''
        if from_tag:
            # Get the raw HTML content of the start tag
            content += str(from_tag) + '\n'
            element = from_tag.find_next()
            while element and element != to_tag:
                # Include top-level divs and any tables
                if (element.name == 'div' and element.find_parent('div') is None) or \
                   element.name == 'table':
                    content += str(element) + '\n'
                element = element.find_next()
        return content

    def _get_section_html(self, section_name: str) -> Optional[str]:
        """
        Get the HTML content for a specific section.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
            
        Returns:
            HTML content of the section if found, None otherwise
        """
        if not self.main_document:
            raise ValueError("No document content loaded. Call read_file() first.")

        # Extract section number and any subsection letter
        parts = section_name.split()
        if len(parts) != 2:
            raise ValueError("Section name must be in format 'Item X' or 'Item XY'")
            
        section_num = parts[1]
        is_subsection = len(section_num) > 1 and section_num[-1].isalpha()
        
        # Find the start tag for this section
        start_tag = self._find_section_tag(f"Item {section_num}.")
        if not start_tag:
            return None
            
        # Determine the end tag based on whether this is a main section or subsection
        if is_subsection:
            # For subsections (e.g., 1A), look for next subsection (1B) or next main section (2)
            base_num = section_num[:-1]
            current_letter = section_num[-1]
            next_letter = chr(ord(current_letter) + 1)
            
            # Try next subsection first
            end_tag = self._find_section_tag(f"Item {base_num}{next_letter}.")
            if not end_tag:
                # If no next subsection, try next main section
                next_num = int(base_num) + 1
                end_tag = self._find_section_tag(f"Item {next_num}.")
        else:
            # For main sections (e.g., 1), look for first subsection (1A) or next main section (2)
            end_tag = self._find_section_tag(f"Item {section_num}A.")
            if not end_tag:
                next_num = int(section_num) + 1
                end_tag = self._find_section_tag(f"Item {next_num}.")
        
        # Extract content between the tags
        content = self._extract_content_between_tags(start_tag, end_tag)
        
        # Process XBRL content if present
        if content and ('<xbrl' in content.lower() or '<ix:' in content.lower()):
            content = self._extract_html_from_xbrl(content)
            
        return content

    def display_section_html(self, section_name: str) -> None:
        """
        Display the HTML content of a section in a scrollable window.
        Automatically removes duplicate tables and handles footnote tables properly.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
        """
        html_content = self._get_section_html(section_name)
        if not html_content:
            print(f"No content found for {section_name}")
            return
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Clean page footers
        soup, footer_count = self._clean_page_footers(soup)
        
        # Remove footnote tables
        soup, footnote_count, _ = self._remove_footnote_tables(soup, section_name)
        
        # Process tables - remove duplicates
        seen_tables = set()
        original_table_count = len(soup.find_all('table'))
        
        # Handle regular tables and deduplication
        for table in soup.find_all('table'):
            # Clean the table to normalize it for comparison
            cleaned_table = clean_table_html(table)
            
            # Skip empty tables
            if not cleaned_table.strip():
                table.decompose()
                continue
                
            # Check for duplicates using a hash of the cleaned table
            table_hash = hash(cleaned_table)
            if table_hash in seen_tables:
                # This is a duplicate table, remove it
                table.decompose()
                continue
                
            # Add to seen tables
            seen_tables.add(table_hash)
        
        # Get the final content
        content = str(soup)
        
        # Print debug info
        remaining_tables = len(seen_tables)
        duplicates = original_table_count - remaining_tables
        
        if duplicates > 0:
            print(f"Removed {duplicates} duplicate tables from {section_name}")
        
        # Add styling and wrap content in a scrollable div
        style = _BASE_STYLE
        html_output = f"{style}<div class='sec-content' style='max-height: 600px; overflow-y: auto;'>{content}</div>"
        
        # Display the HTML
        display(HTML(html_output))

    def display_text(self, text: str, line_length: int = 100) -> None:
        """
        Pretty print markdown-formatted text content with proper line wrapping in a notebook.
        Preserves markdown formatting including headers, lists, and paragraph breaks.
        
        Args:
            text: Markdown-formatted text content to display
            line_length: Maximum length of each line (default: 100)
        """
        if not text:
            print("No text content to display")
            return
        
        # Split text into lines
        lines = text.split('\n')
        formatted_lines = []
        
        in_list = False  # Track if we're in a list
        
        for line in lines:
            line = line.rstrip()
            
            # Skip empty lines but preserve them in output
            if not line:
                formatted_lines.append('')
                in_list = False
                continue
            
            # Don't wrap headers (lines starting with #)
            if line.startswith('#'):
                formatted_lines.append(line)
                formatted_lines.append('')  # Add space after headers
                continue
                
            # Don't wrap list items, but indent their wrapped content
            if line.startswith(('* ', '- ', '+ ')) or line.lstrip().startswith(('* ', '- ', '+ ')):
                in_list = True
                indent = len(line) - len(line.lstrip())
                list_marker = line[indent:line.find(' ', indent) + 1]
                content = line[indent + len(list_marker):]
                
                # Wrap the list item content
                wrapped_content = fill(content, width=line_length - indent - len(list_marker),
                                      break_long_words=False, break_on_hyphens=False)
                wrapped_lines = wrapped_content.split('\n')
                
                # First line with list marker
                formatted_lines.append(' ' * indent + list_marker + wrapped_lines[0])
                # Subsequent lines indented to align with first line
                for wrapped_line in wrapped_lines[1:]:
                    formatted_lines.append(' ' * (indent + len(list_marker)) + wrapped_line)
                continue
                
            # For numbered lists (e.g., "1. ", "2. ")
            if re.match(r'^\d+\.\s', line):
                in_list = True
                indent = len(line) - len(line.lstrip())
                number_end = line.find('. ') + 2
                list_marker = line[:number_end]
                content = line[number_end:]
                
                # Wrap the list item content
                wrapped_content = fill(content, width=line_length - number_end,
                                      break_long_words=False, break_on_hyphens=False)
                wrapped_lines = wrapped_content.split('\n')
                
                # First line with list marker
                formatted_lines.append(list_marker + wrapped_lines[0])
                # Subsequent lines indented to align with first line
                for wrapped_line in wrapped_lines[1:]:
                    formatted_lines.append(' ' * number_end + wrapped_line)
                continue
            
            # Handle code blocks (preserve indentation)
            if line.startswith('    ') or line.startswith('\t'):
                formatted_lines.append(line)
                continue
            
            # Regular paragraph text
            if not in_list:
                wrapped_lines = fill(line, width=line_length,
                                    break_long_words=False, break_on_hyphens=False)
                formatted_lines.extend(wrapped_lines.split('\n'))
        
        # Join lines and print
        print('\n'.join(formatted_lines))
    
    def display_tables(self, tables: List[str], max_width: int = 800) -> None:
        """
        Display tables in a notebook with proper formatting.
        
        Args:
            tables: List of HTML table strings
            max_width: Maximum width of the table display in pixels (default: 800)
        """
        if not tables:
            print("No tables to display")
            return
            
        for i, table in enumerate(tables, 1):
            print(f"\nTable {i}:")
            self.display_table(table)
            
    def display_section_tables_from_parsed(self, section_name: str, max_width: int = 800) -> None:
        """
        Display all tables from a specific section using parsed data.
        Automatically identifies and skips footnote tables to prevent duplication.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
            max_width: Maximum width of the table display in pixels (default: 800)
        """
        section_data = self.parse_section(section_name)
        if not section_data['tables']:
            print(f"No tables found in {section_name}")
            return
            
        print(f"Found {len(section_data['tables'])} data tables in {section_name}:")
        self.display_tables(section_data['tables'], max_width)
    
    def display_section_text(self, section_name: str, line_length: int = 100) -> None:
        """
        Parse and display the text content of a section with proper markdown formatting.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
            line_length: Maximum length of each line (default: 100)
        """
        section_data = self.parse_section(section_name)
        if not section_data['text']:
            print(f"No text content found for {section_name}")
            return
            
        # Convert HTML to markdown
        markdown_text = html_to_markdown(section_data['html'])
        
        # Debug: Print a sample of the markdown to see the formatting
        print("\nFirst 500 characters of markdown:")
        print(markdown_text[:500])
        print("\n")
        
        try:
            from IPython.display import Markdown, display, HTML
            # Display the markdown
            display(Markdown(markdown_text))
        except ImportError:
            # Fallback to regular text display if not in a notebook
            self.display_text(markdown_text, line_length)
    
    def _clean_page_footers(self, soup: BeautifulSoup) -> Tuple[BeautifulSoup, int]:
        """
        Remove page footer text and copyright statements from SEC filings.
        
        Page footers typically appear at the bottom of each page and follow patterns like:
        "Company Name | YYYY Form 10-K | Page Number"
        
        Copyright statements typically follow patterns like:
        "Copyright© 2017 S&P, a division of McGraw Hill Financial. All rights reserved."
        
        Args:
            soup: BeautifulSoup object containing the HTML content
            
        Returns:
            Tuple of (modified soup, count of items removed)
        """
        removal_count = 0
        
        # Look for text nodes and paragraphs that match footer patterns
        patterns = [
            # Company | Form 10-K | Page
            r'[\w\s,\.]+\s*\|\s*\d{4}\s*Form\s*10-K\s*\|\s*\d+\s*$',
            # Page X of Y
            r'^\s*Page\s*\d+\s*of\s*\d+\s*$',
            # Just page numbers at end of line
            r'^.*\s{3,}\d+\s*$',
            # Copyright statements
            r'^\s*Copyright[©\s]\s*\d{4}\s+[\w\s&,\.]+\s*All\s*rights\s*reserved\.?\s*$',
            # Alternative copyright format
            r'^\s*©\s*\d{4}\s+[\w\s&,\.]+\s*All\s*rights\s*reserved\.?\s*$',
            # General copyright line
            r'^\s*Copyright[©\s]\s*\d{4}\s+[\w\s&,\.]+\s*$'
        ]
        
        # Compile the patterns for efficiency
        compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        # Check text nodes
        for text in soup.find_all(string=True):
            if text.parent.name not in ['script', 'style', 'head', 'title']:
                text_content = text.strip()
                for pattern in compiled_patterns:
                    if pattern.match(text_content):
                        # Replace the footer/copyright with empty string
                        text.replace_with('')
                        
                        # If parent element is now empty, remove it completely
                        if text.parent and not text.parent.get_text(strip=True):
                            text.parent.decompose()
                            
                        removal_count += 1
                        break
        
        # Check paragraph elements
        for p in soup.find_all(['p', 'div', 'span']):
            text_content = p.get_text().strip()
            for pattern in compiled_patterns:
                if pattern.match(text_content):
                    # Remove the entire paragraph
                    p.decompose()
                    removal_count += 1
                    break
        
        if removal_count > 0:
            print(f"Removed {removal_count} page footers and copyright statements")
            
        return soup, removal_count
    
    def _remove_footnote_tables(self, soup: BeautifulSoup, section_name: str) -> Tuple[BeautifulSoup, int, List[str]]:
        """
        Remove footnote tables from the soup and handle duplicate footnotes.
        
        Specifically designed to handle SEC filing footnotes that may appear in both table form
        and text form. When both forms exist, it removes the table form and keeps the text form.
        
        Args:
            soup: BeautifulSoup object containing the HTML content
            section_name: Name of the section for logging purposes
            
        Returns:
            Tuple of (modified soup, footnote count, list of footnote texts)
        """
        from bs4 import NavigableString
        footnote_count = 0
        footnote_texts = []
        
        # First, identify all footnote tables
        footnote_tables = []
        for table in soup.find_all('table'):
            if self._is_footnote_table(table):
                footnote_tables.append(table)
        
        # Process each footnote table
        for table in footnote_tables:
            # Extract the footnote reference and text
            rows = table.find_all('tr')
            if not rows:
                table.decompose()  # Remove empty tables
                continue
                
            # Find the data row with the footnote content
            data_row = None
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2 and cells[0].get_text(strip=True) and cells[1].get_text(strip=True):
                    data_row = row
                    break
                    
            if not data_row:
                table.decompose()  # Remove tables without proper data
                continue
                
            cells = data_row.find_all(['td', 'th'])
            if len(cells) < 2:
                table.decompose()  # Remove tables without proper cells
                continue
                
            footnote_ref = cells[0].get_text(strip=True)
            footnote_text = cells[1].get_text(strip=True)
            
            # Store footnote text for later use
            footnote_content = f"{footnote_ref} {footnote_text}"
            footnote_texts.append(footnote_content)
            
            # Check if this footnote also appears as text nearby
            # Look at the next few siblings to see if the same content appears as text
            next_element = table.next_sibling
            duplicate_found = False
            
            # Normalize the footnote reference to handle different formats
            # Convert -1 to (1), etc.
            normalized_ref = footnote_ref
            if re.match(r'^-\d+$', footnote_ref):
                normalized_ref = f"({footnote_ref[1:]})"  # Convert -1 to (1)
            
            # Look through the next few elements
            search_distance = 5  # Check next 5 elements
            for _ in range(search_distance):
                if not next_element:
                    break
                    
                # If it's a text node or paragraph
                if isinstance(next_element, NavigableString):
                    sibling_text = str(next_element).strip()
                    if normalized_ref in sibling_text or footnote_ref in sibling_text:
                        duplicate_found = True
                        break
                elif hasattr(next_element, 'name') and next_element.name in ['p', 'div', 'span']:
                    sibling_text = next_element.get_text(strip=True)
                    if normalized_ref in sibling_text or footnote_ref in sibling_text:
                        duplicate_found = True
                        break
                        
                next_element = next_element.next_sibling
            
            # If no duplicate text found, create a text node with the footnote
            if not duplicate_found:
                footnote_node = soup.new_tag('p')
                footnote_node.string = footnote_content
                table.replace_with(footnote_node)
            else:
                # Duplicate found, just remove the table
                table.decompose()
                
            footnote_count += 1
        
        if footnote_count > 0:
            print(f"Processed {footnote_count} footnote tables in {section_name}")
            
        return soup, footnote_count, footnote_texts
    
    def parse_section(self, section_name: str, output_format: str = 'simple') -> Dict:
        """
        Parse a section from the document and return its content.
        
        Args:
            section_name: Name of the section to parse (e.g., 'Item 1', 'Item 1A')
            output_format: Format of the output, either 'simple' or 'llm'
                - 'simple': Returns dict with {'text': markdown_text, 'html': html, 'tables': [tables]}
                - 'llm': Returns dict with {'text': str} where text is markdown with clean HTML tables
            
        Returns:
            Dict containing section data in the specified format
        """
        if output_format not in ['simple', 'llm']:
            raise ValueError("output_format must be either 'simple' or 'llm'")

        html_content = self._get_section_html(section_name)
        if not html_content:
            return {'text': '', 'html': '', 'tables': []} if output_format == 'simple' else {'text': ''}
            
        # Process XBRL content if present
        if '<xbrl' in html_content.lower() or '<ix:' in html_content.lower():
            html_content = self._extract_html_from_xbrl(html_content)
        
        # Parse HTML content
        section_soup = BeautifulSoup(html_content, 'lxml')
        
        # Clean page footers
        section_soup, footer_count = self._clean_page_footers(section_soup)
        
        # Remove footnote tables for all output formats to ensure consistency
        section_soup, footnote_count, footnote_texts = self._remove_footnote_tables(section_soup, section_name)
        
        if output_format == 'simple':
            # Extract tables with duplicate removal
            tables = []
            seen_tables = set()  # To track duplicate tables
            original_table_count = len(section_soup.find_all('table'))
            
            for table in section_soup.find_all('table'):
                
                # Clean the table to normalize it for comparison
                cleaned_table = clean_table_html(table)
                
                # Skip empty tables
                if not cleaned_table.strip():
                    table.decompose()
                    continue
                    
                # Check for duplicates using a hash of the cleaned table
                table_hash = hash(cleaned_table)
                if table_hash in seen_tables:
                    # This is a duplicate table, remove it
                    table.decompose()
                    continue
                    
                # Add to seen tables
                seen_tables.add(table_hash)
                tables.append(cleaned_table)
                table.decompose()  # Remove table from text content
            
            # Print debug info
            duplicates = original_table_count - len(tables)
            if duplicates > 0:
                print(f"Removed {duplicates} duplicate tables from {section_name} in simple format")
            
            # Store both HTML and converted markdown text
            html = str(section_soup)
            text = html_to_markdown(html)
            
            return {
                'text': text,
                'html': html,
                'tables': tables
            }
        else:  # llm format
            # Process tables and text while preserving original positions
            result = []
            
            # Create a copy to work with
            soup_copy = BeautifulSoup(str(section_soup), 'lxml')
            
            # Process tables - handle deduplication and replace regular tables with placeholders
            table_replacements = {}
            seen_tables = set()  # To track duplicate tables
            
            # Handle regular tables and create placeholders
            for i, table in enumerate(soup_copy.find_all('table')):
                
                # For regular tables, handle deduplication and create placeholders
                cleaned_table = clean_table_html(table)
                
                # Skip empty tables
                if not cleaned_table.strip():
                    table.decompose()
                    continue
                    
                # Check for duplicates using a hash of the cleaned table
                table_hash = hash(cleaned_table)
                if table_hash in seen_tables:
                    # This is a duplicate table, remove it
                    table.decompose()
                    continue
                    
                # Add to seen tables
                seen_tables.add(table_hash)
                
                # Create placeholder
                placeholder = f"TABLE_PLACEHOLDER_{i}"
                table_replacements[placeholder] = cleaned_table
                placeholder_tag = soup_copy.new_tag('div')
                placeholder_tag.string = placeholder
                table.replace_with(placeholder_tag)
            
            # Convert to markdown with placeholders intact
            markdown_with_placeholders = html_to_markdown(str(soup_copy))
            
            # Replace placeholders with actual tables
            final_text = markdown_with_placeholders
            for placeholder, table_html in table_replacements.items():
                if table_html.strip():
                    final_text = final_text.replace(placeholder, f"\n\n{table_html}\n\n")
            
            # Print debug info
            original_table_count = sum(1 for _ in section_soup.find_all('table'))
            duplicates = original_table_count - len(table_replacements) - footnote_count
            print(f"Found {len(table_replacements)} unique tables in section {section_name} (removed {duplicates} duplicates, converted {footnote_count} footnotes to text)")
            
            return {'text': final_text}

    def display_table(self, table_content: str) -> None:
        """
        Display a table in a notebook. Automatically detects if it's HTML or markdown format.
        
        Args:
            table_content: Table content in either HTML or markdown format
        """
        # Detect format
        is_html = '<table' in table_content.lower()
        
        if is_html:
            style = _BASE_STYLE
            html_output = f"{style}<div class='table-wrapper'>{table_content}</div>"
            display(HTML(html_output))
        else:
            # For markdown, display as-is
            print(table_content)

    def _is_footnote_table(self, table: Tag) -> bool:
        """
        Determine if a table is likely a footnote table using targeted detection for SEC filings.
        
        Specifically looks for the common SEC filing footnote patterns:
        1. Simple tables with 1-2 rows and exactly 2 columns
        2. First column contains just a reference marker like -1, -2, (1), (2), etc.
        3. Second column contains longer explanatory text
        4. Table appears isolated (not part of a larger data table)
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            True if the table appears to be a footnote table, False otherwise
        """
        # Check if table has rows
        rows = table.find_all('tr')
        if not rows:
            return False
            
        # Check if the table has at most 3 rows (to handle header rows)
        if len(rows) > 3:
            return False
            
        # Find the first non-empty row with cells
        data_row = None
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2 and any(cell.get_text(strip=True) for cell in cells):
                data_row = row
                break
                
        if not data_row:
            return False
            
        # Check if the row has exactly 2 cells
        cells = data_row.find_all(['td', 'th'])
        if len(cells) != 2:
            return False
            
        # Check if the first cell contains a footnote reference pattern
        first_cell_text = cells[0].get_text(strip=True)
        
        # Expanded patterns to catch more SEC footnote references
        footnote_patterns = [
            r'^-\d+$',                # -1, -2, etc.
            r'^\(\d+\)$',            # (1), (2), etc.
            r'^\d+\.$',               # 1., 2., etc.
            r'^\d+$',                 # 1, 2, etc.
            r'^\*+$',                 # *, **, etc.
            r'^\[\d+\]$',             # [1], [2], etc.
            r'^[a-zA-Z]\)$',          # a), b), etc.
            r'^\([a-zA-Z]\)$',        # (a), (b), etc.
            r'^[†‡§¶]$'  # †, ‡, §, ¶
        ]
        
        # Check if the first cell matches any footnote pattern
        is_footnote_ref = any(re.match(pattern, first_cell_text) for pattern in footnote_patterns)
        
        # Also consider very short first cells (1-3 chars) as potential footnote references
        if not is_footnote_ref and len(first_cell_text) <= 3:
            is_footnote_ref = True
        
        # Check if the second cell contains substantial text (footnote explanation)
        second_cell_text = cells[1].get_text(strip=True)
        has_explanation = len(second_cell_text) > 15  # Increased threshold for explanation text
        
        # Additional check: if the table is alone and follows another table
        # This helps identify isolated footnote tables that follow main data tables
        is_isolated_table = False
        if len(rows) == 1 or (len(rows) == 2 and len(rows[0].find_all(['td', 'th'])) <= 2):
            prev_element = table.find_previous_sibling()
            if prev_element and prev_element.name == 'table':
                is_isolated_table = True
        
        # Return true if it matches footnote patterns or is an isolated simple table after another table
        return (is_footnote_ref and has_explanation) or (is_isolated_table and len(first_cell_text) <= 5 and has_explanation)
    
    def display_section_tables(self, section_name: str) -> None:
        """
        Display all tables found in a specific section.
        Automatically identifies and skips footnote tables to prevent duplication.
        Also removes duplicate tables to ensure clean output.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
        """
        html_content = self._get_section_html(section_name)
        if not html_content:
            print(f"No content found for {section_name}")
            return
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # First, remove footnote tables
        soup, footnote_count, footnote_texts = self._remove_footnote_tables(soup, section_name)
        
        # Find remaining tables after footnote removal
        tables = soup.find_all('table')
        
        if not tables:
            print(f"No data tables found in {section_name} (found {footnote_count} footnote tables)")
            
            # Display footnotes as text if there are any
            if footnote_texts:
                print("\nFootnotes:")
                for footnote in footnote_texts:
                    print(f"  {footnote}")
            return
            
        # Use the same style as display_section_html_interactive
        style = _BASE_STYLE
        
        # Handle duplicates for remaining tables
        data_tables = []
        seen_tables = set()  # To track duplicate tables
        original_table_count = len(tables)
        
        for table in tables:
            # Clean the table to normalize it for comparison
            cleaned_table = clean_table_html(table)
            
            # Skip empty tables
            if not cleaned_table.strip():
                continue
                
            # Check for duplicates using a hash of the cleaned table
            table_hash = hash(cleaned_table)
            if table_hash in seen_tables:
                # This is a duplicate table, skip it
                continue
                
            # Add to seen tables and data_tables
            seen_tables.add(table_hash)
            data_tables.append(table)
        
        # Print debug info
        duplicates = original_table_count - len(data_tables)
        if duplicates > 0:
            print(f"Removed {duplicates} duplicate tables from {section_name}")
        
        # Display each data table with consistent styling
        for i, table in enumerate(data_tables, 1):
            # Clean up any existing styles
            for element in table.find_all(True):
                if 'style' in element.attrs:
                    del element['style']
                if 'bgcolor' in element.attrs:
                    del element['bgcolor']
                if 'color' in element.attrs:
                    del element['color']
            
            html_output = f"{style}<div class='sec-wrapper'>"
            html_output += f"<div class='table-title'>Table {i}</div>"
            html_output += f"<div class='sec-content'>{str(table)}</div>"
            html_output += "</div>"
            display(HTML(html_output))
        
        # Display footnotes as text if there are any
        if footnote_texts:
            print("\nFootnotes:")
            for footnote in footnote_texts:
                print(f"  {footnote}")
            
    def _extract_html_from_xbrl(self, html_content: str) -> str:
        """
        Extract clean HTML content from XBRL documents.
        
        Args:
            html_content: HTML content potentially containing XBRL tags
            
        Returns:
            Cleaned HTML content with XBRL tags properly handled
        """
        # Check if content contains XBRL tags
        if '<xbrl' not in html_content.lower() and '<ix:' not in html_content.lower():
            return html_content
            
        try:
            if XBRL_AVAILABLE:
                # Use the dedicated XBRL parser if available
                parser = XbrlParser()
                parsed_content = parser.parse_string(html_content)
                return str(parsed_content)
            else:
                # Fallback to regex-based extraction if XBRL parser is not available
                # Remove XBRL namespace declarations
                content = re.sub(r'xmlns:xbrl="[^"]*"', '', html_content)
                content = re.sub(r'xmlns:ix="[^"]*"', '', content)
                
                # Remove XBRL tags but keep their content
                content = re.sub(r'<ix:[^>]*>(.*?)</ix:[^>]*>', r'\1', content)
                content = re.sub(r'<xbrl[^>]*>(.*?)</xbrl[^>]*>', r'\1', content)
                
                return content
        except Exception as e:
            print(f"Warning: Error processing XBRL content: {str(e)}")
            return html_content  # Return original content if processing fails
