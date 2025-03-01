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
            'fiscal_year_end': r'FISCAL YEAR END:\s*(.+?)\n'
        }
        
        # Extract metadata using regex
        for field, pattern in patterns.items():
            match = re.search(pattern, self.raw_content)
            if match:
                metadata[field] = match.group(1).strip()
        
        self.metadata = metadata
        return metadata
    
    def _normalize_text(self, text: str) -> str:
        """Normalize unicode characters and convert to lowercase."""
        return unicodedata.normalize('NFKD', text).lower()

    def display_section_tables(self, section_name: str) -> None:
        """
        Display all tables found in a specific section.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
        """
        html_content = self._get_section_html(section_name)
        if not html_content:
            print(f"No content found for {section_name}")
            return
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Find all tables
        tables = soup.find_all('table')
        
        if not tables:
            print(f"No tables found in {section_name}")
            return
            
        # Use the same style as display_section_html_interactive
        style = _BASE_STYLE
        
        # Display each table with consistent styling
        for i, table in enumerate(tables, 1):
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

    def _find_section_tag(self, section_text: str) -> Optional[Tag]:
        """Find the second occurrence of a section tag (first is usually TOC)."""
        tags = [
            tag for tag in self.main_document.find_all(
                text=lambda text: text and section_text.lower() in self._normalize_text(str(text))
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
        """Get the HTML content for a specific section.
        
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
        return self._extract_content_between_tags(start_tag, end_tag)
       
 
    def display_section_html(self, section_name: str) -> None:
        """
        Display the HTML content of a section in a scrollable window.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
        """
        html_content = self._get_section_html(section_name)
        if not html_content:
            print(f"No content found for {section_name}")
            return
        
        # Parse the HTML to make it prettier
        soup = BeautifulSoup(html_content, 'lxml')
        content = str(soup)
        
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
            
    def display_section_tables(self, section_name: str, max_width: int = 800) -> None:
        """
        Display all tables from a specific section.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
            max_width: Maximum width of the table display in pixels (default: 800)
        """
        section_data = self.parse_section(section_name)
        if not section_data['tables']:
            print(f"No tables found in {section_name}")
            return
            
        print(f"Found {len(section_data['tables'])} tables in {section_name}:")
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
        markdown_text = self._html_to_markdown(section_data['html'])
        
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
        
        # Parse HTML content
        section_soup = BeautifulSoup(html_content, 'lxml')
        
        if output_format == 'simple':
            # Extract tables
            tables = []
            for table in section_soup.find_all('table'):
                tables.append(str(table))
                table.decompose()  # Remove table from text content
            
            # Store both HTML and converted markdown text
            html = str(section_soup)
            text = self._html_to_markdown(html)
            
            return {
                'text': text,
                'html': html,
                'tables': tables
            }
        else:  # llm format
            result = []
            temp_soup = BeautifulSoup(str(section_soup), 'lxml')  # Create a copy to work with
            
            # Create a copy of the soup to work with
            current_soup = BeautifulSoup(str(section_soup), 'lxml')
            
            # Process all elements in order
            current_text = []
            for element in current_soup.body.children:
                if isinstance(element, str):
                    if element.strip():
                        current_text.append(element)
                elif element.name == 'table':
                    # First add any accumulated text
                    if current_text:
                        text_content = ''.join(current_text)
                        if text_content.strip():
                            result.append(self._html_to_markdown(text_content))
                        current_text = []
                    
                    # Add the cleaned table
                    cleaned_table = self._clean_table_html(element)
                    result.append(f"\n\n{cleaned_table}\n\n")
                else:
                    # For non-table elements, add their string content
                    text = str(element)
                    if text.strip():
                        current_text.append(text)
            
            # Add any remaining text
            if current_text:
                text_content = ''.join(current_text)
                if text_content.strip():
                    result.append(self._html_to_markdown(text_content))
            
            return {'text': '\n'.join(result)}

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


    def _clean_table_html(self, table_soup: Tag) -> str:
        """Clean HTML tables by removing unnecessary styling and attributes.
        
        Args:
            table_soup: BeautifulSoup table element
            
        Returns:
            Cleaned HTML table string
        """
        # Remove unnecessary table attributes
        attrs_to_keep = {'class'}
        for attr in list(table_soup.attrs):
            if attr not in attrs_to_keep:
                del table_soup[attr]
        
        # Process all cells
        for cell in table_soup.find_all(['td', 'th']):
            # Remove all attributes except rowspan and colspan
            attrs_to_keep = {'rowspan', 'colspan'}
            for attr in list(cell.attrs):
                if attr not in attrs_to_keep:
                    del cell[attr]
            
            # Clean the text content
            if cell.string:
                cell.string = cell.string.strip()
        
        return str(table_soup)

    
    def _is_bold(self, tag: Tag) -> bool:
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
        
    def _get_header_level(self, tag: Tag) -> Optional[int]:
        """Determine header level based on HTML structure.
        
        Args:
            tag: BeautifulSoup tag to check
            
        Returns:
            Header level (1-6) if tag is a header, None otherwise
        """
        if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            return int(tag.name[1])
        return None
        
    def _process_text(self, text: str) -> str:
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
        
    def _process_tag(self, tag: Union[Tag, str], depth: int = 0) -> str:
        """Process a BeautifulSoup tag and convert it to markdown.
        
        Args:
            tag: BeautifulSoup tag or string to process
            depth: Current nesting depth for lists
            
        Returns:
            Markdown formatted text
        """
        if isinstance(tag, str):
            return self._process_text(tag)
            
        if not tag.name:  # NavigableString
            return self._process_text(str(tag))
            
        # Get text content
        parts = []
        for child in tag.children:
            processed = self._process_tag(child, depth + 1)
            if processed:
                parts.append(processed)
                
        content = ' '.join(filter(None, parts))
        content = self._process_text(content)
        
        if not content:
            return ''
            
        # Handle headers and bold text
        header_level = self._get_header_level(tag)
        if header_level is not None:
            # Clean content and format as header
            clean_content = content.strip().strip('#').strip()
            hashes = '#' * header_level
            return f"\n\n{hashes} {clean_content}\n\n"
        elif self._is_bold(tag):
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
            
    def _html_to_markdown(self, html_content: str) -> str:
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
        markdown = self._process_tag(soup.body if soup.body else soup)
        
        # Clean up whitespace while preserving structure
        lines = markdown.split('\n')
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
            
        markdown = '\n'.join(cleaned_lines)
        
        return markdown.strip()
        
        # Process the document and convert to markdown
        markdown = ''.join(process_tag(child) for child in soup.children)
        
        # Clean up extra newlines while preserving intentional spacing
        lines = markdown.splitlines()
        cleaned_lines = []
        prev_empty = True  # Track consecutive empty lines
        
        for line in lines:
            line = line.rstrip()
            is_empty = not line.strip()
            
            if is_empty:
                if not prev_empty:  # Only add empty line if previous line wasn't empty
                    cleaned_lines.append('')
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        
        # Ensure text ends with a newline
        if cleaned_lines and cleaned_lines[-1]:
            cleaned_lines.append('')
            
        return '\n'.join(cleaned_lines)

    def _clean_html(self, html_content: str) -> str:
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
