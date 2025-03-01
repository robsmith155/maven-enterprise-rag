"""
SEC Filing Parser for 10-K Reports.

This module provides functionality to parse SEC 10-K filing reports, with a focus on
extracting specific sections (like Item 1, Item 1A) in a format suitable for RAG workflows.
"""

import re
import unicodedata
from typing import Dict, Optional, Tuple, List
from bs4 import BeautifulSoup, Tag
from IPython.display import HTML, display, clear_output
from textwrap import fill
from ipywidgets import widgets, Layout

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

    def _display_section_page(self, section_name: str, page: int, elements_per_page: int, total_pages: int, content: str) -> None:
        """
        Internal method to display a single page of section content.
        
        Args:
            section_name: Name of the section to display
            page: Current page number
            elements_per_page: Number of elements to display per page
            total_pages: Total number of pages
            content: HTML content to display
        """
        # Parse the content
        soup = BeautifulSoup(content, 'lxml')
        
        # Find all block-level elements
        block_elements = ['p', 'div', 'table', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'pre', 'blockquote']
        all_elements = list(soup.find_all(block_elements))
        
        # Calculate which elements should be on this page
        start_idx = (page - 1) * elements_per_page
        end_idx = min(start_idx + elements_per_page, len(all_elements))
        
        # Create a container for this page's content
        page_content = ''
        for element in all_elements[start_idx:end_idx]:
            page_content += str(element)
        
        # Add styling
        style = '''
        <style>
            .sec-wrapper {
                font-family: Arial, sans-serif;
                line-height: 1.5;
                padding: 20px;
                max-width: 800px;
                margin: 20px auto;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .sec-content {
                color: #333;
            }
            .sec-content p {
                margin: 10px 0;
            }
            .sec-content a {
                color: #0066cc;
                text-decoration: none;
            }
            .sec-content table {
                border-collapse: collapse;
                margin: 15px 0;
                width: 100%;
                background-color: white;
            }
            .sec-content th, .sec-content td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
                color: #333;
                background-color: white;
            }
            .sec-content th {
                background-color: #f8f9fa;
                font-weight: 600;
            }
            .sec-content tr:nth-child(even) {
                background-color: #fcfcfc;
            }
            .pagination-info {
                text-align: center;
                margin: 10px 0;
                padding: 10px;
                color: #444;
                font-weight: 500;
                border-top: 1px solid #eee;
            }
            /* Override any external styles */
            .sec-content * {
                color: #333 !important;
                background-color: transparent !important;
            }
            .sec-content table, .sec-content th, .sec-content td {
                background-color: white !important;
            }
        </style>'''
        
        # Add pagination info
        pagination_info = f"<div class='pagination-info'>Page {page} of {total_pages}</div>"
        
        # Wrap the content in a div with our styling
        html_output = f"{style}<div class='sec-wrapper'><div class='sec-content'>{page_content}</div>{pagination_info}</div>"
        
        # Display the HTML
        display(HTML(html_output))
    
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
    
    def normalize_text(self, text: str) -> str:
        """Normalize unicode characters and convert to lowercase."""
        return unicodedata.normalize('NFKD', text).lower()

    def display_section_tables(self, section_name: str) -> None:
        """
        Display all tables found in a specific section.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
        """
        html_content = self.get_section_html(section_name)
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
        style = '''
        <style>
            .sec-wrapper {
                font-family: Arial, sans-serif;
                line-height: 1.5;
                padding: 20px;
                max-width: 800px;
                margin: 20px auto;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .sec-content {
                color: #333;
            }
            .sec-content p {
                margin: 10px 0;
            }
            .sec-content a {
                color: #0066cc;
                text-decoration: none;
            }
            .sec-content table {
                border-collapse: collapse;
                margin: 15px 0;
                width: 100%;
                background-color: white;
            }
            .sec-content th, .sec-content td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
                color: #333;
                background-color: white;
            }
            .sec-content th {
                background-color: #f8f9fa;
                font-weight: 600;
            }
            .sec-content tr:nth-child(even) {
                background-color: #fcfcfc;
            }
            .table-title {
                text-align: center;
                margin: 10px 0;
                padding: 10px;
                color: #444;
                font-weight: 500;
                border-bottom: 1px solid #eee;
            }
            /* Override any external styles */
            .sec-content table, .sec-content th, .sec-content td {
                color: #333 !important;
                background-color: white !important;
            }
            .sec-content tr:nth-child(even) td {
                background-color: #fcfcfc !important;
            }
        </style>'''
        
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

    def find_section_tag(self, section_text: str) -> Optional[Tag]:
        """Find the second occurrence of a section tag (first is usually TOC)."""
        tags = [
            tag for tag in self.main_document.find_all(
                text=lambda text: text and section_text.lower() in self.normalize_text(str(text))
            )
        ]
        return tags[1].parent if len(tags) >= 2 else None

    def extract_content_between_tags(self, from_tag: Tag, to_tag: Optional[Tag]) -> str:
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

    def get_section_html(self, section_name: str) -> Optional[str]:
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
        start_tag = self.find_section_tag(f"Item {section_num}.")
        if not start_tag:
            return None
            
        # Determine the end tag based on whether this is a main section or subsection
        if is_subsection:
            # For subsections (e.g., 1A), look for next subsection (1B) or next main section (2)
            base_num = section_num[:-1]
            current_letter = section_num[-1]
            next_letter = chr(ord(current_letter) + 1)
            
            # Try next subsection first
            end_tag = self.find_section_tag(f"Item {base_num}{next_letter}.")
            if not end_tag:
                # If no next subsection, try next main section
                next_num = int(base_num) + 1
                end_tag = self.find_section_tag(f"Item {next_num}.")
        else:
            # For main sections (e.g., 1), look for first subsection (1A) or next main section (2)
            end_tag = self.find_section_tag(f"Item {section_num}A.")
            if not end_tag:
                next_num = int(section_num) + 1
                end_tag = self.find_section_tag(f"Item {next_num}.")
        
        # Extract content between the tags
        return self.extract_content_between_tags(start_tag, end_tag)
        if not self.main_document:
            raise ValueError("No document content loaded. Call read_file() first.")
        
        # Get the document content
        doc_str = str(self.main_document)
        
        # Find all section headers, looking for exact matches
        # This pattern matches 'Item X' or 'ITEM X' followed by a delimiter
        pattern = fr'(?:Item|ITEM)\s*{re.escape(section_name.split()[1])}(?=\s*[:.\-]|\s+[A-Z])'  
        
        # Find all matches
        matches = list(re.finditer(pattern, doc_str))
        
        # Skip matches in the first 20% of the document (likely TOC)
        content_matches = [m for m in matches if m.start() > len(doc_str) // 5]
        
        if not content_matches:
            return None
            
        # Get the start of our section
        start_pos = content_matches[0].start()
        
        # Find the start of the next section
        next_section = int(section_name.split()[1][0]) + 1
        next_pattern = fr'(?:Item|ITEM)\s*{next_section}(?=\s*[:.\-]|\s+[A-Z])'
        
        # Look for the next section
        next_match = re.search(next_pattern, doc_str[start_pos:])
        end_pos = start_pos + next_match.start() if next_match else len(doc_str)
        
        # Get the content between the current section and the next
        content = doc_str[start_pos:end_pos].strip()
        
        return content
        """
        Get the HTML content for a specific section.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
            
        Returns:
            HTML content of the section if found, None otherwise
        """
        if not self.main_document:
            raise ValueError("No document content loaded. Call read_file() first.")
        
        # Extract section number and letter (if any)
        section_parts = section_name.split()
        if len(section_parts) != 2:
            raise ValueError("Section name must be in format 'Item X' or 'Item XY'")
            
        section_num = section_parts[1]
        
        # Create regex pattern for section header
        # This looks for the section header at the start of a line or after a clear break
        header_pattern = f'(?:^|[>\n\r])\s*(?:Item|ITEM)\s*{section_num}\s*[:.\-]\s*'
        
        # Find all section headers in the document
        doc_str = str(self.main_document)
        matches = list(re.finditer(header_pattern, doc_str, re.MULTILINE))
        
        # Filter out matches that are likely in the table of contents
        # (they usually appear in the first 20% of the document)
        doc_length = len(doc_str)
        cutoff_pos = doc_length // 5  # 20% mark
        content_matches = [m for m in matches if m.start() > cutoff_pos]
        
        if not content_matches:
            return None
            
        # Get the start position of our section
        start_pos = content_matches[0].start()
        
        # Find the next section header
        # First try to find the next sequential section
        next_section_patterns = []
        
        # If we have a letter (e.g., 1A), try next letter first
        if len(section_num) > 1 and section_num[-1].isalpha():
            base_num = section_num[:-1]
            current_letter = section_num[-1]
            next_letter = chr(ord(current_letter) + 1)
            next_section_patterns.append(f'(?:^|[>\n\r])\s*(?:Item|ITEM)\s*{base_num}{next_letter}\s*[:.\-]\s*')
        
        # Then try next number
        base_num = int(section_num[0])
        next_section_patterns.append(f'(?:^|[>\n\r])\s*(?:Item|ITEM)\s*{base_num + 1}\s*[:.\-]\s*')
        
        # Find the closest next section
        end_pos = len(doc_str)
        for pattern in next_section_patterns:
            matches = list(re.finditer(pattern, doc_str[start_pos:], re.MULTILINE))
            if matches:
                section_end = start_pos + matches[0].start()
                if section_end < end_pos:
                    end_pos = section_end
        
        # Extract the section content
        section_content = doc_str[start_pos:end_pos]
        
        # Clean up any empty divs or paragraphs at the start/end
        section_content = re.sub(r'^\s*<(?:div|p)[^>]*>\s*</(?:div|p)>\s*', '', section_content)
        section_content = re.sub(r'\s*<(?:div|p)[^>]*>\s*</(?:div|p)>\s*$', '', section_content)
        
        return section_content
    
    def _find_item_section(self, section_name: str) -> Tuple[Optional[BeautifulSoup], Optional[BeautifulSoup]]:
        """
        Find the start and end tags for a specific section.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
            
        Returns:
            Tuple of (start_tag, end_tag) for the section
        """
        if not self.main_document:
            return None, None
        
        # Create pattern for the section
        section_number = section_name.split()[-1]
        
        # Find all potential section headers
        headers = []
        for pattern in self.ITEM_PATTERNS:
            # Convert pattern to handle the specific section number
            specific_pattern = pattern.replace(r'(\d+[A-Z]?)', re.escape(section_number))
            headers.extend(self.main_document.find_all(string=re.compile(specific_pattern)))
        
        if len(headers) < 2:
            return None, None
        
        # Use the second occurrence (first is usually in TOC)
        section_start = headers[1].parent
        
        # Find the next section (if any)
        next_section = None
        for pattern in self.ITEM_PATTERNS:
            next_number = str(int(section_number[0]) + 1)
            if len(section_number) > 1:
                next_number += section_number[1:]
            specific_pattern = pattern.replace(r'(\d+[A-Z]?)', re.escape(next_number))
            next_headers = self.main_document.find_all(string=re.compile(specific_pattern))
            if next_headers:
                next_section = next_headers[1].parent if len(next_headers) > 1 else next_headers[0].parent
                break
        
        return section_start, next_section
    
    def _display_section_page(self, section_name: str, page: int, chunk_size: int, total_pages: int, content: str) -> None:
        """
        Internal method to display a single page of section content.
        """
        # Calculate start and end positions for the current page
        start_pos = (page - 1) * chunk_size
        end_pos = min(start_pos + chunk_size, len(content))
        
        # Get the chunk of content for this page
        page_content = content[start_pos:end_pos]
        
        # Try to find complete tags
        if page > 1:
            first_tag_start = page_content.find('<')
            if first_tag_start > 0:
                page_content = page_content[first_tag_start:]
        
        if page < total_pages:
            last_tag_end = page_content.rfind('>')
            if last_tag_end > 0:
                page_content = page_content[:last_tag_end + 1]
        
        # Add styling
        style = '''
        <style>
            .sec-wrapper {
                font-family: Arial, sans-serif;
                line-height: 1.5;
                padding: 20px;
                max-width: 800px;
                margin: 20px auto;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                color: #333;
            }
            .sec-content {
                width: 100%;
            }
            .sec-content p {
                margin: 10px 0;
                color: #333;
            }
            .sec-content a {
                color: #0066cc;
                text-decoration: none;
            }
            .sec-content table {
                border-collapse: collapse;
                margin: 15px 0;
                width: 100%;
                background-color: white;
            }
            .sec-content th, .sec-content td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
                color: #333;
                background-color: white;
            }
            .sec-content th {
                background-color: #f8f9fa;
                font-weight: 600;
            }
            .sec-content tr:nth-child(even) {
                background-color: #fcfcfc;
            }
            .pagination-info {
                text-align: center;
                margin: 10px 0;
                font-family: Arial, sans-serif;
                color: #444;
                background-color: white;
                padding: 10px;
                border-radius: 4px;
                font-weight: 500;
            }
        </style>'''
        
        # Add pagination info
        pagination_info = f"<div class='pagination-info'>Page {page} of {total_pages}</div>"
        
        # Wrap the content in a div with our styling
        html_output = f"{style}<div class='sec-content'>{page_content}</div>{pagination_info}"
        
        # Display the HTML
        display(HTML(html_output))

    def display_section_html_interactive(self, section_name: str) -> None:
        """
        Display the HTML content of a section with interactive pagination buttons.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
        """
        html_content = self.get_section_html(section_name)
        if not html_content:
            print(f"No content found for {section_name}")
            return
        
        # Parse the HTML to handle tables properly
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Find all top-level elements
        elements = []
        for element in soup.children:
            if isinstance(element, Tag):
                elements.append(str(element))
        
        # Group elements into pages, keeping tables intact
        pages = []
        current_page = []
        current_size = 0
        target_size = 5000  # Target characters per page
        
        for element in elements:
            # Check if this is a table
            is_table = '<table' in element.lower()
            element_size = len(element)
            
            # If adding this element would exceed our target size and we're not in the middle of a table
            if current_size + element_size > target_size and current_page and not is_table:
                # Complete the current page
                pages.append('\n'.join(current_page))
                current_page = []
                current_size = 0
            
            # Add the element to the current page
            current_page.append(element)
            current_size += element_size
            
            # If this was a table, complete the page to avoid splitting tables across pages
            if is_table:
                pages.append('\n'.join(current_page))
                current_page = []
                current_size = 0
        
        # Add any remaining content as the last page
        if current_page:
            pages.append('\n'.join(current_page))
        
        total_pages = len(pages)
        current_page = 1
        
        # Create navigation buttons
        button_style = {'description_width': 'initial'}
        layout = Layout(width='100px', margin='0 10px')
        
        prev_button = widgets.Button(
            description='Previous',
            button_style='',
            layout=layout,
            disabled=True,
            style=button_style
        )
        
        next_button = widgets.Button(
            description='Next',
            button_style='',
            layout=layout,
            disabled=total_pages <= 1,
            style=button_style
        )
        
        page_label = widgets.HTML(
            value=f'<div style="color: #444; font-family: Arial; padding: 5px;">Page {current_page} of {total_pages}</div>'
        )
        
        # Create button container
        button_box = widgets.HBox(
            [prev_button, page_label, next_button],
            layout=Layout(
                display='flex',
                justify_content='center',
                align_items='center',
                margin='10px 0'
            )
        )
        
        # Add styling
        style = '''
        <style>
            .sec-wrapper {
                font-family: Arial, sans-serif;
                line-height: 1.5;
                padding: 20px;
                max-width: 800px;
                margin: 20px auto;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .sec-content {
                color: #333;
            }
            .sec-content p {
                margin: 10px 0;
            }
            .sec-content a {
                color: #0066cc;
                text-decoration: none;
            }
            .sec-content table {
                border-collapse: collapse;
                margin: 15px 0;
                width: 100%;
                background-color: white;
            }
            .sec-content th, .sec-content td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
                color: #333;
                background-color: white;
            }
            .sec-content th {
                background-color: #f8f9fa;
                font-weight: 600;
            }
            .sec-content tr:nth-child(even) {
                background-color: #fcfcfc;
            }
            .pagination-info {
                text-align: center;
                margin: 10px 0;
                padding: 10px;
                color: #444;
                font-weight: 500;
                border-top: 1px solid #eee;
            }
        </style>'''
        
        def update_display(page):
            clear_output(wait=True)
            
            # Get content for this page
            page_idx = page - 1
            page_content = pages[page_idx]
            
            # Add pagination info
            pagination_info = f"<div class='pagination-info'>Page {page} of {total_pages}</div>"
            
            # Wrap everything in our styled container
            html_output = f"{style}<div class='sec-wrapper'><div class='sec-content'>{page_content}</div>{pagination_info}</div>"
            
            # Display the HTML and buttons
            display(HTML(html_output))
            display(button_box)
            
            # Update button states
            prev_button.disabled = page <= 1
            next_button.disabled = page >= total_pages
            page_label.value = f'<div style="color: #444; font-family: Arial; padding: 5px;">Page {page} of {total_pages}</div>'
        
        def on_prev_click(b):
            nonlocal current_page
            if current_page > 1:
                current_page -= 1
                update_display(current_page)
        
        def on_next_click(b):
            nonlocal current_page
            if current_page < total_pages:
                current_page += 1
                update_display(current_page)
        
        prev_button.on_click(on_prev_click)
        next_button.on_click(on_next_click)
        
        # Initial display
        update_display(current_page)

    def display_section_html(self, section_name: str, chunk_size: int = 10000, page: int = 1) -> None:
        """
        Display the HTML content of a section in a Jupyter notebook with pagination.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
            chunk_size: Number of characters per page (default: 10000)
            page: Page number to display (default: 1)
        """
        html_content = self.get_section_html(section_name)
        if not html_content:
            print(f"No content found for {section_name}")
            return
        
        # Parse the HTML to make it prettier
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Split content into chunks
        content = str(soup)
        total_chars = len(content)
        total_pages = (total_chars + chunk_size - 1) // chunk_size
        
        # Validate page number
        if page < 1 or page > total_pages:
            print(f"Invalid page number. Please choose a page between 1 and {total_pages}")
            return
        
        # Calculate start and end positions for the current page
        start_pos = (page - 1) * chunk_size
        end_pos = min(start_pos + chunk_size, total_chars)
        
        # Add some basic CSS styling
        style = '''
        <style>
            .sec-content {
                font-family: Arial, sans-serif;
                line-height: 1.5;
                padding: 20px;
                max-width: 800px;
                margin: 0 auto;
            }
            .sec-content table {
                border-collapse: collapse;
                margin: 15px 0;
                width: 100%;
            }
            .sec-content th, .sec-content td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            .sec-content th {
                background-color: #f5f5f5;
            }
            .sec-content h1, .sec-content h2, .sec-content h3 {
                color: #333;
                margin-top: 20px;
            }
            .pagination-info {
                text-align: center;
                margin: 10px 0;
                font-family: Arial, sans-serif;
                color: #666;
            }
        </style>'''
        
        # Add pagination info
        pagination_info = f"<div class='pagination-info'>Page {page} of {total_pages}</div>"
        
        # Get the chunk of content for this page
        page_content = content[start_pos:end_pos]
        
        # Try to find complete tags
        if page > 1:
            # Find the first complete tag start if we're not on the first page
            first_tag_start = page_content.find('<')
            if first_tag_start > 0:
                page_content = page_content[first_tag_start:]
        
        if page < total_pages:
            # Find the last complete tag end
            last_tag_end = page_content.rfind('>')
            if last_tag_end > 0:
                page_content = page_content[:last_tag_end + 1]
        
        # Wrap the content in a div with our styling
        html_output = f"{style}<div class='sec-content'>{page_content}</div>{pagination_info}"
        
        # Display the HTML
        display(HTML(html_output))
        
        # Print navigation help
        print(f"\nTo view other pages, call this method with page=N (1 to {total_pages})")

    def display_text(self, text: str, line_length: int = 100) -> None:
        """
        Pretty print text content with proper line wrapping in a notebook.
        
        Args:
            text: Text content to display
            line_length: Maximum length of each line (default: 100)
        """
        if not text:
            print("No text content to display")
            return
        
        # Use textwrap to format the text
        formatted_text = fill(text, width=line_length, break_long_words=False, break_on_hyphens=False)
        print(formatted_text)
    
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
            
        style = f'''
        <style>
            .sec-table-wrapper {{
                padding: 20px;
                max-width: {max_width}px;
                margin: 20px auto;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .sec-table-container {{
                width: 100%;
                overflow-x: auto;
                font-family: Arial, sans-serif;
            }}
            .sec-table-container table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
                background-color: white;
            }}
            .sec-table-container th, .sec-table-container td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
                line-height: 1.5;
            }}
            .sec-table-container th {{
                background-color: #f8f9fa;
                font-weight: 600;
                color: #333;
            }}
            .sec-table-container tr:nth-child(even) {{
                background-color: #fcfcfc;
            }}
            .sec-table-container tr:hover {{
                background-color: #f8f9fa;
            }}
            .table-number {{
                font-weight: 600;
                color: #444;
                margin: 0 0 15px 0;
                font-size: 1.1em;
            }}
        </style>'''
        
        for i, table in enumerate(tables, 1):
            # Create a container for each table with a number
            html_output = f"{style}<div class='sec-table-wrapper'>"
            html_output += f"<div class='table-number'>Table {i}</div>"
            html_output += f"<div class='sec-table-container'>{table}</div>"
            html_output += "</div>"
            display(HTML(html_output))
            
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
        Parse and display the text content of a section with proper formatting.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
            line_length: Maximum length of each line (default: 100)
        """
        section_data = self.parse_section(section_name)
        if not section_data['text']:
            print(f"No text content found for {section_name}")
            return
        
        self.display_text(section_data['text'], line_length)
    
    def parse_section(self, section_name: str) -> Dict:
        """
        Parse a section and extract text and tables.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
            
        Returns:
            Dictionary containing 'text' and 'tables' keys
        """
        html_content = self.get_section_html(section_name)
        if not html_content:
            return {'text': '', 'tables': []}
        
        # Parse HTML content
        section_soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract tables
        tables = []
        for table in section_soup.find_all('table'):
            tables.append(str(table))
            table.decompose()  # Remove table from text content
        
        # Clean and extract text
        text = self._clean_html(str(section_soup))
        
        return {
            'text': text,
            'tables': tables
        }

    def display_table(self, table_content: str) -> None:
        """
        Display a table in a notebook. Automatically detects if it's HTML or markdown format.
        
        Args:
            table_content: Table content in either HTML or markdown format
        """
        # Detect format
        is_html = '<table' in table_content.lower()
        
        if is_html:
            style = '''
            <style>
                .table-wrapper {
                    font-family: Arial, sans-serif;
                    margin: 20px 0;
                    max-width: 100%;
                    overflow-x: auto;
                }
                .table-wrapper table {
                    border-collapse: collapse;
                    width: 100%;
                    background: white;
                }
                .table-wrapper th, .table-wrapper td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                .table-wrapper th {
                    background-color: #f8f9fa;
                }
                .table-wrapper tr:nth-child(even) {
                    background-color: #fcfcfc;
                }
            </style>'''
            html_output = f"{style}<div class='table-wrapper'>{table_content}</div>"
            display(HTML(html_output))
        else:
            # For markdown, display as-is
            print(table_content)

    def clean_table_for_llm(self, table_html: str, format: str = 'compact_html') -> str:
        """
        Clean an HTML table to make it more suitable for LLMs by removing unnecessary formatting
        and converting to a more token-efficient format.
        
        Args:
            table_html: HTML string containing a table
            format: Output format ('markdown' or 'compact_html')
            
        Returns:
            Cleaned table in specified format
        """
        # Parse the table
        soup = BeautifulSoup(table_html, 'lxml')
        table = soup.find('table')
        if not table:
            return ''
            
        # Warning for markdown format due to SEC filing table structure
        if format == 'markdown':
            print("Warning: Markdown format may not preserve table structure properly due to SEC filing formatting.")
            print("Consider using 'compact_html' format for more accurate table representation.")
            
        if format == 'compact_html':
            # Remove all styling attributes
            for tag in table.find_all(True):
                if 'style' in tag.attrs:
                    del tag['style']
                if 'bgcolor' in tag.attrs:
                    del tag['bgcolor']
                if 'color' in tag.attrs:
                    del tag['color']
                if 'width' in tag.attrs:
                    del tag['width']
                if 'height' in tag.attrs:
                    del tag['height']
                if 'align' in tag.attrs:
                    del tag['align']
                if 'valign' in tag.attrs:
                    del tag['valign']
                if 'class' in tag.attrs:
                    del tag['class']
            return str(table)
            
        # For markdown format
        # Extract headers and rows
        rows = []
        for tr in table.find_all('tr'):
            row = []
            for cell in tr.find_all(['th', 'td']):
                # Clean and normalize the text
                text = cell.get_text(strip=True)
                text = text.replace('|', '\|')  # Escape any | characters
                row.append(text)
            if any(row):  # Skip empty rows
                rows.append(row)
                
        if not rows:
            return ''
            
        # First row becomes headers
        headers = rows.pop(0) if rows else []
        if not headers:
            return ''
            
        # Build markdown table
        md_table = []
        md_table.append('| ' + ' | '.join(headers) + ' |')
        md_table.append('|' + '|'.join([' --- ' for _ in headers]) + '|')
        
        for row in rows:
            # Pad row if needed
            while len(row) < len(headers):
                row.append('')
            md_table.append('| ' + ' | '.join(row) + ' |')
            
        return '\n'.join(md_table)

    def parse_section_with_context(self, section_name: str, table_format: str = 'markdown') -> Dict:
        """
        Parse a section and extract text and tables while preserving their original positions.
        Tables are cleaned and converted to a more token-efficient format.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
            table_format: Format for tables ('markdown' or 'compact_html')
            
        Returns:
            Dictionary containing:
            - 'content': List of dictionaries, each with:
                - 'type': 'text' or 'table'
                - 'content': The actual content
            - 'tables': List of all tables (for convenience)
        """
        html_content = self.get_section_html(section_name)
        if not html_content:
            return {'content': [], 'tables': []}
        
        # Parse HTML content
        section_soup = BeautifulSoup(html_content, 'lxml')
        
        # Initialize content list and tables list
        content = []
        tables = []
        current_text = []
        
        def process_element(element):
            nonlocal current_text, content, tables
            
            if isinstance(element, str):
                cleaned = self._clean_html(str(element))
                if cleaned:
                    current_text.append(cleaned)
                return
                
            if element.name == 'table':
                # If we have accumulated text, add it first
                if current_text:
                    content.append({
                        'type': 'text',
                        'content': '\n'.join(current_text)
                    })
                    current_text = []
                
                # Add the table
                table_html = str(element)
                cleaned_table = self.clean_table_for_llm(table_html, table_format)
                if cleaned_table:  # Only include non-empty tables
                    content.append({
                        'type': 'table',
                        'content': cleaned_table
                    })
                    tables.append(cleaned_table)
                return
            
            # For non-table elements, first process their text content
            if element.string:
                cleaned = self._clean_html(str(element.string))
                if cleaned:
                    current_text.append(cleaned)
            
            # Then recursively process all child elements
            for child in element.children:
                process_element(child)
        
        # Process all top-level elements
        for element in section_soup.children:
            process_element(element)
        
        # Add any remaining text
        if current_text:
            content.append({
                'type': 'text',
                'content': '\n'.join(current_text)
            })
        
        return {
            'content': content,
            'tables': tables
        }

    def display_section_with_context(self, section_name: str) -> None:
        """
        Display a section's content with tables in their original positions.
        
        Args:
            section_name: Name of the section (e.g., 'Item 1', 'Item 1A')
        """
        result = self.parse_section_with_context(section_name)
        
        style = '''
        <style>
            .sec-wrapper {
                font-family: Arial, sans-serif;
                line-height: 1.5;
                padding: 20px;
                max-width: 800px;
                margin: 20px auto;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .sec-content {
                color: #333;
            }
            .sec-content p {
                margin: 10px 0;
            }
            .sec-content table {
                border-collapse: collapse;
                margin: 15px 0;
                width: 100%;
                background-color: white;
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
            .text-content {
                margin: 15px 0;
                white-space: pre-wrap;
            }
        </style>'''
        
        for item in result['content']:
            html_output = f"{style}<div class='sec-wrapper'><div class='sec-content'>"
            
            if item['type'] == 'text':
                # Format text content
                text = item['content'].replace('\n', '<br>')
                html_output += f"<div class='text-content'>{text}</div>"
            else:  # table
                # Clean up any existing styles from the table
                table_soup = BeautifulSoup(item['content'], 'lxml')
                table = table_soup.find('table')
                if table:
                    for element in table.find_all(True):
                        if 'style' in element.attrs:
                            del element['style']
                        if 'bgcolor' in element.attrs:
                            del element['bgcolor']
                        if 'color' in element.attrs:
                            del element['color']
                    html_output += str(table)
            
            html_output += "</div></div>"
            display(HTML(html_output))
    
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
