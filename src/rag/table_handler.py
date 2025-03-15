# src/rag/table_handler.py

import uuid
from typing import List, Dict, Any, Optional
import re
from bs4 import BeautifulSoup
from openai import OpenAI
from .chunking import Chunk


class TableHandler:
    """Handler for processing tables in SEC filings."""
    
    def __init__(
        self,
        table_context_size: int = 500,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize the table handler.
        
        Args:
            table_context_size: Number of characters of context to include around tables
            model: OpenAI model to use for table description
        """
        self.table_context_size = table_context_size
        self.model = model
        self.client = OpenAI()
    
    def extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tables from HTML text.
        
        Args:
            text: HTML text containing tables
            
        Returns:
            List of dictionaries with table information
        """
        tables = []
        
        # Parse HTML
        soup = BeautifulSoup(text, 'html.parser')
        
        # Find all tables
        html_tables = soup.find_all('table')
        
        for i, table in enumerate(html_tables):
            # Get table HTML
            table_html = str(table)
            
            # Get table position in text
            start = text.find(table_html)
            end = start + len(table_html)
            
            if start == -1:
                continue
            
            # Get context before and after table
            context_before = text[max(0, start - self.table_context_size):start]
            context_after = text[end:min(len(text), end + self.table_context_size)]
            
            # Store table information
            tables.append({
                'table_id': i,
                'table_html': table_html,
                'context_before': context_before,
                'context_after': context_after,
                'start': start,
                'end': end
            })
        
        return tables
    
    def describe_table(self, table_html: str, context: str) -> str:
        """
        Generate a description of a table using OpenAI.
        
        Args:
            table_html: HTML content of the table
            context: Text surrounding the table
            
        Returns:
            Description of the table
        """
        prompt = f"""
        Please analyze the following table from an SEC filing and provide a detailed description.
        Focus on the key financial metrics, trends, and insights that would be relevant for answering questions.
        
        Context surrounding the table:
        {context}
        
        Table:
        {table_html}
        
        Your description should:
        1. Identify the main purpose of the table
        2. Highlight key metrics and their values
        3. Note any significant trends or changes
        4. Explain the relationship between different columns/rows
        5. Provide a concise summary of what the table shows
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert at interpreting SEC filing tables."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating table description: {e}")
            return f"[Table description unavailable: {str(e)}]"
    
    def convert_table_to_markdown(self, table_html: str) -> str:
        """
        Convert an HTML table to markdown format.
        
        Args:
            table_html: HTML content of the table
            
        Returns:
            Markdown representation of the table
        """
        try:
            soup = BeautifulSoup(table_html, 'html.parser')
            table = soup.find('table')
            
            if not table:
                return "[No table found]"
            
            # Extract headers
            headers = []
            header_row = table.find('thead')
            if header_row:
                headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
            else:
                # Try to get headers from first row
                first_row = table.find('tr')
                if first_row:
                    headers = [th.get_text().strip() for th in first_row.find_all(['th', 'td'])]
            
            # If no headers found, use placeholder
            if not headers:
                headers = ["Column " + str(i+1) for i in range(len(table.find('tr').find_all(['th', 'td'])))]
            
            # Build markdown table
            markdown = "| " + " | ".join(headers) + " |\n"
            markdown += "| " + " | ".join(["---" for _ in headers]) + " |\n"
            
            # Extract rows
            rows = table.find_all('tr')
            
            # Skip header row if it was included in headers
            start_idx = 1 if headers and len(rows) > 0 else 0
            
            for row in rows[start_idx:]:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = [cell.get_text().strip() for cell in cells]
                    # Pad row if needed
                    while len(row_data) < len(headers):
                        row_data.append("")
                    # Truncate row if needed
                    row_data = row_data[:len(headers)]
                    markdown += "| " + " | ".join(row_data) + " |\n"
            
            return markdown
            
        except Exception as e:
            print(f"Error converting table to markdown: {e}")
            return f"[Table conversion failed: {str(e)}]"
    
    def process_tables(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Process tables in text and create chunks.
        
        Args:
            text: HTML text containing tables
            metadata: Metadata to attach to chunks
            
        Returns:
            List of Chunk objects for tables
        """
        chunks = []
        
        # Extract tables
        tables = self.extract_tables(text)
        
        for table_info in tables:
            table_html = table_info['table_html']
            context = table_info['context_before'] + table_info['context_after']
            
            # Generate table description
            description = self.describe_table(table_html, context)
            
            # Convert table to markdown
            markdown = self.convert_table_to_markdown(table_html)
            
            # Create metadata
            table_metadata = metadata.copy()
            table_metadata['contains_table'] = True
            table_metadata['table_id'] = table_info['table_id']
            table_metadata['start_char'] = table_info['start']
            table_metadata['end_char'] = table_info['end']
            
            # Create chunk for table description
            description_chunk = Chunk(
                text=f"Table Description: {description}\n\nOriginal Table:\n{markdown}",
                metadata=table_metadata,
                chunk_id=str(uuid.uuid4())
            )
            chunks.append(description_chunk)
            
            # Create chunk for table with context
            context_chunk = Chunk(
                text=f"Context before table:\n{table_info['context_before']}\n\nTable:\n{markdown}\n\nContext after table:\n{table_info['context_after']}",
                metadata=table_metadata,
                chunk_id=str(uuid.uuid4())
            )
            chunks.append(context_chunk)
        
        return chunks


# src/rag/table_handler.py

import re
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup

def process_tables(html_content: str) -> str:
    """
    Process tables in HTML content.
    
    Args:
        html_content: HTML content with tables
        
    Returns:
        Processed HTML content
    """
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Process tables
    tables = soup.find_all('table')
    for table in tables:
        # Skip footnote tables
        if _is_footnote_table(table):
            # Convert to text
            table_text = _table_to_text(table)
            # Replace table with text
            table.replace_with(soup.new_string(table_text))
        else:
            # Clean table
            _clean_table(table)
    
    return str(soup)

def _is_footnote_table(table) -> bool:
    """
    Check if a table is a footnote table.
    
    Args:
        table: BeautifulSoup table element
        
    Returns:
        True if the table is a footnote table, False otherwise
    """
    # Check table size
    rows = table.find_all('tr')
    if len(rows) < 2:
        return False
    
    # Check for footnote indicators
    footnote_indicators = ['footnote', 'note:', 'notes:', '(1)', '(2)', '(3)', '[1]', '[2]', '[3]']
    table_text = table.get_text().lower()
    
    for indicator in footnote_indicators:
        if indicator in table_text:
            return True
    
    return False

def _table_to_text(table) -> str:
    """
    Convert a table to text.
    
    Args:
        table: BeautifulSoup table element
        
    Returns:
        Text representation of the table
    """
    rows = table.find_all('tr')
    text_rows = []
    
    for row in rows:
        cells = row.find_all(['td', 'th'])
        text_cells = [cell.get_text().strip() for cell in cells]
        text_row = ' | '.join(text_cells)
        if text_row:
            text_rows.append(text_row)
    
    return '\n'.join(text_rows)

def _clean_table(table) -> None:
    """
    Clean a table in place.
    
    Args:
        table: BeautifulSoup table element
    """
    # Add class for styling
    table['class'] = table.get('class', []) + ['cleaned-financial-table']
    
    # Process cells
    for cell in table.find_all(['td', 'th']):
        # Clean whitespace
        cell_text = cell.get_text().strip()
        cell.string = cell_text
        
        # Handle currency and numbers
        if re.search(r'^\$?\(?\d[\d,\.]*\)?$', cell_text):
            cell['class'] = cell.get('class', []) + ['numeric-cell']


def remove_html_tables(content: str) -> str:
    """
    Remove HTML tables from content while preserving other text
    
    Args:
        content: Content string that may contain HTML tables
        
    Returns:
        Content with HTML tables removed
    """
    # First try to match complete tables
    table_pattern = re.compile(r'<table[\s\S]*?</table>', re.IGNORECASE | re.DOTALL)
    cleaned_content = re.sub(table_pattern, '[TABLE REMOVED]', content)
    
    # If there are still opening table tags, remove everything from <table to the end of the string
    # This handles incomplete tables that might be cut off
    open_table_pattern = re.compile(r'<table[\s\S]*', re.IGNORECASE)
    cleaned_content = re.sub(open_table_pattern, '[INCOMPLETE TABLE REMOVED]', cleaned_content)
    
    # Also look for any remaining table-related tags
    tr_pattern = re.compile(r'<tr[\s\S]*?</tr>', re.IGNORECASE | re.DOTALL)
    td_pattern = re.compile(r'<td[\s\S]*?</td>', re.IGNORECASE | re.DOTALL)
    th_pattern = re.compile(r'<th[\s\S]*?</th>', re.IGNORECASE | re.DOTALL)
    
    cleaned_content = re.sub(tr_pattern, '', cleaned_content)
    cleaned_content = re.sub(td_pattern, '', cleaned_content)
    cleaned_content = re.sub(th_pattern, '', cleaned_content)
    
    return cleaned_content