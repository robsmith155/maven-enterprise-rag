import glob
import json
import os
import re
import tiktoken
from typing import Dict, List

from .benchmark.dataclasses import CompanyReport, QuestionSpec


def load_preprocessed_reports_metadata(preprocessed_dir: str) -> List[CompanyReport]:
    """
    Load preprocessed report metadata from the specified directory.
    
    Args:
        preprocessed_dir: Directory containing preprocessed reports
        
    Returns:
        List of CompanyReport objects
    """
    company_reports = []
    
    # Find all JSON report files
    report_files = glob.glob(os.path.join(preprocessed_dir, "report_*.json"))
    
    for report_file in report_files:
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # Create CompanyReport object
            company_report = CompanyReport(
                company_name=report_data.get('company_name', ''),
                ticker=report_data.get('ticker', ''),
                year=report_data.get('year', 0),
                raw_file_path=report_data.get('file_path', ''),
                industry=report_data.get('industry', ''),
                available_sections=set(report_data.get('available_sections', [])),
                accession_number=report_data.get('accession_number', '')
            )
            
            company_reports.append(company_report)
            
        except Exception as e:
            print(f"Error loading report file {report_file}: {e}")
    
    return company_reports


def extract_sections_from_reports(question_spec: QuestionSpec, preprocessed_dir: str, max_tokens_per_section: int = 4000) -> Dict[str, Dict[int, Dict[str, str]]]:
    """
    Extract sections from preprocessed reports.
    
    Args:
        question_spec: Question specification
        preprocessed_dir: Directory containing preprocessed reports
        max_tokens_per_section: Maximum tokens per section
        
    Returns:
        Dictionary of extracted content
    """
    extracted_content = {}
    
    for report in question_spec.reports:
        company = report.ticker
        year = report.year
        
        if company not in extracted_content:
            extracted_content[company] = {}
        if year not in extracted_content[company]:
            extracted_content[company][year] = {}
        
        # Find the preprocessed report file
        report_file = os.path.join(preprocessed_dir, f"report_{report.ticker}_{report.year}.json")
        
        if not os.path.exists(report_file):
            print(f"Warning: Preprocessed report file not found: {report_file}")
            continue
        
        try:
            # Load the preprocessed report
            with open(report_file, 'r', encoding='utf-8') as f:
                preprocessed_report = json.load(f)
            
            # Extract the specified sections
            # Use the same path format as in determine_question_specification
            processed_file_path = os.path.join(preprocessed_dir, f"report_{report.ticker}_{report.year}.json")
            for section_id in question_spec.sections_to_include.get(processed_file_path, []):
                if section_id in preprocessed_report.get('section_content', {}):
                    section_content = preprocessed_report['section_content'][section_id]
                    
                    # Check token count and truncate if necessary
                    encoding = tiktoken.encoding_for_model("gpt-4")
                    tokens = encoding.encode(section_content)
                    
                    if len(tokens) > max_tokens_per_section:
                        # Truncate to max_tokens_per_section
                        truncated_tokens = tokens[:max_tokens_per_section]
                        section_content = encoding.decode(truncated_tokens)
                    
                    # Store the content
                    extracted_content[company][year][section_id] = section_content
        except Exception as e:
            print(f"Error extracting sections from {report_file}: {e}")
    
    return extracted_content


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