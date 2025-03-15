import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class SourceInformation:
    company: str
    year: str
    section: str
    subsection: Optional[str]
    span_text: str
    span_location: Dict[str, Any]
    contains_table: bool
    table_row: Optional[str]
    table_column: Optional[str]


@dataclass
class BenchmarkQuestion:
    id: str
    question: str
    answer: str
    source_information: List[SourceInformation]
    reasoning_path: List[str]
    question_type: str
    difficulty: str


class DataLoader:
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the benchmark questions and SEC filings
        """
        self.data_dir = data_dir
        self.benchmark_questions = []
        self.sec_filings = {}
        
        # Company name to ticker mapping
        self.company_to_ticker = {
            "Amazon.com Inc.": "AMZN",
            "AMAZON COM INC": "AMZN",
            "Alphabet Inc.": "GOOG",
            "ALPHABET INC": "GOOG",
            "Apple Inc.": "AAPL",
            "APPLE INC": "AAPL",
            "NVIDIA CORP": "NVDA",
            "NVIDIA Corporation": "NVDA",
            "Microsoft Corporation": "MSFT",
            "MICROSOFT CORP": "MSFT",
            "Walt Disney Co": "DIS",
            "WALT DISNEY CO": "DIS",
            "Pfizer Inc.": "PFE",
            "PFIZER INC": "PFE",
            "AT&T Inc.": "T",
            "AT&T INC": "T",
            "Uber Technologies, Inc.": "UBER",
            "UBER TECHNOLOGIES INC": "UBER",
            "Lyft, Inc.": "LYFT",
            "LYFT INC": "LYFT"
        }
        
        # Ticker to company name mapping (canonical name)
        self.ticker_to_company = {
            "AMZN": "Amazon.com Inc.",
            "GOOG": "Alphabet Inc.",
            "AAPL": "Apple Inc.",
            "NVDA": "NVIDIA Corporation",
            "MSFT": "Microsoft Corporation",
            "DIS": "Walt Disney Co",
            "PFE": "Pfizer Inc.",
            "T": "AT&T Inc.",
            "UBER": "Uber Technologies, Inc.",
            "LYFT": "Lyft, Inc."
        }
        
    def get_ticker_for_company(self, company_name: str) -> str:
        """
        Get the ticker for a company name.
        
        Args:
            company_name: Company name
            
        Returns:
            Ticker symbol or original company name if not found
        """
        return self.company_to_ticker.get(company_name, company_name)
        
    def load_benchmark_questions(self, benchmark_file: str) -> List[BenchmarkQuestion]:
        """
        Load benchmark questions from a JSON file.
        
        Args:
            benchmark_file: Path to the benchmark questions JSON file
            
        Returns:
            List of BenchmarkQuestion objects
        """
        file_path = os.path.join(self.data_dir, benchmark_file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        benchmark_questions = []
        
        for q_data in questions_data:
            # Convert source_information list
            source_info_list = []
            for src in q_data['source_information']:
                source_info_list.append(SourceInformation(
                    company=src['company'],
                    year=src['year'],
                    section=src['section'],
                    subsection=src.get('subsection'),
                    span_text=src['span_text'],
                    span_location=src['span_location'],
                    contains_table=src['contains_table'],
                    table_row=src.get('table_row'),
                    table_column=src.get('table_column')
                ))
            
            # Create BenchmarkQuestion object
            question = BenchmarkQuestion(
                id=q_data['id'],
                question=q_data['question'],
                answer=q_data['answer'],
                source_information=source_info_list,
                reasoning_path=q_data['reasoning_path'],
                question_type=q_data['question_type'],
                difficulty=q_data['difficulty']
            )
            
            benchmark_questions.append(question)
        
        self.benchmark_questions = benchmark_questions
        return benchmark_questions
    
    def load_preprocessed_reports(self, reports_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Load preprocessed SEC filing reports.
        
        Args:
            reports_dir: Directory containing the preprocessed reports
            
        Returns:
            Dictionary with structure: {ticker: {year: {section: content}}}
        """
        reports_path = os.path.join(self.data_dir, reports_dir)
        
        # Find all JSON report files
        report_files = [f for f in os.listdir(reports_path) if f.startswith('report_') and f.endswith('.json')]
        
        sec_filings = {}
        
        for report_file in report_files:
            try:
                with open(os.path.join(reports_path, report_file), 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                company = report_data.get('company_name', '')
                ticker = report_data.get('ticker', '')
                year = report_data.get('year', '')
                
                if not ticker or not year:
                    # Try to extract ticker from filename if not in the data
                    # Format: report_TICKER_YEAR.json
                    if '_' in report_file and report_file.startswith('report_'):
                        parts = report_file.replace('.json', '').split('_')
                        if len(parts) >= 3:
                            ticker = parts[1]
                            try:
                                year = int(parts[2])
                            except ValueError:
                                continue
                
                if not ticker or not year:
                    continue
                
                if ticker not in sec_filings:
                    sec_filings[ticker] = {}
                
                if year not in sec_filings[ticker]:
                    sec_filings[ticker][year] = {}
                
                # Store sections
                for section_id, content in report_data.get('section_content', {}).items():
                    sec_filings[ticker][year][section_id] = content
                
            except Exception as e:
                print(f"Error loading report file {report_file}: {e}")
        
        self.sec_filings = sec_filings
        return sec_filings
    
    def get_section_content(self, company: str, year: str, section: str) -> Optional[str]:
        """
        Get the content of a specific section from a SEC filing.
        
        Args:
            company: Company name or ticker
            year: Year of the filing
            section: Section ID (e.g., "Item 1A")
            
        Returns:
            Section content or None if not found
        """
        # Convert company name to ticker if needed
        ticker = self.get_ticker_for_company(company)
        
        # Convert year to int if it's a string
        if isinstance(year, str):
            try:
                year = int(year)
            except ValueError:
                pass
        
        if ticker in self.sec_filings and year in self.sec_filings[ticker] and section in self.sec_filings[ticker][year]:
            return self.sec_filings[ticker][year][section]
        
        return None