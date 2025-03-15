import os
import re
import json
from datetime import datetime
from tqdm import tqdm
import traceback
import tiktoken
from dataclasses import dataclass, asdict
from typing import Set, Optional, Dict, List, Any, Tuple
from bs4 import BeautifulSoup

from .sec_filing_parser import SECFilingParser

@dataclass
class SECReport:
    """Representation of an SEC 10-K filing report"""
    company_ticker: str
    company_name: Optional[str]  # Will be populated later
    year: int
    file_path: str
    accession_number: str

@dataclass
class SectionStats:
    """Statistics about a section in a report"""
    word_count: int
    token_count: int
    table_count: int
    has_content: bool

@dataclass
class CompanyReport:
    """Representation of a company's SEC filing report with parsed sections"""
    company_name: str
    ticker: str
    year: int
    file_path: str
    industry: Optional[str]
    available_sections: Set[str]
    section_stats: Dict[str, SectionStats]
    section_content: Dict[str, str]  
    accession_number: str

def count_words(text: str) -> int:
    """Count the number of words in a text string"""
    return len(text.split())

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximate token count if tiktoken fails
        return len(text.split()) * 1.3  # Rough approximation

def count_tables(html_content: str) -> int:
    """Count the number of tables in HTML content"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        return len(soup.find_all('table'))
    except Exception as e:
        return 0

# Helper function to make dataclasses JSON serializable
def serialize_report(report):
    """Convert a CompanyReport to a JSON-serializable dictionary"""
    report_dict = asdict(report)
    # Convert sets to lists for JSON serialization
    report_dict['available_sections'] = list(report_dict['available_sections'])
    # Convert SectionStats objects to dictionaries
    for section_id, stats in report_dict['section_stats'].items():
        report_dict['section_stats'][section_id] = asdict(stats)
    return report_dict

def find_sec_reports(base_dir: str) -> List[SECReport]:
    """
    Scan the base directory to find all SEC 10-K reports.
    
    Args:
        base_dir: Base directory containing SEC filings (e.g., './data/raw2/sec-edgar-filings')
        
    Returns:
        List of SECReport objects
    """
    sec_reports = []
    
    # Get all company tickers (directories in the base_dir)
    print("Finding all company directories...")
    company_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Found {len(company_dirs)} company directories")
    
    print("Finding all SEC 10-K reports...")
    for ticker in tqdm(company_dirs, desc="Scanning companies"):
        # Path to 10-K reports for this company
        report_type_dir = os.path.join(base_dir, ticker, "10-K")
        
        if not os.path.exists(report_type_dir):
            continue
        
        # Get all filing directories
        try:
            filing_dirs = [d for d in os.listdir(report_type_dir) if os.path.isdir(os.path.join(report_type_dir, d))]
        except Exception as e:
            print(f"Error reading directory {report_type_dir}: {e}")
            continue
        
        for filing_dir in filing_dirs:
            # Extract year from the accession number
            match = re.match(r'\d+-(\d{2})-\d+', filing_dir)
            if match:
                year_suffix = match.group(1)
                year = 2000 + int(year_suffix) if int(year_suffix) < 50 else 1900 + int(year_suffix)
                
                # Full path to the submission file
                file_path = os.path.join(report_type_dir, filing_dir, "full-submission.txt")
                
                if os.path.exists(file_path):
                    report = SECReport(
                        company_ticker=ticker,
                        company_name=None,
                        year=year,
                        file_path=file_path,
                        accession_number=filing_dir
                    )
                    sec_reports.append(report)
    
    print(f"Found {len(sec_reports)} SEC 10-K reports")
    return sec_reports


def process_sec_reports(sec_reports: List[SECReport], output_dir: str, sections_to_extract: List[str], stats_save_interval: int = 100) -> Tuple[Dict, Dict, Dict]:
    """
    Process SEC reports to extract sections and calculate statistics.
    
    Args:
        sec_reports: List of SECReport objects to process
        output_dir: Directory to save output files
        sections_to_extract: List of section IDs to extract
        stats_save_interval: Interval for saving progress statistics
        
    Returns:
        Tuple of (report_success_stats, section_failures, section_stats_summary)
    """
    
    # Create data structures to track failures and statistics
    section_failures = {section_id: {"count": 0, "errors": {}} for section_id in sections_to_extract}
    section_statistics = {section_id: {"word_count": [], "token_count": [], "table_count": []} for section_id in sections_to_extract}
    
    report_success_stats = {"total": 0, "with_sections": {}}
    report_failure_count = 0
    
    # Process reports to get available sections
    print("Processing reports to extract metadata and available sections...")
    successful_report_count = 0

    section_content_dict = {}
    
    for idx, sec_report in enumerate(tqdm(sec_reports, desc="Processing reports")):
        report_id = f"{sec_report.company_ticker}_{sec_report.year}_{sec_report.accession_number}"
        report_sections = {"success": [], "failed": []}
        section_stats_dict = {}
        
        try:
            # Parse basic metadata
            # Create parser instance
            parser = SECFilingParser()
            parser.read_file(sec_report.file_path)
            metadata = parser.extract_metadata()
            
            company_name = metadata.get("company_name", sec_report.company_ticker)
            industry = metadata.get("industry", "Unknown")
            
            # Get available sections - check all important sections
            available_sections = set()
            for section_id in sections_to_extract:
                try:
                    # Get both text and HTML versions of the section
                    section_result = parser.parse_section(section_id, output_format='llm')
                    section_content = section_result.get('text', '')
                    
                    # # Try to get HTML content for table counting
                    # try:
                    #     html_content = parser.parse_section(section_id, output_format='html')
                    # except:
                    #     html_content = ""
                    
                    if section_content and len(section_content.strip()) > 100:
                        # Store the section content
                        section_content_dict[section_id] = section_content

                        # Calculate statistics
                        word_count = count_words(section_content)
                        token_count = count_tokens(section_content)
                        table_count = count_tables(section_content)
                        
                        # Store section statistics
                        section_stats = SectionStats(
                            word_count=word_count,
                            token_count=token_count,
                            table_count=table_count,
                            has_content=True
                        )
                        section_stats_dict[section_id] = section_stats
                        
                        # Add to available sections
                        available_sections.add(section_id)
                        report_sections["success"].append(section_id)
                        
                        # Track successful section stats
                        if section_id not in report_success_stats["with_sections"]:
                            report_success_stats["with_sections"][section_id] = 0
                        report_success_stats["with_sections"][section_id] += 1
                        
                        # Add to section statistics
                        section_statistics[section_id]["word_count"].append(word_count)
                        section_statistics[section_id]["token_count"].append(token_count)
                        section_statistics[section_id]["table_count"].append(table_count)
                        
                    else:
                        # Section was found but content was too short or empty
                        error_type = "empty_or_short_content"
                        error_message = "Content too short or empty"
                        
                        # Store empty section statistics
                        section_stats = SectionStats(
                            word_count=0,
                            token_count=0,
                            table_count=0,
                            has_content=False
                        )
                        section_stats_dict[section_id] = section_stats
                        
                        report_sections["failed"].append({
                            "section": section_id,
                            "error_type": error_type,
                            "error_message": error_message
                        })
                        
                        # Save failure details
                        failure_detail = {
                            "report_path": sec_report.file_path,
                            "company": sec_report.company_ticker,
                            "year": sec_report.year,
                            "section_id": section_id,
                            "error_type": error_type,
                            "error_message": error_message
                        }
                        
                        failure_filename = f"section_failure_{sec_report.company_ticker}_{sec_report.year}_{section_id}_{error_type}.json"
                        with open(os.path.join(output_dir, "failures", failure_filename), "w") as f:
                            json.dump(failure_detail, f, indent=2)
                        
                        # Track section failure stats
                        section_failures[section_id]["count"] += 1
                        if error_type not in section_failures[section_id]["errors"]:
                            section_failures[section_id]["errors"][error_type] = 0
                        section_failures[section_id]["errors"][error_type] += 1
                        
                except Exception as e:
                    error_message = str(e)
                    error_type = type(e).__name__
                    stack_trace = traceback.format_exc()
                    
                    # Store error section statistics
                    section_stats = SectionStats(
                        word_count=0,
                        token_count=0,
                        table_count=0,
                        has_content=False
                    )
                    section_stats_dict[section_id] = section_stats
                    
                    report_sections["failed"].append({
                        "section": section_id,
                        "error_type": error_type,
                        "error_message": error_message
                    })
                    
                    # Save failure details
                    failure_detail = {
                        "report_path": sec_report.file_path,
                        "company": sec_report.company_ticker,
                        "year": sec_report.year,
                        "section_id": section_id,
                        "error_type": error_type,
                        "error_message": error_message,
                        "stack_trace": stack_trace
                    }
                    
                    failure_filename = f"section_failure_{sec_report.company_ticker}_{sec_report.year}_{section_id}_{error_type}.json"
                    with open(os.path.join(output_dir, "failures", failure_filename), "w") as f:
                        json.dump(failure_detail, f, indent=2)
                    
                    # Track section failure stats
                    section_failures[section_id]["count"] += 1
                    if error_type not in section_failures[section_id]["errors"]:
                        section_failures[section_id]["errors"][error_type] = 0
                    section_failures[section_id]["errors"][error_type] += 1
            
            # Only save reports with at least one valid section
            if available_sections:
                # Create report object
                report = CompanyReport(
                    company_name=company_name,
                    ticker=sec_report.company_ticker,
                    year=sec_report.year,
                    file_path=sec_report.file_path,
                    industry=industry,
                    available_sections=available_sections,
                    section_stats=section_stats_dict,
                    section_content=section_content_dict,
                    accession_number=sec_report.accession_number
                )
                
                # Save report to file
                report_filename = f"report_{sec_report.company_ticker}_{sec_report.year}.json"
                with open(os.path.join(output_dir, "reports", report_filename), "w") as f:
                    json.dump(serialize_report(report), f, indent=2)
                
                successful_report_count += 1
                report_success_stats["total"] += 1
            else:
                # Track report with no valid sections
                report_failure_count += 1
                
                # Save failure details
                failure_detail = {
                    "report_path": sec_report.file_path,
                    "company": sec_report.company_ticker,
                    "year": sec_report.year,
                    "error_type": "no_valid_sections",
                    "error_message": "No valid sections found in report",
                    "section_failures": report_sections["failed"]
                }
                
                failure_filename = f"report_failure_{sec_report.company_ticker}_{sec_report.year}_no_valid_sections.json"
                with open(os.path.join(output_dir, "failures", failure_filename), "w") as f:
                    json.dump(failure_detail, f, indent=2)
            
        except Exception as e:
            error_message = str(e)
            error_type = type(e).__name__
            stack_trace = traceback.format_exc()
            
            report_failure_count += 1
            
            # Save failure details
            failure_detail = {
                "report_path": sec_report.file_path,
                "company": sec_report.company_ticker,
                "year": sec_report.year,
                "error_type": error_type,
                "error_message": error_message,
                "stack_trace": stack_trace,
                "section_failures": report_sections["failed"]
            }
            
            failure_filename = f"report_failure_{sec_report.company_ticker}_{sec_report.year}_{error_type}.json"
            with open(os.path.join(output_dir, "failures", failure_filename), "w") as f:
                json.dump(failure_detail, f, indent=2)
        
        # Save running statistics periodically
        if (idx + 1) % stats_save_interval == 0 or (idx + 1) == len(sec_reports):
            # Calculate current section statistics
            section_stats_summary = calculate_section_statistics(section_statistics, sections_to_extract)
            
            # Save current progress summary
            progress_summary = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reports_processed": idx + 1,
                "total_reports": len(sec_reports),
                "successful_reports": successful_report_count,
                "failed_reports": report_failure_count,
                "section_failures": section_failures,
                "section_statistics": section_stats_summary,
                "report_success_stats": report_success_stats
            }
            
            with open(os.path.join(output_dir, f"processing_progress.json"), "w") as f:
                json.dump(progress_summary, f, indent=2)
    
    print(f"Found {successful_report_count} valid reports with parseable sections")
    
    # Calculate final section statistics
    section_stats_summary = calculate_section_statistics(section_statistics, sections_to_extract)
    
    return report_success_stats, section_failures, section_stats_summary, report_failure_count


def calculate_section_statistics(section_statistics: Dict, sections_to_extract: List[str]) -> Dict:
    """
    Calculate summary statistics for each section based on collected data.
    
    Args:
        section_statistics: Dictionary containing raw statistics data
        sections_to_extract: List of section IDs
        
    Returns:
        Dictionary of summary statistics for each section
    """
    section_stats_summary = {}
    for section_id in sections_to_extract:
        word_counts = section_statistics[section_id]["word_count"]
        token_counts = section_statistics[section_id]["token_count"]
        table_counts = section_statistics[section_id]["table_count"]
        
        if word_counts:
            section_stats_summary[section_id] = {
                "word_count": {
                    "min": min(word_counts),
                    "max": max(word_counts),
                    "avg": sum(word_counts) / len(word_counts),
                    "median": sorted(word_counts)[len(word_counts) // 2],
                    "total": sum(word_counts)
                },
                "token_count": {
                    "min": min(token_counts),
                    "max": max(token_counts),
                    "avg": sum(token_counts) / len(token_counts),
                    "median": sorted(token_counts)[len(token_counts) // 2],
                    "total": sum(token_counts)
                },
                "table_count": {
                    "min": min(table_counts),
                    "max": max(table_counts),
                    "avg": sum(table_counts) / len(table_counts),
                    "median": sorted(table_counts)[len(table_counts) // 2],
                    "total": sum(table_counts)
                },
                "count": len(word_counts)
            }
        else:
            section_stats_summary[section_id] = {
                "word_count": {"min": 0, "max": 0, "avg": 0, "median": 0, "total": 0},
                "token_count": {"min": 0, "max": 0, "avg": 0, "median": 0, "total": 0},
                "table_count": {"min": 0, "max": 0, "avg": 0, "median": 0, "total": 0},
                "count": 0
            }
    
    return section_stats_summary

def display_summary_statistics(
    sec_reports: List[SECReport], 
    report_success_stats: Dict, 
    section_failures: Dict, 
    section_stats_summary: Dict, 
    report_failure_count: int,
    sections_to_extract: List[str]
):
    """
    Display summary statistics for the processed reports.
    
    Args:
        sec_reports: List of all SEC reports
        report_success_stats: Statistics about successful reports
        section_failures: Information about section failures
        section_stats_summary: Summary statistics for each section
        report_failure_count: Count of failed reports
        sections_to_extract: List of section IDs that were extracted
    """
    print("\n=== PROCESSING SUMMARY ===")
    print(f"Total reports processed: {len(sec_reports)}")
    print(f"Reports with at least one valid section: {report_success_stats['total']} ({report_success_stats['total']/len(sec_reports)*100:.1f}%)")
    print(f"Reports with no valid sections: {report_failure_count} ({report_failure_count/len(sec_reports)*100:.1f}%)")
    
    print("\n=== SECTION SUCCESS RATES ===")
    for section_id in sections_to_extract:
        success_count = report_success_stats["with_sections"].get(section_id, 0)
        failure_count = section_failures[section_id]["count"]
        total = success_count + failure_count
        success_rate = (success_count / total * 100) if total > 0 else 0
        
        print(f"{section_id}: {success_count}/{total} successful ({success_rate:.1f}%)")
    
    print("\n=== SECTION STATISTICS ===")
    for section_id in sections_to_extract:
        stats = section_stats_summary[section_id]
        if stats["count"] > 0:
            print(f"\n{section_id} statistics ({stats['count']} sections):")
            print(f"  Word count: avg={stats['word_count']['avg']:.1f}, median={stats['word_count']['median']}, min={stats['word_count']['min']}, max={stats['word_count']['max']}")
            print(f"  Token count: avg={stats['token_count']['avg']:.1f}, median={stats['token_count']['median']}, min={stats['token_count']['min']}, max={stats['token_count']['max']}")
            print(f"  Table count: avg={stats['table_count']['avg']:.1f}, median={stats['table_count']['median']}, min={stats['table_count']['min']}, max={stats['table_count']['max']}")
    
    print("\n=== TOP FAILURE REASONS BY SECTION ===")
    for section_id in sections_to_extract:
        if section_failures[section_id]["count"] > 0:
            print(f"\n{section_id} failures ({section_failures[section_id]['count']} total):")
            # Sort errors by frequency
            sorted_errors = sorted(
                section_failures[section_id]["errors"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for error_type, count in sorted_errors[:5]:  # Show top 5 errors
                print(f"  - {error_type}: {count} occurrences ({count/section_failures[section_id]['count']*100:.1f}%)")

def save_final_summary(
    output_dir: str,
    sec_reports: List[SECReport],
    report_success_stats: Dict,
    report_failure_count: int,
    section_failures: Dict,
    section_stats_summary: Dict,
    sections_to_extract: List[str]
):
    """
    Save the final summary statistics to a JSON file.
    
    Args:
        output_dir: Directory to save the summary file
        sec_reports: List of all SEC reports
        report_success_stats: Statistics about successful reports
        report_failure_count: Count of failed reports
        section_failures: Information about section failures
        section_stats_summary: Summary statistics for each section
        sections_to_extract: List of section IDs that were extracted
    """
    # Save final summary
    final_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_reports": len(sec_reports),
            "successful_reports": report_success_stats["total"],
            "failed_reports": report_failure_count,
            "section_failures": section_failures,
            "section_success_rates": {
                section_id: {
                    "success_count": report_success_stats["with_sections"].get(section_id, 0),
                    "failure_count": section_failures[section_id]["count"],
                    "success_rate": (report_success_stats["with_sections"].get(section_id, 0) / 
                                    (report_success_stats["with_sections"].get(section_id, 0) + section_failures[section_id]["count"]) * 100)
                    if (report_success_stats["with_sections"].get(section_id, 0) + section_failures[section_id]["count"]) > 0 else 0
                }
                for section_id in sections_to_extract
            }
        },
        "section_statistics": section_stats_summary
    }
    
    with open(os.path.join(output_dir, "sec_report_processing_final_summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\nFinal summary saved to {os.path.join(output_dir, 'sec_report_processing_final_summary.json')}")

def serialize_report(report):
    """Convert a CompanyReport to a JSON-serializable dictionary"""
    report_dict = asdict(report)
    # Convert sets to lists for JSON serialization
    report_dict['available_sections'] = list(report_dict['available_sections'])
    # Convert SectionStats objects to dictionaries
    for section_id, stats in report_dict['section_stats'].items():
        try:
            # Try to convert to dict using asdict
            report_dict['section_stats'][section_id] = asdict(stats)
        except TypeError:
            # If stats is not a dataclass instance, check if it's already a dict
            if isinstance(stats, dict):
                report_dict['section_stats'][section_id] = stats
            else:
                # Create a dict with the same structure as SectionStats
                report_dict['section_stats'][section_id] = {
                    'word_count': getattr(stats, 'word_count', 0),
                    'token_count': getattr(stats, 'token_count', 0),
                    'table_count': getattr(stats, 'table_count', 0),
                    'has_content': getattr(stats, 'has_content', False)
                }
    return report_dict


def process_all_sec_reports(base_dir: str, sections_to_extract: list, output_dir: str = None, company_tickers: list = None):
    """
    Main function to process all SEC reports in the base directory.
    
    Args:
        base_dir: Base directory containing SEC filings (e.g., './data/raw/sec-edgar-filings')
        output_dir: Directory to save output files (default: creates a timestamped directory)
        
    Returns:
        Path to the output directory containing all results
    """
    # Define the sections to extract (excluding Item 3 and Item 6)
    # sections_to_extract = ["Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8"]
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = f"sec_processing_results_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "failures"), exist_ok=True)
    
    # Step 1: Find all SEC reports
    sec_reports = find_sec_reports(base_dir)

    # Filter reports
    if company_tickers is not None:
        sec_reports = [r for r in sec_reports if r.company_ticker in company_tickers]
    
    # Save the list of reports to process
    with open(os.path.join(output_dir, "sec_reports_to_process.json"), "w") as f:
        json.dump([asdict(report) for report in sec_reports], f, indent=2)
    
    # Step 2: Process the reports
    report_success_stats, section_failures, section_stats_summary, report_failure_count = process_sec_reports(
        sec_reports=sec_reports,
        output_dir=output_dir,
        sections_to_extract=sections_to_extract,
        stats_save_interval=100
    )
    
    # Step 3: Display summary statistics
    display_summary_statistics(
        sec_reports=sec_reports,
        report_success_stats=report_success_stats,
        section_failures=section_failures,
        section_stats_summary=section_stats_summary,
        report_failure_count=report_failure_count,
        sections_to_extract=sections_to_extract
    )
    
    # Step 4: Save final summary
    save_final_summary(
        output_dir=output_dir,
        sec_reports=sec_reports,
        report_success_stats=report_success_stats,
        report_failure_count=report_failure_count,
        section_failures=section_failures,
        section_stats_summary=section_stats_summary,
        sections_to_extract=sections_to_extract
    )
    
    print(f"All processing results saved to {output_dir}")
    return output_dir
