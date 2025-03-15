import os
import random
from typing import Dict, List

from .dataclasses import CompanyReport, QuestionCategory, QuestionType, QuestionDifficulty, QuestionSpec

def get_relevant_sections_for_category(category: QuestionCategory) -> List[str]:
    """
    Get the relevant sections for a given question category.
    
    Args:
        category: Question category
        
    Returns:
        List of relevant section IDs
    """
    # Map categories to relevant sections
    category_to_sections = {
        QuestionCategory.FINANCIAL_METRIC: ["Item 7", "Item 8"],
        QuestionCategory.RISK_FACTOR: ["Item 1A"],
        QuestionCategory.BUSINESS_OVERVIEW: ["Item 1"],
        QuestionCategory.MANAGEMENT_DISCUSSION: ["Item 7"],
        QuestionCategory.FORWARD_LOOKING: ["Item 7"],
        QuestionCategory.SEGMENT_ANALYSIS: ["Item 7", "Item 8"],
        QuestionCategory.RISK_FACTOR: ["Item 5"],
        QuestionCategory.TABLE_ANALYSIS: ["Item 7", "Item 8", "Item 7A"]
    }
    
    return category_to_sections.get(category, ["Item 1", "Item 1A", "Item 5", "Item 7", "Item 7A", "Item 8"])


def determine_question_specification(companies: Dict[str, Dict[int, CompanyReport]], processed_reports_dir: str) -> QuestionSpec:
    """
    Determine the specification for a question.
    
    Args:
        companies: Dictionary of companies with their reports by year
        processed_reports_dir: Path to directory containing the processed SEC 10-K filing reports
        
    Returns:
        QuestionSpec object
    """
   
    # Ensure we have enough data for each question type
    available_companies = list(companies.keys())
    
    if len(available_companies) < 2:
        # If we have only one company, we can only do single company questions
        valid_question_types = [
            QuestionType.SINGLE_COMPANY_SINGLE_YEAR,
            QuestionType.SINGLE_COMPANY_MULTI_YEAR
        ]
    else:
        # We have multiple companies, so all question types are valid
        # Exclude NULL_QUESTION as it's handled separately above
        valid_question_types = [
            QuestionType.SINGLE_COMPANY_SINGLE_YEAR,
            QuestionType.SINGLE_COMPANY_MULTI_YEAR,
            QuestionType.MULTI_COMPANY_SINGLE_YEAR,
            #QuestionType.MULTI_COMPANY_MULTI_YEAR
        ]
    
    # Choose a random question type from valid options
    question_type = random.choice(valid_question_types)
    
    # Choose a random category
    category = random.choices(
        list(QuestionCategory),
        weights=[25, 10, 10, 15, 10, 15, 10, 5],  # Adjust weights as needed
        k=1
    )[0]
    
    # Choose a random difficulty
    difficulty = random.choices(
        list(QuestionDifficulty),
        weights=[40, 50, 10],  # 40% Easy, 50% Medium, 10% Hard
        k=1
    )[0]
    
    # Select reports based on question type
    selected_reports = []
    sections_to_include = {}
    
    if question_type == QuestionType.SINGLE_COMPANY_SINGLE_YEAR:
        # Choose a random company
        company_name = random.choice(available_companies)
        available_years = list(companies[company_name].keys())
        
        if not available_years:
            raise ValueError(f"No years available for company {company_name}")
        
        # Choose a random year
        year = random.choice(available_years)
        report = companies[company_name][year]
        selected_reports.append(report)
        
        # Choose sections to include based on question category
        if report.available_sections:
            # Get relevant sections for this category
            relevant_sections = get_relevant_sections_for_category(category)
            # Filter to only include sections that are available in this report
            available_relevant_sections = [section for section in relevant_sections if section in report.available_sections]
            
            if available_relevant_sections:
                # Choose 1-3 relevant sections
                num_sections = min(random.randint(1, 3), len(available_relevant_sections))
                selected_sections = random.sample(available_relevant_sections, num_sections)
            else:
                # Fallback to random sections if no relevant sections are available
                num_sections = min(random.randint(1, 3), len(report.available_sections))
                selected_sections = random.sample(list(report.available_sections), num_sections)
            
            # Use the processed report path instead of raw file path
            processed_file_path = os.path.join(processed_reports_dir, f"report_{report.ticker}_{report.year}.json")
            sections_to_include[processed_file_path] = selected_sections
    
    elif question_type == QuestionType.SINGLE_COMPANY_MULTI_YEAR:
        # Choose a random company
        company_name = random.choice(available_companies)
        available_years = list(companies[company_name].keys())
        
        if len(available_years) < 2:
            # Not enough years, try a different question type
            return determine_question_specification(companies)
        
        # Choose 2-3 random years
        num_years = min(random.randint(2, 3), len(available_years))
        selected_years = random.sample(available_years, num_years)
        
        for year in selected_years:
            report = companies[company_name][year]
            selected_reports.append(report)
            
            # Choose sections to include based on question category
            if report.available_sections:
                # Get relevant sections for this category
                relevant_sections = get_relevant_sections_for_category(category)
                # Filter to only include sections that are available in this report
                available_relevant_sections = [section for section in relevant_sections if section in report.available_sections]
                
                if available_relevant_sections:
                    # Choose 1-2 relevant sections
                    num_sections = min(random.randint(1, 2), len(available_relevant_sections))
                    selected_sections = random.sample(available_relevant_sections, num_sections)
                else:
                    # Fallback to random sections if no relevant sections are available
                    num_sections = min(random.randint(1, 2), len(report.available_sections))
                    selected_sections = random.sample(list(report.available_sections), num_sections)
                
                # Use the processed report path instead of raw file path
                processed_file_path = os.path.join(processed_reports_dir, f"report_{report.ticker}_{report.year}.json")
                sections_to_include[processed_file_path] = selected_sections
    
    elif question_type == QuestionType.MULTI_COMPANY_SINGLE_YEAR:
        # Choose 2-3 random companies
        num_companies = min(random.randint(2, 3), len(available_companies))
        selected_companies = random.sample(available_companies, num_companies)
        
        # Find years that all selected companies have in common
        common_years = set()
        for i, company in enumerate(selected_companies):
            company_years = set(companies[company].keys())
            if i == 0:
                common_years = company_years
            else:
                common_years &= company_years
        
        if not common_years:
            # No common years, try a different question type
            return determine_question_specification(companies)
        
        # Choose a random year from common years
        year = random.choice(list(common_years))
        
        for company in selected_companies:
            report = companies[company][year]
            selected_reports.append(report)
            
            # Choose sections to include based on question category
            if report.available_sections:
                # Get relevant sections for this category
                relevant_sections = get_relevant_sections_for_category(category)
                # Filter to only include sections that are available in this report
                available_relevant_sections = [section for section in relevant_sections if section in report.available_sections]
                
                if available_relevant_sections:
                    # Choose 1-2 relevant sections
                    num_sections = min(random.randint(1, 2), len(available_relevant_sections))
                    selected_sections = random.sample(available_relevant_sections, num_sections)
                else:
                    # Fallback to random sections if no relevant sections are available
                    num_sections = min(random.randint(1, 2), len(report.available_sections))
                    selected_sections = random.sample(list(report.available_sections), num_sections)
                
                # Use the processed report path instead of raw file path
                processed_file_path = os.path.join(processed_reports_dir, f"report_{report.ticker}_{report.year}.json")
                sections_to_include[processed_file_path] = selected_sections
    
    else:  # MULTI_COMPANY_MULTI_YEAR
        # Choose 2-3 random companies
        num_companies = min(random.randint(2, 3), len(available_companies))
        selected_companies = random.sample(available_companies, num_companies)
        
        # Find years that all selected companies have in common
        common_years = set()
        for i, company in enumerate(selected_companies):
            company_years = set(companies[company].keys())
            if i == 0:
                common_years = company_years
            else:
                common_years &= company_years
        
        if len(common_years) < 2:
            # Not enough common years, try a different question type
            return determine_question_specification(companies)
        
        # Choose 2 random years from common years
        selected_years = random.sample(list(common_years), 2)
        
        for company in selected_companies:
            for year in selected_years:
                report = companies[company][year]
                selected_reports.append(report)
                
                # Choose sections to include based on question category
                if report.available_sections:
                    # Get relevant sections for this category
                    relevant_sections = get_relevant_sections_for_category(category)
                    # Filter to only include sections that are available in this report
                    available_relevant_sections = [section for section in relevant_sections if section in report.available_sections]
                    
                    if available_relevant_sections:
                        # Choose 1 relevant section
                        num_sections = min(1, len(available_relevant_sections))
                        selected_sections = random.sample(available_relevant_sections, num_sections)
                    else:
                        # Fallback to random sections if no relevant sections are available
                        num_sections = min(1, len(report.available_sections))
                        selected_sections = random.sample(list(report.available_sections), num_sections)
                    
                    # Use the processed report path instead of raw file path
                    processed_file_path = os.path.join(processed_reports_dir, f"report_{report.ticker}_{report.year}.json")
                    sections_to_include[processed_file_path] = selected_sections
    
    # Ensure we have at least one report and one section
    if not selected_reports or not sections_to_include:
        # Try again with a different specification
        return determine_question_specification(companies)
    
    return QuestionSpec(
        question_type=question_type,
        category=category,
        difficulty=difficulty,
        reports=selected_reports,
        sections_to_include=sections_to_include
    )