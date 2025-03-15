import uuid
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

# Company Name Extraction and Ticker Mapping Functions

def build_company_ticker_database() -> Dict[str, str]:
    """
    Build a database mapping company names to tickers.
    Returns a dictionary with various name forms as keys and tickers as values.
    """
    # Major companies mapping (simplified for testing)
    major_companies = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "google": "GOOG",
        "alphabet": "GOOG",
        "meta": "META",
        "facebook": "META",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        "berkshire hathaway": "BRK-A",
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "johnson & johnson": "JNJ",
        "walmart": "WMT",
        "visa": "V",
        "mastercard": "MA",
        "procter & gamble": "PG",
        "p&g": "PG",
        "exxon mobil": "XOM",
        "exxonmobil": "XOM",
        "ibm": "IBM",
        "intel": "INTC",
        "cisco": "CSCO",
        "oracle": "ORCL",
        "netflix": "NFLX",
        "adobe": "ADBE",
        "salesforce": "CRM",
        "disney": "DIS",
        "coca cola": "KO",
        "coca-cola": "KO",
        "pepsi": "PEP",
        "pepsico": "PEP",
        "mcdonalds": "MCD",
        "mcdonald's": "MCD",
        "boeing": "BA",
        "general electric": "GE",
        "ge": "GE",
        "at&t": "T",
        "verizon": "VZ",
        "home depot": "HD",
        "goldman sachs": "GS",
        "bank of america": "BAC",
        "bofa": "BAC",
        "citigroup": "C",
        "citi": "C",
        "wells fargo": "WFC",
        "chevron": "CVX",
        "merck": "MRK",
        "pfizer": "PFE",
        "ups": "UPS",
        "united parcel service": "UPS",
        "fedex": "FDX",
        "3m": "MMM",
        "caterpillar": "CAT",
        "dow": "DOW",
        "american express": "AXP",
        "amex": "AXP"
    }
    
    # Expand with variations
    expanded_mapping = {}
    for name, ticker in major_companies.items():
        # Original name
        expanded_mapping[name] = ticker
        
        # Without spaces
        expanded_mapping[name.replace(" ", "")] = ticker
        
        # Without special characters
        expanded_mapping[name.replace("&", "and")] = ticker
        
        # With "inc", "corp", etc.
        expanded_mapping[f"{name} inc"] = ticker
        expanded_mapping[f"{name} corp"] = ticker
        expanded_mapping[f"{name} corporation"] = ticker
        
    return expanded_mapping


def extract_company_names(question: str) -> List[str]:
    """
    Extract potential company names from a question using rule-based methods.
    
    Args:
        question: The user question
        
    Returns:
        List of potential company names
    """
    companies = []
    
    # Method 1: Look for capitalized words that might be company names
    capitalized_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    capitalized_matches = re.findall(capitalized_pattern, question)
    companies.extend(capitalized_matches)
    
    # Method 2: Look for words followed by Inc, Corp, etc.
    company_suffix_pattern = r'([A-Za-z\s]+)(?:\s+(?:Inc|Corp|Corporation|Company|Co|Ltd))'
    suffix_matches = re.findall(company_suffix_pattern, question)
    companies.extend(suffix_matches)
    
    # Method 3: Look for explicit mentions like "Company X" or "X Company"
    explicit_pattern = r'(?:company|corporation|firm)\s+([A-Za-z\s]+)|([A-Za-z\s]+)\s+(?:company|corporation|firm)'
    explicit_matches = re.findall(explicit_pattern, question, re.IGNORECASE)
    for match in explicit_matches:
        companies.extend([m for m in match if m])
    
    # Method 4: Direct lookup in our company database
    company_ticker_db = build_company_ticker_database()
    for company_name in company_ticker_db.keys():
        if company_name.lower() in question.lower():
            companies.append(company_name)
    
    # Clean and deduplicate
    cleaned_companies = []
    for company in companies:
        company = company.strip()
        if len(company) > 1 and company.lower() not in ["the", "a", "an"]:
            cleaned_companies.append(company)
    
    return list(set(cleaned_companies))


def map_companies_to_tickers(company_names: List[str], typo_tolerance: bool = True) -> Dict[str, str]:
    """
    Map company names to tickers with typo tolerance.
    
    Args:
        company_names: List of potential company names
        typo_tolerance: Whether to use fuzzy matching for typo tolerance
        
    Returns:
        Dictionary mapping company names to their tickers
    """
    company_ticker_db = build_company_ticker_database()
    results = {}
    
    for name in company_names:
        # Try exact match
        if name.lower() in company_ticker_db:
            results[name] = company_ticker_db[name.lower()]
            continue
            
        # Try without common words
        cleaned_name = re.sub(r'\b(inc|corp|corporation|company|co|ltd)\b', '', name.lower()).strip()
        if cleaned_name in company_ticker_db:
            results[name] = company_ticker_db[cleaned_name]
            continue
        
        # If typo tolerance is enabled, try fuzzy matching
        if typo_tolerance:
            # Simple character-level similarity
            best_match = None
            best_score = 0
            
            for db_name in company_ticker_db:
                # Skip very short names to avoid false matches
                if len(db_name) < 3 or len(name) < 3:
                    continue
                
                # Calculate similarity (Jaccard similarity of character sets)
                set1 = set(db_name.lower())
                set2 = set(name.lower())
                similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                
                # Check for substring matching (for partial matches)
                substring_bonus = 0
                if db_name.lower() in name.lower() or name.lower() in db_name.lower():
                    substring_bonus = 0.2  # Bonus for substring match
                
                # Combine scores
                final_score = similarity + substring_bonus
                
                # Update best match if this is better
                if final_score > 0.7 and final_score > best_score:  # Threshold of 0.7
                    best_score = final_score
                    best_match = db_name
            
            if best_match:
                results[name] = company_ticker_db[best_match]
    
    return results


def extract_tickers_from_question(question: str) -> List[str]:
    """
    Extract stock tickers directly mentioned in the question.
    
    Args:
        question: User question
        
    Returns:
        List of tickers
    """
    # Look for tickers in format $AAPL or AAPL
    ticker_pattern = r'\$([A-Z]{1,5})|(?<!\w)([A-Z]{2,5})(?!\w)'
    matches = re.findall(ticker_pattern, question)
    
    # Flatten and clean
    tickers = []
    for match in matches:
        ticker = match[0] if match[0] else match[1]
        if ticker and ticker not in ["A", "I", "AM", "PM", "CEO", "CFO", "CTO", "COO"]:
            tickers.append(ticker)
    
    return tickers

def extract_metadata_filters(question: str, include_adjacent_years: bool = True) -> Dict[str, Any]:
    """
    Extract metadata filters from question for hybrid search.
    
    Args:
        question: User question
        include_adjacent_years: Whether to include the next year for each year mentioned
        
    Returns:
        Dictionary of metadata filters
    """
    filters = {}
    
    # Extract company tickers
    direct_tickers = extract_tickers_from_question(question)
    
    # Extract company names and map to tickers
    company_names = extract_company_names(question)
    company_ticker_map = map_companies_to_tickers(company_names)
    mapped_tickers = list(company_ticker_map.values())
    
    # Combine all tickers
    all_tickers = list(set(direct_tickers + mapped_tickers))
    
    if all_tickers:
        filters["ticker"] = all_tickers
    
    # Extract years
    year_pattern = r'\b(20\d{2}|19\d{2})\b'
    years = re.findall(year_pattern, question)
    
    if years:
        # Convert to integers
        int_years = [int(year) for year in years]
        
        # Include adjacent years if requested
        if include_adjacent_years:
            expanded_years = set(int_years)
            for year in int_years:
                # Add the next year to capture reports that might contain information about the previous year
                expanded_years.add(year + 1)
            
            filters["year"] = list(expanded_years)
        else:
            filters["year"] = int_years
    
    # Extract document types
    doc_types = ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"]
    found_doc_types = [dt for dt in doc_types if dt in question]
    
    if found_doc_types:
        filters["doc_type"] = found_doc_types
    
    return filters


# def extract_metadata_filters(question: str) -> Dict[str, Any]:
#     """
#     Extract metadata filters from question for hybrid search.
    
#     Args:
#         question: User question
        
#     Returns:
#         Dictionary of metadata filters
#     """
#     filters = {}
    
#     # Extract company tickers
#     direct_tickers = extract_tickers_from_question(question)
    
#     # Extract company names and map to tickers
#     company_names = extract_company_names(question)
#     company_ticker_map = map_companies_to_tickers(company_names)
#     mapped_tickers = list(company_ticker_map.values())
    
#     # Combine all tickers
#     all_tickers = list(set(direct_tickers + mapped_tickers))
    
#     if all_tickers:
#         filters["ticker"] = all_tickers
    
#     # Extract years
#     year_pattern = r'\b(20\d{2}|19\d{2})\b'
#     years = re.findall(year_pattern, question)
    
#     if years:
#         filters["year"] = [int(year) for year in years]
    
#     # Extract document types
#     doc_types = ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"]
#     found_doc_types = [dt for dt in doc_types if dt in question]
    
#     if found_doc_types:
#         filters["doc_type"] = found_doc_types
    
#     return filters