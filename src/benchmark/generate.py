import json
import os
import tiktoken
import uuid
from openai import OpenAI
from tqdm import tqdm
from typing import Dict, Tuple

from .dataclasses import BenchmarkQuestion, QuestionCategory, QuestionDifficulty, QuestionSpec, QuestionType
from .utils import determine_question_specification
from ..utils import extract_sections_from_reports, load_preprocessed_reports_metadata, remove_html_tables


def prepare_question_generation_input(
    question_spec: QuestionSpec,
    extracted_content: Dict[str, Dict[str, Dict[str, str]]],
    max_total_tokens: int = 100000,
    exclude_tables: bool = False
) -> Tuple[str, Dict]:
    """
    Prepare the input for question generation.
    
    Args:
        question_spec: Specification for the question
        extracted_content: Extracted content from reports
        max_total_tokens: Maximum total tokens for the input
        exclude_tables: Whether to exclude tables from the content
        
    Returns:
        Tuple of (formatted_content, metadata)
    """
    # Initialize tokenizer
    encoding = tiktoken.encoding_for_model("gpt-4")
    
    # Format content for LLM input
    formatted_content = ""
    content_metadata = {}
    
    # Track token count
    total_tokens = 0
    
    # Format based on question type
    if question_spec.question_type in [QuestionType.SINGLE_COMPANY_SINGLE_YEAR, QuestionType.SINGLE_COMPANY_MULTI_YEAR]:
        # Single company questions
        company = list(extracted_content.keys())[0]
        content_metadata["company"] = company
        
        formatted_content += f"COMPANY: {company}\n\n"
        
        # Add each year's content
        for year in sorted(extracted_content[company].keys()):
            formatted_content += f"YEAR: {year}\n\n"
            content_metadata[f"year_{year}"] = {}
            
            # Add each section
            for section_id, content in extracted_content[company][year].items():
                # Filter tables if requested
                processed_content = remove_html_tables(content) if exclude_tables else content
                
                section_header = f"SECTION: {section_id}\n\n"
                section_tokens = encoding.encode(section_header + processed_content)
                
                # Check if adding this section would exceed our token limit
                if total_tokens + len(section_tokens) > max_total_tokens:
                    # If we're already over limit, skip this section
                    continue
                
                formatted_content += section_header + processed_content + "\n\n"
                total_tokens += len(section_tokens)
                
                # Store metadata about this section
                content_metadata[f"year_{year}"][section_id] = {
                    "start_index": len(formatted_content) - len(processed_content) - 2,  # Account for newlines
                    "end_index": len(formatted_content) - 2,
                    "token_count": len(section_tokens)
                }
    
    else:  # Multi-company questions
        # Add each company's content
        for company in sorted(extracted_content.keys()):
            formatted_content += f"COMPANY: {company}\n\n"
            content_metadata[company] = {}
            
            # Add each year's content
            for year in sorted(extracted_content[company].keys()):
                formatted_content += f"YEAR: {year}\n\n"
                content_metadata[company][f"year_{year}"] = {}
                
                # Add each section
                for section_id, content in extracted_content[company][year].items():
                    # Filter tables if requested
                    processed_content = remove_html_tables(content) if exclude_tables else content
                    
                    section_header = f"SECTION: {section_id}\n\n"
                    section_tokens = encoding.encode(section_header + processed_content)
                    
                    # Check if adding this section would exceed our token limit
                    if total_tokens + len(section_tokens) > max_total_tokens:
                        # If we're already over limit, skip this section
                        continue
                    
                    formatted_content += section_header + processed_content + "\n\n"
                    total_tokens += len(section_tokens)
                    
                    # Store metadata about this section
                    content_metadata[company][f"year_{year}"][section_id] = {
                        "start_index": len(formatted_content) - len(processed_content) - 2,
                        "end_index": len(formatted_content) - 2,
                        "token_count": len(section_tokens)
                    }
    
    return formatted_content, content_metadata


def generate_benchmark_question(
        question_spec: QuestionSpec,
        formatted_content: str,
        content_metadata: Dict,
        model: str = "gpt-4o-mini",
        exclude_tables: bool = False
    ) -> BenchmarkQuestion:
    """
    Generate a synthetic benchmark question using the OpenAI API.
    
    Args:
        question_spec: Specification for the question
        formatted_content: Formatted content for LLM input
        content_metadata: Metadata about the content
        model: OpenAI model to use for generation
        
    Returns:
        A BenchmarkQuestion object
    """

    # Initialize the OpenAI client
    client = OpenAI()
    
    system_prompt = """
    You are an expert financial analyst tasked with creating high-quality question-answer pairs for a RAG benchmarking dataset.
    Your goal is to generate answerable questions suitable for RAG benchmarking based on SEC 10-K filing data that is provided to you. You msut also extract the source information.

    For each question you generate, you must:
    1. Create a clear, specific question that requires understanding the provided documents
    2. Make questions conversational and natural, as if asked by a real person (use shortened company names like "Apple" instead of "Apple Inc.")
    3. Provide a concise, comprehensive answer based EXCLUSIVELY on the documents provided - NEVER include information not present in the documents
    4. Include COMPLETE, EXACT references to where the information was found - extract full sentences or paragraphs, not just fragments. If the information you use is from different parts (e.g. sentences that are not connected or different sections), you MUST output them as separate references. Do NOT join references together.
    5. Detail the reasoning path that shows how to derive the answer step by step
    6. Identify the specific spans of text that contain the information needed, ensuring they include complete context. Don't just output a few words, output everything so that the context can be understood (typically at lest one or more sentences).
    7. NEVER fabricate or hallucinate information - if the answer cannot be fully derived from the provided documents, state this explicitly

    The output must follow the exact JSON format specified, with all fields properly populated.
    """
    
    # Customize user prompt based on question type and category
    type_descriptions = {
        QuestionType.SINGLE_COMPANY_SINGLE_YEAR: "a question about a single company in a specific year, but use multiple pieces of information from different areas of the provided context.",
        QuestionType.SINGLE_COMPANY_MULTI_YEAR: "a question comparing different years for the same company",
        QuestionType.MULTI_COMPANY_SINGLE_YEAR: "a question comparing different companies in the same year",
        QuestionType.MULTI_COMPANY_MULTI_YEAR: "a question comparing different companies across multiple years"
    }
    
    category_descriptions = {
        QuestionCategory.FINANCIAL_METRIC: "financial metrics or performance",
        QuestionCategory.RISK_FACTOR: "risk factors or uncertainties",
        QuestionCategory.BUSINESS_OVERVIEW: "business operations or strategy",
        QuestionCategory.MANAGEMENT_DISCUSSION: "management's discussion and analysis",
        QuestionCategory.FORWARD_LOOKING: "forward-looking statements or future outlook",
        QuestionCategory.SEGMENT_ANALYSIS: "business segment performance or analysis",
        QuestionCategory.TABLE_ANALYSIS: "analysis of tabular financial data"
    }
    
    difficulty_descriptions = {
        QuestionDifficulty.EASY: "straightforward, requiring direct lookup or simple comparison",
        QuestionDifficulty.MEDIUM: "moderately complex, requiring understanding multiple parts or basic analysis",
        QuestionDifficulty.HARD: "challenging, requiring synthesizing multiple pieces of information"
    }
    
    type_desc = type_descriptions.get(question_spec.question_type, "")
    category_desc = category_descriptions.get(question_spec.category, "")
    difficulty_desc = difficulty_descriptions.get(question_spec.difficulty, "")

    if exclude_tables:
        tables_prompt = "\nNOTE: HTML tables have been removed from the provided content. Generate questions and answers that do not require table data."
    else:
        tables_prompt = "\nNOTE: HTML tables are included in the content in HTML format. You may generate questions that require analyzing table data."
    
    user_prompt = f"""
    Please generate a {difficulty_desc} question using the provided SEC filing content below. The question should be {type_desc}.
    
    {tables_prompt}

    Base your question and answer ONLY on the following SEC 10-K filing content:
    --------------------------
    START OF SEC 10-K FILING CONTENT:

    {formatted_content}

    END OF SEC FILING CONTENT
    ------------------------------

    Your response must be a valid JSON object matching this structure:
    {{
    "question": "The question text",
    "answer": "The comprehensive answer",
    "source_information": [
        {{
        "company": "Company name",
        "ticker": "Company ticker symbol",
        "year": "Year of the data",
        "section": "Section ID (e.g., Item 7)",
        "subsection": "Subsection name if applicable",
        "span_text": "The exact text from the provided content/ that supports this part of the answer. This should be the full context which covers the specific information needed to answer part of the question. Generally one or more sentences.",
        "span_location": {{
            "document_id": "ID of the document",
            "start_offset": 12345,
            "end_offset": 12678
        }},
        "contains_table": false,
        "table_row": null,
        "table_column": null
        }}
    ],
    "reasoning_path": [
        "Step 1: Find...",
        "Step 2: Calculate...",
        "Step 3: Compare..."
    ],
    "question_type": "{question_spec.question_type.value}",
    "difficulty": "{question_spec.difficulty.value}",
    "contains_tables": false
    }}

    REMINDER OF IMPORTANT GUIDELINES:
    1. Make sure the generated questions and answers are suitable for a RAG benchmarking dataset
    2. Ensure all span_text values are EXACT and COMPLETE quotes from the documents, including full sentences or short paragraphs for proper context (i.e. if I read the output I should understand what it is refering to)
    3. If information is used from non-continuous parts of the context, you MUST output the source information in multiple dictionaries in the source_information part of the output. Do NOT combine multiple references together.
    4. In the extracted span text, do NOT change the wording (e.g. If the original reports says 'Google's vision is to remain ...', output exactly like this. Do NOT change to something like 'Our vision is to remain...' .)
    5. For span_location, provide accurate character offsets.
    6. Make the question conversational and natural, using shortened company names (e.g., "Apple" instead of "Apple Inc.").
    7. Set contains_tables to true if ANY of the source_information items has contains_table set to true.
    8. DO NOT fabricate or hallucinate information - all answers must be directly supported by the provided documents. Do NOT use what you know about the companies in your answer.
    9. If the information in the documents is insufficient to fully answer the question, explicitly state this in the answer.
    """
    
    # Call the OpenAI API using the new format
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=4000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Extract the response content
        response_content = response.choices[0].message.content
        
        # Parse the JSON response
        try:
            # Find the JSON part of the response (in case there's any preamble)
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            json_str = response_content[json_start:json_end]
            
            # Parse the JSON
            question_data = json.loads(json_str)

               # Create a mapping of company+ticker+year to processed report paths from question_spec
            processed_report_paths = {}
            
            # Extract all the processed report paths from question_spec.sections_to_include
            for processed_path in question_spec.sections_to_include.keys():
                # Extract ticker and year from the filename
                # Expected format: path/to/report_TICKER_YEAR.json
                filename = os.path.basename(processed_path)
                if filename.startswith("report_") and filename.endswith(".json"):
                    parts = filename.replace("report_", "").replace(".json", "").split("_")
                    if len(parts) >= 2:
                        ticker = parts[0]
                        year = parts[1]
                        # Create keys for lookup
                        ticker_year_key = f"{ticker}_{year}"
                        processed_report_paths[ticker_year_key] = processed_path
            
            # Add processed report paths to source information
            for source in question_data.get("source_information", []):
                company = source.get("company", "")
                ticker = source.get("ticker", "")
                year = source.get("year", "")
                
                # Try to find the processed report path
                ticker_year_key = f"{ticker}_{year}"
                
                if ticker_year_key in processed_report_paths:
                    source["processed_path"] = processed_report_paths[ticker_year_key]
            
            
            # Create and return the BenchmarkQuestion object
            return BenchmarkQuestion(**question_data)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response content: {response_content}")
            raise
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise


def generate_sec_benchmark_dataset(
    preprocessed_dir: str,
    output_dir: str,
    output_file: str = 'benchmark_dataset.json',
    num_questions: int = 100,
    model: str = "gpt-4o-mini",
    max_tokens_per_section: int = 8000,
    max_total_tokens: int = 100000,
    max_retries: int = 10,  # Add a maximum retry limit
    exclude_tables: bool = False
):
    """
    Main function to generate benchmark questions from preprocessed SEC filings.
    
    Args:
        preprocessed_dir: Directory containing preprocessed reports
        output_dir: Directory to save output files
        output_file: Path to output file
        num_questions: Number of questions to generate
        model: OpenAI model to use for generation
        max_tokens_per_section: Maximum tokens to include per section
        max_total_tokens: Maximum total tokens for the input to OpenAI
        max_retries: Maximum number of retries per question
    """
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    # Load available preprocessed reports
    print(f"Loading preprocessed reports from {preprocessed_dir}...")
    available_reports = load_preprocessed_reports_metadata(preprocessed_dir)
    print(f"Loaded {len(available_reports)} preprocessed reports")
    
    if not available_reports:
        raise ValueError(f"No preprocessed reports found in {preprocessed_dir}")
    
    # Group reports by company and year
    companies = {}
    for report in available_reports:
        if report.ticker not in companies:
            companies[report.ticker] = {}
        companies[report.ticker][report.year] = report
    
    print(f"Found data for {len(companies)} companies")
    for company, years in companies.items():
        print(f"  - {company}: {len(years)} years ({', '.join(map(str, sorted(years.keys())))})")
    
    # Generate questions
    benchmark_questions = []
    
    print(f"Generating {num_questions} benchmark questions...")
    with tqdm(total=num_questions) as pbar:
        question_count = 0
        total_attempts = 0
        
        while question_count < num_questions and total_attempts < num_questions * 3:  # Add an overall limit
            total_attempts += 1
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Generate question specification
                    question_spec = determine_question_specification(companies, preprocessed_dir)
                    print(question_spec)
                    
                    # For regular questions, proceed with normal generation
                    # Extract relevant sections
                    extracted_content = extract_sections_from_reports(
                        question_spec,
                        preprocessed_dir,
                        # max_tokens_per_section
                    )
                    
                    # Check if we have enough content to generate a question
                    if not extracted_content or all(not year_data for company_data in extracted_content.values() for year_data in company_data.values()):
                        print(f"Warning: No content extracted for question specification. Retrying...")
                        retry_count += 1
                        continue
                    
                    # Prepare input for question generation
                    formatted_content, content_metadata = prepare_question_generation_input(
                        question_spec,
                        extracted_content,
                        max_total_tokens,
                        exclude_tables
                    )
                    
                    # Check if formatted content is too short
                    if len(formatted_content) < 500:  # Arbitrary minimum length
                        print(f"Warning: Formatted content too short ({len(formatted_content)} chars). Retrying...")
                        retry_count += 1
                        continue
                    
                    # Generate benchmark question
                    benchmark_question = generate_benchmark_question(
                        question_spec,
                        formatted_content,
                        content_metadata,
                        model
                    )
                    
                    # Add unique ID
                    benchmark_question.id = str(uuid.uuid4())
                    
                    # Add to list
                    benchmark_questions.append(benchmark_question)
                    
                    # Update progress bar and counters
                    question_count += 1
                    pbar.update(1)
                    
                    # Break out of retry loop on success
                    break
                
                except Exception as e:
                    print(f"Error generating question (attempt {retry_count+1}/{max_retries}): {e}")
                    retry_count += 1
                    
                    # If we've reached max retries, print a more detailed error
                    if retry_count >= max_retries:
                        print(f"Failed to generate question after {max_retries} attempts. Moving on...")
    
    # Check if we generated enough questions
    if len(benchmark_questions) < num_questions:
        print(f"Warning: Only generated {len(benchmark_questions)}/{num_questions} questions after {total_attempts} attempts")
    
    # Save to output file
    output_path = f"{output_dir}/{output_file}"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {len(benchmark_questions)} questions to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            [question.model_dump() for question in benchmark_questions],
            f,
            indent=2,
            ensure_ascii=False
        )
    
    print("Done!")
    return benchmark_questions