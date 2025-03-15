from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, field_validator, Field
from typing import List, Dict, Optional, Union, Literal, Set, Tuple

# Define the same models and enums from the original script
class QuestionType(Enum):
    """Enumeration of question types for the benchmark dataset"""
    SINGLE_COMPANY_SINGLE_YEAR = "single_company_single_year"  # E.g., "What were Apple's key risk factors in 2022?"
    SINGLE_COMPANY_MULTI_YEAR = "single_company_multi_year"    # E.g., "How did Apple's revenue change from 2021 to 2022?"
    MULTI_COMPANY_SINGLE_YEAR = "multi_company_single_year"    # E.g., "Compare Apple and Microsoft's R&D spending in 2022"
    MULTI_COMPANY_MULTI_YEAR = "multi_company_multi_year"      # E.g., "How did Apple and Microsoft's revenue growth compare from 2020-2022?"
    #NULL_QUESTION = "null_question"                           # Questions without context in the dataset     

class QuestionCategory(Enum):
    """Categories of questions based on content focus"""
    FINANCIAL_METRIC = "financial_metric"          # Questions about specific financial metrics
    RISK_FACTOR = "risk_factor"                    # Questions about risk factors
    BUSINESS_OVERVIEW = "business_overview"        # Questions about business operations and strategy
    MANAGEMENT_DISCUSSION = "management_discussion" # Questions about MD&A
    FORWARD_LOOKING = "forward_looking"            # Questions about future outlook
    SEGMENT_ANALYSIS = "segment_analysis"          # Questions about business segments
    COMPANY_STOCK = "company_stock"                # Questions about company stock
    TABLE_ANALYSIS = "table_analysis"              # Questions specifically about tabular data

class QuestionDifficulty(Enum):
    """Difficulty levels for questions"""
    EASY = "easy"          # Direct lookup or simple comparison
    MEDIUM = "medium"      # Requires understanding multiple parts or basic analysis
    HARD = "hard"          # Requires complex reasoning or synthesizing multiple pieces of information

@dataclass
class CompanyReport:
    """Representation of a company's SEC filing report"""
    company_name: str
    ticker: str
    year: int
    raw_file_path: str
    industry: Optional[str]
    available_sections: Set[str]
    accession_number: Optional[str] = None

@dataclass
class QuestionSpec:
    """Specification for generating a question"""
    question_type: QuestionType
    category: QuestionCategory
    difficulty: QuestionDifficulty
    reports: List[CompanyReport]
    sections_to_include: Dict[str, List[str]]  # Map of file_path to list of sections

class SpanLocation(BaseModel):
    """Location of a text span within a document"""
    document_id: str
    start_offset: int
    end_offset: int
    
class SourceInformation(BaseModel):
    """Source information model."""
    company: str
    ticker: str 
    year: str
    section: str
    subsection: Optional[str] = None
    span_text: str
    span_location: SpanLocation
    contains_table: bool
    table_row: Optional[Union[str, int]] = None
    table_column: Optional[Union[str, int]] = None
    processed_path: Optional[str] = None
    
    # Pydantic v2 validator
    @field_validator('table_row', 'table_column', mode='before')
    @classmethod
    def convert_int_to_str(cls, v):
        if isinstance(v, int):
            return str(v)
        return v


class BenchmarkQuestion(BaseModel):
    """A synthetic question for the benchmark dataset"""
    id: Optional[str] = None
    question: str
    answer: str
    source_information: List[SourceInformation]
    reasoning_path: List[str]
    question_type: str
    difficulty: Literal["easy", "medium", "hard"]
    contains_tables: bool = False
    
    @field_validator('source_information')
    @classmethod
    def validate_sources(cls, v, info):
        # Allow empty source_information for null questions
        values = info.data
        if 'question_type' in values and values['question_type'] == 'null_question':
            return v
        
        # For non-null questions, require at least one source
        if not v:
            raise ValueError("At least one source must be provided for non-null questions")
        return v


class BenchmarkDatasetItem(BaseModel):
    """A complete item in the benchmark dataset"""
    id: str
    created_at: datetime = Field(default_factory=datetime.now)
    question_data: BenchmarkQuestion
    input_documents: List[str]


class BenchmarkDataset(BaseModel):
    """The complete benchmark dataset"""
    dataset_name: str
    description: str
    version: str
    created_at: datetime = Field(default_factory=datetime.now)
    items: List[BenchmarkDatasetItem]
