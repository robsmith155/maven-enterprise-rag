import os
import json
import re
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
import markdown
import html
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging
from openai import OpenAI
from datetime import datetime

from ..processing.sec_filing_parser import SECFilingParser
from .pipeline import load_existing_pipeline

# Set up logger for retry attempts
logger = logging.getLogger(__name__)

# Function to find the best matching text in a document
def find_text_in_document(document_text: str, chunk_text: str) -> Tuple[int, float, int]:
    """
    Find the best matching position of chunk_text in document_text.
    Returns (start_position, match_ratio, length_of_match)
    """
    # Clean up both texts for better matching
    doc_text = re.sub(r'\s+', ' ', document_text).strip()
    chunk = re.sub(r'\s+', ' ', chunk_text).strip()
    
    # For very short chunks, extend the search text
    if len(chunk) < 20 and len(chunk.split()) < 5:
        return 0, 0, 0  # Too short to reliably match
    
    # Try exact match first (fastest)
    position = doc_text.find(chunk)
    if position >= 0:
        return position, 1.0, len(chunk)
    
    # Try with a shorter version of the chunk (first 100 chars)
    search_text = chunk[:min(100, len(chunk))]
    if len(search_text) > 20:  # Only if it's still substantial
        position = doc_text.find(search_text)
        if position >= 0:
            # Find how much of the chunk matches from this position
            i = 0
            while position + i < len(doc_text) and i < len(chunk) and doc_text[position + i] == chunk[i]:
                i += 1
            return position, 0.9, i
    
    # Use sequence matcher for fuzzy matching
    best_ratio = 0
    best_position = 0
    best_length = 0
    
    # For efficiency, use a sliding window approach
    window_size = min(200, len(chunk) * 2)  # Reasonable window size
    step_size = max(1, window_size // 4)    # Overlap windows for better matching
    
    for i in range(0, len(doc_text) - window_size + 1, step_size):
        window = doc_text[i:i + window_size]
        matcher = SequenceMatcher(None, window, chunk)
        match = matcher.find_longest_match(0, len(window), 0, len(chunk))
        ratio = match.size / len(chunk) if len(chunk) > 0 else 0
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_position = i + match.a
            best_length = match.size
    
    return best_position, best_ratio, best_length

# Function to highlight text in HTML content using BeautifulSoup
def highlight_text_in_html(html_content: str, text_to_find: str) -> str:
    """
    Highlight the given text in HTML content using BeautifulSoup for proper parsing.
    Returns the HTML with highlighted text and a unique ID for scrolling.
    """
    if not text_to_find or not html_content:
        return html_content, ""
    
    # Clean the text for better matching
    text_to_find = re.sub(r'\s+', ' ', text_to_find).strip()
    
    # Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Generate a unique ID for the highlight
    highlight_id = f"highlight-{abs(hash(text_to_find)) % 10000}"
    
    # Find all text nodes
    text_found = False
    
    # Function to process a text node
    def process_text_node(node):
        nonlocal text_found
        if text_found:
            return
        
        if node.string:
            node_text = node.string.strip()
            if not node_text:
                return
                
            # Clean the node text for better matching
            clean_node_text = re.sub(r'\s+', ' ', node_text)
            
            # Check if text_to_find is in this node
            if text_to_find in clean_node_text:
                # Split the text and wrap the matching part in a highlight span
                start_idx = clean_node_text.find(text_to_find)
                end_idx = start_idx + len(text_to_find)
                
                # Create new nodes
                before_text = node.string[:start_idx]
                highlight_text = node.string[start_idx:end_idx]
                after_text = node.string[end_idx:]
                
                # Replace the current node with three new nodes
                highlight_span = soup.new_tag('span')
                highlight_span['style'] = 'background-color: yellow; font-weight: bold;'
                highlight_span['id'] = highlight_id
                highlight_span.string = highlight_text
                
                # Replace the node with the three parts
                if before_text:
                    node.insert_before(before_text)
                node.insert_before(highlight_span)
                if after_text:
                    node.insert_before(after_text)
                
                # Remove the original node
                node.extract()
                text_found = True
    
    # Traverse all text nodes
    for element in soup.find_all(text=True):
        process_text_node(element)
    
    # If exact match wasn't found, try a more aggressive approach with fuzzy matching
    if not text_found:
        # Get all text from the document
        all_text = soup.get_text()
        position, match_ratio, match_length = find_text_in_document(all_text, text_to_find)
        
        if match_ratio > 0.7:  # Good enough match
            # This is more complex - we need to find which node contains this position
            # For simplicity, we'll just add a marker at the beginning of the document
            marker = soup.new_tag('div')
            marker['id'] = highlight_id
            marker['style'] = 'position: absolute; visibility: hidden;'
            if soup.body:
                soup.body.insert(0, marker)
            else:
                soup.insert(0, marker)
    
    return str(soup), highlight_id

# Function to extract reference numbers from an answer
def extract_cited_references(answer_text: str) -> List[int]:
    """
    Extract the reference numbers cited in the answer text.
    Returns a list of integers representing the reference numbers.
    """
    # Look for reference patterns like [1], [2], etc.
    ref_pattern = r'\[(\d+)\]'
    matches = re.findall(ref_pattern, answer_text)
    
    # Convert to integers and remove duplicates
    cited_refs = []
    for match in matches:
        try:
            ref_num = int(match)
            if ref_num not in cited_refs:
                cited_refs.append(ref_num)
        except ValueError:
            pass
    
    return cited_refs

# Function to format the answer with nice styling
def format_answer(answer_text: str) -> str:
    """Format the answer text with nice styling."""
    # Split into answer and references if possible
    parts = re.split(r'References:', answer_text, flags=re.IGNORECASE, maxsplit=1)
    
    answer_part = parts[0].strip()
    references_part = parts[1].strip() if len(parts) > 1 else ""
    
    # Convert markdown to HTML
    answer_html = markdown.markdown(answer_part)
    
    # Style the references section if it exists
    if references_part:
        # Convert references to HTML with nice formatting
        references_html = "<h3>References</h3>"
        
        # Process each reference line
        for line in references_part.split('\n'):
            line = line.strip()
            if line:
                # Try to extract reference number
                ref_match = re.match(r'^\[(\d+)\]', line)
                if ref_match:
                    ref_num = ref_match.group(1)
                    ref_content = line[ref_match.end():].strip()
                    references_html += f'<p><strong>[{ref_num}]</strong> {ref_content}</p>'
                else:
                    references_html += f'<p>{line}</p>'
        
        # Combine answer and references
        result_html = f"""
        <div class="answer-container">
            <div class="answer-content">
                {answer_html}
            </div>
            <div class="references-section">
                {references_html}
            </div>
        </div>
        """
    else:
        # Just the answer without references
        result_html = f"""
        <div class="answer-container">
            <div class="answer-content">
                {answer_html}
            </div>
        </div>
        """
    
    return result_html

# Function to display raw chunk text
def display_raw_chunk(source):
    """Display the raw chunk text in a formatted way."""
    if not source:
        display(HTML("<p>No source information available.</p>"))
        return
    
    text = source.get("text", "")
    doc_title = source.get("document_title", "Unknown")
    doc_id = source.get("document_id", "Unknown")
    chunk_id = source.get("chunk_id", "Unknown")
    
    display(HTML(f"""
    <h3>Raw Chunk Text</h3>
    <p><strong>Document:</strong> {doc_title} (ID: {doc_id})</p>
    <p><strong>Chunk ID:</strong> {chunk_id}</p>
    <pre style="white-space: pre-wrap; background-color: #f5f5f5; padding: 10px; border: 1px solid #eee;">{html.escape(text)}</pre>
    """))

# Function to display all chunks sent to the model
def display_all_chunks(chunks, output_area):
    """Display all chunks that were sent to the model."""
    with output_area:
        clear_output()
        display(HTML("<h3>All Chunks Sent to the Model</h3>"))
        
        if not chunks:
            display(HTML("<p>No chunks were sent to the model.</p>"))
            return
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            doc_title = chunk.get("document_title", "Unknown")
            doc_id = chunk.get("document_id", "Unknown")
            
            display(HTML(f"""
            <div style="margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9;">
                <h4>Chunk {i+1}: {doc_title} (ID: {doc_id})</h4>
                <pre style="white-space: pre-wrap; background-color: #f5f5f5; padding: 10px; border: 1px solid #eee;">{html.escape(chunk_text)}</pre>
            </div>
            """))

# Function to update reference display based on view mode
def update_reference_display(reference_dropdown, view_mode, reference_output, source=None):
    """Update the reference display based on the selected view mode."""
    # Clear the output area
    reference_output.clear_output()
    
    # Get the selected source
    if source is None:
        selected_source = reference_dropdown.value if reference_dropdown.value else None
    else:
        selected_source = source
    
    if not selected_source:
        with reference_output:
            display(HTML("<p>No reference selected.</p>"))
        return
    
    # Get the view mode
    selected_mode = view_mode.value
    
    # Display based on the selected mode
    with reference_output:
        if selected_mode == "Raw Chunk Text":
            display_raw_chunk(selected_source)
        else:  # Original Document
            # Try to get the original document
            original_file_path = selected_source.get("original_file_path", None)
            processed_file_path = selected_source.get("processed_file_path", None)
            section = selected_source.get("section", None)
            chunk_text = selected_source.get("text", "")
            
            if not (original_file_path and os.path.exists(original_file_path)):
                display(HTML("<p>Original document not available. Showing raw chunk text instead.</p>"))
                display_raw_chunk(selected_source)
                return
            
            try:
                # Create a parser instance
                parser = SECFilingParser()
                
                # Load the document
                parser.load_filing(original_file_path)
                
                # If we have a section, display that section
                if section:
                    # Display the section with the chunk text highlighted
                    section_html = parser._get_section_html(section)
                    
                    if section_html:
                        # Highlight the chunk text in the section HTML
                        highlighted_html, highlight_id = highlight_text_in_html(section_html, chunk_text)
                        
                        # Display the section title
                        display(HTML(f"<h3>Section: {section}</h3>"))
                        
                        # Display the highlighted HTML with a scroll script
                        display(HTML(highlighted_html))
                        
                        # Add a script to scroll to the highlighted text
                        if highlight_id:
                            display(HTML(f"""
                            <script>
                                setTimeout(function() {{
                                    var element = document.getElementById('{highlight_id}');
                                    if (element) {{
                                        element.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                                    }}
                                }}, 500);
                            </script>
                            """))
                    else:
                        display(HTML(f"<p>Section '{section}' not found in the document. Showing raw chunk text instead.</p>"))
                        display_raw_chunk(selected_source)
                else:
                    # No section specified, try to find the chunk in the full document
                    full_html = parser.get_html()
                    
                    if full_html:
                        # Highlight the chunk text in the full HTML
                        highlighted_html, highlight_id = highlight_text_in_html(full_html, chunk_text)
                        
                        # Display the highlighted HTML with a scroll script
                        display(HTML(highlighted_html))
                        
                        # Add a script to scroll to the highlighted text
                        if highlight_id:
                            display(HTML(f"""
                            <script>
                                setTimeout(function() {{
                                    var element = document.getElementById('{highlight_id}');
                                    if (element) {{
                                        element.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                                    }}
                                }}, 500);
                            </script>
                            """))
                    else:
                        display(HTML("<p>Could not load the full document. Showing raw chunk text instead.</p>"))
                        display_raw_chunk(selected_source)
            except Exception as e:
                display(HTML(f"<p>Error displaying original document: {str(e)}. Showing raw chunk text instead.</p>"))
                display_raw_chunk(selected_source)

# OpenAI API call with retry mechanism
@retry(
    retry=retry_if_exception_type((Exception)),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(min=1, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def chat_completion_with_backoff(messages, model, max_tokens, temperature):
    """Make OpenAI API call with exponential backoff retry logic"""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response

# RAG with OpenAI function
def rag_with_openai(
    pipeline,
    question: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1000,
    temperature: float = 0.7,
    top_k: int = 5,
    initial_retrieval_multiplier: int = 3,
    debug_mode: bool = False
):
    """
    Perform RAG with OpenAI API
    
    Args:
        pipeline: The RAG pipeline to use
        question: The question to answer
        model: The OpenAI model to use
        max_tokens: Maximum tokens for completion
        temperature: Temperature for sampling
        top_k: Number of chunks to retrieve
        initial_retrieval_multiplier: Multiplier for initial retrieval in hybrid search
        debug_mode: Whether to enable debug mode
        
    Returns:
        Dictionary with question, answer, model, token usage, and sources
    """
    # Set system prompt
    system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Your task is to provide accurate, concise answers based solely on the information in the context.

Guidelines:
1. Only use information present in the provided context
2. If the answer cannot be determined from the context, say "I don't have enough information to answer this question" rather than making up information
3. Use numbered references to cite your sources, starting from [1] and continuing sequentially (e.g., [1], [2])
4. Only cite sources that you actually use in your answer - do not include unused references
5. If you cannot answer the question, do not include any references
6. Present financial data clearly, using tables where appropriate
7. Be concise and direct in your answer

Format your answer as follows:
[Your detailed answer with numbered citations like [1], [2], etc.]

References:
[1] Document title and ID
[2] Document title and ID
..."""
    
    # Determine if we're using hybrid search
    is_hybrid_search = hasattr(pipeline, 'config') and hasattr(pipeline.config, 'use_hybrid_search') and pipeline.config.use_hybrid_search
    
    # Determine initial retrieval limit
    initial_limit = top_k * initial_retrieval_multiplier if is_hybrid_search else top_k
    
    # Retrieve chunks
    retrieved_chunks = pipeline.search(query=question, limit=initial_limit)
    
    # For hybrid search, sort and filter
    if is_hybrid_search and len(retrieved_chunks) > top_k:
        retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x.get('score', 0.0), reverse=True)
        retrieved_chunks = retrieved_chunks[:top_k]
    
    # Format context and prepare sources
    context_texts = []
    sources = []
    
    for i, chunk in enumerate(retrieved_chunks):
        # Extract text and metadata
        text = chunk.get("text", "")
        source_id = chunk.get("meta_document_id", f"doc_{i}")
        source_title = chunk.get("meta_document_title", "Unknown")
        chunk_id = chunk.get("id", "")
        
        # Extract additional metadata
        processed_file_path = chunk.get("meta_processed_file_path", None)
        original_file_path = chunk.get("meta_original_file_path", None)
        accession_number = chunk.get("meta_accession_number", None)
        section = chunk.get("meta_section", None)
        
        # Format the chunk with sequential reference numbers starting from 1
        formatted_chunk = f"[{i+1}] Document: {source_title} (ID: {source_id})\nChunk ID: {chunk_id}\n{text}\n"
        context_texts.append(formatted_chunk)
        
        # Store source information with sequential reference numbers
        source_info = {
            "index": i + 1,  # Sequential numbering starting from 1
            "original_index": chunk.get("index", i),  # Store original index if available
            "document_id": source_id,
            "document_title": source_title,
            "chunk_id": chunk_id,
            "text": text,
            "section": section
        }
        
        # Add additional metadata if available
        if processed_file_path:
            source_info["processed_file_path"] = processed_file_path
        if original_file_path:
            source_info["original_file_path"] = original_file_path
        if accession_number:
            source_info["accession_number"] = accession_number
            
        sources.append(source_info)
    
    # If no chunks were retrieved, provide a message
    if not retrieved_chunks:
        context_texts = ["No relevant documents were found in the database."]
    
    # Combine all context
    combined_context = "\n\n".join(context_texts)
    
    # Create user message
    user_message = f"""Question: {question}

Context:
{combined_context}

Please provide a comprehensive answer based solely on the information in the context above. 
Use numbered references like [1], [2] to cite your sources, and include a "References:" section at the end listing all cited sources."""
    
    # Create messages for OpenAI API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # Make request to OpenAI API with retry mechanism
    try:
        response = chat_completion_with_backoff(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Process the response
        answer = response.choices[0].message.content
        
        # Create result dictionary
        result = {
            "question": question,
            "answer": answer,
            "model": model,
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
            "sources": sources,
            "context": combined_context
        }
    except Exception as e:
        logger.error(f"Failed to get response after multiple retries: {str(e)}")
        answer = f"I'm sorry, but I encountered an error when processing your question. Error: {str(e)}"
        
        # Create result dictionary with error information
        result = {
            "question": question,
            "answer": answer,
            "model": model,
            "error": str(e),
            "sources": sources,
            "context": combined_context
        }
    
    return result


# Handle query submission
def on_submit_button_clicked(
    b, 
    query_input, 
    output_area, 
    reference_output, 
    reference_dropdown, 
    view_mode, 
    pipeline, 
    last_result,
    model,
    max_tokens,
    temperature,
    top_k,
    initial_retrieval_multiplier,
    view_chunks_button=None
):
    """Handle query submission button click."""
    # Clear outputs
    output_area.clear_output()
    reference_output.clear_output()
    reference_dropdown.options = []
    reference_dropdown.disabled = True
    view_mode.disabled = True
    if view_chunks_button is not None:
        view_chunks_button.disabled = True
    
    question = query_input.value
    if not question:
        with output_area:
            print("Please enter a question.")
        return
    
    with output_area:
        print(f"Processing query: {question}")
        
        try:
            # Call the RAG with OpenAI function
            result = rag_with_openai(
                pipeline=pipeline,
                question=question,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                initial_retrieval_multiplier=initial_retrieval_multiplier,
                debug_mode=False
            )
            
            # Update the last result reference
            last_result.update(result)
            
            # Display the answer with nice formatting
            clear_output()
            display(HTML(f"<h2>Question</h2><p>{question}</p>"))
            display(HTML("<h2>Answer</h2>"))
            display(HTML(format_answer(result["answer"])))
            
            # Display token usage if available
            if "total_tokens" in result:
                display(HTML(f"""
                <div style="margin-top: 20px; font-size: 0.8em; color: #666;">
                    <p>Token usage: {result["prompt_tokens"]} prompt + {result["completion_tokens"]} completion = {result["total_tokens"]} total</p>
                </div>
                """))
            
            # Extract cited reference numbers from the answer
            cited_refs = extract_cited_references(result["answer"])
            
            # Update reference dropdown with only the cited references
            if cited_refs and result["sources"]:
                # Filter sources to only include those that were cited
                cited_sources = [s for s in result["sources"] if s["index"] in cited_refs]
                
                if cited_sources:
                    # Create options with clear reference numbers
                    reference_options = [(f"[{s['index']}] {s['document_title']} (ID: {s['document_id']})", s) for s in cited_sources]
                    reference_dropdown.options = reference_options
                    reference_dropdown.disabled = False
                    view_mode.disabled = False
                    
                    # Add a note about the references
                    display(HTML(f"""
                    <div style="margin-top: 10px; font-size: 0.9em;">
                        <p>References cited in the answer: {', '.join(f'[{ref}]' for ref in cited_refs)}</p>
                    </div>
                    """))
                else:
                    display(HTML("""
                    <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                        <p>No matching sources found for the cited references.</p>
                    </div>
                    """))
            else:
                display(HTML("""
                <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                    <p>No references were cited in the answer.</p>
                </div>
                """))
            
            # Enable the View All Chunks button if it exists
            if view_chunks_button is not None and result["sources"]:
                view_chunks_button.disabled = False
            
        except Exception as e:
            import traceback
            print(f"Error processing query: {str(e)}")
            traceback.print_exc()

# Function to create a RAG chatbot
def create_rag_chatbot(
    run_dir: str, 
    config_name: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1000,
    temperature: float = 0.7,
    top_k: int = 5,
    initial_retrieval_multiplier: int = 3
):
    """
    Create a simple RAG chatbot interface in a notebook.
    
    Args:
        run_dir: Directory containing the pipeline
        config_name: Name of the pipeline configuration
        model: The OpenAI model to use
        max_tokens: Maximum tokens for completion
        temperature: Temperature for sampling
        top_k: Number of chunks to retrieve
        initial_retrieval_multiplier: Multiplier for initial retrieval in hybrid search
    """
    # Load the pipeline
    try:
        pipeline = load_existing_pipeline(run_dir, config_name)
        print(f"Successfully loaded pipeline from {run_dir}/{config_name}")
    except Exception as e:
        print(f"Error loading pipeline: {str(e)}")
        return
    
    # Create widgets
    query_input = widgets.Text(
        value='',
        placeholder='Enter your question about SEC filings...',
        description='Query:',
        layout=widgets.Layout(width='80%')
    )
    
    submit_button = widgets.Button(
        description='Submit',
        button_style='primary',
        tooltip='Submit your question'
    )
    
    view_chunks_button = widgets.Button(
        description='View All Chunks',
        button_style='info',
        tooltip='View all chunks sent to the model',
        disabled=True
    )
    
    output_area = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            overflow_y='auto',
            max_height='500px'
        )
    )
    
    reference_dropdown = widgets.Dropdown(
        options=[],
        description='View Reference:',
        disabled=True,
        layout=widgets.Layout(width='80%')
    )
    
    # Add view mode selector for references
    view_mode = widgets.RadioButtons(
        options=['Original Document', 'Raw Chunk Text'],
        description='View Mode:',
        disabled=True,
        layout=widgets.Layout(width='80%')
    )
    
    reference_output = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            overflow_y='auto',
            max_height='600px'
        )
    )
    
    # Store the last query result
    last_result = {'sources': [], 'answer': ''}
    
    # Remove any existing event handlers to prevent duplicates
    submit_button._click_handlers.callbacks = []
    view_chunks_button._click_handlers.callbacks = []
    
    # Define the on_reference_change function
    def on_reference_change(change):
        update_reference_display(reference_dropdown, view_mode, reference_output)
    
    # Define the on_view_chunks_clicked function
    def on_view_chunks_clicked(b):
        display_all_chunks(last_result.get('sources', []), reference_output)
    
    # Remove any existing observers to prevent duplicates
    reference_dropdown._trait_notifiers.get('value', {}).get('change', [])[:] = []
    view_mode._trait_notifiers.get('value', {}).get('change', [])[:] = []
    
    # Set up event handlers
    submit_button.on_click(
        lambda b: on_submit_button_clicked(
            b, 
            query_input, 
            output_area, 
            reference_output, 
            reference_dropdown, 
            view_mode, 
            pipeline, 
            last_result,
            model,
            max_tokens,
            temperature,
            top_k,
            initial_retrieval_multiplier,
            view_chunks_button
        )
    )
    
    # Set up view chunks button handler
    view_chunks_button.on_click(on_view_chunks_clicked)
    
    # Set up reference dropdown observer
    reference_dropdown.observe(on_reference_change, names='value')
    view_mode.observe(on_reference_change, names='value')
    
    # Create the UI layout
    input_area = widgets.HBox([query_input, submit_button])
    button_area = widgets.HBox([view_chunks_button])
    reference_area = widgets.VBox([reference_dropdown, view_mode, reference_output])
    
    # Display the UI
    display(widgets.HTML("<h1>SEC Filing RAG Chatbot</h1>"))
    display(input_area)
    display(output_area)
    display(button_area)
    display(widgets.HTML("<h2>References</h2>"))
    display(reference_area)
    
    # Return the widgets for further customization if needed
    return {
        'query_input': query_input,
        'submit_button': submit_button,
        'output_area': output_area,
        'reference_dropdown': reference_dropdown,
        'view_mode': view_mode,
        'reference_output': reference_output,
        'view_chunks_button': view_chunks_button
    }