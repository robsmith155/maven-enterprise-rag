import json
from typing import List, Dict, Any, Optional, Set
import re
from difflib import SequenceMatcher
import os
import glob
from dataclasses import dataclass

# Import IPython-related modules only when needed for the interactive components
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    INTERACTIVE_MODE = True
except ImportError:
    INTERACTIVE_MODE = False

def get_similarity_ratio(a, b):
    """Get the similarity ratio between two strings"""
    return SequenceMatcher(None, a, b).ratio()

def find_best_match(span_text, section_content, min_similarity=0.9):
    """
    Find the best matching text in the section content
    Returns: (is_match, best_match, similarity_ratio, context_before, context_after)
    """
    # Debug info
    print(f"Finding best match for text of length {len(span_text)} in content of length {len(section_content)}")
    print(f"First 50 chars of span_text: {span_text[:50]}...")
    print(f"First 50 chars of section_content: {section_content[:50]}...")
    
    # Check for exact match first
    if span_text in section_content:
        print("Found exact match!")
        # Find the position of the exact match
        start_idx = section_content.find(span_text)
        
        # Get context around the match
        context_size = 200  # Characters of context before and after
        context_before = section_content[max(0, start_idx - context_size):start_idx]
        context_after = section_content[start_idx + len(span_text):min(len(section_content), start_idx + len(span_text) + context_size)]
        
        return True, span_text, 1.0, context_before, context_after
    
    # Try to find the best matching substring
    best_match = ""
    best_ratio = 0
    best_context_before = ""
    best_context_after = ""
    
    # First, try a direct similarity check between the entire span and content
    # This helps with very short spans
    direct_ratio = get_similarity_ratio(span_text, section_content[:min(len(section_content), len(span_text) * 2)])
    if direct_ratio > best_ratio:
        best_ratio = direct_ratio
        best_match = section_content[:min(len(section_content), len(span_text))]
        best_context_before = ""
        best_context_after = section_content[min(len(section_content), len(span_text)):min(len(section_content), len(span_text) + 200)]
    
    # Split the content into sentences for more natural matching
    try:
        sentences = re.split(r'(?<=[.!?])\s+', section_content)
        print(f"Split content into {len(sentences)} sentences")
        
        # For very short spans, combine adjacent sentences
        if len(span_text) < 100:
            combined_sentences = []
            for i in range(len(sentences) - 1):
                combined_sentences.append(sentences[i] + " " + sentences[i+1])
            sentences.extend(combined_sentences)
            print(f"Added {len(combined_sentences)} combined sentences")
        
        # Check each sentence for similarity
        for sentence in sentences:
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            ratio = get_similarity_ratio(span_text, sentence)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = sentence
                
                # Find the position of this sentence in the original content
                if sentence in section_content:
                    start_idx = section_content.find(sentence)
                    context_size = 200
                    best_context_before = section_content[max(0, start_idx - context_size):start_idx]
                    best_context_after = section_content[start_idx + len(sentence):min(len(section_content), start_idx + len(sentence) + context_size)]
    except Exception as e:
        print(f"Error in sentence matching: {e}")
    
    # If we didn't find a good sentence match, try a sliding window approach
    if best_ratio < min_similarity:
        try:
            print("Trying sliding window approach")
            window_size = len(span_text) + 50  # Add some context
            step_size = max(1, min(50, len(span_text) // 4))  # Reasonable step size
            
            # Limit the number of windows to check to avoid excessive computation
            max_windows = 1000
            window_count = min(max_windows, (len(section_content) - len(span_text)) // step_size)
            
            for i in range(0, window_count * step_size, step_size):
                if i + window_size > len(section_content):
                    break
                    
                window = section_content[i:i+window_size]
                ratio = get_similarity_ratio(span_text, window)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = window
                    
                    # Get context around this window
                    context_size = 200
                    best_context_before = section_content[max(0, i - context_size):i]
                    best_context_after = section_content[i + len(window):min(len(section_content), i + len(window) + context_size)]
            
            print(f"Best match from sliding window: ratio={best_ratio:.2f}")
        except Exception as e:
            print(f"Error in sliding window matching: {e}")
    
    print(f"Final best match: ratio={best_ratio:.2f}, length={len(best_match)}")
    return best_ratio >= min_similarity, best_match, best_ratio, best_context_before, best_context_after

class QuestionReviewer:
    def __init__(self, questions, output_dir='.', min_similarity=0.7):
        self.questions = questions
        self.output_dir = output_dir
        self.current_idx = 0
        self.current_source_idx = 0
        self.rejected_indices = set()
        self.edited_questions = {}
        self.min_similarity = min_similarity
        self.section_contents = {}  # Cache for section contents
        self.debug_output = widgets.Output()  # For debugging output
        
        # Create widgets
        self.question_text = widgets.HTML()
        self.answer_text = widgets.HTML()
        self.source_info = widgets.HTML()
        self.comparison_view = widgets.HTML()
        self.context_view = widgets.HTML()
        
        self.edit_source_text = widgets.Textarea(
            description='Edit Source:',
            layout=widgets.Layout(width='90%', height='100px')
        )
        
        # Navigation buttons
        self.prev_question_button = widgets.Button(description='Prev Question')
        self.next_question_button = widgets.Button(description='Next Question')
        self.prev_source_button = widgets.Button(description='Prev Source')
        self.next_source_button = widgets.Button(description='Next Source')
        
        # Action buttons
        self.accept_question_button = widgets.Button(description='Accept Question', button_style='success')
        self.reject_question_button = widgets.Button(description='Reject Question', button_style='danger')
        self.use_best_match_button = widgets.Button(description='Use Best Match', button_style='warning')
        self.update_source_button = widgets.Button(description='Update Source', button_style='info')
        self.save_results_button = widgets.Button(description='Save Results', button_style='primary')
        
        # Set up button callbacks
        self.prev_question_button.on_click(self.prev_question_handler)
        self.next_question_button.on_click(self.next_question_handler)
        self.prev_source_button.on_click(self.prev_source_handler)
        self.next_source_button.on_click(self.next_source_handler)
        self.accept_question_button.on_click(self.accept_current_question)
        self.reject_question_button.on_click(self.reject_current_question)
        self.use_best_match_button.on_click(self.use_current_best_match)
        self.update_source_button.on_click(self.update_current_source)
        self.save_results_button.on_click(self.save_current_results)
        
        # Layout
        self.nav_buttons = widgets.HBox([
            self.prev_question_button, self.next_question_button, 
            self.prev_source_button, self.next_source_button
        ])
        
        self.action_buttons = widgets.HBox([
            self.accept_question_button, self.reject_question_button,
            self.use_best_match_button, self.update_source_button, 
            self.save_results_button
        ])
        
        self.output = widgets.Output()
        
        # Main container
        self.container = widgets.VBox([
            widgets.HTML(value="<h2>Question Review Tool</h2>"),
            widgets.HTML(value="<h3>Question:</h3>"),
            self.question_text,
            widgets.HTML(value="<h3>Answer:</h3>"),
            self.answer_text,
            widgets.HTML(value="<h3>Source Information:</h3>"),
            self.source_info,
            widgets.HTML(value="<h3>Source Comparison:</h3>"),
            self.comparison_view,
            widgets.HTML(value="<h3>Context Around Match:</h3>"),
            self.context_view,
            self.edit_source_text,
            self.nav_buttons,
            self.action_buttons,
            self.output,
            widgets.HTML(value="<h3>Debug Output:</h3>"),
            self.debug_output
        ])
    
    def get_section_content(self, source):
        """Get the content of a section from the processed report path"""
        company = source.company
        year = source.year
        section = source.section
        
        # Check if we've already loaded this section
        cache_key = f"{company}_{year}_{section}"
        if cache_key in self.section_contents:
            return self.section_contents[cache_key]
        
        # Use the processed_path from the source
        if not hasattr(source, 'processed_path') or not source.processed_path:
            with self.debug_output:
                print(f"No processed_path available for source: company={company}, year={year}, section={section}")
            return ""
        
        report_file = source.processed_path
        
        with self.debug_output:
            print(f"Using processed_path from source: {report_file}")
            print(f"File exists: {os.path.exists(report_file)}")
        
        if not os.path.exists(report_file):
            with self.debug_output:
                print(f"Report file not found: {report_file}")
            return ""
        
        # Load the report
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            with self.debug_output:
                print(f"Report loaded successfully")
                print(f"Report keys: {list(report.keys())}")
                if "sections" in report:
                    print(f"Available sections: {list(report['sections'].keys())}")
                if "section_content" in report:
                    print(f"Available section_content: {list(report['section_content'].keys())}")
            
            # First check if section exists in "sections" key (old format)
            if "sections" in report and section in report["sections"]:
                content = report["sections"][section]["content"]
                self.section_contents[cache_key] = content
                
                with self.debug_output:
                    print(f"Section {section} found in 'sections', content length: {len(content)}")
                    print(f"First 100 chars: {content[:100]}...")
                
                return content
            # Then check if section exists in "section_content" key (new format)
            elif "section_content" in report and section in report["section_content"]:
                content = report["section_content"][section]
                self.section_contents[cache_key] = content
                
                with self.debug_output:
                    print(f"Section {section} found in 'section_content', content length: {len(content)}")
                    print(f"First 100 chars: {content[:100]}...")
                
                return content
            # Check if section is in available_sections but not found in either location
            elif "available_sections" in report and section in report["available_sections"]:
                with self.debug_output:
                    print(f"Section {section} is listed in available_sections but content not found")
                return ""
            else:
                with self.debug_output:
                    print(f"Section {section} not found in report {company} {year}")
                return ""
        except Exception as e:
            with self.debug_output:
                print(f"Error loading report: {e}")
                import traceback
                traceback.print_exc()
            return ""
    
            
    def display(self):
        """Display the reviewer interface"""
        display(self.container)
        self.update_display()
    
    def update_display(self):
        """Update the display with the current question"""
        with self.debug_output:
            clear_output()
            print(f"Updating display for question {self.current_idx + 1}, source {self.current_source_idx + 1}")
        
        if self.current_idx >= len(self.questions):
            self.question_text.value = "<b>End of questions</b>"
            self.answer_text.value = ""
            self.source_info.value = ""
            self.comparison_view.value = ""
            self.context_view.value = ""
            self.edit_source_text.value = ""
            return
        
        # Get the current question (either original or edited)
        if self.current_idx in self.edited_questions:
            question = self.edited_questions[self.current_idx]
        else:
            question = self.questions[self.current_idx]
        
        # Display question and answer
        self.question_text.value = f"<p><b>{question.question}</b></p>"
        self.answer_text.value = f"<p>{question.answer}</p>"
        
        # Display source information
        source_count = len(question.source_information)
        source_html = f"<p>Source {self.current_source_idx + 1} of {source_count}</p>"
        
        if source_count > 0 and self.current_source_idx < source_count:
            source = question.source_information[self.current_source_idx]
            
            with self.debug_output:
                print(f"Processing source: company={source.company}, year={source.year}, section={source.section}")
                if hasattr(source, 'subsection') and source.subsection:
                    print(f"Subsection: {source.subsection}")
                if hasattr(source, 'processed_path') and source.processed_path:
                    print(f"Processed path: {source.processed_path}")
            
            source_html += f"<p><b>Company:</b> {source.company}<br>"
            source_html += f"<b>Year:</b> {source.year}<br>"
            source_html += f"<b>Section:</b> {source.section}<br>"
            if hasattr(source, 'subsection') and source.subsection:
                source_html += f"<b>Subsection:</b> {source.subsection}<br>"
            if hasattr(source, 'processed_path') and source.processed_path:
                source_html += f"<b>Processed path:</b> {source.processed_path}<br>"
            source_html += "</p>"
            
            # Get the section content
            section_content = self.get_section_content(source)
            
            with self.debug_output:
                print(f"Section content length: {len(section_content)}")
                if len(section_content) > 0:
                    print(f"First 100 chars of section content: {section_content[:100]}...")
            
            # Check if span_text exists in the source
            if not hasattr(source, 'span_text') or not source.span_text:
                with self.debug_output:
                    print("No span_text found in source, using empty string")
                span_text = ""
            else:
                span_text = source.span_text
                
            with self.debug_output:
                print(f"Span text length: {len(span_text)}")
                if span_text:
                    print(f"First 100 chars of span text: {span_text[:100]}...")
            
            # Update the edit source text
            self.edit_source_text.value = span_text
            
            if len(section_content) > 0 and span_text:
                with self.debug_output:
                    is_match, best_match, similarity, context_before, context_after = find_best_match(
                        span_text, section_content, self.min_similarity
                    )
                
                # Create comparison view
                comparison_html = "<div style='display: flex; flex-direction: column;'>"
                
                # LLM extracted text
                comparison_html += "<div style='margin-bottom: 20px;'>"
                comparison_html += "<h4>LLM Extracted Text:</h4>"
                comparison_html += f"<div style='padding: 10px; border: 1px solid #ccc; background-color: #f8f8f8;'>{span_text}</div>"
                comparison_html += "</div>"
                
                # Best match from report
                match_status = f"✅ Exact Match (similarity: 1.00)" if span_text == best_match else f"⚠️ Similar Match (similarity: {similarity:.2f})"
                if not is_match:
                    match_status = f"❌ No Good Match (best similarity: {similarity:.2f})"
                
                comparison_html += "<div>"
                comparison_html += f"<h4>Best Match from Report: {match_status}</h4>"
                comparison_html += f"<div style='padding: 10px; border: 1px solid #ccc; background-color: #f0f8ff;'>{best_match}</div>"
                comparison_html += "</div>"
                
                comparison_html += "</div>"
                
                self.comparison_view.value = comparison_html
                
                # Create context view
                context_html = "<div style='margin-top: 20px;'>"
                context_html += "<div style='white-space: pre-wrap; font-family: monospace; padding: 10px; border: 1px solid #ccc; background-color: #f5f5f5;'>"
                context_html += f"{context_before}<mark style='background-color: #FFEB3B;'>{best_match}</mark>{context_after}"
                context_html += "</div>"
                context_html += "</div>"
                
                self.context_view.value = context_html
            else:
                if not section_content:
                    self.comparison_view.value = "<p>No section content found</p>"
                elif not span_text:
                    self.comparison_view.value = "<p>No span text found in source</p>"
                else:
                    self.comparison_view.value = "<p>Unable to process content</p>"
                self.context_view.value = ""
        else:
            source_html += "<p>No sources available</p>"
            self.comparison_view.value = ""
            self.context_view.value = ""
            self.edit_source_text.value = ""
        
        self.source_info.value = source_html
        
        # Update status
        with self.output:
            clear_output()
            status = "REJECTED" if self.current_idx in self.rejected_indices else "ACCEPTED"
            print(f"Question {self.current_idx + 1} of {len(self.questions)} - Status: {status}")
    
    def prev_question_handler(self, b):
        """Go to the previous question"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.current_source_idx = 0
            self.update_display()
    
    def next_question_handler(self, b):
        """Go to the next question"""
        if self.current_idx < len(self.questions) - 1:
            self.current_idx += 1
            self.current_source_idx = 0
            self.update_display()
    
    def prev_source_handler(self, b):
        """Go to the previous source"""
        if self.current_source_idx > 0:
            self.current_source_idx -= 1
            self.update_display()
    
    def next_source_handler(self, b):
        """Go to the next source"""
        question = self.questions[self.current_idx]
        if self.current_idx in self.edited_questions:
            question = self.edited_questions[self.current_idx]
            
        if self.current_source_idx < len(question.source_information) - 1:
            self.current_source_idx += 1
            self.update_display()
    
    def accept_current_question(self, b):
        """Accept the current question"""
        if self.current_idx in self.rejected_indices:
            self.rejected_indices.remove(self.current_idx)
        self.next_question_handler(None)
    
    def reject_current_question(self, b):
        """Reject the current question"""
        self.rejected_indices.add(self.current_idx)
        self.next_question_handler(None)
    
    def use_current_best_match(self, b):
        """Use the best match as the source text"""
        with self.debug_output:
            print("Using best match")
        
        if self.current_idx >= len(self.questions):
            return
            
        # Get the current question
        if self.current_idx in self.edited_questions:
            question = self.edited_questions[self.current_idx]
        else:
            question = self.questions[self.current_idx].copy()
            self.edited_questions[self.current_idx] = question
        
        if self.current_source_idx < len(question.source_information):
            source = question.source_information[self.current_source_idx]
            section_content = self.get_section_content(source)
            
            # Check if span_text exists
            if not hasattr(source, 'span_text') or not source.span_text:
                with self.debug_output:
                    print("No span_text found in source, cannot use best match")
                return
            
            span_text = source.span_text
            
            if len(section_content) > 0 and span_text:
                is_match, best_match, similarity, _, _ = find_best_match(span_text, section_content, self.min_similarity)
                
                if is_match:
                    # Update the source text
                    source.span_text = best_match
                    self.update_display()
            else:
                with self.debug_output:
                    print("No section content or span_text found, cannot use best match")
    
    def update_current_source(self, b):
        """Update the source text with the edited text"""
        with self.debug_output:
            print("Updating source")
        
        if self.current_idx >= len(self.questions):
            return
            
        # Get the current question
        if self.current_idx in self.edited_questions:
            question = self.edited_questions[self.current_idx]
        else:
            question = self.questions[self.current_idx].copy()
            self.edited_questions[self.current_idx] = question
        
        if self.current_source_idx < len(question.source_information):
            source = question.source_information[self.current_source_idx]
            
            # Update the source text
            source.span_text = self.edit_source_text.value
            self.update_display()
    
    def save_current_results(self, button):
        """Save the reviewed questions to the output directory"""
        with self.output:
            clear_output()
            try:
                if not self.output_dir:
                    print("Error: No output directory specified")
                    return
                    
                # Ensure output directory exists
                os.makedirs(self.output_dir, exist_ok=True)
                
                # Create output filename
                output_path = os.path.join(self.output_dir, "benchmark_dataset_reviewed.json")
                
                # Create the final list of questions
                final_questions = []
                for i, question in enumerate(self.questions):
                    if i in self.rejected_indices:
                        continue
                        
                    if i in self.edited_questions:
                        final_questions.append(self.edited_questions[i])
                    else:
                        final_questions.append(question)
                
                # Save to file
                with open(output_path, 'w') as f:
                    json.dump([q.__dict__ for q in final_questions], f, indent=2)
                
                print(f"Successfully saved {len(final_questions)} accepted questions to {output_path}")
                print(f"Rejected {len(self.rejected_indices)} questions")
            except Exception as e:
                print(f"Error saving results: {e}")
                import traceback
                traceback.print_exc()

def review_questions(questions, output_dir='.', min_similarity=0.7):
    """
    Start the interactive question review process
    
    Args:
        questions: List of BenchmarkQuestion objects
        min_similarity: Minimum similarity threshold (0.0-1.0)
        
    Returns:
        The reviewer object, which has a save_results method to get the final questions
    """
    reviewer = QuestionReviewer(questions, output_dir, min_similarity)
    reviewer.display()
    return reviewer