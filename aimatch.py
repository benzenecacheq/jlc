#!/usr/bin/env python3
###############################################################################
# PDF2Parts - Lumber List Scanner and Parts Matcher
#
# This program uses the Claude API to scan handwritten lumber lists from PDF files or images,
# then iteratively searches through multiple parts databases to find matches.
# 
# Supports:
# - PDF files (converts to images for processing)
# - Image files (.jpg, .png, .gif, .webp)
# - Multiple parts databases
# - Intelligent matching with Claude AI
###############################################################################
import os
import sys
import csv
import json
import base64
import argparse
import subprocess
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import anthropic
from dataclasses import dataclass
import re
import shutil
from util import *

# PDF processing imports
try:
    from pdf2image import convert_from_path
    from PIL import Image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: pdf2image not installed. PDF support will be disabled.")
    print("Install with: pip install pdf2image")

###############################################################################
class AIMatcher:
    def __init__(self, api_key: str, databases):
        """Initialize with Claude API key"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.databases = databases

    def load_training_data(self, csv_path: str) -> bool:
        """Load training data from CSV file (original_text, correct_sku columns)"""
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as file:
                # Try to detect the CSV format
                sample = file.read(1024)
                file.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(file, delimiter=delimiter)
                training_examples = []
                
                for row in reader:
                    # Clean up whitespace in all fields
                    clean_row = {k.strip(): v.strip() if v else '' for k, v in row.items()}
                    
                    # Check for required columns
                    if 'original_text' not in clean_row or 'correct_sku' not in clean_row:
                        print(f"Warning: Training file {csv_path} missing required columns (original_text, correct_sku)")
                        continue
                    
                    original_text = clean_row['original_text']
                    correct_sku = clean_row['correct_sku']
                    
                    if original_text and correct_sku:
                        training_examples.append({
                            'original_text': original_text,
                            'correct_sku': correct_sku
                        })
                
                self.training_data.extend(training_examples)
                print(f"✓ Loaded {len(training_examples)} training examples from {Path(csv_path).name}")
                return True
                
        except Exception as e:
            print(f"✗ Error loading training data from {csv_path}: {e}")
            return False
    
    def _build_training_examples_text(self, max_examples: int = None) -> str:
        """Build training examples text for inclusion in prompts"""
        if not self.training_data:
            return ""
        
        examples = []
        limit = max_examples if max_examples is not None else len(self.training_data)
        for example in self.training_data[:limit]:
            examples.append(f'  "{example["original_text"]}" -> {example["correct_sku"]}')
        
        return f"""
TRAINING EXAMPLES (use these to understand common patterns and improve accuracy):
======================================================================
{chr(10).join(examples)}
======================================================================
"""
    
    def find_all_matches_ai(self, scanned_items, debug: bool = False, output_dir: str = ".") -> None:
        """Find matches using batch processing: all items in single Claude call per database"""
        if debug:
            print("\n" + "="*60)
            print("SEARCHING FOR MATCHES USING BATCH PROCESSING")
            print("(All items processed together for efficiency)")
            print("="*60)

        # Create debug log file for detailed output
        debug_log_file = Path(output_dir) / "matching_debug.log"
        with open(debug_log_file, 'w', encoding='utf-8') as log_file:
            log_file.write("DETAILED MATCHING DEBUG LOG\n")
            log_file.write("=" * 60 + "\n\n")

        # Process each database
        for db_name in self.databases:
            if debug:
                print(f"\nProcessing database: {db_name}")
            
            # Batch match all items against this database
            self._batch_match_items(db_name, debug, debug_log_file)

        if debug:
            print(f"\n✓ Detailed matching log saved to: {debug_log_file}")

    def _batch_match_items(self, scanned_items, database_name: str, debug: bool, debug_log_file) -> None:
        """Match all scanned items against a single database in one AI call"""
        database = self.databases[database_name]
        parts = database['parts']
        headers = database['headers']
        
        if not parts:
            print(f"  No parts in database {database_name}")
            return
        
        print(f"  Batch matching {len(scanned_items)} items against {len(parts)} parts...")
        
        # Format all items for batch processing
        items_text = ""
        for i, item in enumerate(scanned_items, 1):
            items_text += f"[{i}] Quantity: {item.quantity} | Description: {item.description} | Original: {item.original_text}\n"
        
        # Format database
        db_entries = []
        header_line = "|".join(headers)
        db_entries.append(f"HEADERS: {header_line}")
        
        for part in parts:
            row_data = []
            for header in headers:
                value = part.get(header, '').strip()
                row_data.append(value)
            db_entries.append("|".join(row_data))
        
        db_text = "\n".join(db_entries)
        training_examples = self._build_training_examples_text()
        
        # Build batch prompt
        batch_prompt = load_prompt("batch_match_prompt", 
                                    items_text=items_text,
                                    training_examples=training_examples,
                                    training_instruction="Use the training examples above to understand common patterns",
                                    database_name=database_name,
                                    column_info="",
                                    db_text=db_text,
                                    max_matches=3)
        
        # Write prompt to debug file
        with open(debug_log_file, 'a', encoding='utf-8') as log_file:
            log_file.write(f"\n=== BATCH MATCHING PROMPT FOR {database_name} ===\n")
            log_file.write(f"Items to match: {len(scanned_items)}\n")
            log_file.write(f"Database parts: {len(parts)}\n")
            log_file.write(f"Prompt length: {len(batch_prompt)} characters\n")
            log_file.write(f"Training examples: {len(training_examples)} characters\n")
            log_file.write("\n--- PROMPT CONTENT ---\n")
            log_file.write(batch_prompt)
            log_file.write("\n--- END PROMPT ---\n\n")
        
        try:
            if os.getenv("BATCH_RESPONSE") and os.path.exists(os.getenv("BATCH_RESPONSE")):
                with open(os.getenv("BATCH_RESPONSE"), "r") as f:
                    response_text = f.read()
            else:
                model = "claude-sonnet-4-20250514"
                max_tokens = 8000
                messages = [{"role": "user", "content": batch_prompt}]
                response = call_ai_with_retry(self.client, messages, 
                                              model=model, max_tokens=max_tokens)
                response_text = response.content[0].text

            # Write response to debug file
            with open(debug_log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n=== RESPONSE ===\n")
                log_file.write(response_text)
                log_file.write(f"\n=== END RESPONSE ===\n\n")
            
            # Parse the batch response
            self._parse_batch_response(response_text, database_name, debug, debug_log_file)
            
        except Exception as e:
            print(f"  ✗ Error in batch matching for {database_name}: {e}")
            with open(debug_log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n=== ERROR ===\n")
                log_file.write(f"Error in batch matching: {e}\n")
                log_file.write(f"Error type: {type(e).__name__}\n")
                log_file.write(f"=== END ERROR ===\n\n")
            exit(1)

    def _parse_batch_response(self, scanned_items, response_text: str, database_name: str, debug: bool, debug_log_file) -> None:
        """Parse the batch response and assign matches to items"""
        try:
            print(f"  Parsing Claude response...")
            
            # Extract JSON from response - handle both bare JSON and markdown code blocks
            json_text = None
            
            # First try to find JSON in markdown code blocks
            json_start = response_text.find('```json')
            
            if json_start != -1:
                # Find the start of the JSON array after ```json
                array_start = response_text.find('[', json_start)
                
                if array_start != -1:
                    # Count brackets to find the matching closing bracket
                    bracket_count = 0
                    for i, char in enumerate(response_text[array_start:], array_start):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_text = response_text[array_start:i+1]
                                break
            else:
                # Try to find bare JSON array using bracket counting
                start_pos = response_text.find('[')
                if start_pos != -1:
                    # Count brackets to find the matching closing bracket
                    bracket_count = 0
                    for i, char in enumerate(response_text[start_pos:], start_pos):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_text = response_text[start_pos:i+1]
                                break
            
            if not json_text:
                if json_start != -1:
                    print(f"  ✗ Response appears to have been truncated.  "
                           "You may need to increase token limit.")
                else:
                    print(f"  ✗ No JSON array found in response")
                with open(debug_log_file, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"\n=== JSON EXTRACTION ERROR ===\n")
                    log_file.write(f"Could not find JSON array in response\n")
                    log_file.write(f"Response length: {len(response_text)}\n")
                    log_file.write(f"First 500 chars: {response_text[:500]}\n")
                    log_file.write(f"Last 500 chars: {response_text[-500:]}\n")
                    log_file.write(f"=== END JSON EXTRACTION ERROR ===\n\n")
                return
            print(f"  Extracted JSON length: {len(json_text)} characters")
            
            batch_matches = json.loads(json_text)
            print(f"  Parsed {len(batch_matches)} match entries")
            
            # Group matches by item index
            item_matches = {}
            for match in batch_matches:
                item_idx = match.get('item_index', 1) - 1
                if item_idx not in item_matches:
                    item_matches[item_idx] = []
                
                part_match = PartMatch(
                    description="",  # Will be set when assigning to items
                    part_number=match.get('part_number', ''),
                    database_name=database_name,
                    database_description=match.get('description', ''),
                    confidence=match.get('confidence', 'low'),
                    reason=match.get('reason', '')
                )
                item_matches[item_idx].append(part_match)
            
            print(f"  Grouped matches for {len(item_matches)} items")
            
            # Assign matches to items
            for i, item in enumerate(scanned_items):
                if i in item_matches:
                    matches = item_matches[i]
                else:
                    # try to match with keyword matching
                    # matches = self.keyword_matcher.match_item(item, database_name, debug=False)
                    # keyword matching needs work so don't use for now
                    matches = []

                if len(matches):
                    # Update the description field for each match
                    for match in matches:
                        match.description = item.description
                    item.matches.extend(matches)
                    
                    if debug:
                        print(f"  [{i+1:2d}] {item.description} - {len(matches)} matches")
                    
                    # Log to debug file
                    with open(debug_log_file, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n=== ITEM {i+1}: {item.description} ===\n")
                        log_file.write(f"Database: {database_name}\n")
                        log_file.write(f"Matches found: {len(matches)}\n")
                        for match in matches:
                            log_file.write(f"  - {match.confidence}: {match.part_number} | {match.database_description}\n")
                        log_file.write("-" * 50 + "\n")
                elif debug:
                    print(f"  [{i+1:2d}] {item.description} - 0 matches")
                        
        except Exception as e:
            print(f"  ✗ Error parsing batch response: {e}")
            with open(debug_log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n=== PARSING ERROR ===\n")
                log_file.write(f"Error parsing batch response: {e}\n")
                log_file.write(f"Error type: {type(e).__name__}\n")
                log_file.write(f"Response text: {response_text[:1000]}...\n")
                log_file.write(f"=== END PARSING ERROR ===\n\n")
