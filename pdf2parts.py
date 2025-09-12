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
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import anthropic
from dataclasses import dataclass
import re
import shutil
from util import *
from match import Matcher
from scan import Scanner

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
def load_prompt(name, **kwargs):
    """
    Read a file and replace all text enclosed in backquotes (`) with
    corresponding values from kwargs.

    Args:
        file_path (str): Path to the file to process
        **kwargs: Key-value pairs to substitute for backquoted fields

    Returns:
        str: The file content with all backquoted fields replaced
    """

    # Read the file content
    script_dir = str(Path(__file__).parent)
    fname = script_dir + "/" + name
    print(f"Loading prompt {fname}")
    try:
        with open(fname, 'r') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading {fname}: {e}", file=sys.stderr)
        exit(1)

    # Find all backquoted fields and replace them
    import re

    def replace_match(match):
        field_name = match.group(1)
        if field_name in kwargs:
            return str(kwargs[field_name])
        else:
            print(f"Unable to replace {field_name} in template", file=sys.stderr)
            exit(1)

    # The pattern matches any text between backquotes
    pattern = r"`([^`]+)`"
    result = re.sub(pattern, replace_match, content)

    return result

###############################################################################
class LumberListMatcher:
    def __init__(self, api_key: str):
        """Initialize with Claude API key"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.databases = {}
        self.scanned_items = []
        self.training_data = []
        
        self.scanner = Scanner(self.client, self.databases)
        self.keyword_matcher = Matcher(self.databases)

    def load_database(self, csv_path: str, database_name: str, output_dir: str = None) -> bool:
        """Load a parts database from CSV file"""
        try:
            parts = []
            with open(csv_path, 'r', newline='', encoding='utf-8') as file:
                # Try to detect the CSV format
                sample = file.read(1024)
                file.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(file, delimiter=delimiter)
                first_column_name = None
                last_first_column_value = None
                
                for row in reader:
                    # Clean up whitespace in all fields
                    clean_row = {k.strip(): v.strip() if v else '' for k, v in row.items()}
                    
                    # Get the first column name and value
                    if first_column_name is None:
                        first_column_name = list(clean_row.keys())[0]
                    
                    first_column_value = clean_row.get(first_column_name, '')
                    
                    # If first column is empty, use the last non-empty value
                    if not first_column_value and last_first_column_value:
                        clean_row[first_column_name] = last_first_column_value
                    elif first_column_value:
                        last_first_column_value = first_column_value
                    
                    parts.append(clean_row)
                    
            self.databases[database_name] = {
                'parts': parts,
                'headers': list(parts[0].keys()) if parts else []
            }
            
            # Save the fixed CSV to output directory if provided
            if output_dir and parts:
                output_path = Path(output_dir) / f"{database_name}_fixed.csv"
                with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.DictWriter(outfile, fieldnames=parts[0].keys(), delimiter=delimiter)
                    writer.writeheader()
                    writer.writerows(parts)
                print(f"✓ Saved fixed CSV to: {output_path}")
            
            print(f"✓ Loaded {len(parts)} parts from {database_name}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading {database_name}: {e}")
            return False
    
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
    
    def generate_report(self, output_file: str = "lumber_match_report.txt") -> None:
        """Generate a detailed report of all matches"""
        with open(output_file, 'w') as f:
            f.write("LUMBER LIST MATCHING REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Databases searched: {', '.join(self.databases.keys())}\n")
            f.write(f"Total items scanned: {len(self.scanned_items)}\n\n")
            
            matched_count = sum(1 for item in self.scanned_items if item.matches)
            f.write(f"Items with matches: {matched_count}/{len(self.scanned_items)}\n\n")
            
            for i, item in enumerate(self.scanned_items, 1):
                f.write(f"[{i:2d}] ITEM: {item.description}\n")
                f.write(f"     Original: {item.original_text}\n")
                f.write(f"     Quantity: {item.quantity}\n")
                
                if item.matches:
                    f.write("     MATCHES:\n")
                    for match in item.matches:
                        f.write(f"       {match.confidence.upper():8s} | {match.part_number:15s} | {match.database_name:15s}\n")
                        f.write(f"                | {match.database_description}\n")
                else:
                    f.write("     MATCHES: None found\n")
                f.write("\n")
        
        print(f"\n✓ Detailed report saved to: {output_file}")
    
    def export_csv(self, output_file: str = "lumber_matches.csv") -> None:
        """Export matches to CSV for further processing"""
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Item_Number', 'Quantity', 'Description', 'Original_Text', 
                           'Match_Found', 'Part_Number', 'Database', 'Database_Description', 
                           'Confidence'])
            
            for i, item in enumerate(self.scanned_items, 1):
                if item.matches:
                    for match in item.matches:
                        writer.writerow([
                            i, item.quantity, item.description, item.original_text,
                            'Yes', match.part_number, match.database_name, 
                            match.database_description, match.confidence
                        ])
                else:
                    writer.writerow([
                        i, item.quantity, item.description, item.original_text,
                        'No', '', '', '', ''
                    ])
        
        print(f"✓ CSV export saved to: {output_file}")

    def find_all_matches_ai(self, debug: bool = False, output_dir: str = ".") -> None:
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

    def _batch_match_items(self, database_name: str, debug: bool, debug_log_file) -> None:
        """Match all scanned items against a single database in one AI call"""
        database = self.databases[database_name]
        parts = database['parts']
        headers = database['headers']
        
        if not parts:
            print(f"  No parts in database {database_name}")
            return
        
        print(f"  Batch matching {len(self.scanned_items)} items against {len(parts)} parts...")
        
        # Format all items for batch processing
        items_text = ""
        for i, item in enumerate(self.scanned_items, 1):
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
            log_file.write(f"Items to match: {len(self.scanned_items)}\n")
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
            # Fall back to individual matching
            # self._fallback_individual_matching(database_name, debug, debug_log_file)
            exit(1)

    def _parse_batch_response(self, response_text: str, database_name: str, debug: bool, debug_log_file) -> None:
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
            for i, item in enumerate(self.scanned_items):
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

    def _fallback_individual_matching(self, database_name: str, debug: bool, debug_log_file) -> None:
        """Fallback to individual matching if batch fails"""
        for i, item in enumerate(self.scanned_items, 1):
            matches = self.keyword_matcher.match_item(item.description, database_name, debug=False)
            item.matches.extend(matches)
            
            if debug:
                print(f"  [{i:2d}] {item.description} - {len(matches)} matches (fallback)")

###############################################################################
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Scan handwritten lumber lists and match items against parts databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf parts1.csv parts2.csv
  %(prog)s -k YOUR_API_KEY photo.png lumber_db.csv hardware_db.csv
  %(prog)s --output-dir ./results lumber_list.pdf database.csv
  %(prog)s --training-data training1.csv training2.csv document.pdf database.csv
  %(prog)s --use-keyword-matching document.pdf database.csv
  
Note: All output files (reports, debug files, CSV exports) will be saved in a 
subdirectory named after the input file (e.g., 'document_results/', 'lumber_list_results/')
        """
    )
    
    parser.add_argument('document', 
                       help='Path to the lumber list document (image file: .jpg, .png, .gif, .webp or PDF file: .pdf)')
    
    parser.add_argument('databases', nargs='+',
                       help='One or more CSV files containing parts databases')
    
    parser.add_argument('-k', '--api-key',
                       help='Anthropic API key (can also use ANTHROPIC_API_KEY env var)')
    
    parser.add_argument('-o', '--output-dir', default='.',
                       help='Base directory to save output files. A subdirectory named after the input file will be created (default: current directory)')
    
    parser.add_argument('--report-name', default='lumber_match_report.txt',
                       help='Name for the text report file (default: lumber_match_report.txt)')
    
    parser.add_argument('--csv-name', default='lumber_matches.csv',
                       help='Name for the CSV export file (default: lumber_matches.csv)')
    
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    parser.add_argument('--use-claude-matching', action='store_true',
                       help='Use Claude AI for intelligent parts matching (slower but more accurate)')
    
    parser.add_argument('--full-database', action='store_true',
                       help='Send full database to Claude for matching (use with large SKU lists)')
    
    parser.add_argument('--verbose-matching', action='store_true',
                       help='Show detailed matching debug output on console (default: save to files)')
    
    parser.add_argument('--test-match', '-tm', action='store_true', help='Test matching function')

    parser.add_argument('--training-data',
                       nargs='*',
                       default=[],
                       help='One or more CSV files containing training data (original_text, correct_sku columns)')
    
    parser.add_argument('--use-keyword-matching', '-mk',
                       action='store_true',
                       help='Use keyword-based matching instead of Claude AI (for comparison)')
    
    return parser.parse_args()

###############################################################################
def main():
    """Main program execution"""
    args = parse_arguments()
    
    if not args.quiet:
        print("LUMBER LIST SCANNER AND PARTS MATCHER")
        print("="*50)
    
    # Get API key from argument, environment, or user input
    api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        api_key = input("Enter your Anthropic API key: ").strip()
        if not api_key:
            print("Error: API key is required")
            sys.exit(1)
    
    # Check document file exists
    if not os.path.exists(args.document):
        print(f"Error: Document file not found: {args.document}")
        sys.exit(1)
    
    # Create output directory based on input file name
    input_file_stem = Path(args.document).stem
    output_dir = Path(args.output_dir) / f"{input_file_stem}_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.quiet:
        print(f"Output directory: {output_dir}")
    
    # Initialize matcher
    matcher = LumberListMatcher(api_key)
    
    # Load databases from command line arguments
    if not args.quiet:
        print(f"\nLoading {len(args.databases)} parts database(s)...")
    
    databases_loaded = 0
    for db_path in args.databases:
        if not os.path.exists(db_path):
            print(f"Warning: Database file not found: {db_path}")
            continue
        
        # Use filename as database name
        db_name = Path(db_path).stem
        
        if matcher.load_database(db_path, db_name, str(output_dir)):
            databases_loaded += 1
        elif not args.quiet:
            print(f"Failed to load database: {db_path}")
    
    if databases_loaded == 0:
        print("Error: No databases loaded successfully")
        sys.exit(1)
    
    # Count tokens in loaded databases
    if not args.quiet:
        total_db_tokens = 0
        for db_name, db_info in matcher.databases.items():
            # Build database text similar to what's used in matching
            parts = db_info['parts']
            headers = db_info['headers']
            
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
            db_tokens = count_tokens(matcher.client, db_text)
            total_db_tokens += db_tokens
            print(f"  {db_name} database tokens: {db_tokens:,}")
        
        print(f"  Total database tokens: {total_db_tokens:,}")
    
    # Load training data if provided
    if args.training_data:
        if not args.quiet:
            print(f"\nLoading {len(args.training_data)} training data file(s)...")
        
        training_loaded = 0
        for training_path in args.training_data:
            if not os.path.exists(training_path):
                print(f"Warning: Training file not found: {training_path}")
                continue
            
            if matcher.load_training_data(training_path):
                training_loaded += 1
            elif not args.quiet:
                print(f"Failed to load training data: {training_path}")
        
        if not args.quiet and training_loaded > 0:
            # Count tokens in training data
            training_text = matcher._build_training_examples_text()
            training_tokens = count_tokens(matcher.client, training_text)
            print(f"✓ Loaded {training_loaded} training data file(s) with {len(matcher.training_data)} total examples")
            print(f"  Training data tokens: {training_tokens:,}")
    
    if args.test_match:
        matcher.keyword_matcher._load_attributes()
        print("Testing match_lumber_items()")
        print("Enter strings to test (type 'exit' to quit):")
        print("-" * 40)
        
        while True:
            # Get input from user
            user_input = input("Enter a string: ")
            
            # Check if user wants to exit
            if user_input.lower() == "exit":
                print("Exiting test function.")
                break
            
            # Call cleanup function and print result
            result = matcher.keyword_matcher.match_lumber_item(user_input)
            if len(result) == 0:
                print(f'No result matched "{user_input}".')
            else:
                print("-" * 40)
                print(f'Original: "{user_input}"')
                print(f'matched:')
                for match in result:
                   print(f' -> {match.part_number}: {match.database_description}')

            print("-" * 40)

    # Scan document
    if not args.quiet:
        print(f"\nScanning document: {args.document}")
    items = matcher.scanner.scan_document_with_database_context(args.document, output_dir=str(output_dir), 
                                                        verbose=args.verbose_matching)
    if not items:
        print("Error: No items found in document")
        sys.exit(1)
    matcher.scanned_items = items
    
    # Find matches using selected approach
    if args.use_keyword_matching:
        matcher.keyword_matcher.find_all_matches(matcher.scanned_items, 
                                    debug=args.verbose_matching, output_dir=str(output_dir))
    else:
        matcher.find_all_matches_ai(debug=args.verbose_matching, output_dir=str(output_dir))
    
    # Generate outputs with specified names and directory
    if not args.quiet:
        print("\nGenerating reports...")
    
    report_path = output_dir / args.report_name
    csv_path = output_dir / args.csv_name
    
    matcher.generate_report(str(report_path))
    matcher.export_csv(str(csv_path))
    
    if not args.quiet:
        print("\n✓ Processing complete!")
    
    # Summary
    matched_items = sum(1 for item in matcher.scanned_items if item.matches)
    total_items = len(matcher.scanned_items)
    match_rate = (matched_items / total_items * 100) if total_items > 0 else 0
    
    print(f"\nSUMMARY:")
    print(f"  Items scanned: {total_items}")
    print(f"  Items matched: {matched_items}")
    print(f"  Match rate: {match_rate:.1f}%")
    print(f"  Databases searched: {len(matcher.databases)}")
    print(f"  Output files saved to: {output_dir}")
    print(f"    Report: {args.report_name}")
    print(f"    CSV: {args.csv_name}")

###############################################################################
if __name__ == "__main__":
    main()
