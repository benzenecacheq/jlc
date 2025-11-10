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
from scan import Scanner
from match import RulesMatcher
from aimatch import AIMatcher

###############################################################################
def load_database(csv_path: str, database_name: str, output_dir: str=None, quiet=False) -> bool:
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
                clean_row = {re.sub('[^a-zA-Z0-9 ]','', k.strip()): v.strip() if v else '' 
                                for k, v in row.items()}
                
                # Get the first column name and value
                if first_column_name is None:
                    first_column_name = list(clean_row.keys())[0]
                
                first_column_value = clean_row.get(first_column_name, '')
                
                # If first column is empty, use the last non-empty value
                if not first_column_value and last_first_column_value:
                    clean_row[first_column_name] = last_first_column_value
                elif first_column_value:
                    last_first_column_value = first_column_value
                
                # add database name
                clean_row['database'] = database_name
                parts.append(clean_row)
                
        # Save the fixed CSV to output directory if provided
        if output_dir and parts:
            output_path = Path(output_dir) / f"{database_name}_fixed.csv"
            with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=parts[0].keys(), delimiter=delimiter)
                writer.writeheader()
                writer.writerows(parts)
            if not quiet:
               print(f"✓ Saved fixed CSV to: {output_path}")
        
        if not quiet:
           print(f"✓ Loaded {len(parts)} parts from {database_name}")
        return {
                   'parts': parts,
                   'headers': list(parts[0].keys()) if parts else []
               }
    
    except Exception as e:
        print(f"✗ Error loading {database_name}: {e}")
        return {}
    
###############################################################################
def generate_report(databases, scanned_items, output_file: str = "lumber_match_report.txt") -> None:
    """Generate a detailed report of all matches"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("LUMBER LIST MATCHING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Databases searched: {', '.join(databases.keys())}\n")
        f.write(f"Total items scanned: {len(scanned_items)}\n\n")
        
        matched_count = sum(1 for item in scanned_items if item.matches)
        f.write(f"Items with matches: {matched_count}/{len(scanned_items)}\n")
        multiple_match = sum(1 for item in scanned_items if len(item.matches) > 1)
        f.write(f"Items with multiple matches: {multiple_match}\n\n")
        
        for i, item in enumerate(scanned_items, 1):
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

###############################################################################
def export_csv(scanned_items, output_file: str = "lumber_matches.csv") -> None:
    """Export matches to CSV for further processing"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Item_Number', 'Quantity', 'Description', 'Original_Text', 
                       'Match_Found', 'Part_Number', 'Database', 'Database_Description', 
                       'Confidence'])
        
        for i, item in enumerate(scanned_items, 1):
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

###############################################################################
def run_viewer(csv_path: str, database_path: str, original_text: bool) -> None:
    """
    Run the viewer program to display results in a formatted table.
    
    Args:
        csv_path: Path to the CSV file with matching results
        database_path: Path to the first database file for the viewer
    """
    try:
        # Get the path to the viewer script (same directory as this script)
        script_dir = Path(__file__).parent
        viewer_path = script_dir / "viewer.py"
        
        if not viewer_path.exists():
            print(f"Error: Viewer script not found at {viewer_path}")
            return
        
        print(f"\nRunning viewer with results...")
        print(f"  CSV file: {csv_path}")
        print(f"  Database: {database_path}")
    
        # Run the viewer program
        args = [sys.executable, str(viewer_path), csv_path, database_path]
        if original_text:
           print("  Flags: -O")
           args.append("-O")
        args.append("-w45")
        result = subprocess.run(args, capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"Error: Viewer program exited with code {result.returncode}")
            
    except FileNotFoundError:
        print("Error: Python interpreter not found")
    except Exception as e:
        print(f"Error running viewer: {e}")

###############################################################################
def errexit(error_string):
    print(f"ERROR: {error_string}", file=sys.stderr)
    exit(1)

def run_matcher(document, api_key, database_names, training_data, use_ai_matching, output_dir, 
                output_file_name, error_func=errexit, notify_func=print, debug=False):
    # Load databases from command line arguments
    notify_func(f"\nLoading {len(database_names)} parts database(s)...")
    
    databases = {}
    databases_loaded = 0
    for db_path in database_names:
        if not os.path.exists(db_path):
            error_func(f"Database file not found: {db_path}")
            return None, None
        
        # Use filename as database name
        db_name = Path(db_path).stem
        db = load_database(db_path, db_name, str(output_dir))
        if db:
            databases_loaded += 1
            databases[db_path] = db
        else:
            error_func(f"Failure loading database {db_path}")
            return None, None
    
    # Initialize matcher
    scanner = Scanner(api_key, databases)
    matcher = RulesMatcher(databases, global_debug=debug)
    ai_matcher = AIMatcher(api_key, databases)
    
    if databases_loaded == 0:
        error_func("No databases loaded")
        return None, None
    
    # Count tokens in loaded databases
    if debug:
        total_db_tokens = 0
        for db_name, db_info in databases.items():
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
            db_tokens = count_tokens(ai_matcher.client, db_text)
            total_db_tokens += db_tokens
            notify_func(f"  {db_name} database tokens: {db_tokens:,}")
        
        notify_func(f"  Total database tokens: {total_db_tokens:,}")
    
    # Load training data if provided
    if training_data:
        notify_func(f"\nLoading {len(training_data)} training data file(s)...")
        
        training_loaded = 0
        for training_path in training_data:
            if not os.path.exists(training_path):
                error_func(f"Missing training file {training_path}")
                return None, None
            
            if ai_matcher.load_training_data(training_path):
                training_loaded += 1
            else:
                error_func(f"Failure loading training file {training_path}")
                return None, None
        
        if training_loaded > 0:
            # Count tokens in training data
            training_text = ai_matcher._build_training_examples_text()
            training_tokens = count_tokens(ai_matcher.client, training_text)
            notify_func(f"✓ Loaded {training_loaded} training data file(s) with {len(ai_matcher.training_data)} total examples")
            notify_func(f"  Training data tokens: {training_tokens:,}")
    
    # Scan document
    notify_func(f"Scanning document: {document}")
    scanned_items = scanner.scan_document(document, output_dir=str(output_dir), verbose=debug)
    if not scanned_items:
        error_func("No items found in document")
        return None, None
    
    # Find matches using selected approach
    notify_func(f"Matching items in document: {document}")
    matcher.find_all_matches(scanned_items, output_dir=str(output_dir))
    if use_ai_matching:
        ai_matcher.find_all_matches_ai(scanned_items, 
                             debug=debug, output_dir=str(output_dir))

    notify_func(f"Matching complete. Exporting results to {str(output_dir / output_file_name)}")
    export_csv(scanned_items, str(output_dir / output_file_name))

    return databases, scanned_items

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
  %(prog)s --view document.pdf database.csv
  
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
    
    parser.add_argument('--keyword-file',
                        help='Name of a .csv file that provides additional keywords')
                        
    parser.add_argument('-o', '--output-dir', default="",
                        help='Base directory to save output files. A subdirectory named after the input '
                             'file will be created (default: current directory)')
    parser.add_argument("-c", "--csv-file-name", default="matches.csv", help="Output matches to this file")
    
    parser.add_argument('-O', '--original-text', action='store_true',
                        help='Show the original text in the viewer')
    parser.add_argument('--report-name', default='lumber_match_report.txt',
                        help='Name for the text report file (default: lumber_match_report.txt)')
    
    parser.add_argument('--verbose-matching', action='store_true',
                        help='Show detailed matching debug output on console (default: save to files)')
    
    parser.add_argument('--training-data', nargs='*', default=[],
                        help='One or more CSV files containing training data (original_text, correct_sku columns)')
    
    parser.add_argument('-ai', '--use-ai-matching', action='store_true', help='Use AI for matching')
    
    parser.add_argument('-v', '--view', action='store_true',
                        help='Run the viewer program to display results in a formatted table')
    
    return parser.parse_args()

###############################################################################
def main():
    """Main program execution"""
    args = parse_arguments()

    if args.keyword_file:
        os.environ["MATCHER_KEYWORDS"] = args.keyword_file
    
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
    if args.output_dir == "":
        args.output_dir = Path(args.document).parent
    output_dir = Path(args.output_dir) / f"{input_file_stem}_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    databases, scanned_items = run_matcher(args.document, api_key, args.databases, args.training_data, 
                                           args.use_ai_matching, output_dir, args.csv_file_name,
                                           debug=args.verbose_matching)

    # Generate outputs with specified names and directory
    print("\nGenerating report...")
    
    report_path = output_dir / args.report_name
    generate_report(databases, scanned_items, str(report_path))
    
    print("\n✓ Processing complete!")
    
    # Summary
    matched_items = sum(1 for item in scanned_items if item.matches)
    multiple_match = sum(1 for item in scanned_items if len(item.matches) > 1)
    total_items = len(scanned_items)
    match_rate = (matched_items / total_items * 100) if total_items > 0 else 0
    
    print(f"\nSUMMARY:")
    print(f"  Items scanned: {total_items}")
    print(f"  Items matched: {matched_items} ({multiple_match} with multiple matches)")
    print(f"  Match rate: {match_rate:.1f}%")
    print(f"  Databases searched: {len(databases)}")
    print(f"  Output files saved to: {output_dir}")
    print(f"    Report: {args.report_name}")
    print(f"    CSV: matches.csv")
    
    # Run viewer if requested
    if args.view:
        run_viewer(output_dir / args.csv_file_name, args.databases[0], args.original_text)

###############################################################################
if __name__ == "__main__":
    main()
