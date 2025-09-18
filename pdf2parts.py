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
def load_database(csv_path: str, database_name: str, output_dir: str = None) -> bool:
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
            print(f"✓ Saved fixed CSV to: {output_path}")
        
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
    with open(output_file, 'w') as f:
        f.write("LUMBER LIST MATCHING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Databases searched: {', '.join(databases.keys())}\n")
        f.write(f"Total items scanned: {len(scanned_items)}\n\n")
        
        matched_count = sum(1 for item in scanned_items if item.matches)
        f.write(f"Items with matches: {matched_count}/{len(scanned_items)}\n\n")
        
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
    with open(output_file, 'w', newline='') as f:
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
    
    parser.add_argument('-tm', '--test-match', action='store_true', help='Test matching function')

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
    
    # Load databases from command line arguments
    print(f"\nLoading {len(args.databases)} parts database(s)...")
    
    databases = {}
    databases_loaded = 0
    for db_path in args.databases:
        if not os.path.exists(db_path):
            print(f"Warning: Database file not found: {db_path}")
            continue
        
        # Use filename as database name
        db_name = Path(db_path).stem
        db = load_database(db_path, db_name, str(output_dir))
        if db:
            databases_loaded += 1
            databases[db_name] = db
        else:
            print(f"Failed to load database: {db_path}")
            exit(1)
    
    # Initialize matcher
    scanner = Scanner(api_key, databases)
    matcher = RulesMatcher(databases, global_debug=args.verbose_matching)
    ai_matcher = AIMatcher(api_key, databases)
    
    if databases_loaded == 0:
        print("Error: No databases loaded successfully")
        sys.exit(1)
    
    # Count tokens in loaded databases
    if not args.quiet:
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
            
            if ai_matcher.load_training_data(training_path):
                training_loaded += 1
            elif not args.quiet:
                print(f"Failed to load training data: {training_path}")
        
        if not args.quiet and training_loaded > 0:
            # Count tokens in training data
            training_text = ai_matcher._build_training_examples_text()
            training_tokens = count_tokens(ai_matcher.client, training_text)
            print(f"✓ Loaded {training_loaded} training data file(s) with {len(ai_matcher.training_data)} total examples")
            print(f"  Training data tokens: {training_tokens:,}")
    
    if args.test_match:
        matcher._load_attributes()
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
            result = matcher.match_lumber_item(user_input)
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
    items = scanner.scan_document_with_database_context(args.document, output_dir=str(output_dir), 
                                                        verbose=args.verbose_matching)
    if not items:
        print("Error: No items found in document")
        sys.exit(1)
    scanned_items = items
    
    # Find matches using selected approach
    matcher.find_all_matches(scanned_items, output_dir=str(output_dir))
    if args.use_ai_matching:
        ai_matcher.find_all_matches_ai(scanned_items, 
                             debug=args.verbose_matching, output_dir=str(output_dir))
    
    # Generate outputs with specified names and directory
    if not args.quiet:
        print("\nGenerating reports...")
    
    report_path = output_dir / args.report_name
    csv_path = output_dir / args.csv_name
    
    generate_report(databases, scanned_items, str(report_path))
    export_csv(scanned_items, str(csv_path))
    
    if not args.quiet:
        print("\n✓ Processing complete!")
    
    # Summary
    matched_items = sum(1 for item in scanned_items if item.matches)
    total_items = len(scanned_items)
    match_rate = (matched_items / total_items * 100) if total_items > 0 else 0
    
    print(f"\nSUMMARY:")
    print(f"  Items scanned: {total_items}")
    print(f"  Items matched: {matched_items}")
    print(f"  Match rate: {match_rate:.1f}%")
    print(f"  Databases searched: {len(databases)}")
    print(f"  Output files saved to: {output_dir}")
    print(f"    Report: {args.report_name}")
    print(f"    CSV: {args.csv_name}")
    
    # Run viewer if requested
    if args.view:
        run_viewer(str(csv_path), args.databases[0], args.quiet)

def run_viewer(csv_path: str, database_path: str, quiet: bool = False) -> None:
    """
    Run the viewer program to display results in a formatted table.
    
    Args:
        csv_path: Path to the CSV file with matching results
        database_path: Path to the first database file for the viewer
        quiet: Whether to suppress output
    """
    try:
        # Get the path to the viewer script (same directory as this script)
        script_dir = Path(__file__).parent
        viewer_path = script_dir / "viewer.py"
        
        if not viewer_path.exists():
            print(f"Error: Viewer script not found at {viewer_path}")
            return
        
        if not quiet:
            print(f"\nRunning viewer with results...")
            print(f"  CSV file: {csv_path}")
            print(f"  Database: {database_path}")
        
        # Run the viewer program
        result = subprocess.run([
            sys.executable, str(viewer_path), csv_path, database_path
        ], capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"Error: Viewer program exited with code {result.returncode}")
        elif not quiet:
            print("✓ Viewer completed successfully")
            
    except FileNotFoundError:
        print("Error: Python interpreter not found")
    except Exception as e:
        print(f"Error running viewer: {e}")

###############################################################################
if __name__ == "__main__":
    main()
