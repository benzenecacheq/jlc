#!/usr/bin/env python3
"""
CSV and Database Table Combiner

This program takes a CSV file with matching results and a database file,
then creates a table showing:
- Item number from the CSV
- Original text from the CSV  
- Part number (if exists, otherwise blank)
- Item description from the database that matches that part number
- Confidence value from the CSV
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

def load_database(database_file: str) -> Dict[str, str]:
    """
    Load the database file and create a mapping from part number to description.
    
    Args:
        database_file: Path to the database CSV file
        
    Returns:
        Dictionary mapping part numbers to descriptions
    """
    part_to_description = {}
    
    try:
        with open(database_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                part_number = row.get('Item Number', '').strip()
                description = row.get('Item Description', '').strip()
                if part_number and description:
                    part_to_description[part_number] = description
    except FileNotFoundError:
        print(f"Error: Database file '{database_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading database file '{database_file}': {e}")
        sys.exit(1)
    
    return part_to_description

def process_csv_file(csv_file: str, database_mapping: Dict[str, str]) -> List[Dict]:
    """
    Process the CSV file and combine with database information.
    
    Args:
        csv_file: Path to the CSV file with matching results
        database_mapping: Dictionary mapping part numbers to descriptions
        
    Returns:
        List of dictionaries containing the combined data
    """
    results = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract data from CSV
                item_number = row.get('Item_Number', '').strip()
                original_text = row.get('Original_Text', '').strip()
                part_number = row.get('Part_Number', '').strip()
                confidence = row.get('Confidence', '').strip()
                
                # Get description from database if part number exists
                item_description = ""
                if part_number and part_number in database_mapping:
                    item_description = database_mapping[part_number]
                
                # Create result row
                result_row = {
                    'item_number': item_number,
                    'original_text': original_text,
                    'part_number': part_number if part_number else '',
                    'item_description': item_description,
                    'confidence': confidence
                }
                
                results.append(result_row)
                
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file '{csv_file}': {e}")
        sys.exit(1)
    
    return results

def print_table(results: List[Dict]) -> None:
    """
    Print the results in a formatted table.
    
    Args:
        results: List of dictionaries containing the combined data
    """
    if not results:
        print("No data to display.")
        return
    
    # Calculate column widths
    col_widths = {
        'item_number': max(len('Item Number'), max(len(str(r['item_number'])) for r in results)),
        'original_text': max(len('Original Text'), max(len(r['original_text']) for r in results)),
        'part_number': max(len('Part Number'), max(len(r['part_number']) for r in results)),
        'item_description': max(len('Item Description'), max(len(r['item_description']) for r in results)),
        'confidence': max(len('Confidence'), max(len(str(r['confidence'])) for r in results))
    }
    
    # Ensure minimum widths for readability
    col_widths['item_number'] = max(col_widths['item_number'], 12)
    col_widths['original_text'] = max(col_widths['original_text'], 20)
    col_widths['part_number'] = max(col_widths['part_number'], 12)
    col_widths['item_description'] = max(col_widths['item_description'], 30)
    col_widths['confidence'] = max(col_widths['confidence'], 10)
    
    # Print header
    header = (f"{'Item Number':<{col_widths['item_number']}} | "
              f"{'Original Text':<{col_widths['original_text']}} | "
              f"{'Part Number':<{col_widths['part_number']}} | "
              f"{'Item Description':<{col_widths['item_description']}} | "
              f"{'Confidence':<{col_widths['confidence']}}")
    
    print(header)
    print("-" * len(header))
    
    # Print data rows
    for row in results:
        data_row = (f"{row['item_number']:<{col_widths['item_number']}} | "
                   f"{row['original_text']:<{col_widths['original_text']}} | "
                   f"{row['part_number']:<{col_widths['part_number']}} | "
                   f"{row['item_description']:<{col_widths['item_description']}} | "
                   f"{row['confidence']:<{col_widths['confidence']}}")
        print(data_row)

def save_to_csv(results: List[Dict], output_file: str) -> None:
    """
    Save the results to a CSV file.
    
    Args:
        results: List of dictionaries containing the combined data
        output_file: Path to the output CSV file
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                fieldnames = ['item_number', 'original_text', 'part_number', 'item_description', 'confidence']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving to CSV file '{output_file}': {e}")

def main():
    """Main function to run the program."""
    if len(sys.argv) != 3:
        print("Usage: python combine_csv_db.py <csv_file> <database_file>")
        print(f"Example: python {sys.argv[0]} 1_results/lumber_matches.csv 1_results/skulist_fixed.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    database_file = sys.argv[2]
    
    print(f"Loading database from: {database_file}")
    database_mapping = load_database(database_file)
    print(f"Loaded {len(database_mapping)} items from database")
    
    print(f"\nProcessing CSV file: {csv_file}")
    results = process_csv_file(csv_file, database_mapping)
    print(f"Processed {len(results)} rows from CSV")
    
    print("\n" + "="*100)
    print("COMBINED RESULTS TABLE")
    print("="*100)
    print_table(results)
    
    # Save to CSV
    output_file = "combined_results.csv"
    save_to_csv(results, output_file)

if __name__ == "__main__":
    main()
