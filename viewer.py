#!/usr/bin/env python3
"""
CSV and Database Table Combiner

This program takes a CSV file with matching results and a database file,
then creates a table showing:
- Item number from the CSV
- Quantity from the CSV
- Original text from the CSV  
- Processed text from the CSV
- Part number (if exists, otherwise blank)
- Item description from the database that matches that part number
- Confidence value from the CSV
"""

import argparse
import csv
import sys
import re
from typing import Dict, List, Optional

import pdf2parts
import match
from pathlib import Path

def load_database(database_file: str) -> Dict[str, str]:
    """
    Load the database file and create a mapping from part number to description.
    
    Args:
        database_file: Path to the database CSV file
        
    Returns:
        Dictionary mapping part numbers to descriptions
    """
    part_to_description = {}
    part_to_type = {}
    
    try:
        with open(database_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                part_number = row.get('Item Number', '').strip()
                description = row.get('Item Description', '').strip()
                # irritatingly, the first column has a funky leading character 
                # so we have to do this the hard way
                type = ""
                for key,val in row.items():
                   if "Terms" in key:
                       type = val
                if part_number and description:
                    part_to_description[part_number] = description
                    part_to_type[part_number] = type

    except FileNotFoundError:
        print(f"Error: Database file '{database_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading database file '{database_file}': {e}")
        sys.exit(1)
    
    return part_to_description, part_to_type

def process_csv_file(csv_file: str, matcher, description_mapping: Dict[str, str], 
                                             type_mapping: Dict[str, str]) -> List[Dict]:
    """
    Process the CSV file and combine with database information.
    
    Args:
        csv_file: Path to the CSV file with matching results
        description_mapping: Dictionary mapping part numbers to descriptions
        type_mapping: Dictionary mapping part numbers to types
        
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
                quantity = row.get('Quantity', '').strip()
                processed_text = row.get('Description', '').strip()
                original_text = row.get('Original_Text', '').strip()
                part_number = row.get('Part_Number', '').strip()
                confidence = row.get('Confidence', '').strip()
                
                # Get description from database if part number exists
                item_description = ""
                item_type = ""
                if part_number and part_number in description_mapping:
                    item_description = description_mapping[part_number]
                    item_type = type_mapping[part_number]
                
                # Create result row
                result_row = {
                    'item_number': item_number,
                    'quantity': quantity,
                    'processed_text': processed_text,
                    'original_text': original_text,
                    'part_number': part_number if part_number else '',
                    'item_description': item_description,
                    'item_type' : item_type,
                    'item_components' : str(matcher.parse_lumber_item(item_description)),
                    'confidence': confidence,
                }
                
                results.append(result_row)
                
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file '{csv_file}': {e}")
        sys.exit(1)
    
    return results

def print_table(results: List[Dict], fields:Dict) -> None:
    """
    Print the results in a formatted table.
    
    Args:
        results: List of dictionaries containing the combined data
    """
    if not results:
        print("No data to display.")
        return
    
    # Calculate column widths
    col_widths = {}
    order = sorted(fields, key=lambda f: fields[f]["order"])
    for field in order:
        col_widths[field] = min(fields[field]["max_width"], max(len(str(r[field])) for r in results))

    # some overrides:
    col_widths['item_number'] = 5
    col_widths['item_type'] = 10
    col_widths['confidence'] = 5
    
    # Print header
    header = ""
    first = True
    for field in order:
        header += "" if first else " | "
        header += f"{fields[field]['header']:<{col_widths[field]}}"
        first = False
    
    print(header)
    print("-" * len(header))
    
    # Print data rows
    previous_item_number = None
    for row in results:
        # Show item number, quantity, and original text only if different from the previous row
        is_new_item = row['item_number'] != previous_item_number
        first = True
        data_row = ""
        for field in order:
            if field == 'confidence':
                # Format confidence to 2 decimal places
                confidence_value = row['confidence']
                try:
                    confidence_formatted = f"{float(confidence_value):.2f}"
                except (ValueError, TypeError):
                    confidence_formatted = confidence_value
                display_text = str(confidence_formatted)
            else:
                display_text = row[field] if (is_new_item or fields[field]['show_dups']) else ""
            data_row += ('' if first else ' | ') + f"{display_text[:col_widths[field]]:<{col_widths[field]}}"
            first = False

        data_row.strip()
        print(data_row)
        
        # Update previous item number for next iteration
        previous_item_number = row['item_number']

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
                fieldnames = ['item_number', 'quantity', 'processed_text', 'original_text', 'part_number', 'item_description', 'item_type', 'confidence']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving to CSV file '{output_file}': {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Print out concise list of matching items")

    # Positional arguments (no switches needed)
    parser.add_argument('match.csv', help='Path to the match CSV file')
    parser.add_argument('skulist.csv', help='Path to the parts database CSV file')
    
    # switches
    parser.add_argument('-O', '--original-text', action='store_true', 
                        help='Include original description in the output')
    parser.add_argument('-c', '--components', action='store_true', 
                        help='Include item components in the output')
    parser.add_argument('-w', '--max-col-width', type=int, default=0,
                        help='Do not allow columns wider than this')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    csv_file = getattr(args, "match.csv")
    database_file = getattr(args, "skulist.csv")

    print(f"Loading database from: {database_file}")
    description_mapping,type_mapping = load_database(database_file)
    print(f"Loaded {len(description_mapping)} items from database")
    
    # to get components, we need to load stuff the way pdf2parts loads stuff
    if True:
        matcherdbs = {}
        db_name = Path(database_file).stem
        db = pdf2parts.load_database(database_file, db_name, quiet=True)
        matcherdbs[db_name] = db
        matcher = match.RulesMatcher(matcherdbs, False)

    print(f"\nProcessing CSV file: {csv_file}")
    results = process_csv_file(csv_file, matcher, description_mapping, type_mapping)
    print(f"Processed {len(results)} rows from CSV")
    
    fields = {}
    maxwidth = 9999 if args.max_col_width == 0 else args.max_col_width
    for i, (name, header) in enumerate([('item_number', 'Item'),             ('quantity', 'Qty'),
                                        ('processed_text', 'Text'),          ('part_number', 'SKU'),
                                        ('item_description', 'Description'), ('item_type', 'Type'),
                                        ('confidence', 'Conf'),              ('original_text', 'Original Text'),
                                        ('item_components', 'Components')]):
        fields[name] = {'order':i, 'header':header, 'max_width':maxwidth, 'show_dups': (i > 2 and 'original' not in name)}

    if not args.original_text:
        del fields['original_text']
        fields['item_components']['order'] -= 1
    if not args.components:
        del fields['item_components']

    print("\n" + "="*120)
    print("COMBINED RESULTS TABLE")
    print("="*120)
    print_table(results, fields)
    
    exit(0)

    # Save to CSV
    output_file = "combined_results.csv"
    save_to_csv(results, output_file)

if __name__ == "__main__":
    main()
