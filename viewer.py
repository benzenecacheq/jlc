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

def process_csv_file(csv_file: str, description_mapping: Dict[str, str], 
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
        'item_number': max(len('Item'), max(len(str(r['item_number'])) for r in results)),
        'quantity': max(len('Qty'), max(len(str(r['quantity'])) for r in results)),
        'processed_text' : max(len('Text'), max(len(r['processed_text']) for r in results)),
        'original_text': max(len('Original Text'), max(len(r['original_text']) for r in results)),
        'part_number': max(len('SKU'), max(len(r['part_number']) for r in results)),
        'item_description': max(len('Item Description'), max(len(r['item_description']) for r in results)),
        'confidence': max(len('Conf'), max(len(str(r['confidence'])) for r in results))
    }
    
    # Ensure minimum widths for readability
    col_widths['item_number'] = max(col_widths['item_number'], 4)
    col_widths['quantity'] = max(col_widths['quantity'], 4)
    col_widths['processed_text'] = max(col_widths['processed_text'], 20)
    col_widths['original_text'] = max(col_widths['original_text'], 20)
    col_widths['part_number'] = max(col_widths['part_number'], 4)
    col_widths['item_description'] = max(col_widths['item_description'], 20)
    col_widths['item_type'] = 6         # truncate if necessary
    col_widths['confidence'] = 5        # truncate if necessary
    
    # Print header
    header = (f"{'Item':<{col_widths['item_number']}} | "
              f"{'Qty':<{col_widths['quantity']}} | "
              f"{'Text':<{col_widths['processed_text']}} | "
              f"{'Original Text':<{col_widths['original_text']}} | "
              f"{'SKU':<{col_widths['part_number']}} | "
              f"{'Item Description':<{col_widths['item_description']}} | "
              f"{'Type':<{col_widths['item_type']}} | "
              f"{'Conf':<{col_widths['confidence']}}")
    
    print(header)
    print("-" * len(header))
    
    # Print data rows
    previous_item_number = None
    for row in results:
        # Show item number, quantity, and original text only if different from the previous row
        is_new_item = row['item_number'] != previous_item_number
        display_item_number = row['item_number'] if is_new_item else ''
        display_quantity = row['quantity'] if is_new_item else ''
        display_processed_text = row['processed_text'] if is_new_item else ''
        display_original_text = row['original_text'] if is_new_item else ''
        
        # Format confidence to 2 decimal places
        confidence_value = row['confidence']
        try:
            confidence_formatted = f"{float(confidence_value):.2f}"
        except (ValueError, TypeError):
            confidence_formatted = confidence_value
        
        data_row = (f"{display_item_number:<{col_widths['item_number']}} | "
                   f"{display_quantity:<{col_widths['quantity']}} | "
                   f"{display_processed_text:<{col_widths['processed_text']}} | "
                   f"{display_original_text:<{col_widths['original_text']}} | "
                   f"{row['part_number']:<{col_widths['part_number']}} | "
                   f"{row['item_description']:<{col_widths['item_description']}} | "
                   f"{row['item_type'][:col_widths['item_type']]:<{col_widths['item_type']}} | "
                   f"{confidence_formatted[:col_widths['confidence']]:<{col_widths['confidence']}}")
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

def main():
    """Main function to run the program."""
    if len(sys.argv) != 3:
        print("Usage: python combine_csv_db.py <csv_file> <database_file>")
        print(f"Example: python {sys.argv[0]} 1_results/lumber_matches.csv 1_results/skulist_fixed.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    database_file = sys.argv[2]
    
    print(f"Loading database from: {database_file}")
    description_mapping,type_mapping = load_database(database_file)
    print(f"Loaded {len(description_mapping)} items from database")
    
    print(f"\nProcessing CSV file: {csv_file}")
    results = process_csv_file(csv_file, description_mapping, type_mapping)
    print(f"Processed {len(results)} rows from CSV")
    
    # add type to list

    
    print("\n" + "="*120)
    print("COMBINED RESULTS TABLE")
    print("="*120)
    print_table(results)
    
    exit(0)

    # Save to CSV
    output_file = "combined_results.csv"
    save_to_csv(results, output_file)

if __name__ == "__main__":
    main()
