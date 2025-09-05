#!/usr/bin/env python3
"""
PDF2Parts - Lumber List Scanner and Parts Matcher

This program uses the Claude API to scan handwritten lumber lists from PDF files or images,
then iteratively searches through multiple parts databases to find matches.

Supports:
- PDF files (converts to images for processing)
- Image files (.jpg, .png, .gif, .webp)
- Multiple parts databases
- Intelligent matching with Claude AI
"""

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
import tempfile
import shutil

# PDF processing imports
try:
    from pdf2image import convert_from_path
    from PIL import Image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: pdf2image not installed. PDF support will be disabled.")
    print("Install with: pip install pdf2image")

@dataclass
class PartMatch:
    """Represents a matched part"""
    description: str
    part_number: str
    database_name: str
    database_description: str
    confidence: str  # "exact", "partial", "similar"

@dataclass
class ScannedItem:
    """Represents an item from the scanned list"""
    quantity: str
    description: str
    original_text: str
    matches: List[PartMatch]

class LumberListMatcher:
    def __init__(self, api_key: str):
        """Initialize with Claude API key"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.databases = {}
        self.scanned_items = []
        
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
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to images and return list of image file paths"""
        if not PDF_SUPPORT:
            raise ImportError("PDF support requires pdf2image. Install with: pip install pdf2image")
        
        # Create temporary directory for images
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        
        try:
            # Convert PDF to images with optimized DPI for API size limits
            # Start with 200 DPI, reduce if still too large
            dpi = 200
            max_size_mb = 4.5  # Leave some buffer under 5MB limit
            
            for attempt in range(3):  # Try up to 3 different DPI settings
                try:
                    images = convert_from_path(pdf_path, dpi=dpi)
                    
                    for i, image in enumerate(images):
                        # Save as JPEG with compression to reduce file size
                        image_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
                        
                        # Convert to RGB if necessary (JPEG doesn't support RGBA)
                        if image.mode in ('RGBA', 'LA', 'P'):
                            # Create white background
                            background = Image.new('RGB', image.size, (255, 255, 255))
                            if image.mode == 'P':
                                image = image.convert('RGBA')
                            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                            image = background
                        elif image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Save with compression
                        image.save(image_path, 'JPEG', quality=85, optimize=True)
                        
                        # Check file size
                        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                        if file_size_mb > max_size_mb:
                            print(f"  Page {i+1} too large ({file_size_mb:.1f}MB), retrying with lower DPI...")
                            raise ValueError(f"Image too large: {file_size_mb:.1f}MB")
                        
                        image_paths.append(image_path)
                        print(f"  Page {i+1} saved: {file_size_mb:.1f}MB")
                    
                    break  # Success, exit the retry loop
                    
                except ValueError as e:
                    if attempt < 2:  # Not the last attempt
                        dpi = int(dpi * 0.8)  # Reduce DPI by 20%
                        print(f"  Retrying with DPI={dpi}...")
                        # Clean up failed images
                        for path in image_paths:
                            if os.path.exists(path):
                                os.remove(path)
                        image_paths = []
                        continue
                    else:
                        raise Exception(f"Could not compress images small enough: {e}")
                
        except Exception as e:
            # Clean up temp directory on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise Exception(f"Error converting PDF to images: {e}")
        
        return image_paths, temp_dir
    
    def scan_document(self, document_path: str, debug_output: bool = True, output_dir: str = ".") -> List[ScannedItem]:
        """Use Claude API to scan and parse the lumber list with debug output"""
        try:
            file_ext = Path(document_path).suffix.lower()
            
            # Handle PDF files
            if file_ext == '.pdf':
                if not PDF_SUPPORT:
                    raise ImportError("PDF support requires pdf2image. Install with: pip install pdf2image")
                
                if debug_output:
                    print(f"Converting PDF to images: {document_path}")
                
                image_paths, temp_dir = self.convert_pdf_to_images(document_path)
                
                if debug_output:
                    print(f"PDF converted to {len(image_paths)} page(s)")
                
                all_items = []
                temp_files_to_cleanup = []
                
                try:
                    # Process each page
                    for i, image_path in enumerate(image_paths):
                        if debug_output:
                            print(f"Processing page {i+1}/{len(image_paths)}")
                        
                        page_items = self._scan_single_image(image_path, debug_output, output_dir, verbose)
                        all_items.extend(page_items)
                        temp_files_to_cleanup.append(image_path)
                    
                    self.scanned_items = all_items
                    return all_items
                    
                finally:
                    # Clean up temporary files
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Handle image files
            else:
                items = self._scan_single_image(document_path, debug_output, output_dir, verbose)
                self.scanned_items = items
                return items
                
        except Exception as e:
            print(f"Error scanning document: {e}")
            return []
    
    def _scan_single_image(self, image_path: str, debug_output: bool = True, output_dir: str = ".", verbose: bool = False) -> List[ScannedItem]:
        """Scan a single image file and return ScannedItems"""
        try:
            # Encode the image
            image_data = self.encode_image(image_path)

            # Determine image format from file extension
            file_ext = Path(image_path).suffix.lower()
            media_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = media_type_map.get(file_ext, 'image/jpeg')

            # Build context from loaded databases (if any)
            parts_context = ""
            if self.databases:
                parts_context = self._build_parts_context()

            # Create the prompt - simplified if no context available
            if parts_context:
                prompt_text = f"""Please scan this handwritten lumber/materials list and extract each item with its quantity and description.

    {parts_context}

    IMPORTANT: Use the parts reference above to help with:
    1. Correcting OCR errors and typos in handwritten text
    2. Standardizing lumber dimensions and terminology
    3. Recognizing common abbreviations (PT=Pressure Treated, DF=Douglas Fir, etc.)
    4. Matching similar items when handwriting is unclear

    Format your response as a JSON array where each item has:
    {{
        "quantity": "the number/amount",
        "description": "the item description (size, material, etc.) - corrected using parts reference",
        "original_text": "the original handwritten text as you read it",
        "confidence": "high|medium|low - how confident you are in the OCR reading"
    }}

    Focus on extracting all lumber, hardware, and construction materials. When handwriting is unclear, use the parts reference to make educated guesses about what the item likely is."""
            else:
                prompt_text = """Please scan this handwritten lumber/materials list and extract each item with its quantity and description.

    Format your response as a JSON array where each item has:
    {
        "quantity": "the number/amount",
        "description": "the item description (size, material, etc.)",
        "original_text": "the original handwritten text as you read it"
    }

    Focus on extracting all lumber, hardware, and construction materials. Be as accurate as possible with dimensions and specifications."""

            # Create the message with image using the current model
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt_text
                            }
                        ]
                    }
                ],
            )

            # Parse the response
            response_text = message.content[0].text

            # Debug: Save the raw response
            if debug_output:
                debug_file = Path(output_dir) / (Path(image_path).stem + "_scan_debug.txt")
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("RAW CLAUDE RESPONSE FROM DOCUMENT SCAN\n")
                    f.write("="*60 + "\n\n")
                    f.write(response_text)
                    f.write("\n\n" + "="*60 + "\n")
                    f.write("PARTS CONTEXT SENT TO CLAUDE:\n")
                    f.write("="*60 + "\n")
                    f.write(parts_context if parts_context else "No parts context provided")
                    f.write("\n")
                print(f"✓ Debug info saved to: {debug_file}")

            # Extract JSON from the response (it might be wrapped in markdown)
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find JSON array - look for opening bracket and find matching closing bracket
                start_pos = response_text.find('[')
                if start_pos != -1:
                    # Count brackets to find the matching closing bracket
                    bracket_count = 0
                    end_pos = start_pos
                    for i, char in enumerate(response_text[start_pos:], start_pos):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_pos = i + 1
                                break
                    json_text = response_text[start_pos:end_pos]
                else:
                    json_text = response_text

            # Debug: Save the extracted JSON
            if debug_output:
                json_debug_file = Path(output_dir) / (Path(image_path).stem + "_extracted_json.json")
                with open(json_debug_file, 'w', encoding='utf-8') as f:
                    f.write(json_text)
                print(f"✓ Extracted JSON saved to: {json_debug_file}")

            # Parse JSON
            items_data = json.loads(json_text)

            # Convert to ScannedItem objects
            scanned_items = []
            for item_data in items_data:
                item = ScannedItem(
                    quantity=item_data.get('quantity', ''),
                    description=item_data.get('description', ''),
                    original_text=item_data.get('original_text', ''),
                    matches=[]
                )
                scanned_items.append(item)

            # Debug: Save parsed items in readable format
            if debug_output:
                items_debug_file = Path(output_dir) / (Path(image_path).stem + "_parsed_items.txt")
                with open(items_debug_file, 'w', encoding='utf-8') as f:
                    f.write("PARSED ITEMS FROM DOCUMENT SCAN\n")
                    f.write("="*60 + "\n\n")
                    for i, item in enumerate(scanned_items, 1):
                        f.write(f"[{i:2d}] ITEM:\n")
                        f.write(f"     Quantity: '{item.quantity}'\n")
                        f.write(f"     Description: '{item.description}'\n")
                        f.write(f"     Original Text: '{item.original_text}'\n")
                        if hasattr(item, 'confidence') or 'confidence' in item_data:
                            confidence = item_data.get('confidence', 'unknown')
                            f.write(f"     Confidence: {confidence}\n")
                        f.write("\n")
                print(f"✓ Parsed items saved to: {items_debug_file}")

            print(f"✓ Scanned {len(scanned_items)} items from image")

            # Report confidence levels if available
            if items_data and 'confidence' in items_data[0]:
                high_conf = sum(1 for item in items_data if item.get('confidence') == 'high')
                medium_conf = sum(1 for item in items_data if item.get('confidence') == 'medium')
                low_conf = sum(1 for item in items_data if item.get('confidence') == 'low')
                print(f"  Confidence: {high_conf} high, {medium_conf} medium, {low_conf} low")

            # Print items to console only if verbose
            if verbose:
                print("\nSCANNED ITEMS:")
                print("-" * 50)
                for i, item in enumerate(scanned_items, 1):
                    print(f"[{i:2d}] {item.quantity:8s} | {item.description}")
                    if item.original_text != item.description:
                        print(f"     Original: {item.original_text}")
            else:
                # Save detailed output to file instead
                items_file = Path(output_dir) / "scanned_items.txt"
                with open(items_file, 'w', encoding='utf-8') as f:
                    f.write("SCANNED ITEMS:\n")
                    f.write("-" * 50 + "\n")
                    for i, item in enumerate(scanned_items, 1):
                        f.write(f"[{i:2d}] {item.quantity:8s} | {item.description}\n")
                        if item.original_text != item.description:
                            f.write(f"     Original: {item.original_text}\n")
                print(f"✓ Detailed item list saved to: {items_file}")

            return scanned_items

        except Exception as e:
            print(f"✗ Error scanning image: {e}")
            return []

    def _build_parts_context(self) -> str:
        """Build a context string from the parts databases to help with OCR"""
        if not self.databases:
            return ""

        context_parts = []
        context_parts.append("PARTS REFERENCE (to help with OCR and typo correction):")
        context_parts.append("=" * 60)

        # Collect sample items from each database
        total_samples = 0
        max_samples_per_db = 30  # Reduced to avoid overwhelming context
        max_total_samples = 100  # Reduced overall limit

        for db_name, database in self.databases.items():
            parts = database['parts']
            headers = database['headers']

            if not parts or not headers:
                continue

            context_parts.append(f"\n{db_name.upper()} DATABASE SAMPLE:")
            context_parts.append("-" * 40)

            # Get key columns
            item_col = headers[0] if headers else ''
            desc_col = None

            # Find description column
            for header in headers:
                if 'description' in header.lower() or 'desc' in header.lower():
                    desc_col = header
                    break

            if not desc_col and len(headers) > 1:
                desc_col = headers[1]

            # Add sample entries
            samples_from_this_db = 0
            for part in parts:
                if total_samples >= max_total_samples or samples_from_this_db >= max_samples_per_db:
                    break

                item_num = part.get(item_col, '').strip()
                item_desc = part.get(desc_col, '').strip() if desc_col else ''

                if item_num and item_desc:
                    context_parts.append(f"  {item_num}: {item_desc}")
                    samples_from_this_db += 1
                    total_samples += 1

            if samples_from_this_db > 0:
                context_parts.append(f"  ... ({len(parts)} total items in {db_name})")

        context_parts.append(f"\nNote: Use this reference to correct OCR errors and match unclear handwriting.")
        context_parts.append("=" * 60)

        return "\n".join(context_parts)

    # Also add this method to help with debugging matches
    def debug_matching_process(self, item: ScannedItem, database_name: str) -> None:
        """Debug helper to see what's happening during matching"""
        print(f"\nDEBUG: Matching item '{item.description}' against {database_name}")
        print(f"  Original text: '{item.original_text}'")
        print(f"  Quantity: '{item.quantity}'")

        database = self.databases[database_name]
        parts = database['parts']
        headers = database['headers']

        print(f"  Database has {len(parts)} parts with headers: {headers}")

        # Show first few parts for comparison
        if parts:
            print("  First 3 database entries:")
            for i, part in enumerate(parts[:3]):
                print(f"    [{i+1}] {part}")
        else:
            print("  Database is empty!")


    def _fallback_match(self, item: ScannedItem, database_name: str) -> List[PartMatch]:
        """Simple fallback matching when Claude fails"""
        database = self.databases[database_name]
        parts = database['parts']
        headers = database['headers']
        
        if not parts or not headers:
            return []
        
        item_num_col = headers[0]
        desc_col = headers[1] if len(headers) > 1 else headers[0]
        
        matches = []
        item_desc_lower = item.description.lower()
        
        for part in parts[:50]:  # Limit search for performance
            part_num = part.get(item_num_col, '').strip()
            part_desc = part.get(desc_col, '').strip()
            
            if not part_num or not part_desc:
                continue
            
            part_desc_lower = part_desc.lower()
            
            # Simple keyword matching
            if self._keyword_match(item_desc_lower, part_desc_lower):
                match = PartMatch(
                    description=item.description,
                    part_number=part_num,
                    database_name=database_name,
                    database_description=part_desc,
                    confidence="low"
                )
                matches.append(match)
                
                if len(matches) >= 3:
                    break
        
        return matches
    
    def _keyword_match(self, item_desc: str, part_desc: str) -> bool:
        """Check for keyword overlap"""
        # Split into words and remove common filler words
        stop_words = {'x', 'and', 'the', 'a', 'an', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        item_words = set(re.findall(r'\w+', item_desc.lower())) - stop_words
        part_words = set(re.findall(r'\w+', part_desc.lower())) - stop_words
        
        if len(item_words) == 0 or len(part_words) == 0:
            return False
        
        # Calculate overlap ratio
        overlap = len(item_words.intersection(part_words))
        total_unique = len(item_words.union(part_words))
        
        overlap_ratio = overlap / total_unique if total_unique > 0 else 0
        
        return overlap_ratio > 0.3  # At least 30% word overlap

    def match_item_in_database(self, item: ScannedItem, database_name: str) -> List[PartMatch]:
        """Use Claude to intelligently find matches for an item in a specific database"""
        database = self.databases[database_name]
        parts = database['parts']
        headers = database['headers']
        
        if not parts:
            return []
        
        # Get the primary columns for part numbers and descriptions
        item_num_col = headers[0] if headers else 'item_number'
        desc_col = None
        
        # Find the best description column
        for header in headers:
            if 'description' in header.lower() or 'desc' in header.lower():
                desc_col = header
                break
        
        if not desc_col:
            desc_col = headers[1] if len(headers) > 1 else headers[0]
        
        # Create a condensed database sample for Claude (to fit in context)
        max_parts_for_claude = 500  # Limit to avoid context overflow
        sample_parts = parts[:max_parts_for_claude] if len(parts) > max_parts_for_claude else parts
        
        # Format database entries for Claude
        db_entries = []
        for part in sample_parts:
            part_num = part.get(item_num_col, '').strip()
            part_desc = part.get(desc_col, '').strip()
            if part_num and part_desc:
                db_entries.append(f"{part_num}|{part_desc}")
        
        if not db_entries:
            return []
        
        # Prepare the matching request for Claude
        db_text = "\n".join(db_entries[:100])  # Further limit for the API call
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user", 
                        "content": f"""I need to match this lumber/construction item against a parts database.

ITEM TO MATCH:
Quantity: {item.quantity}
Description: {item.description}
Original text: {item.original_text}

PARTS DATABASE ({database_name}):
Format: PART_NUMBER|DESCRIPTION
{db_text}

Please find the best matches from the database for the item. Consider:
- Exact dimension matches (2x4, 2x6, etc.)
- Material equivalents (PT = Pressure Treated, DF = Douglas Fir, etc.)
- Length conversions and variations
- Common construction terminology
- Partial matches where core specs align

Return your response as JSON in this format:
[
  {{
    "part_number": "exact part number from database",
    "confidence": "exact|high|medium|low",
    "reason": "brief explanation of why this matches"
  }}
]

Return up to 3 best matches, ordered by confidence. If no reasonable matches, return empty array []."""
                    }
                ]
            )
            
            # Parse Claude's response
            response_text = response.content[0].text
            
            # Extract JSON from response
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                claude_matches = json.loads(json_match.group(0))
            else:
                return []
            
            # Convert Claude's matches to PartMatch objects
            matches = []
            for claude_match in claude_matches:
                part_num = claude_match.get('part_number', '')
                confidence = claude_match.get('confidence', 'low')
                reason = claude_match.get('reason', '')
                
                # Find the full part info from database
                matching_part = None
                for part in sample_parts:
                    if part.get(item_num_col, '').strip() == part_num:
                        matching_part = part
                        break
                
                if matching_part:
                    part_desc = matching_part.get(desc_col, '').strip()
                    match = PartMatch(
                        description=item.description,
                        part_number=part_num,
                        database_name=database_name,
                        database_description=f"{part_desc} (Claude: {reason})",
                        confidence=confidence
                    )
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            print(f"Warning: Claude matching failed for {database_name}: {e}")
            # Fallback to simple keyword matching
            return self._fallback_match(item, database_name)

    def match_item_with_full_database(self, item: ScannedItem, database_name: str) -> List[PartMatch]:
        """Use Claude to match against a large database by sending the full database context"""
        database = self.databases[database_name]
        parts = database['parts']
        headers = database['headers']
        
        if not parts:
            return []
        
        # Format the entire database with ALL columns for Claude
        db_entries = []
        header_line = "|".join(headers)
        db_entries.append(f"HEADERS: {header_line}")
        
        for part in parts:
            row_data = []
            for header in headers:
                value = part.get(header, '').strip()
                row_data.append(value)
            db_entries.append("|".join(row_data))
        
        if len(db_entries) <= 1:  # Only headers, no data
            return []
        
        # Calculate approximate token usage and split if necessary
        db_text = "\n".join(db_entries)
        estimated_tokens = len(db_text) // 4  # Rough estimate: 4 chars per token
        
        if estimated_tokens > 80000:  # Leave room for response and other content
            # Split database into chunks and process separately
            chunk_size = len(parts) // 3  # Split into 3 chunks
            
            all_matches = []
            for i in range(0, len(parts), chunk_size):
                chunk_parts = parts[i:i + chunk_size]
                chunk_entries = [f"HEADERS: {header_line}"]
                
                for part in chunk_parts:
                    row_data = []
                    for header in headers:
                        value = part.get(header, '').strip()
                        row_data.append(value)
                    chunk_entries.append("|".join(row_data))
                
                chunk_text = "\n".join(chunk_entries)
                chunk_matches = self._match_with_claude_chunk_enhanced(item, chunk_text, database_name, f"chunk_{i//chunk_size + 1}", headers)
                all_matches.extend(chunk_matches)
            
            # Deduplicate and sort
            unique_matches = []
            seen_parts = set()
            for match in all_matches:
                if match.part_number not in seen_parts:
                    unique_matches.append(match)
                    seen_parts.add(match.part_number)
            
            # Sort by confidence
            confidence_order = {"exact": 0, "high": 1, "medium": 2, "low": 3}
            unique_matches.sort(key=lambda x: confidence_order.get(x.confidence, 4))
            
            return unique_matches[:3]
        
        else:
            # Database fits in one request
            return self._match_with_claude_chunk_enhanced(item, db_text, database_name, "full_db", headers)

    def _match_with_claude_chunk_enhanced(self, item: ScannedItem, db_text: str, database_name: str, chunk_name: str, headers: List[str]) -> List[PartMatch]:
        """Enhanced matching using all available database columns"""
        try:
            # Create a description of what each column contains
            column_descriptions = []
            for header in headers:
                header_lower = header.lower()
                if 'customer' in header_lower or 'term' in header_lower:
                    column_descriptions.append(f"- {header}: Alternative names, categories, or search terms for the product")
                elif 'item' in header_lower and 'number' in header_lower:
                    column_descriptions.append(f"- {header}: Unique part/SKU number")
                elif 'description' in header_lower or 'desc' in header_lower:
                    column_descriptions.append(f"- {header}: Detailed product description with dimensions and specifications")
                elif 'stock' in header_lower or 'multiple' in header_lower or 'unit' in header_lower:
                    column_descriptions.append(f"- {header}: Unit of measure (LF=Linear Feet, EA=Each, etc.)")
                else:
                    column_descriptions.append(f"- {header}: Additional product information")
            
            column_info = "\n".join(column_descriptions)
            
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Match this construction/lumber item against the parts database using ALL available information.

ITEM TO MATCH:
Quantity: {item.quantity}
Description: {item.description}
Original handwritten text: {item.original_text}

PARTS DATABASE ({database_name} - {chunk_name}):
The database has the following columns:
{column_info}

Database format (pipe-separated values):
{db_text}

MATCHING INSTRUCTIONS:
1. Use ALL columns for matching, not just the description:
   - Customer Terms can provide alternative names or categories
   - Item Description has the technical specs
   - Stocking Multiple shows the unit type (LF, EA, etc.)

2. Look for these construction material matches:
   - Exact dimension matches (2x4, 2x6, 3 1/2 x 3 1/2, etc.)
   - Material types: PT/Pressure Treated, DF/Douglas Fir, Cedar, Pine, POC (Port Orford Cedar), GLU LAM/Glulam beams
   - Length specifications: 8', 10', 12', 16', 20' etc.
   - Lumber grades: S1S2E (Surfaced 1 Side, 2 Edges), S4S, Construction Grade
   - Product categories: Posts, Beams, Fascia, Advantage lumber, etc.

3. Consider unit compatibility:
   - LF (Linear Feet) for dimensional lumber
   - EA (Each) for individual items
   - Board feet for lumber volume

4. Match handwritten abbreviations and variations:
   - "PT" = "Pressure Treated"
   - "DF" = "Douglas Fir" 
   - "GLU LAM" = "Glulam"
   - Dimension variations: "2x4" = "2 X 4" = "2 by 4"

Respond with JSON array of up to 5 best matches:
[
  {{
    "part_number": "exact item number from database",
    "confidence": "exact|high|medium|low",
    "reason": "specific explanation using ALL relevant columns - mention which columns helped with the match"
  }}
]

If no reasonable matches found, return []"""
                    }
                ]
            )
            
            response_text = response.content[0].text
            
            # Extract JSON
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if not json_match:
                return []
            
            claude_matches = json.loads(json_match.group(0))
            
            # Convert to PartMatch objects with full row information
            matches = []
            for claude_match in claude_matches:
                part_num = claude_match.get('part_number', '')
                confidence = claude_match.get('confidence', 'low')
                reason = claude_match.get('reason', '')
                
                # Find the full row data for this part number
                full_row_info = ""
                for line in db_text.split('\n')[1:]:  # Skip header line
                    if line and '|' in line:
                        row_parts = line.split('|')
                        if len(row_parts) > 0 and row_parts[0].strip() == part_num:
                            # Create a readable description using all columns
                            row_info = []
                            for i, header in enumerate(headers):
                                if i < len(row_parts) and row_parts[i].strip():
                                    row_info.append(f"{header}: {row_parts[i].strip()}")
                            full_row_info = " | ".join(row_info)
                            break
                
                if part_num:
                    match = PartMatch(
                        description=item.description,
                        part_number=part_num,
                        database_name=database_name,
                        database_description=f"{full_row_info} || Claude reasoning: {reason}",
                        confidence=confidence
                    )
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            print(f"Warning: Enhanced Claude matching failed for {database_name}: {e}")
            return []
    
    def find_all_matches(self, use_claude: bool = False, full_database: bool = False) -> None:
        """Find matches for all scanned items across all databases"""
        print("\n" + "="*60)
        if use_claude:
            print("SEARCHING FOR MATCHES USING CLAUDE AI MATCHING")
        else:
            print("SEARCHING FOR MATCHES USING BASIC KEYWORD MATCHING")
        print("="*60)
        
        for i, item in enumerate(self.scanned_items, 1):
            print(f"\n[{i:2d}] Searching for: {item.description}")
            print("-" * 50)
            
            all_matches = []
            for db_name in self.databases:
                if use_claude:
                    if full_database:
                        matches = self.match_item_with_full_database(item, db_name)
                    else:
                        matches = self.match_item_in_database(item, db_name)
                else:
                    matches = self._fallback_match(item, db_name)
                all_matches.extend(matches)
            
            # Remove duplicates and sort by confidence
            unique_matches = []
            seen_parts = set()
            for match in all_matches:
                key = (match.part_number, match.database_name)
                if key not in seen_parts:
                    unique_matches.append(match)
                    seen_parts.add(key)
            
            confidence_order = {"exact": 0, "high": 1, "medium": 2, "low": 3}
            unique_matches.sort(key=lambda x: confidence_order.get(x.confidence, 4))
            
            item.matches = unique_matches[:3]  # Keep top 3 matches
            
            if item.matches:
                for match in item.matches:
                    print(f"  ✓ {match.confidence.upper():8s} | {match.part_number:15s} | {match.database_name:15s} | {match.database_description}")
            else:
                print("  ✗ No matches found")
    
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

    def scan_document_with_database_context(self, document_path: str, debug_output: bool = True, output_dir: str = ".", verbose: bool = False) -> List[ScannedItem]:
        """Use Claude API to scan the lumber list with database context for better matching"""
        try:
            file_ext = Path(document_path).suffix.lower()
            
            # Handle PDF files
            if file_ext == '.pdf':
                if not PDF_SUPPORT:
                    raise ImportError("PDF support requires pdf2image. Install with: pip install pdf2image")
                
                if debug_output:
                    print(f"Converting PDF to images: {document_path}")
                
                image_paths, temp_dir = self.convert_pdf_to_images(document_path)
                
                if debug_output:
                    print(f"PDF converted to {len(image_paths)} page(s)")
                
                all_items = []
                
                try:
                    # Process each page
                    for i, image_path in enumerate(image_paths):
                        if debug_output:
                            print(f"Processing page {i+1}/{len(image_paths)}")
                        
                        page_items = self._scan_single_image_with_database_context(image_path, debug_output, output_dir, verbose)
                        all_items.extend(page_items)
                    
                    self.scanned_items = all_items
                    return all_items
                    
                finally:
                    # Clean up temporary files
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Handle image files
            else:
                items = self._scan_single_image_with_database_context(document_path, debug_output, output_dir, verbose)
                self.scanned_items = items
                return items
                
        except Exception as e:
            print(f"Error scanning document: {e}")
            return []
    
    def _scan_single_image_with_database_context(self, image_path: str, debug_output: bool = True, output_dir: str = ".", verbose: bool = False) -> List[ScannedItem]:
        """Scan a single image file with database context and return ScannedItems"""
        try:
            # Encode the image
            image_data = self.encode_image(image_path)

            # Determine image format from file extension
            file_ext = Path(image_path).suffix.lower()
            media_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = media_type_map.get(file_ext, 'image/jpeg')

            # Build comprehensive database context
            database_context = self._build_comprehensive_database_context()

            # Create enhanced prompt with database examples
            prompt_text = f"""Please scan this handwritten lumber/materials list and extract each item with its quantity and description.

{database_context}

CRITICAL INSTRUCTIONS:
1. Use the database examples above to format your descriptions to MATCH the database style
2. For lumber: Use format like "2X4  16" not "2x4 x 16' lumber"
3. For hardware: Use format like "#8 SCREWS" not "2x4 construction screws #8"
4. Keep descriptions SHORT and match the database terminology
5. When unsure, pick the closest database format

Examples of good matches:
- Handwritten "2x12 20'" → Description: "2X12  20" (matches database format)
- Handwritten "4x4 12 PT" → Description: "4X4  12  PT" (matches database style)
- Handwritten "#8 screws" → Description: "#8 SCREWS" (matches database style)

Format your response as a JSON array where each item has:
{{
    "quantity": "the number/amount",
    "description": "item description formatted to match database style",
    "original_text": "the original handwritten text as you read it",
    "confidence": "high|medium|low - how confident you are in the match"
}}

Focus on making descriptions that will match items in the provided database."""

            # Create the message with image using the current model
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt_text
                            }
                        ]
                    }
                ],
            )

            # Parse the response
            response_text = message.content[0].text

            # Debug output
            if debug_output:
                debug_file = Path(output_dir) / (Path(image_path).stem + "_scan_with_context_debug.txt")
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("CLAUDE RESPONSE WITH DATABASE CONTEXT\n")
                    f.write("="*60 + "\n\n")
                    f.write(response_text)
                    f.write("\n\n" + "="*60 + "\n")
                    f.write("DATABASE CONTEXT PROVIDED:\n")
                    f.write("="*60 + "\n")
                    f.write(database_context)
                print(f"✓ Debug info saved to: {debug_file}")

            # Extract and parse JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find JSON array - look for opening bracket and find matching closing bracket
                start_pos = response_text.find('[')
                if start_pos != -1:
                    # Count brackets to find the matching closing bracket
                    bracket_count = 0
                    end_pos = start_pos
                    for i, char in enumerate(response_text[start_pos:], start_pos):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_pos = i + 1
                                break
                    json_text = response_text[start_pos:end_pos]
                else:
                    json_text = response_text

            if debug_output:
                json_debug_file = Path(output_dir) / (Path(image_path).stem + "_context_extracted_json.json")
                with open(json_debug_file, 'w', encoding='utf-8') as f:
                    f.write(json_text)
                print(f"✓ Extracted JSON saved to: {json_debug_file}")

            items_data = json.loads(json_text)

            # Convert to ScannedItem objects
            scanned_items = []
            for item_data in items_data:
                item = ScannedItem(
                    quantity=item_data.get('quantity', ''),
                    description=item_data.get('description', ''),
                    original_text=item_data.get('original_text', ''),
                    matches=[]
                )
                scanned_items.append(item)

            print(f"✓ Scanned {len(scanned_items)} items with database context")

            # Print items to console only if verbose
            if verbose:
                print("\nSCANNED ITEMS (WITH DATABASE CONTEXT):")
                print("-" * 50)
                for i, item in enumerate(scanned_items, 1):
                    print(f"[{i:2d}] {item.quantity:8s} | {item.description}")
                    if item.original_text != item.description:
                        print(f"     Original: {item.original_text}")
            else:
                # Save detailed output to file instead
                items_file = Path(output_dir) / "scanned_items.txt"
                with open(items_file, 'w', encoding='utf-8') as f:
                    f.write("SCANNED ITEMS (WITH DATABASE CONTEXT):\n")
                    f.write("-" * 50 + "\n")
                    for i, item in enumerate(scanned_items, 1):
                        f.write(f"[{i:2d}] {item.quantity:8s} | {item.description}\n")
                        if item.original_text != item.description:
                            f.write(f"     Original: {item.original_text}\n")
                print(f"✓ Detailed item list saved to: {items_file}")

            return scanned_items

        except Exception as e:
            print(f"✗ Error scanning document: {e}")
            return []

    def _build_comprehensive_database_context(self) -> str:
        """Build database context showing variety of items and formats"""
        if not self.databases:
            return ""

        context_parts = []
        context_parts.append("DATABASE EXAMPLES (match this format in your descriptions):")
        context_parts.append("=" * 70)

        # Collect diverse samples from each database
        total_samples = 0
        max_total_samples = 80  # Limit for context size

        categories = {
            'lumber_dimensional': [],
            'lumber_specialty': [],
            'hardware': [],
            'other': []
        }

        for db_name, database in self.databases.items():
            parts = database['parts']
            headers = database['headers']

            if not parts or not headers:
                continue

            for part in parts:
                if total_samples >= max_total_samples:
                    break

                item_num = part.get('Item Number', '').strip()
                item_desc = part.get('Item Description', '').strip()
                customer_terms = part.get('Customer Terms', '').strip()

                if not item_num or not item_desc:
                    continue

                # Categorize the item
                desc_lower = item_desc.lower()
                terms_lower = customer_terms.lower()

                if any(term in desc_lower for term in ['x', 'post', 'beam']):
                    if any(term in terms_lower for term in ['glu lam', 'advantage', 'boral']):
                        categories['lumber_specialty'].append(f"  {item_num}: {item_desc}")
                    else:
                        categories['lumber_dimensional'].append(f"  {item_num}: {item_desc}")
                elif any(term in desc_lower for term in ['screw', 'nail', 'bolt', 'hardware']):
                    categories['hardware'].append(f"  {item_num}: {item_desc}")
                else:
                    categories['other'].append(f"  {item_num}: {item_desc}")

                total_samples += 1

        # Add samples from each category
        if categories['lumber_dimensional']:
            context_parts.append("\nDIMENSIONAL LUMBER FORMAT:")
            for item in categories['lumber_dimensional'][:15]:
                context_parts.append(item)

        if categories['lumber_specialty']:
            context_parts.append("\nSPECIALTY LUMBER FORMAT:")
            for item in categories['lumber_specialty'][:15]:
                context_parts.append(item)

        if categories['hardware']:
            context_parts.append("\nHARDWARE FORMAT:")
            for item in categories['hardware'][:10]:
                context_parts.append(item)

        if categories['other']:
            context_parts.append("\nOTHER ITEMS FORMAT:")
            for item in categories['other'][:10]:
                context_parts.append(item)

        context_parts.append("\nKEY PATTERNS TO MATCH:")
        context_parts.append("- Lumber dimensions: '2X4', '2X6', '4X4' (not '2x4', '2x6')")
        context_parts.append("- Spacing: '2X4  16' (spaces around dimensions)")
        context_parts.append("- Materials: 'POC', 'CEDAR', 'PINE' (all caps)")
        context_parts.append("- Treatments: 'PT', 'KD', 'S4S' (abbreviated)")
        context_parts.append("- Product types: 'ADVANTAGE', 'GLU LAM', 'POST'")
        context_parts.append("=" * 70)

        return "\n".join(context_parts)

    def _parse_lumber_item(self, description: str, debug: bool = False) -> dict:
        """Parse a scanned lumber item description into components"""
        desc = description.lower().strip()
        components = {}

        # Remove common words that don't help with matching
        desc = re.sub(r'\b(lumber|wood|board|construction|grade|treated|pressure)\b', '', desc)
        desc = ' '.join(desc.split())  # Clean up whitespace

        if debug:
            print(f"    Cleaned description: '{desc}'")

        # Extract dimensions - handle various formats
        # 2x4, 2x6, 4x4, etc.
        dim_match = re.search(r'(\d+(?:\s*1/2)?)\s*x\s*(\d+(?:\s*1/2)?)', desc)
        if dim_match:
            width = dim_match.group(1).replace(' ', '')
            height = dim_match.group(2).replace(' ', '')
            components['dimensions'] = f"{width}x{height}"
            if debug:
                print(f"    Found dimensions: {components['dimensions']}")

        # Extract length - look for numbers followed by ' or ft
        length_match = re.search(r'(\d+)(?:\s*(?:\'|ft|feet))?(?:\s|$)', desc)
        if length_match:
            components['length'] = length_match.group(1)
            if debug:
                print(f"    Found length: {components['length']}")

        # Extract material/treatment indicators
        if 'pt' in desc or 'pressure' in desc:
            components['treatment'] = 'pt'

        # Look for specific material types
        for material in ['cedar', 'pine', 'fir', 'poc']:
            if material in desc:
                components['material'] = material
                break

        # Look for product type indicators
        if 'glu' in desc or 'glulam' in desc or 'glue' in desc:
            components['type'] = 'glulam'
        elif 'post' in desc:
            components['type'] = 'post'
        elif 'advantage' in desc or 'adv' in desc:
            components['type'] = 'advantage'

        return components

    def match_lumber_item(self, item: ScannedItem, database_name: str, debug: bool = False) -> List[PartMatch]:
        """Specialized matching for lumber database format"""
        if debug:
            print(f"\nDEBUG: Lumber-specific matching for '{item.description}' in {database_name}")

        database = self.databases[database_name]
        parts = database['parts']
        headers = database['headers']

        if not parts:
            return []

        # Parse the scanned item to extract lumber components
        item_components = self._parse_lumber_item(item.description, debug)
        if debug:
            print(f"  Parsed item components: {item_components}")

        matches = []
        confidence_scores = []

        for part in parts:
            item_number = part.get('Item Number', '').strip()
            item_desc = part.get('Item Description', '').strip()
            customer_terms = part.get('Customer Terms', '').strip()
            stocking_multiple = part.get('Stocking Multiple', '').strip()

            if not item_number or not item_desc:
                continue

            # Parse the database entry
            db_components = self._parse_database_entry(item_desc, customer_terms, debug)

            # Calculate match score
            match_score = self._calculate_lumber_match_score(item_components, db_components, debug)

            if match_score > 0.5:  # Threshold for considering it a match
                confidence = "high" if match_score > 0.8 else "medium" if match_score > 0.65 else "low"

                match = PartMatch(
                    description=item.description,
                    part_number=item_number,
                    database_name=database_name,
                    database_description=f"{item_desc} | Terms: {customer_terms} | Unit: {stocking_multiple} | Score: {match_score:.2f}",
                    confidence=confidence
                )
                matches.append(match)
                confidence_scores.append(match_score)

                if debug:
                    print(f"    MATCH: {item_number} -> {item_desc} (score: {match_score:.2f})")

        # Sort by match score (highest first)
        # Use a stable sort that handles ties by using the index as a secondary key
        sorted_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i], reverse=True)
        sorted_matches = [matches[i] for i in sorted_indices]

        if debug:
            print(f"  Found {len(sorted_matches)} matches")

        return sorted_matches[:3]  # Return top 3 matches

    def _calculate_lumber_match_score(self, item_components: dict, db_components: dict, debug: bool = False) -> float:
        """Calculate how well an item matches a database entry"""
        score = 0.0
        max_score = 0.0

        # Dimensions are most important (40% weight)
        if 'dimensions' in item_components or 'dimensions' in db_components:
            max_score += 0.4
            if ('dimensions' in item_components and 'dimensions' in db_components and
                item_components['dimensions'] == db_components['dimensions']):
                score += 0.4
                if debug:
                    print(f"      Dimension match: {item_components['dimensions']} == {db_components['dimensions']}")
            elif debug and 'dimensions' in item_components and 'dimensions' in db_components:
                print(f"      Dimension mismatch: {item_components['dimensions']} != {db_components['dimensions']}")

        # Length is important (25% weight)
        if 'length' in item_components or 'length' in db_components:
            max_score += 0.25
            if ('length' in item_components and 'length' in db_components and
                item_components['length'] == db_components['length']):
                score += 0.25
                if debug:
                    print(f"      Length match: {item_components['length']} == {db_components['length']}")
            elif debug and 'length' in item_components and 'length' in db_components:
                print(f"      Length mismatch: {item_components['length']} != {db_components['length']}")

        # Material type (20% weight)
        if 'material' in item_components or 'material' in db_components:
            max_score += 0.2
            if ('material' in item_components and 'material' in db_components and
                item_components['material'] == db_components['material']):
                score += 0.2
                if debug:
                    print(f"      Material match: {item_components['material']} == {db_components['material']}")

        # Product type (10% weight)
        if 'type' in item_components or 'type' in db_components:
            max_score += 0.1
            if ('type' in item_components and 'type' in db_components and
                item_components['type'] == db_components['type']):
                score += 0.1
                if debug:
                    print(f"      Type match: {item_components['type']} == {db_components['type']}")

        # Treatment (5% weight)
        if 'treatment' in item_components or 'treatment' in db_components:
            max_score += 0.05
            if ('treatment' in item_components and 'treatment' in db_components and
                item_components['treatment'] == db_components['treatment']):
                score += 0.05
                if debug:
                    print(f"      Treatment match: {item_components['treatment']} == {db_components['treatment']}")

        # Normalize score
        final_score = score / max_score if max_score > 0 else 0

        if debug:
            print(f"      Final score: {score:.2f}/{max_score:.2f} = {final_score:.2f}")

        return final_score

    def _parse_database_entry(self, item_desc: str, customer_terms: str, debug: bool = False) -> dict:
        """Parse a database entry into components"""
        desc = item_desc.lower().strip()
        terms = customer_terms.lower().strip()
        components = {}
        
        if debug:
            print(f"    Parsing DB entry: '{item_desc}' | Terms: '{customer_terms}'")
        
        # Extract dimensions from database format: "2X4", "3 1/2  X  3 1/2", etc.
        dim_match = re.search(r'(\d+(?:\s*1/2)?)\s*x\s*(\d+(?:\s*1/2)?)', desc)
        if dim_match:
            width = dim_match.group(1).replace(' ', '')
            height = dim_match.group(2).replace(' ', '')
            components['dimensions'] = f"{width}x{height}"
        
        # Extract length
        length_match = re.search(r'\b(\d+)\s+(?!x|1/2)', desc)  # Number not followed by x or 1/2
        if length_match:
            components['length'] = length_match.group(1)
        
        # Material identification
        if 'poc' in desc:
            components['material'] = 'poc'
        elif 'cedar' in desc or 'wrc' in desc or 'cdr' in desc:
            components['material'] = 'cedar'
        elif 'spf' in desc or 'pine' in desc:
            components['material'] = 'pine'
        elif 'df' in desc:
            components['material'] = 'fir'
        
        # Product type from terms and description
        if 'glu lam' in terms or 'glulam' in desc or 'glu lam' in desc:
            components['type'] = 'glulam'
        elif 'post' in desc:
            components['type'] = 'post'
        elif 'advantage' in terms or 'adv' in terms or 'advantage' in desc:
            components['type'] = 'advantage'
        elif 'fascia' in terms:
            components['type'] = 'fascia'
        elif 'boral' in desc or 'bor' in terms:
            components['type'] = 'boral'
        
        # Treatment indicators
        if 'trtd' in desc or 'treated' in desc:
            components['treatment'] = 'pt'
        
        return components

    def hybrid_match_item(self, item: ScannedItem, database_name: str, debug: bool = False) -> List[PartMatch]:
        """Hybrid matching: lumber-specific for lumber items, general for others"""
        if debug:
            print(f"\nDEBUG: Hybrid matching for '{item.description}' in {database_name}")

        # Determine if this looks like a lumber item
        desc_lower = item.description.lower()
        is_lumber = bool(re.search(r'\d+\s*x\s*\d+', desc_lower)) or any(term in desc_lower for term in
                         ['post', 'beam', 'lumber', 'cedar', 'pine', 'pt', 'advantage', 'glu lam', 'glulam'])

        if debug:
            print(f"  Classified as lumber: {is_lumber}")

        if is_lumber:
            # Use lumber-specific matching
            return self.match_lumber_item(item, database_name, debug)
        else:
            # Use enhanced general matching for hardware, etc.
            return self.match_general_item(item, database_name, debug)

    def match_general_item(self, item: ScannedItem, database_name: str, debug: bool = False) -> List[PartMatch]:
        """General matching for non-lumber items (hardware, screws, etc.)"""
        if debug:
            print(f"  Using general matching for non-lumber item")

        database = self.databases[database_name]
        parts = database['parts']
        headers = database['headers']

        if not parts:
            return []

        matches = []
        item_desc_lower = item.description.lower()

        # Extract key terms from the scanned item
        item_terms = set(re.findall(r'\w+', item_desc_lower))

        for part in parts:
            item_number = part.get('Item Number', '').strip()
            item_desc = part.get('Item Description', '').strip()
            customer_terms = part.get('Customer Terms', '').strip()
            stocking_multiple = part.get('Stocking Multiple', '').strip()

            if not item_number or not item_desc:
                continue

            # Combine description and customer terms for matching
            full_desc = f"{item_desc} {customer_terms}".lower()
            db_terms = set(re.findall(r'\w+', full_desc))

            # Calculate overlap
            overlap = len(item_terms.intersection(db_terms))
            total_unique = len(item_terms.union(db_terms))

            if total_unique > 0:
                overlap_ratio = overlap / total_unique

                # More lenient threshold for general items
                if overlap_ratio > 0.2:  # 20% overlap
                    confidence = "high" if overlap_ratio > 0.5 else "medium" if overlap_ratio > 0.35 else "low"

                    match = PartMatch(
                        description=item.description,
                        part_number=item_number,
                        database_name=database_name,
                        database_description=f"{item_desc} | Terms: {customer_terms} | Overlap: {overlap_ratio:.2f}",
                        confidence=confidence
                    )
                    matches.append(match)

                    if debug:
                        print(f"    General match: {item_number} -> {item_desc} (overlap: {overlap_ratio:.2f})")

        # Sort by overlap ratio (stored in description for now)
        matches.sort(key=lambda x: float(x.database_description.split('Overlap: ')[1].split()[0]), reverse=True)

        return matches[:3]

    def find_all_matches_hybrid(self, debug: bool = False, output_dir: str = ".") -> None:
        """Find matches using hybrid approach: lumber-specific + general matching"""
        if debug:
            print("\n" + "="*60)
            print("SEARCHING FOR MATCHES USING HYBRID MATCHING")
            print("(Lumber-specific for lumber items, general for hardware/other)")
            print("="*60)

        # Create debug log file for detailed output
        debug_log_file = Path(output_dir) / "matching_debug.log"
        with open(debug_log_file, 'w', encoding='utf-8') as log_file:
            log_file.write("DETAILED MATCHING DEBUG LOG\n")
            log_file.write("=" * 60 + "\n\n")

        for i, item in enumerate(self.scanned_items, 1):
            if debug:
                print(f"\n[{i:2d}] Searching for: {item.description}")
                print(f"     Original: {item.original_text}")
                print(f"     Quantity: {item.quantity}")
                print("-" * 50)
            else:
                print(f"  [{i:2d}] {item.description}")

            all_matches = []
            for db_name in self.databases:
                if debug:
                    print(f"\n  Checking database: {db_name}")

                # Run matching with appropriate debug level
                matches = self.hybrid_match_item(item, db_name, debug=debug)
                all_matches.extend(matches)

                if debug:
                    print(f"    Found {len(matches)} matches in {db_name}")
                
                # Always write to debug log file for detailed analysis
                with open(debug_log_file, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"\n=== ITEM {i}: {item.description} ===\n")
                    log_file.write(f"Database: {db_name}\n")
                    log_file.write(f"Original: {item.original_text}\n")
                    log_file.write(f"Quantity: {item.quantity}\n")
                    log_file.write(f"Matches found: {len(matches)}\n")
                    for match in matches:
                        log_file.write(f"  - {match.confidence}: {match.part_number} | {match.database_description}\n")
                    log_file.write("-" * 50 + "\n")

            # Remove duplicates and sort by confidence
            unique_matches = []
            seen_parts = set()
            for match in all_matches:
                key = (match.part_number, match.database_name)
                if key not in seen_parts:
                    unique_matches.append(match)
                    seen_parts.add(key)

            confidence_order = {"high": 0, "medium": 1, "low": 2}
            unique_matches.sort(key=lambda x: confidence_order.get(x.confidence, 3))

            item.matches = unique_matches[:3]  # Keep top 3 matches

            if debug:
                if item.matches:
                    for match in item.matches:
                        print(f"  ✓ {match.confidence.upper():8s} | {match.part_number:15s} | {match.database_description[:80]}...")
                else:
                    print("  ✗ No matches found")
        
        print(f"\n✓ Detailed matching log saved to: {debug_log_file}")

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
    
    parser.add_argument('-o', '--output-dir',
                       default='.',
                       help='Base directory to save output files. A subdirectory named after the input file will be created (default: current directory)')
    
    parser.add_argument('--report-name',
                       default='lumber_match_report.txt',
                       help='Name for the text report file (default: lumber_match_report.txt)')
    
    parser.add_argument('--csv-name',
                       default='lumber_matches.csv',
                       help='Name for the CSV export file (default: lumber_matches.csv)')
    
    parser.add_argument('-q', '--quiet',
                       action='store_true',
                       help='Reduce output verbosity')
    
    parser.add_argument('--use-claude-matching',
                       action='store_true',
                       help='Use Claude AI for intelligent parts matching (slower but more accurate)')
    
    parser.add_argument('--full-database',
                       action='store_true',
                       help='Send full database to Claude for matching (use with large SKU lists)')
    
    parser.add_argument('--verbose-matching',
                       action='store_true',
                       help='Show detailed matching debug output on console (default: save to files)')
    
    return parser.parse_args()

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
    
    # Scan document
    if not args.quiet:
        print(f"\nScanning document: {args.document}")
    items = matcher.scan_document_with_database_context(args.document, output_dir=str(output_dir), verbose=args.verbose_matching)
    
    if not items:
        print("Error: No items found in document")
        sys.exit(1)
    
    # Find matches
    if not args.quiet:
        matcher.find_all_matches_hybrid(debug=args.verbose_matching, output_dir=str(output_dir))
    elif not args.quiet:
        matcher.find_all_matches(
            use_claude=args.use_claude_matching,
            full_database=args.full_database
        )
    else:
        # Silent matching for quiet mode
        for item in matcher.scanned_items:
            all_matches = []
            for db_name in matcher.databases:
                if args.use_claude_matching:
                    if args.full_database:
                        matches = matcher.match_item_with_full_database(item, db_name)
                    else:
                        matches = matcher.match_item_in_database(item, db_name)
                else:
                    matches = matcher._fallback_match(item, db_name)
                all_matches.extend(matches)
            
            # Remove duplicates and sort by confidence
            unique_matches = []
            seen_parts = set()
            for match in all_matches:
                key = (match.part_number, match.database_name)
                if key not in seen_parts:
                    unique_matches.append(match)
                    seen_parts.add(key)
            
            confidence_order = {"exact": 0, "high": 1, "medium": 2, "low": 3}
            unique_matches.sort(key=lambda x: confidence_order.get(x.confidence, 4))
            item.matches = unique_matches[:3]
    
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

if __name__ == "__main__":
    main()
