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
@dataclass
class PartMatch:
    """Represents a matched part"""
    description: str
    part_number: str
    database_name: str
    database_description: str
    confidence: str  # "exact", "partial", "similar"
    reason: str = ""  # Explanation for the match

###############################################################################
@dataclass
class ScannedItem:
    """Represents an item from the scanned list"""
    quantity: str
    description: str
    original_text: str
    matches: List[PartMatch]

###############################################################################
class LumberListMatcher:
    def __init__(self, api_key: str):
        """Initialize with Claude API key"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.databases = {}
        self.scanned_items = []
        self.training_data = []
        
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
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using Anthropic's tokenizer"""
        try:
            return self.client.count_tokens(text)
        except Exception as e:
            # Fallback to rough estimate if counting fails
            return len(text) // 4

    def _call_ai_with_retry(self, messages, max_tokens=4000, max_retries=3, model=None):
        """Call AI API with retry logic (works with both Anthropic and OpenAI)"""
        import time
        
        # Set default model
        if model is None:
            model = "claude-sonnet-4-20250514"
        
        for attempt in range(max_retries):
            try:
                if True:
                    response = self.client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=messages
                    )
                    return response
                elif self.ai_provider == 'openai':
                    response = self.client.chat.completions.create(
                        model=model,
                        max_completion_tokens=max_tokens,
                        messages=messages
                    )
                    return response
            except Exception as e:
                error_str = str(e)
                if "rate_limit_error" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 60, 120, 240 seconds
                        wait_time = 60 * (2 ** attempt)
                        print(f"  Rate limited, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        print(f"  ✗ Max retries ({max_retries}) exceeded for rate limit")
                        raise e
                else:
                    # Non-rate-limit error, don't retry
                    raise e
        
        raise Exception("Max retries exceeded")
    
    def _fix_missing_specifications(self, scanned_items: List[ScannedItem]) -> List[ScannedItem]:
        """Post-process scanned items to carry forward missing material specifications"""
        if not scanned_items:
            # Post-process to fix missing specifications
            scanned_items = self._fix_missing_specifications(scanned_items)
            return scanned_items
        
        fixed_items = []
        last_material = None
        
        for item in scanned_items:
            # Check if this looks like a lumber item missing material specs
            desc = item.description.upper()
            original = item.original_text.upper()
            
            # Pattern: dimension + length but no material (e.g., "4X8 16'" instead of "4X8 PT 16'")
            lumber_pattern = r'^(\d+X\d+)\s+(\d+[\'"]?)$'
            match = re.match(lumber_pattern, desc)
            
            if match and last_material:
                # This looks like a lumber item missing material specs
                dimension = match.group(1)
                length = match.group(2)
                
                # Reconstruct with carried-forward material
                fixed_desc = f"{dimension} {last_material} {length}"
                fixed_original = f"{dimension} {last_material} {length}"
                
                # Create new item with fixed description
                fixed_item = ScannedItem(
                    quantity=item.quantity,
                    description=fixed_desc,
                    original_text=fixed_original,
                    matches=item.matches
                )
                fixed_items.append(fixed_item)
            else:
                # Check if this item has material specs to carry forward
                # Look for common lumber materials in the description
                material_match = re.search(r'\b(PT|DF|CEDAR|PINE|POC|DFH2|DFHC|ADVANTAGE|GLU\s*LAM)\b', desc)
                if material_match:
                    last_material = material_match.group(1)
                
                fixed_items.append(item)
        
        return fixed_items
    
    def _should_skip_scanning(self, document_path: str, output_dir: str) -> bool:
        """Check if we can skip scanning based on file modification times"""
        try:
            # Check if input file exists
            if not os.path.exists(document_path):
                return False
            
            # Get input file modification time
            input_mtime = os.path.getmtime(document_path)
            
            # Check if scan_prompt file exists and get its modification time
            script_dir = Path(__file__).parent
            scan_prompt_path = script_dir / "scan_prompt"
            if not os.path.exists(scan_prompt_path):
                return False
            prompt_mtime = os.path.getmtime(scan_prompt_path)
            
            # Check if we have cached results
            output_path = Path(output_dir)
            cached_files = list(output_path.glob("page_*_context_extracted_json.json"))
            if not cached_files:
                return False
            
            # Check if all cached files are newer than both input and prompt
            for cached_file in cached_files:
                if os.path.getmtime(cached_file) < input_mtime or os.path.getmtime(cached_file) < prompt_mtime:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _load_cached_scan_results(self, output_dir: str) -> List[ScannedItem]:
        """Load previously scanned results from cache files"""
        try:
            all_items = []
            output_path = Path(output_dir)
            
            # Find all cached JSON files
            cached_files = sorted(output_path.glob("page_*_context_extracted_json.json"))
            
            for cached_file in cached_files:
                with open(cached_file, 'r', encoding='utf-8') as f:
                    json_text = f.read()
                
                if json_text.strip():
                    items_data = json.loads(json_text)
                    
                    for item_data in items_data:
                        item = ScannedItem(
                            quantity=item_data.get('quantity', ''),
                            description=item_data.get('description', ''),
                            original_text=item_data.get('original_text', ''),
                            matches=[]
                        )
                        all_items.append(item)
            
            self.scanned_items = all_items
            print(f"✓ Loaded {len(all_items)} cached items from {len(cached_files)} pages")
            return all_items
            
        except Exception as e:
            print(f"Error loading cached results: {e}")
            return []
    
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
            # Check if we can skip scanning
            if self._should_skip_scanning(document_path, output_dir):
                if debug_output:
                    print("✓ Using cached scan results (input file and scan_prompt unchanged)")
                return self._load_cached_scan_results(output_dir)
            
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
            prompt_text = load_prompt("scan_prompt", database_context=database_context)

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

            # Post-process to fix missing specifications
            scanned_items = self._fix_missing_specifications(scanned_items)
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

    def find_all_matches_keyword(self, debug: bool = False, output_dir: str = ".") -> None:
        """Find matches using original keyword-based matching (for comparison)"""
        if debug:
            print("\n" + "="*60)
            print("SEARCHING FOR MATCHES USING KEYWORD MATCHING")
            print("(Original keyword-based approach for comparison)")
            print("="*60)

        # Create debug log file for detailed output
        debug_log_file = Path(output_dir) / "matching_debug.log"
        with open(debug_log_file, 'w', encoding='utf-8') as log_file:
            log_file.write("DETAILED MATCHING DEBUG LOG (KEYWORD MATCHING)\n")
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

                # Use original keyword matching
                matches = self.keyword_match_item(item, db_name, debug=debug)
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

    def keyword_match_item(self, item: ScannedItem, database_name: str, debug: bool = False) -> List[PartMatch]:
        if debug:
            print(f"\nDEBUG: Keyword matching for '{item.description}' in {database_name}")

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
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8000,
                    messages=[{"role": "user", "content": batch_prompt}]
                )
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
                    matches = self.keyword_match_item(item, database_name, debug=False)

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
            matches = self.keyword_match_item(item, database_name, debug=False)
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
    
    parser.add_argument('--training-data',
                       nargs='*',
                       default=[],
                       help='One or more CSV files containing training data (original_text, correct_sku columns)')
    
    parser.add_argument('--use-keyword-matching',
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
            db_tokens = matcher._count_tokens(db_text)
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
            training_tokens = matcher._count_tokens(training_text)
            print(f"✓ Loaded {training_loaded} training data file(s) with {len(matcher.training_data)} total examples")
            print(f"  Training data tokens: {training_tokens:,}")
    
    # Scan document
    if not args.quiet:
        print(f"\nScanning document: {args.document}")
    items = matcher.scan_document_with_database_context(args.document, output_dir=str(output_dir), 
                                                        verbose=args.verbose_matching)
    
    if not items:
        print("Error: No items found in document")
        sys.exit(1)
    
    # Find matches using selected approach
    if args.use_keyword_matching:
        matcher.find_all_matches_keyword(debug=args.verbose_matching, output_dir=str(output_dir))
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
