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
class Scanner:
    def __init__(self, client, databases):
        """Initialize with Claude API key"""
        self.client = client
        self.databases = databases

        self.keyword_matcher = Matcher(self.databases)

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
                
                page_items = []
                if json_text.strip():
                    items_data = json.loads(json_text)
                    
                    for item_data in items_data:
                        item = ScannedItem(
                            quantity=item_data.get('quantity', ''),
                            description=item_data.get('description', ''),
                            original_text=item_data.get('original_text', ''),
                            matches=[]
                        )
                        page_items.append(item)

                    page_items = self._fix_missing_specifications(page_items)
                    all_items.extend(page_items)
            
            print(f"✓ Loaded {len(all_items)} cached items from {len(cached_files)} pages")
            return all_items
            
        except Exception as e:
            print(f"Error loading cached results: {e}")
            return []
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def convert_pdf_to_images(self, pdf_path: str, output_dir:str) -> List[str]:
        """Convert PDF pages to images and return list of image file paths"""
        if not PDF_SUPPORT:
            raise ImportError("PDF support requires pdf2image. Install with: pip install pdf2image")
        
        # Create temporary directory for images
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
                        image_path = os.path.join(output_dir, f"page_{i+1}.jpg")
                        
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
            raise Exception(f"Error converting PDF to images: {e}")
        
        return image_paths
    
    #########################################
    # NOT USED
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
                
                image_paths = self.convert_pdf_to_images(document_path, output_dir)
                
                if debug_output:
                    print(f"PDF converted to {len(image_paths)} page(s)")
                
                all_items = []
                
                # Process each page
                for i, image_path in enumerate(image_paths):
                    if debug_output:
                        print(f"Processing page {i+1}/{len(image_paths)}")
                    
                    page_items = self._scan_single_image_with_database_context(image_path, debug_output, output_dir, verbose)
                    all_items.extend(page_items)
                
                return all_items
                    
            # Handle image files
            else:
                items = self._scan_single_image_with_database_context(document_path, debug_output, output_dir, verbose)
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

