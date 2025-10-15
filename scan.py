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
import pdb
import os
import sys
import csv
import json
import base64
import platform
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
    if platform.system() == "Windows":
        poppler_path = None
        script_dir = Path(__file__).parent
        # make sure that the poppler library is there
        if not os.path.exists(str(script_dir) + "/poppler/Library/bin/pdftoppm.exe"):
            print(f"Warning: poppler package not installed.  "
                  f"Please install it in {str(script_dir / 'poppler')}")
        else:
            poppler_path = str(script_dir) + "/poppler/Library/bin"
            
except ImportError:
    PDF_SUPPORT = False
    print("Warning: pdf2image not installed. PDF support will be disabled.")
    print("Install with: pip install pdf2image")

###############################################################################
class Scanner:
    def __init__(self, api_key, databases):
        """Initialize with Claude API key"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.databases = databases

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

                    debug_file = Path(output_dir) / (Path(cached_file).stem + "_context_extracted_json_fixed.json")
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        for item in page_items:
                           print(item, file=f)

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
                    if platform.system() == "Windows":
                        print(f"Calling convert_from_path({pdf_path}, dpi={dpi}, popplerpath={poppler_path})")
                        images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
                    else:
                        images = convert_from_path(pdf_path, dpi=dpi)
                    print(f"Converted {len(images)} pages")
                    
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
    
    def scan_document(self, document_path: str, debug_output: bool = True, output_dir: str = ".", verbose: bool = False) -> List[ScannedItem]:
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
                    
                    page_items = self._scan_single_image(image_path, debug_output, output_dir, verbose)
                    all_items.extend(page_items)
                
                return all_items
                    
            # Handle image files
            else:
                items = self._scan_single_image(document_path, debug_output, output_dir, verbose)
                return items
                
        except Exception as e:
            print(f"Error scanning document: {e}")
            return []
    
    def _scan_single_image(self, image_path: str, debug_output: bool = True, output_dir: str = ".", verbose: bool = False) -> List[ScannedItem]:
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

            # Create enhanced prompt with database examples
            prompt_text = load_prompt("scan_prompt")

            model = "claude-opus-4-20250514"
            model = "claude-sonnet-4-20250514"
            model = "claude-sonnet-4-5-20250929"

            # Create the message with image using the current model
            message = call_ai_with_retry(
                client=self.client,
                model=model,
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
                temperature=0
            )

            # Parse the response
            response_text = message.content[0].text

            # Debug output
            if debug_output:
                debug_file = Path(output_dir) / (Path(image_path).stem + "_scan_debug.txt")
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("PROMPT TEXT\n")
                    f.write("="*60 + "\n")
                    f.write(prompt_text)
                    f.write("="*60 + "\n\n")
                    f.write("CLAUDE RESPONSE\n")
                    f.write("="*60 + "\n\n")
                    f.write(response_text)
                    f.write("\n\n" + "="*60 + "\n")
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
                    # don't write the output to the file; this will force rescan of the page
                    print(f"✗ Error scanning document: No items found in page!")
                    return []

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
