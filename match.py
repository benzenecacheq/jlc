#!/usr/bin/env python3
###############################################################################
# Match items in database programmatically.
###############################################################################

import os
import sys
import csv
import json
import anthropic
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from util import *
import re


class Matcher:
    def __init__(self, databases):
        """Initialize with Claude API key"""
        self.databases = databases
        
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

    def find_all_matches(self, scanned_items, debug: bool = False, output_dir: str = ".") -> None:
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

        for i, item in enumerate(scanned_items, 1):
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
                matches = self.match_item(item, db_name, debug=debug)
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

    def match_item(self, item: ScannedItem, database_name: str, debug: bool = False) -> List[PartMatch]:
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

