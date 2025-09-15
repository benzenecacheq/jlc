#!/usr/bin/env python3
###############################################################################
# Match items in database programmatically.
###############################################################################

import pdb
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
        self.attributes = None
        self.attrmap = None
        self.debug_item = None
        self.current_item = None
        self.debug_part = None
    
    def _load_attributes(self):
        if self.attributes is not None:
            return
        # load attributes from the database.  Map the attribute to the first value and the entry
        # in the database
        self.attributes = {}    # map alternative attribute names
        self.attrmap = {}       # list of database entries by attribute

        for fname,database in self.databases.items():
            for entry in database["parts"]:
                # Assume the first column is the attributes
                attributes = entry[database["headers"][0]].split(',')
                for i,attr in enumerate(attributes):
                    attributes[i] = attr.lower().strip().replace(' ', '')

                # the first attributes in the list is the name we will map others to
                name = attributes[0]
                if name not in self.attrmap:
                    self.attrmap[name] = []
                self.attrmap[name].append(entry)

                for attr in attributes:
                    if attr not in self.attributes:
                       self.attributes[attr] = name

                entry["attr"] = name

    def _looks_like_dimension(self, word, requirex=False):
        # dimensions look like 2x4 or 4x8x1/2
        if requirex and 'x' not in word: 
            return False
        invalid_pattern = "[^0-9 x/-]"
        return re.search(invalid_pattern, word) is None

    def _dimensions_match(self, word1, word2):
        # replace everythign that's not a number with a space and compare
        return re.sub(r'[^0-9]', ' ', word1) == re.sub(r'[^0-9]', ' ', word2)

    def _looks_like_fraction(self, word):
        valid_pattern = "[0-9]* */ *[0-9]*"
        return re.fullmatch(valid_pattern, word) is not None

    def _looks_like_length(self, word):
        if word == "104-1/4" or word == "116-1/4":
            return True
        if word == "1" or word == "2": # these are probably a grade.
            return False

        return re.match(r'(\d+)(?:\s*(?:\'|ft|feet))?(?:\s|$)', word) != None
        return re.search(r'(\d+)(?:\s*(?:\'|ft|feet))?(?:\s|$)', word) != None

    def _lengths_equal(self, word1, word2):
        # strip out anything that's not a number
        if type(word1) != type('') or type(word2) != type(''):
            return False
        return re.sub(r'[^0-9]', '', word1) == re.sub(r'[^0-9]', '', word2)

    def _cleanup(self, description):
        # perform obvious fixing of things that look like OCR errors and other
        # things that will confuse the matcher
        # this takes a string and returns a list of words
        desc = do_subs("cleanup",description.lower()).strip().split()
        i = 0
        gotdim = False
        gotlen = False
        gotattr = False
        gotgrade = False
        while i < len(desc):
            word = desc[i]

            # look for two words divided by /.  We see PT/DF which should actually
            # be changed to "DF PT" or just PT
            if '/' in word and not word.startswith('/') and not word.endswith('/'):
                index = word.find('/')
                next = word[index+1:]
                word = word[:index]
                if word+next in self.attributes:
                    # join together without /
                    desc[i] = word+next
                    continue
                elif next+word in self.attributes:
                    desc[i] = next+word
                    continue
                elif word in self.attributes or next in self.attributes:
                    # split in two
                    desc[i] = word
                    desc.insert(i+1, next)
                    continue    # reprocess word we just truncated

            # look for # sign and if it's a grade maybe split the word
            if '#' in word:
                # is there a number after it?
                index = word.find('#')
                if index == 0 and word[1:].isdigit():
                    # assume this is a grade
                    gotgrade = True
                else:
                    if word[:index] in self.attributes:
                        # break the word here
                        if index == len(word)-1:
                            # just smoke the # if it's at the end of an attr
                            desc[i] = word[:-1]
                        else:
                            # break into two
                            desc[i] = word[:index]
                            next = word[index:]
                            desc.insert(i+1, next)
                        continue    # reprocess word we just truncated

            # sometimes we see an H instead of a #
            if 'h' in word:
                # this only matters if the stuff before the h is an attr and the stuff
                # after the h is a number
                index = word.find('h')
                if index == len(word)-2 and word[index+1].isdigit() and word[:index] in self.attributes:
                    desc[i] = word[:index] + "#" + word[index+1:]
                    continue    # reprocess with number sign instead
                
            # look for cases where we will join words
            if i < len(desc)-1:
                nextword = desc[i+1]
                # sometimes there are multi-word attributes that may have 
                # an attribute as a subset, such as KD and KD Doug Fir is three word example
                if i < len(desc)-2 and word + nextword + desc[i+2] in self.attributes:
                    desc[i] = self.attributes[word + nextword + desc[i+2]]
                    del desc[i+1]
                    del desc[i+1]
                    continue
                # two word example, such as KD DF
                if word + nextword in self.attributes:
                    desc[i] = self.attributes[word + nextword]
                    del desc[i+1]
                    continue

                # sometimes we see an H instead of a # at the end of a word that should be a #
                if 'h' in word:
                    # this only matters if the stuff before the h is an attr and the next word is a number
                    index = word.find('h')
                    if index == len(word)-1 and nextword.isdigit() and word[:index] in self.attributes:
                        # just smoke the #
                        desc[i] = word[:index]
                        continue
                
                # look for number followed by feet or ft
                if word.isdigit() and nextword in ["ft", "feet", "'"]:
                    desc[i] += "'"
                    del desc[i+1]
                    continue

                # sometimes we see dimensions with spaces in them like 2 x 4
                if i > 0 and word == "x":
                    check = desc[i-1] + word + desc[i+1]
                    if self._looks_like_dimension(check):
                        desc[i-1] = check
                        del desc[i]
                        del desc[i]
                        i -= 1
                        continue

                if (word[-1].isdigit() and  self._looks_like_dimension(word)
                    and self._looks_like_fraction(nextword)):
                    # this was something like '1 1/2' and we want to make it
                    # 1-1/2
                    desc[i] += '-' + nextword
                    del desc[i+1]
                    continue

            # look for grade. currently 1 and 2 are the only grades I've seen
            if not gotgrade and word in ["1","2"]:
                found = gotlen
                if not found:
                    # unlikely given it's a 1 or 2, but this could be a length.  
                    # To make sure it's not, check to see if there is a length later.
                    for next in desc[i+1:]:
                        if self._looks_like_length(next):
                            found = True
                            break
                if found:
                    gotgrade = True
                    desc[i] = '#' + word

            if self._looks_like_dimension(word, requirex=True):
                gotdim = True
            elif word in self.attributes:
                # change to universal attribute name
                desc[i] = self.attributes[word]
                gotattr = True
            elif self._looks_like_length(word):
                gotlen = True

            i += 1

        return desc

    def _parse_lumber_item(self, description: str, debug: bool = False) -> dict:
        if self.attributes is None:
            self._load_attributes()

        desc = self._cleanup(description)
        if debug:
            print(f"    Cleaned description: '{desc}'")

        components = {}

        for d in desc:
            if self._looks_like_dimension(d, requirex=True) and 'dimensions' not in components:
                components['dimensions'] = d
                if debug:
                    print(f'    Found dimensions: {d}')
            elif d in self.attributes:
                if 'attributes' not in components:
                    components['attributes'] = []
                components['attributes'].append(d)
                if debug:
                    print(f'    Found attribute: {d}')
            elif self._looks_like_length(d) and 'length' not in components:
                components['length'] = d
                if debug:
                    print(f'    Found length: {d}')
            else:
                if 'other' not in components:
                    components['other'] = []
                if debug:
                    print(f'    Found other: {d}')
                components['other'].append(d)

        return components

    def _calculate_lumber_match_score(self, item_components: dict, db_components: dict, debug: bool = False) -> float:
        """Calculate how well an item matches a database entry"""
        score = 0.0

        for name,dbc in db_components.items():
            if name not in item_components:
                continue
            if type(dbc) == type([]) and name in item_components:
                for i in item_components[name]:
                    found = False
                    for d in dbc:
                        if fuzzy_match(d,i):
                            score += 0.05 * len(i)
                            break
            elif name == "dimensions" and self._dimensions_match(dbc, item_components.get(name)):
                score += 0.3
            elif name == "length" and self._lengths_equal(dbc, item_components.get(name)):
                score += 0.3
            elif dbc == item_components.get(name):
                score += 0.1
        if "length" not in item_components and "length" not in db_components:
            score *= 1.6
        if "dimensions" not in item_components and "dimensions" not in db_components:
            # this is probably not lumber
            score *= 1.6
        return score

    def match_lumber_item(self, item: str, database_name: str=None, debug: bool = False) -> List[PartMatch]:
        """Specialized matching for lumber database format"""
        if database_name is None:
            matches = []
            for db in sorted(self.databases):
                matches += self.match_lumber_item(item, db, debug)
            return matches
        if debug:
            print(f"\nDEBUG: Lumber-specific matching for '{item}' in {database_name}")

        database = self.databases[database_name]
        parts = []
        headers = database['headers']

        # Parse the scanned item to extract lumber components
        item_components = self._parse_lumber_item(item, debug)
        if debug:
            print(f"  Parsed item components: {item} -> {item_components}")
        if self.debug_item == self.current_item and self.debug_part is None:
            pdb.set_trace()     # debug the processing of this item.

        if "attributes" in item_components:
            # I would really expect only one attribute.
            for attr in item_components["attributes"]:
                parts += self.attrmap[attr]
        if len(parts) == 0:
            parts = database['parts']

        matches = []
        confidence_scores = []

        for part in parts:
            item_number = part.get('Item Number', '').strip()
            item_desc = part.get('Item Description', '').strip()
            customer_terms = part.get('Customer Terms', '').strip()
            stocking_multiple = part.get('Stocking Multiple', '').strip()
            attr = part.get('attr')

            if not item_number or not item_desc:
                continue
            if (self.debug_item == self.current_item and 
                self.debug_part.lower() == item_number.lower()):
                pdb.set_trace()

            # Parse the database entry
            if 'components' in part:
                db_components = part['components']
            else:
                db_components = self._parse_lumber_item(item_desc, debug)
                part['components'] = db_components

            if 'attributes' not in db_components:
                db_components['attributes'] = part.get('attr')

            # Calculate match score
            match_score = self._calculate_lumber_match_score(item_components, db_components, debug)
            if match_score > 0.50:  # Threshold for considering it a match
                match = PartMatch(
                    description=item,
                    part_number=item_number,
                    database_name=database_name,
                    database_description=f"{item_desc} | Attr: {attr} | components: {db_components} | Score: {match_score:.2f}",
                    score=match_score,
                    confidence=str(match_score)
                )
                matches.append(match)

                if debug:
                    print(f"    MATCH: {item_number} -> {item_desc} (score: {match_score:.2f})")

        # Sort by match score (highest first)
        # Use a stable sort that handles ties by using the index as a secondary key
        sorted_matches = sorted(matches, key=lambda m: -m.score)

        if debug:
            print(f"  Found {len(sorted_matches)} matches")

        # only return matches that are at least as good as the best score and a max of 5
        n = 5
        for i,match in enumerate(sorted_matches[:n]):
           if match.score < sorted_matches[0].score:
               n = i
               break
        return sorted_matches[:n]  # Return top matches

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

    def match_general_item(self, item: str, database_name: str, debug: bool = False) -> List[PartMatch]:
        """General matching for non-lumber items (hardware, screws, etc.)"""
        if debug:
            print(f"  Using general matching for non-lumber item")

        database = self.databases[database_name]
        parts = database['parts']
        headers = database['headers']

        if not parts:
            return []

        matches = []
        item_desc_lower = item.lower()

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
                        description=item,
                        part_number=item_number,
                        database_name=database_name,
                        database_description=f"{item_desc} | Terms: {customer_terms} | Overlap: {overlap_ratio:.2f}",
                        score=overlap_ratio,
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
        if os.getenv("DEBUG_ITEM"):
            di = os.getenv("DEBUG_ITEM")
            if ':' in di:
                self.debug_part = di[di.find(':')+1:]
                di = di[:di.find(':')]
            self.debug_item = int(di)

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
            self.current_item = i
            if debug:
                print(f"\n[{i:2d}] Searching for: {item.description}")
                print(f"     Original: {item.original_text}")
                print(f"     Quantity: {item.quantity}")
                print("-" * 50)
            print(f"  [{i:2d}] {item.description}")

            all_matches = []
            for db_name in self.databases:
                if debug:
                    print(f"\n  Checking database: {db_name}")

                # Use original keyword matching
                matches = self.match_item(item.description, db_name, debug=debug)
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

    def match_item(self, item: str, database_name: str, debug: bool = False) -> List[PartMatch]:
        if debug:
            print(f"\nDEBUG: Keyword matching for '{item}' in {database_name}")

        # Determine if this looks like a lumber item
        desc_lower = item.lower()
        is_lumber = bool(re.search(r'\d+\s*x\s*\d+', desc_lower)) or any(term in desc_lower for term in
                         ['post', 'beam', 'lumber', 'cedar', 'pine', 'pt', 'advantage', 'glu lam', 'glulam'])

        if debug:
            print(f"  Classified as lumber: {is_lumber}")

        matches = self.match_lumber_item(item, database_name, debug)
        return matches
        if len(matches) > 0:
            return matches

        # Use enhanced general matching for hardware, etc.
        return self.match_general_item(item, database_name, debug)

