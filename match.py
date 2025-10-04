#!/usr/bin/env python3
###############################################################################
# Match items in database programmatically.
###############################################################################

import pdb
import os
import sys
import csv
import json
import copy
import anthropic
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from util import *
import re


class RulesMatcher:
    def __init__(self, databases, global_debug=False):
        self.debug = global_debug
        self.settings = None            # these get loaded from matcher.json

        self.databases = databases
        self.attrs = None
        self.attrmap = None
        self.debug_item = None
        self.current_item = None
        self.debug_part = None

        self.default_categories = None  # Encourage this category if none is otherwise present

        # Idenfity things that are more likely hardware than lumber
        self.hardware_categories = None
        self.hardware_terms = None
    
        self.default_categories = None  # if no category, assume these

        # most of the words in the description are ignored unless the match
        # below are some exceptions.
        self.keywords = None            # these are parts that contain certain keywords
                                        # loaded from the settings
        self.detractors = None          # penalize some words if they appear in only one
                                        # of the items being matched
        self.detractor_map = {}         # so we don't have to keep recalculating

        self.sku_parts = None           # parts that are in the SKU-only lookup
        self.board_parts = None         # parts that have dimensions and length
        self.dim_parts = None           # parts with dimensions but no length
        self.hardware_parts = None
        self.keyword_parts = None

        self.sku2partmap = None

    def get_setting(self, key):
        if key not in self.settings:
            raise Exception(f"Error in matcher.json: Missing setting for '{key}'")
        return self.settings[key]

    def _load_attrs(self):
        if self.settings is not None:
            return

        # load the global settings from the file
        exe_dir = str(Path(__file__).parent)
        fn = exe_dir + "/matcher.json"
        try:
            with open(fn, "r") as file:
                self.settings = json.load(file)
        except Exception as e:
            print(f"Error loading global settings from {fn}: {e}")
            raise e

        # Load misc items from settings json
        self.default_categories = self.get_setting("default categories")
        self.keywords = self.get_setting("keywords")
        self.detractors = self.get_setting("detractors")

        # Get the hardware characteristics
        self.hardware_categories = self.get_setting('hardware categories')
        self.hardware_terms = self.get_setting('hardware terms')

        # load attributes from the database.  Map the attribute to the first value and the entry
        # in the database
        self.attrs = {}    # map alternative attribute names
        self.attrmap = {}       # list of database entries by attribute
        self.sku2partmap = {}

        for fname,database in self.databases.items():
            for entry in database["parts"]:
                # Assume the first column is the attributes
                attrs = entry[database["headers"][0]].split(',')
                for i,attr in enumerate(attrs):
                    attrs[i] = attr.lower().strip().replace(' ', '')

                # the first attributes in the list is the name we will map others to
                name = attrs[0]
                if name not in self.attrmap:
                    self.attrmap[name] = []
                self.attrmap[name].append(entry)

                for attr in attrs:
                    if attr not in self.attrs:
                       self.attrs[attr] = name

                entry['attr'] = name

                self.sku2partmap[entry['Item Number']] = entry

        # load the board parts
        self.board_parts = []
        self.dim_parts = []
        self.sku_parts = []
        self.hardware_parts = []
        self.keyword_parts = {k:[] for k in self.keywords if k not in self.attrs}

        for fname,database in self.databases.items():
            for entry in database["parts"]:
                db_components = self.parse_lumber_item(entry["Item Description"], is_db=True)
                entry['components'] = db_components
                attrs = db_components.get('attrs')
                eattr = entry['attr']
                db_components['attrs'] = [eattr]
                if attrs is not None and attrs != [eattr]:
                    if eattr in attrs:
                        attrs.remove(eattr)
                    if 'other' not in db_components:
                        db_components['other'] = attrs
                    else:
                        db_components['other'] = list(set(db_components['other'] + attrs))
                if 'attrs' not in db_components:
                    db_components['attrs'] = [entry['attr']]
                elif entry['attr'] not in db_components['attrs']:
                    db_components['attrs'].append(entry['attr'])
                if 'dimensions' in db_components and ('length' in db_components or 
                                                      entry['Stocking Multiple'].lower() == 'lf'):
                    self.board_parts.append(entry)
                elif 'dimensions' in db_components:
                    self.dim_parts.append(entry)
                
                keywords = self._has_keyword(db_components, threshold=0.9)
                for k in keywords:
                    if k not in self.attrs:
                        self.keyword_parts[k].append(entry)

            self.sku_parts += self.select_parts(database["parts"], 'usuallyusethesku')
            self.hardware_parts += self.select_parts(database["parts"], self.hardware_categories)

    def _has_keyword(self, components, threshold = 0.7):
        # keyword can be a list or a single item
        items = ((components["attrs"] if 'attrs' in components else []) +
                 (components["other"] if 'other' in components else []))
        return fuzzy_match(sorted(self.keywords), items, threshold=threshold)
       
    def _looks_like_dimension(self, word, requirex=False):
        # dimensions look like 2x4 or 4x8x1/2
        if requirex and 'x' not in word: 
            return False
        invalid_pattern = "[^0-9 'x/-]"
        return re.search(invalid_pattern, word) is None

    def _countx(self, dimensions):
        if type(dimensions) == type(''):
            return dimensions.count('x')
        elif type(dimensions) == type({}) and 'dimensions' in dimensions:
            return dimensions['dimensions'].count('x')
        else:
            print("Expected either a string or item_components")
            return 0

    def _dimensions_match(self, word1, word2):
        # replace everythign that's not a number with a space and compare
        newword1 = " ".join(re.sub(r'[^0-9]', ' ', word1).split())
        newword2 = " ".join(re.sub(r'[^0-9]', ' ', word2).split())

        if newword1 != newword2 and '/' in (word1+word2) and len(word1) > 4:
            # sometimes there are issues with fractions
            return fuzzy_match(word1, word2, is_dimension=True)

        return newword1 == newword2

    def _looks_like_fraction(self, word, stop_at_x=True):
        if stop_at_x and 'x' in word:
            word = word[:word.find('x')]
        valid_pattern = "[0-9]* */ *[0-9]*"
        return re.fullmatch(valid_pattern, word) is not None

    def _looks_like_length(self, word, ok1and2=False):
        if not re.search(r'\d', word):
            return False                # no digits
        if word in ["92-1/4","104-1/4","116-1/4"]:
            return True
        if word == "1" or word == "2":  # these are probably a grade.
            return ok1and2

        return re.match(r'([\d\-/]+)(?:\s*(?:\'|ft|feet))?(?:\s|$)', word) != None
        return re.match(r'(\d+)(?:\s*(?:\'|ft|feet))?(?:\s|$)', word) != None
        return re.search(r'(\d+)(?:\s*(?:\'|ft|feet))?(?:\s|$)', word) != None

    def _lengths_equal(self, word1, word2):
        if type(word1) != type('') or type(word2) != type(''):
            return False
        # remove leading zeroes
        word1,word2 = word1.lstrip('0'), word2.lstrip('0')
        # strip out anything that's not a number
        ret = re.sub(r'[^0-9/]', '', word1) == re.sub(r'[^0-9/]', '', word2)
        if not ret and '/' in (word1+word2):
            # The scan has trouble with fractions so do a fuzzy match
            return fuzzy_match(word1, word2, is_dimension=True)
        return ret

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
                if word+next in self.attrs:
                    # join together without /
                    desc[i] = word+next
                    continue
                elif next+word in self.attrs:
                    desc[i] = next+word
                    continue
                elif word in self.attrs or next in self.attrs:
                    # split in two
                    desc[i] = word
                    desc.insert(i+1, next)
                    continue    # reprocess word we just truncated
                word = desc[i]

            # look for # sign and if it's a grade maybe split the word
            if '#' in word:
                # is there a number after it?
                index = word.find('#')
                if index == 0 and word[1:].isdigit():
                    # assume this is a grade
                    gotgrade = True
                else:
                    if word[:index] in self.attrs:
                        # break the word here
                        if index == len(word)-1:
                            # just smoke the # if it's at the end of an attr
                            desc[i] = word[:-1]
                        elif index == 0:
                            # just smoke the leading #
                            desc[i] = word[1:]
                        else:
                            # break into two
                            desc[i] = word[:index]
                            next = word[index:]
                            desc.insert(i+1, next)
                        continue    # reprocess word we just truncated

            dash = word.find('-')
            if dash > 0 and dash < len(word)-1:
                # if either side is a useful word, split it
                split_it = word[:dash] in self.attrs or word[dash+1:] in self.attrs
                split_it |= word[:dash] in self.keywords or word[dash+1:] in self.keywords
                 
                # if it's a dimension then unless the dash is followed by a fraction, split it
                split_it |= self._looks_like_dimension(word) and not self._looks_like_fraction(word[dash+1:])

                if split_it:
                    desc[i] = word[:dash]
                    desc.insert(i+1, word[dash+1:])
                    continue

            # sometimes we see an H instead of a #
            if 'h' in word:
                # this only matters if the stuff before the h is an attr and the stuff
                # after the h is a number
                index = word.find('h')
                if index == len(word)-2 and word[index+1].isdigit() and word[:index] in self.attrs:
                    desc[i] = word[:index] + "#" + word[index+1:]
                    continue    # reprocess with number sign instead
                
            # look for cases where we will join words
            if i < len(desc)-1:
                nextword = desc[i+1]
                # sometimes there are multi-word attributes that may have 
                # an attribute as a subset, such as KD and KD Doug Fir is three word example
                if i < len(desc)-2 and word + nextword + desc[i+2] in self.attrs:
                    desc[i] = self.attrs[word + nextword + desc[i+2]]
                    del desc[i+1]
                    del desc[i+1]
                    continue
                # two word example, such as KD DF
                if word + nextword in self.attrs:
                    desc[i] = self.attrs[word + nextword]
                    del desc[i+1]
                    continue

                # sometimes we see an H instead of a # at the end of a word that should be a #
                if 'h' in word:
                    # this only matters if the stuff before the h is an attr and the next word is a number
                    index = word.find('h')
                    if index == len(word)-1 and nextword.isdigit() and word[:index] in self.attrs:
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

                if (word[-1].isdigit() and self._looks_like_dimension(word)
                    and self._looks_like_fraction(nextword)):
                    # this was something like '1 1/2' and we want to make it
                    # 1-1/2
                    desc[i] += '-' + nextword
                    del desc[i+1]
                    continue

            # look for grade. currently 1 and 2 are the only grades I've seen
            if not gotgrade and word in ["1","2"] and gotdim:
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
            elif word in self.attrs:
                # change to universal attribute name
                desc[i] = self.attrs[word]
                gotattr = True
            elif self._looks_like_length(word):
                gotlen = True

            i += 1

        return desc

    def parse_lumber_item(self, description: str, is_db=False) -> dict:
        if self.attrs is None:
            self._load_attrs()
        desc = self._cleanup(description)
        if self.debug:
            print(f"    Cleaned description: '{desc}'")

        components = {}

        for d in desc:
            if self._looks_like_dimension(d, requirex=True) and 'dimensions' not in components:
                components['dimensions'] = d
                if self.debug:
                    print(f'    Found dimensions: {d}')
            elif d in self.attrs:
                if 'attrs' not in components:
                    components['attrs'] = []
                if d not in components['attrs']:
                    components['attrs'].append(d)
                if self.debug:
                    print(f'    Found attribute: {d}')
            elif self._looks_like_length(d, ok1and2=is_db and "attrs" not in components) and 'length' not in components:
                components['length'] = d
                if self.debug:
                    print(f'    Found length: {d}')
            else:
                if 'other' not in components:
                    components['other'] = []
                if self.debug:
                    print(f'    Found other: {d}')
                if d not in components['other']:
                    components['other'].append(d)

        return components

    def _get_detractors(self, components, threshold=0.6):
        check = []
        for key,words in components.items():
            if type(words) == type([]):
                check += words
        key = " ".join(check)
        detractors = self.detractor_map.get(key)
        if detractors is not None:
            return detractors
        detractors = fuzzy_match(sorted(self.detractors), check, threshold=threshold) if len(check) > 0 else {}
        self.detractor_map[key] = detractors
        return detractors

    def _calculate_lumber_match_score(self, item_components: dict, db_components: dict, sku: str='', sell_by_foot=False) -> float:
        """Calculate how well an item matches a database entry"""
        score = 0.0

        # some special cases:
        if len(item_components) == 1 and 'other' in item_components and 'other' in db_components:
            # All we have is 'other'
            divisor = (len(item_components['other'])*2 + len(db_components['other'])) / 3
            for i in item_components['other']:
                skumatch = fuzzy_match(i, sku)
                skumatch = skumatch if skumatch > 0.6 else 0.0
                maxmatch = 0.0
                for d in db_components['other']:
                    maxmatch = max(maxmatch, fuzzy_match(i, d))
                    if maxmatch > 0.8:
                        break
                if maxmatch > skumatch:
                    score += maxmatch / divisor
                else:
                    score += skumatch / len(item_components['other'])
            return score

        # check for keywords that are required to be in both if found in either
        ifound = self._get_detractors(item_components)
        dfound = self._get_detractors(db_components, threshold=0.9)     # should be no typos

        found = set(ifound) ^ set(dfound)
        if found:
            penalty = sum([self.detractors[name] * len(name) * (ifound[name] if name in ifound else dfound[name])
                                for name in found])
            score += penalty

        for name,dbc in db_components.items():
            if type(dbc) == type([]) and name in item_components:
                multiplier = 0.05 if name == "attrs" else 0.03
                matches = fuzzy_match(dbc, item_components[name], threshold=0.6)
                for n,s in matches.items():
                    s = 1.0 if s > 0.8 else s
                    l = self.keywords[n] if n in self.keywords else len(n)
                    score += s * multiplier * l
            elif name == "attrs":
                # the item doesn't have a category.  See if we want to assume one
                for d in dbc:
                    if d in self.default_categories:
                        score += self.default_categories[d]
            elif name == "dimensions" and name in item_components:
                score += 0.4 * self._dimensions_match(dbc, item_components[name])
                if "length" not in item_components and "length" not in db_components:
                    score += .25
            elif name == "length" and name in item_components:
                score += 0.3 * self._lengths_equal(dbc, item_components.get(name))
            elif name in item_components and dbc == item_components.get(name):
                score += 0.1

        if "dimensions" not in item_components and "dimensions" not in db_components:
            # this is probably not lumber
            if "length" not in item_components and "length" not in db_components:
                score *= 2.0
            else:
                score *= 1.6

        # sometimes people will put the length in with the dimensions
        dims = item_components.get('dimensions')
        db_dims = db_components.get('dimensions')
        if (dims is not None and db_dims is not None and
            ('length' in item_components) != ('length' in db_components)):
            a,b = (db_components, item_components) if 'length' in db_components else (item_components, db_components)
            length = a['length']
            dims = a['dimensions']

            a = copy.deepcopy(a)
            del a['length']
            a['dimensions'] = dims + 'x' + length  # slightly prefer this order
            score = max(score, self._calculate_lumber_match_score(a, b, sku, sell_by_foot=sell_by_foot) + 0.05)
            a['dimensions'] = length + 'x' + dims
            score = max(score, self._calculate_lumber_match_score(a, b, sku, sell_by_foot=sell_by_foot))

        if 'dimensions' in item_components and 'length' in item_components and 'dimensions' in db_components:
            # sometimes we'll see a length when it's just another number the user has put into the description
            if self._countx(item_components) > self._countx(db_components):
                a = copy.deepcopy(item_components)
                del a['length']
                score = max(score, self._calculate_lumber_match_score(a, db_components, sku, sell_by_foot=sell_by_foot))

        if sell_by_foot:
            # match cases where we are selling by linear feet instead of each item
            length = ""
            new_score = -0.05   # this should be worse than an exact matching length
            acopy = copy.deepcopy(item_components)
            if 'length' in item_components:
                # remove the length from the item and try again
                new_length = acopy['length']
                del acopy['length']
                if 'length' in db_components:
                    del db_components['length']   # this shouldn't happen
                new_score += self._calculate_lumber_match_score(acopy, db_components, sku=sku)
            elif 'dimensions' in item_components:
                dims = acopy.get('dimensions')
                lastx = dims.rfind('x')
                if lastx > 0 and lastx < len(dims)-1:
                    new_length = dims[lastx+1:]
                    acopy['dimensions'] = dims[:lastx]
                    new_score += self._calculate_lumber_match_score(acopy, db_components, sku=sku)

            if new_score > score:
                score = new_score

                # tell top level that we have a length
                item_components["sell_by_foot"] = db_components["sell_by_foot"] = new_length

        return score

    # add a found item to the list of matches
    def add_match(self, matches: List[PartMatch], item: str, part: Dict, score: float, length: str=""):
        item_number = part.get('Item Number', '').strip()
        item_desc = part.get('Item Description', '').strip()
        db_components = self.parse_lumber_item(item_desc)
        attr = part.get('attr')
        match = PartMatch(
            description=item,
            part_number=item_number,
            database_name=part['database'],
            database_description=f"{item_desc} | Attr: {attr} | "
                                 f"components: {db_components} | Score: {score:.2f}",
            score=score,
            lf=length,
            confidence=str(score)
        )
        matches.append(match)

    def _unique_parts(self, parts):
        # remove duplicate parts from the list using the part number as key to a dict
        map = {d["Item Number"]: d for d in parts}
        
        return sorted(map.values(), key=lambda item: item["Item Description"])

    def select_parts(self, parts, categories):
        selected = []
        if type(categories) == type([]):
            for c in categories:
                selected += self.select_parts(parts, c)
            return selected

        # there can be misspellings so use fuzzy matching
        scores = fuzzy_match(sorted(self.attrmap), categories, threshold=0.5)
        for cat, score in scores.items():
             selected += self.attrmap[cat]
        return selected

    def try_sku_match(self, item_str: str, parts_list: List[Dict]) -> List[PartMatch]:
        item_components = self.parse_lumber_item(item_str)
        matches = []
        items = item_components['attrs'] if 'attrs' in item_components else []
        items += item_components['other'] if 'other' in item_components else []

        # clean out non-alphanumeric from items in the list because skus are all alphanumeric
        items = [re.sub(r'[^a-zA-Z0-9]', '', s) for s in items]

        for part in parts_list:
            if (self.debug_item == self.current_item and self.debug_part and
                self.debug_part.lower() == part['Item Number'].lower()):
                print(f"items={items}")
                pdb.set_trace()
            scores = fuzzy_match(items, part['Item Number'], threshold=0.7)
            if len(scores):
                # add any other information that might help improve match
                score = self._calculate_lumber_match_score(item_components, part['components'], sku="") / 10
                self.add_match(matches, item_str, part, max(scores.values()) + score)

        return matches

    def match_lumber_item(self, item: ScannedItem, database_name: str=None, use_original=False) -> List[PartMatch]:
        """Specialized matching for lumber database format"""
        item_desc = item.description
        if use_original:
            # try using the original text
            otext = item.original_text.strip()

            # strip off quantity
            if otext.startswith(str(item.quantity) + " "):
                otext = otext[len(str(item.quantity))+1:]

            item_desc = otext

        if database_name is None:
            matches = []
            for db in sorted(self.databases):
                matches += self.match_lumber_item(item, db)
            return matches
        if self.debug:
            print(f"\nDEBUG: Lumber-specific matching for '{item_desc}' in {database_name}")

        database = self.databases[database_name]
        parts = []
        headers = database['headers']

        # Parse the scanned item to extract lumber components
        item_components = self.parse_lumber_item(item_desc)
        if self.debug:
            print(f"  Parsed item components: {item_desc} -> {item_components}")

        if self.debug_item == self.current_item and self.debug_part is None:
            print(f"  Parsed item components: {item_desc} -> {item_components}")
            pdb.set_trace()     # debug the processing of this item.
            self.parse_lumber_item(item_desc)

        if "attrs" in item_components:
            # I would really expect only one attribute, but sometimes in the part list
            # attributes will be subsets of others, so I need to aggregate
            for attr in item_components["attrs"]:
                for cat,partlist in self.attrmap.items():
                    if fuzzy_match(attr,cat) > 0.6 or attr in cat:
                        parts += partlist

        # If we don't have a category for the item, special treatment applies
        if len(parts) == 0:
            if False: # 'other' in item_components and fuzzy_match(self.hardware_terms, item_components['other'], threshold=0.6):
                parts = self.hardware_parts
                matches = self.try_sku_match(item_desc, parts)
                if len(matches) > 0:
                    return self.sort_matches(item, matches)
            elif 'dimensions' in item_components and 'length' in item_components:
                # if there are dimensions and length, it's probably a board
                parts = self.board_parts
            elif 'dimensions' in item_components:
                # if there are dimensions and length, it's probably a board
                parts = self._unique_parts(self.dim_parts + self.board_parts)
            else:
                # I no longer have a SKU-only list so try the whole database
                matches = self.try_sku_match(item_desc, database['parts'])
                if len(matches) > 0:
                    return self.sort_matches(item, matches)

            #parts = database['parts']
        
        # add any keyword parts to the list
        keywords = self._has_keyword(item_components)
        if keywords:
            for k in keywords:
                if k not in self.attrs:
                    parts += self.keyword_parts[k]
            parts = self._unique_parts(parts)

        matches = []
        confidence_scores = []
        for idx,part in enumerate(parts):
            part_number = part.get('Item Number', '').strip()
            part_desc = part.get('Item Description', '').strip()
            stocking_multiple = part.get('Stocking Multiple', '').strip()
            attr = part.get('attr')

            if not part_number or not part_desc:
                continue

            # Parse the database entry
            if 'components' in part:
                db_components = part['components']
            else:
                db_components = self.parse_lumber_item(part_desc)
                part['components'] = db_components

            sell_by_foot = stocking_multiple.lower() == "lf"
            # Calculate match score

            match_score = self._calculate_lumber_match_score(item_components, db_components, sku=part_number,
                                                             sell_by_foot=sell_by_foot)
            if (self.debug_item == self.current_item and self.debug_part and
                self.debug_part.lower() == part_number.lower()):
                print(f"item_components={item_components}")
                print(f"db_components={db_components}")
                print(f"match_score={match_score}")
                pdb.set_trace()
                match_score = self._calculate_lumber_match_score(item_components, db_components, sku=part_number,
                                                             sell_by_foot=sell_by_foot)
            length = ""
            if sell_by_foot and 'sell_by_foot' in db_components:
                length = db_components['sell_by_foot']
                del db_components['sell_by_foot']
                if 'sell_by_foot' in item_components:
                    del item_components['sell_by_foot']

            if match_score >= 0.7:  # Threshold for considering it a match
                self.add_match(matches, item_desc, part, match_score, length)
                if self.debug:
                    print(f"    MATCH: {part_number} -> {part_desc} (score: {match_score:.2f})")

        matches = self.sort_matches(item, matches)
        if self.debug:
            print(f"  Found {len(sorted_matches)} matches")
        
        if False: #  use_original:
            # see if we get a better answer with the original text
            new_matches = self.match_lumber_item(item, database_name, True)
            if len(matches) == 0 or (len(new_matches) > 0 and matches[0].score <= new_matches[0].score):
                matches = new_matches

        # if we are doing variable length, set the quantity to the qty x length
        if len(matches) > 0 and self.sku2partmap[matches[0].part_number].get("Stocking Multiple") == "LF":
            if matches[0].lf == "": 
                # no length was set so assume quantity is the length
                item.quantity = "1x" + item.quantity
            else:
                item.quantity += "x" + re.sub(r'[^0-9/\.\-]', '', matches[0].lf) + "'"
                matches[0].lf = ""

        return matches

    def sort_matches(self, item, matches, n=5):
        # Sort by match score (highest first)
        # Use a stable sort that handles ties by using the index as a secondary key
        # Prefer precut to variable length
        sorted_matches = sorted(matches, key=lambda m: -m.score + (0 if m.lf == "" else 0.001))

        # only return matches that are at least as good as the best score and a max of N
        for i,match in enumerate(sorted_matches[:n]):
           if match.score < sorted_matches[0].score:
               n = i
               break

        return sorted_matches[:n]  # Return top matches


    def match_general_item(self, item: ScannedItem, database_name: str) -> List[PartMatch]:
        """General matching for non-lumber items (hardware, screws, etc.)"""
        if self.debug:
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
                        description=item_desc_lower,
                        part_number=item_number,
                        database_name=database_name,
                        database_description=f"{item_desc} | Terms: {customer_terms} | Overlap: {overlap_ratio:.2f}",
                        score=overlap_ratio,
                        confidence=confidence,
                    )
                    matches.append(match)

                    if self.debug:
                        print(f"    General match: {item_number} -> {item_desc} (overlap: {overlap_ratio:.2f})")

        # Sort by overlap ratio (stored in description for now)
        matches.sort(key=lambda x: float(x.database_description.split('Overlap: ')[1].split()[0]), reverse=True)

        return matches[:3]

    def find_all_matches(self, scanned_items, output_dir: str = ".") -> None:
        """Find matches using original keyword-based matching (for comparison)"""
        if os.getenv("DEBUG_ITEM"):
            di = os.getenv("DEBUG_ITEM")
            if ':' in di:
                self.debug_part = di[di.find(':')+1:]
                di = di[:di.find(':')]
            self.debug_item = int(di)

        if self.debug:
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
            if self.debug:
                print(f"\n[{i:2d}] Searching for: {item.description}")
                print(f"     Original: {item.original_text}")
                print(f"     Quantity: {item.quantity}")
                print("-" * 50)
            print(f"  [{i:2d}] {item.description}")

            all_matches = []
            for db_name in self.databases:
                if self.debug:
                    print(f"\n  Checking database: {db_name}")

                # Use rules-based
                matches = self.match_item(item, db_name)
                all_matches.extend(matches)

                if self.debug:
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

            if self.debug:
                if item.matches:
                    for match in item.matches:
                        print(f"  ✓ {match.confidence.upper():8s} | {match.part_number:15s} | {match.database_description[:80]}...")
                else:
                    print("  ✗ No matches found")
        
        print(f"\n✓ Detailed matching log saved to: {debug_log_file}")

    def match_item(self, item: ScannedItem, database_name: str) -> List[PartMatch]:
        if self.debug:
            print(f"\nDEBUG: Keyword matching for '{item}' in {database_name}")

        matches = self.match_lumber_item(item, database_name)
        return matches

        if len(matches) > 0:
            return matches

        # Use enhanced general matching for hardware, etc.
        return self.match_general_item(item, database_name)

