#!/usr/bin/env python3).strip('\'"')
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
        self.merged_database = None
        self.attrs = None
        self.attrmap = None
        self.debug_item = None
        self.current_item = None
        self.debug_part = None

        self.default_categories = None  # Encourage this category if none is otherwise present
        self.default_other = None       # Encourage default for other things (like 5ply)

        # Idenfity things that are more likely hardware than lumber
        self.hardware_categories = None
        self.hardware_terms = None
    
        # most of the words in the description are ignored unless the match
        # below are some exceptions.
        self.keywords = None            # these are parts that contain certain keywords
                                        # loaded from the settings
        self.detractors = None          # penalize some words if they appear in only one
                                        # of the items being matched
        self.detractor_map = {}         # so we don't have to keep recalculating
        self.implies = None
        self.substitute_dimensions=None # things that are aliases for dimensions (like penny)

        self.board_parts = None         # parts that have dimensions and length
        self.dim_parts = None           # parts with dimensions but no length
        self.lumber_categories = None
        self.hardware_parts = None
        self.keyword_parts = None

        self.sorted_skus = None
        self.skus_by_letter = None
        self.special_lengths = None
        self.scoring = None

        self.fuzzy_category_matching = False
        self.fuzzy_keyword_matching = False

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
        self.fuzzy_category_matching = self.get_setting("misc")["fuzzy category match"]
        self.fuzzy_keyword_matching = self.get_setting("misc")["fuzzy keyword match"]

        self.default_categories = self.get_setting("default categories")
        self.default_other = self.get_setting("default other")
        self.detractors = self.get_setting("detractors")
        self.special_lengths = set(self.get_setting("special lengths"))
        self.scoring = self.get_setting("scoring")
        self.implies = self.get_setting("implies")
        self.substitute_dimensions = self.get_setting("substitute dimensions")

        # get keywords and load optional additional keywords
        self.keywords = self.get_setting("keywords")
        kwfile = os.getenv("MATCHER_KEYWORDS")
        if kwfile:
            kw = csv2dict(kwfile)
            try:
                kw = { name.lower():float(val) for name,val in kw.items() }
            except:
                raise Exception(f"Invalid format for keyword file {kwfile}")
            self.keywords |= kw

        # Get the hardware characteristics
        self.lumber_categories = self.get_setting('lumber categories')
        self.hardware_categories = self.get_setting('hardware categories')
        self.hardware_terms = self.get_setting('hardware terms')

        # load attributes from the database.  Map the attribute to the first value and the entry
        # in the database
        self.attrs = {}    # map alternative attribute names
        self.attrmap = {}       # list of database entries by attribute

        self.merged_database = {}
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
                self.merged_database[entry['Item Number']] = entry

        self.sorted_skus = sorted(self.merged_database)
        self.skus_by_letter = {}
        i = 0
        while i < len(self.sorted_skus):
            fl = self.sorted_skus[i][0]
            skus = []
            while i < len(self.sorted_skus) and self.sorted_skus[i][0] == fl:
                skus.append(self.sorted_skus[i])
                i += 1
            self.skus_by_letter[fl] = skus

        # load the board parts
        self.board_parts = []
        self.dim_parts = []
        self.hardware_parts = []
        self.keyword_parts = {k:[] for k in self.keywords if k not in self.attrs}

        for entry in self.merged_database.values():
            db_components = self.parse_lumber_item(entry["Item Description"], db_attr=entry["attr"])
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

        self.hardware_parts += self._select_parts(self.merged_database.values(), self.hardware_categories)

    def _has_keyword(self, components, threshold=None):
        if threshold is None:
            threshold = self.scoring["keyword-threshold"]
        # keyword can be a list or a single item
        items = ((components["attrs"] if 'attrs' in components else []) +
                 (components["other"] if 'other' in components else []))
        return fuzzy_match(sorted(self.keywords), items, threshold=threshold)

    def _get_dims(self, word):
        if 'x' not in word:
            return []

        # how many dimensions in this specification
        breaks = [-1] + [m.start() for m in re.finditer('x', word)] + [len(word)]
        return [word[breaks[i]+1:breaks[i+1]] for i in range(len(breaks)-1)]

    def _looks_like_dimension(self, word, requirex=False):
        if word == "x": 
            return False

        # dimensions look like 2x4 or 4x8x1/2
        if requirex and 'x' not in word: 
            return False
        # if there is a - followed by something that isn't a fraction, bail
        dashes = [m.start() for m in re.finditer(re.escape('-'), word)]
        for dash in dashes:
            x = word[dash:].find('x')
            check = (word[dash+1:dash+x] if x > 0 else word[dash+1:]).strip('\'"')
            if not self._looks_like_fraction(check):
                return False
        invalid_pattern = "[^0-9 \\.'x/-]"
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
        # replace everything that's not a number with a space and compare
        newword1 = " ".join(re.sub(r'[^0-9]', ' ', word1).split())
        newword2 = " ".join(re.sub(r'[^0-9]', ' ', word2).split())

        if newword1 == newword2:
            return 1.0
        if len(word1) <= 4 or len(word2) <= 4 or "xx" in newword1+newword2 or '/' not in word1+word2:
            return 0

        score = fuzzy_match(word1, word2, is_dimension=True)

        # add extra penalty for cases where leading digit is different
        breaks1 = [-1] + [m.start() for m in re.finditer('x', word1)] + [len(word1)]
        breaks2 = [-1] + [m.start() for m in re.finditer('x', word2)] + [len(word2)]
        if len(breaks1) != len(breaks2):
            return score

        nums1 = [word1[breaks1[i]+1:breaks1[i+1]] for i in range(len(breaks1)-1)]
        nums2 = [word2[breaks2[i]+1:breaks2[i+1]] for i in range(len(breaks2)-1)]

        for n1,n2 in zip(nums1, nums2):
            # see if only the fraction is different.
            if '-' in n1 and '-' in n2:
                # strip off any fractional part
                n1,n2 = n1[:n1.find('-')], n2[:n2.find('-')]

                n1, n2 = re.sub(r'[^0-9]', '', n1), re.sub(r'[^0-9]', '', n2)
                if n1 == "" or n2 == "":
                    continue

                i1, i2 = int(n1), int(n2)
                diff = abs(i1-i2)
                if diff != 0:
                    # penalize by amount of difference
                    score -= 0.01 * abs(diff)
            
        return max(score,0)

    def _looks_like_fraction(self, word, stop_at_x=True):
        if stop_at_x and 'x' in word:
            word = word[:word.find('x')]
        valid_pattern = "[0-9]+ */ *[0-9]+"
        return re.fullmatch(valid_pattern, word) is not None

    def _looks_like_length(self, word, ok1and2=False):
        if not re.search(r'\d', word):
            return False                # no digits
        if word in self.special_lengths:
            return True
        if word == "1" or word == "2":  # these are probably a grade.
            return ok1and2
        
        # see if it's too big to be a reasonable length
        match = re.match(r'\d+', word)
        if match:
            if int(match.group()) > 40:
                return False
        return re.match(r'([\d\-/]+)(?:\s*(?:\'|ft|feet))?(?:\s|$)', word) != None

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

    def _unique_parts(self, parts):
        # remove duplicate parts from the list using the part number as key to a dict
        map = {d["Item Number"]: d for d in parts}
        
        return sorted(map.values(), key=lambda item: item["Item Description"])

    def _select_parts(self, parts, categories):
        selected = []
        if type(categories) == type([]):
            for c in categories:
                selected += self._select_parts(parts, c)
            return self._unique_parts(selected)

        # there can be misspellings so use fuzzy matching
        scores = fuzzy_match(sorted(self.attrmap), categories, threshold=0.5)
        for cat, score in scores.items():
             selected += self.attrmap[cat]
        return selected

    def _deselect_parts(self, parts, categories):
        if type(categories) == type([]):
            return [p for p in parts if p['attr'] not in categories]
        elif type(categories) == type(""):
            return [p for p in parts if p['attr'] != categories]
        else:
            raise Exception("_delect_parts() called with incorrect categories argument")
            

    def _fuzzy_sku_match(self, word):
        # this is suboptimal but it's faster
        word = word.upper()     # skus are all upper case
        skus = self.skus_by_letter.get(word[0])
        if skus is not None:
            return fuzzy_match(skus, word, threshold=0.7)
        return []

    def _is_attr(self, word, is_db=True):
        if not is_db and self.fuzzy_category_matching:
            matches = fuzzy_match(word, self.attrs, threshold=0.8)
            if matches:
                return list(matches)[0]
        elif word in self.attrs:
            return word
        return None

    def _is_keyword(self, word, is_db=True):
        if not is_db and self.fuzzy_keyword_matching:
            matches = fuzzy_match(word, self.keywords, threshold=0.8)
            if matches:
                return list(matches)[0]
        elif word in self.keywords:
            return word
        return None

    def _cleanup(self, description, is_db):
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
                if self._is_attr(word+next, is_db):
                    # join together without /
                    desc[i] = word+next
                    continue
                elif self._is_attr(next+word, is_db):
                    desc[i] = next+word
                    continue
                elif self._is_attr(word, is_db) or self._is_attr(next, is_db):
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
                    if self._is_attr(word[:index], is_db):
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
            if dash == 0 or dash == len(word)-1:
                # remove it
                desc[i] = word[1:] if dash == 0 else word[:-1]
            elif dash > 0:
                withoutdash = word[:dash]+word[dash+1:]
                split_it = True
                if is_db:
                    split_it = False
                elif (self._is_keyword(withoutdash, is_db) or 
                      self._is_attr(withoutdash, is_db) or 
                      self._fuzzy_sku_match(withoutdash)):
                    # the dash is confusing things so smoke it
                    desc[i] = withoutdash
                    split_it = False
                elif word[dash-1].isdigit() and word[dash+1].isdigit():
                    # could be dimension or part of dimension
                    if self._looks_like_dimension(word) or self._looks_like_fraction(word[dash+1:]):
                        split_it = False

                if split_it:
                    desc[i] = word[:dash]
                    desc.insert(i+1, word[dash+1:])
                    continue

            # sometimes we see an H instead of a #
            if 'h' in word:
                # this only matters if the stuff before the h is an attr and the stuff
                # after the h is a number
                index = word.find('h')
                if index == len(word)-2 and word[index+1].isdigit() and self._is_attr(word[:index], is_db):
                    desc[i] = word[:index] + "#" + word[index+1:]
                    continue    # reprocess with number sign instead

            if not is_db:
                # I saw an x followed by a catgory.  Not common but reasonably cheap
                x = word.find('x')
                found = False
                while x > 0 and not found:
                    if x < len(word)-1 and word[x-1].isdigit() and self._is_attr(word[x+1:], is_db):
                        # split it
                        desc[i] = word[:x]
                        desc.insert(i+1, word[x+1:])
                        found = True
                    x = word.find('x', x+1)

                if found:
                    continue

            # look for cases where we will join words
            if i < len(desc)-1:
                nextword = desc[i+1]
                # sometimes there are multi-word attributes that may have 
                # an attribute as a subset, such as KD and KD Doug Fir is three word example
                if i < len(desc)-2 and self._is_attr(word + nextword + desc[i+2], is_db):
                    desc[i] = self.attrs[self._is_attr(word + nextword + desc[i+2], is_db)]
                    del desc[i+1]
                    del desc[i+1]
                    continue
                # two word example, such as KD DF
                if self._is_attr(word + nextword, is_db):
                    desc[i] = self.attrs[self._is_attr(word + nextword, is_db)]
                    del desc[i+1]
                    continue

                # sometimes we see an H instead of a # at the end of a word that should be a #
                if 'h' in word:
                    # this only matters if the stuff before the h is an attr and the next word is a number
                    index = word.find('h')
                    if index == len(word)-1 and nextword.isdigit() and self._is_attr(word[:index], is_db):
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
            elif self._is_attr(word, is_db):
                # change to universal attribute name
                desc[i] = self.attrs[self._is_attr(word, is_db)]
                gotattr = True
            elif self._looks_like_length(word):
                gotlen = True

            i += 1

        return desc

    def parse_lumber_item(self, description: str, db_attr=None) -> dict:
        if self.attrs is None:
            self._load_attrs()
        is_db = db_attr is not None
        desc = self._cleanup(description, is_db)
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
            elif (self._looks_like_length(d, ok1and2=is_db and "attrs" not in components and "other" not in components) 
                  and 'length' not in components):
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

        # add any implied components:
        if "attrs" in components or db_attr is not None:
            for attr in [db_attr] if db_attr is not None else components["attrs"]:
                implied = self.implies[attr] if attr in self.implies else {}
                for k,implication in implied.items():
                    if k == "other":
                        if "other" not in components:
                            components["other"] = []
                        components["other"] += [i for i in implication if i not in components["other"]]
                        if self.debug:
                            print(f"    Added implied other items {implication}")
                    elif k not in components:
                        components[k] = implication
                        if self.debug:
                            print(f"    Added {k} {implication}")

        return components

    # add a found item to the list of matches
    def _add_match(self, matches: List[PartMatch], item: str, part: Dict, score: float, length: str=""):
        item_number = part.get('Item Number', '').strip()
        item_desc = part.get('Item Description', '').strip()
        db_components = self.parse_lumber_item(item_desc)
        attr = part.get('attr')
        match = PartMatch(
            description=item_desc,
            part_number=item_number,
            database_name=part['database'],
            database_description=f"{item_desc} | Attr: {attr} | "
                                 f"components: {db_components} | Score: {score:.2f}",
            type=attr,
            score=score,
            lf=length,
            confidence=str(score)
        )
        matches.append(match)

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

    def _dbgout(self, indent, line, text):
        if indent is not None:
            print(f"{indent*' '}-->line {line}: {text}")

    def _calculate_lumber_match_score(self, item_components: dict, db_components: dict, sku: str='', sell_by_foot=False, 
                                             indent=None) -> float:
        """Calculate how well an item matches a database entry"""
        score = 0.0
        newindent = None
        if indent is not None:
            print(indent*' ' + f"===============================")
            print(indent*' ' + "_calculate_lumber_match_score(")
            print(indent*' ' + f"      item_components={item_components}")
            print(indent*' ' + f"      db_components={db_components}")
            print(indent*' ' + f"      sku={sku}")
            print(indent*' ' + f"      sell_by_foot={sell_by_foot} )")
            print(indent*' ' + f"===============================")
            newindent = indent + 3

        # some special cases:
        if len(item_components) == 1 and 'other' in item_components and 'other' in db_components:
            # All we have is 'other'
            divisor = (len(item_components['other'])*2 + len(db_components['other'])) / 3
            skumatch = {}
            if sku:
                skumatch = fuzzy_match(sku, item_components["other"], threshold=self.scoring["skumatch-threshold"])
            sku_sum = sum(skumatch.values())
            dbmatch = fuzzy_match(item_components["other"], db_components["other"])
            db_sum = sum(dbmatch.values())
            score += db_sum / divisor + sku_sum / len(item_components['other'])
            if indent is not None:
                line = sys._getframe().f_lineno - 1
                print(indent*' ' + f"--> line {line}: score={score}, skumatch={skumatch}, dbmatch={dbmatch}")
                print(indent*' ' + f"=============== Final score: {score} ================")
            return score

        # check for detractors that are required to be in both if found in either
        ifound = self._get_detractors(item_components, threshold=0.7)
        dfound = self._get_detractors(db_components, threshold=0.9)     # should be no typos

        found = set(ifound) ^ set(dfound)
        if found:
            penalty = sum([self.detractors[name] * len(name) * (ifound[name] if name in ifound else dfound[name])
                                for name in found])
            score += penalty
            if indent is not None:
                line = sys._getframe().f_lineno - 1
                print(indent*' ' + f"--> line {line}: score={score}, detractors penalty={penalty}")

        for name,dbc in db_components.items():
            if type(dbc) == type([]):
                matches = []
                if name in item_components:
                    multiplier = self.scoring[name+"-multiplier"]
                    matches = fuzzy_match(dbc, item_components[name], threshold=0.6)
                    for n,s in matches.items():
                        s = 1.0 if s > self.scoring["close-enough"] else s
                        if n in self.substitute_dimensions:
                            # trick to make resulting calculation add up to the number we want
                            l = self.substitute_dimensions[n] / multiplier
                        else:
                            l = self.keywords[n] if n in self.keywords else len(n)
                            
                        score += s * multiplier * l
                        self._dbgout(indent, sys._getframe().f_lineno - 1, 
                                     f"score={score}, n={n}, s={s}, l={l}, multiplier={multiplier}")
                elif name == "attrs":
                    # check for default category
                    for d in dbc:
                        if d in self.default_categories:
                            score += self.default_categories[d]
                            if indent is not None:
                                line = sys._getframe().f_lineno - 1
                                print(indent*' ' + f"--> line {line}: score={score}, d={d}")
                if name == "other":
                    # check for default other
                    default_other = self.default_other if name == "other" else []
                    do = fuzzy_match(self.default_other, dbc, threshold = 0.9)
                    if do and list(do)[0] not in matches:
                        score += self.default_other[list(do)[0]]
                        if indent is not None:
                            line = sys._getframe().f_lineno - 1
                            print(indent*' ' + f"--> line {line}: score={score}, do={do}")
            elif name == "dimensions" and name in item_components:
                match = self._dimensions_match(dbc, item_components[name])
                score += self.scoring["dimensions-multiplier"] * match
                if indent is not None:
                    line = sys._getframe().f_lineno - 1
                    print(indent*' ' + f"--> line {line}: score={score}, dims match={match}")
                if "length" not in item_components and "length" not in db_components:
                    score += self.scoring["no-length-adder"]
                    if indent is not None:
                        line = sys._getframe().f_lineno - 1
                        print(indent*' ' + f"--> line {line}: score={score}, no-length-adder={self.scoring['no-length-adder']}")
            elif name == "length" and name in item_components:
                equal = self._lengths_equal(dbc, item_components.get(name))
                score += self.scoring["length-multiplier"] * equal
                if indent is not None:
                    line = sys._getframe().f_lineno - 1
                    print(indent*' ' + f"--> line {line}: score={score}, lengths equal={equal}, "
                                       f"length_multiplier={self.scoring['length-multiplier']}")
            elif name in item_components and dbc == item_components.get(name):
                score += self.scoring["misc-adder"]     # I don't think this ever happens
                if indent is not None:
                    line = sys._getframe().f_lineno - 1
                    print(indent*' ' + f"--> line {line}: score={score}, DIDN'T EXPECT THIS")

        idims = item_components.get("dimensions")
        ddims = db_components.get("dimensions")
        if not "has_dim" in db_components:
            db_components["has_dim"] = ddims is not None or ("other" in db_components and 
                                       (set(db_components["other"]) & set(self.substitute_dimensions)))
        if not db_components["has_dim"] and idims is None:
            # this is probably not lumber
            if "length" not in item_components and "length" not in db_components:
                score *= self.scoring["no-dim-no-len-multiplier"]
            else:
                score *= self.scoring["no-dim-multiplier"]
            if indent is not None:
                line = sys._getframe().f_lineno - 1
                print(indent*' ' + f"--> line {line}: score={score}, no dims")

        # sometimes people will put the length in with the dimensions
        if idims is not None and ddims is not None and ('length' in item_components) != ('length' in db_components):
            a,b = (db_components, item_components) if 'length' in db_components else (item_components, db_components)
            length = a['length']
            dims = a['dimensions']

            a = copy.deepcopy(a)
            del a['length']
            a['dimensions'] = dims + 'x' + length  # slightly prefer this order
            if 'length' in db_components:
                score = max(score, self._calculate_lumber_match_score(b, a, sku, sell_by_foot=sell_by_foot, indent=newindent))
            else:
                score = max(score, self._calculate_lumber_match_score(a, b, sku, sell_by_foot=sell_by_foot, indent=newindent))
            if indent is not None:
                line = sys._getframe().f_lineno - 1
                print(indent*' ' + f"--> line {line}: score={score}, {'length' in db_components}")

            # try putting length first *only* if it's a fraction 
            if self._looks_like_fraction(length):
                ldla = self.scoring["length-dim-last-add"]
                a['dimensions'] = length + 'x' + dims
                if 'length' in db_components:
                    score = max(score, self._calculate_lumber_match_score(b, a, sku, sell_by_foot=sell_by_foot, indent=newindent)-ldla)
                else:
                    score = max(score, self._calculate_lumber_match_score(a, b, sku, sell_by_foot=sell_by_foot, indent=newindent)-ldla)
                if indent is not None:
                    line = sys._getframe().f_lineno - 1
                    print(indent*' ' + f"--> line {line}: score={score}, {'length' in db_components}")

        if idims is not None and ddims is not None and 'length' in item_components:
            # sometimes we'll see a length when it's just another number the user has put into the description
            if self._countx(item_components) > self._countx(db_components):
                a = copy.deepcopy(item_components)
                del a['length']
                score = max(score, self._calculate_lumber_match_score(a, db_components, sku, 
                                                sell_by_foot=sell_by_foot, indent=newindent))
                if indent is not None:
                    line = sys._getframe().f_lineno - 1
                    print(indent*' ' + f"--> line {line}: score={score}")

        if idims is not None and ddims is not None:
            # sometimes we see a fractional dimension as a fractional part of the last dimension so try separating
            _d = self._get_dims(ddims)
            _i = self._get_dims(idims)
            # this really only matters if only one of them has a fraction in the last dimension
            if len(_d) and len(_i) and ('-' in _d[-1]) != ('-' in _i[-1]):
                a, b, dims = (db_components, item_components, ddims) if '-' in _d[-1]  else (item_components, db_components, idims) 
                # only good if the fraction is the last thing in the dimensions
                a = copy.deepcopy(a)
                dash = dims.rfind('-')
                a["length"] = dims[dash+1:]
                a["dimensions"] = dims[:dash]
                if '-' in _d[-1]:
                    score = max(score, self._calculate_lumber_match_score(b, a, sku))
                else:
                    score = max(score, self._calculate_lumber_match_score(a, b, sku))
                if indent is not None:
                    line = sys._getframe().f_lineno - 1
                    print(indent*' ' + f"--> line {line}: score={score}, {'-' in d[-1]}")
        
        if sell_by_foot:
            # match cases where we are selling by linear feet instead of each item
            length = ""
            new_score = self.scoring["sell-by-foot"]   # this should be worse than an exact matching length
            if "attrs" in db_components and db_components["attrs"][0] in self.get_setting("lf only"):
                new_score = 0
            acopy = copy.deepcopy(item_components)
            if 'length' in item_components:
                # remove the length from the item and try again
                new_length = acopy['length']
                del acopy['length']
                if 'length' in db_components:
                    del db_components['length']   # this shouldn't happen
                new_score += self._calculate_lumber_match_score(acopy, db_components, sku=sku, indent=newindent)
            elif 'dimensions' in item_components:
                dims = acopy.get('dimensions')
                lastx = dims.rfind('x')
                if lastx > 0 and lastx < len(dims)-1:
                    new_length = dims[lastx+1:]
                    acopy['dimensions'] = dims[:lastx]
                    new_score += self._calculate_lumber_match_score(acopy, db_components, sku=sku, indent=newindent)

            if new_score > score:
                if indent is not None:
                    line = sys._getframe().f_lineno - 1
                    print(indent*' ' + f"--> line {line}: old_score={score} score={new_score}")
                score = new_score

                # tell top level that we have a length
                item_components["sell_by_foot"] = db_components["sell_by_foot"] = new_length

        if indent is not None:
            print(indent*' ' + f"=============== Final score: {score} ================")
        return score

    def try_sku_match(self, item_str: str) -> List[PartMatch]:
        matches = []
        item_components = self.parse_lumber_item(item_str)
        if True:
            # use raw text rather than cleaned-up text
            words = item_str.split()
            # add double-words
            words += [words[i] + words[i+1] for i in range(len(words)-1)]
            # add triple-words
            words += [words[i] + words[i+1] + words[i+2] for i in range(len(words)-2)]
        else:
            words = item_components['attrs'] if 'attrs' in item_components else []
            words += item_components['other'] if 'other' in item_components else []
       
        # clean out non-alphanumeric from words in the list because skus are all alphanumeric
        words = [re.sub(r'[^a-zA-Z0-9]', '', s) for s in words]

        # remove any that are all numbers.  This may miss a few but it signicantly cuts
        # the number of incorrect matches
        words = [w for w in words if not w.isdigit()]

        indent = None
        if self.debug_item is not None and self.debug_item == self.current_item:
            indent = 3
            pdb.set_trace()
            if self.debug_part:
                scores = fuzzy_match(self.debug_part, words, threshold=self.scoring["skumatch-threshold"])

        scores = fuzzy_match(self.merged_database, words, threshold=self.scoring["skumatch-threshold"])
        if scores:
            for sku,score in scores.items():
                db_components = self.merged_database[sku]["components"]
                bonus = self._calculate_lumber_match_score(item_components, db_components, sku="", indent=indent) / 10
                if bonus < 0:
                    continue            # something bad happened

                # look for detractors.  We don't care about detractors that are in the database but not 
                # in the item because theoretically this is a SKU match
                ifound = self._get_detractors(item_components, threshold=0.7)
                if False: # ifound:
                    dfound = self._get_detractors(db_components, threshold=0.9)     # should be no typos
                    found = set(ifound) - set(dfound)
                    if found:
                        continue        # probably not a valid sku match

                self._add_match(matches, item_str, self.merged_database[sku], score + bonus)
            
        return matches

    def match_lumber_item(self, item: ScannedItem, use_original=False) -> List[PartMatch]:
        """Specialized matching for lumber database format"""
        item_desc = item.description
        if use_original:
            # try using the original text
            otext = item.original_text.strip()

            # strip off quantity
            if otext.startswith(str(item.quantity) + " "):
                otext = otext[len(str(item.quantity))+1:]

            item_desc = otext

        if self.debug:
            print(f"\nDEBUG: Lumber-specific matching for '{item_desc}' in {database_name}")

        parts = []

        # Parse the scanned item to extract lumber components
        item_components = self.parse_lumber_item(item_desc)
        if self.debug_item == self.current_item and not self.debug_part:
            print(f"  Parsed item components: {item_desc} -> {item_components}")
            pdb.set_trace()     # debug the processing of this item.
            self.parse_lumber_item(item_desc)

        friends = {}
        attrs = item_components.get("attrs")
        if attrs is not None:
            # see if there are any friend categories we should add to the list of parts
            found = fuzzy_match(sorted(self.get_setting("friends")), attrs, threshold=0.8)
            for f in sorted(found):
                friends |= self.get_setting("friends")[f]
                for friend in self.get_setting("friends")[f]:
                    if friend not in attrs:
                        attrs.append(friend)

            # I would really expect only one attribute, but sometimes in the part list
            # attributes will be subsets of others, so I need to aggregate
            for attr in attrs:
                for cat,partlist in self.attrmap.items():
                    if fuzzy_match(attr,cat) > 0.6 or attr in cat:
                        parts += partlist

        # If we don't have a category for the item, special treatment applies
        if len(parts) == 0:
            tried_skumatch = False
            if 'dimensions' in item_components and 'length' in item_components:
                # if there are dimensions and length, it's probably a board
                parts = self.board_parts
            elif 'dimensions' in item_components:
                # if there are dimensions and length, it's probably a board
                parts = self.dim_parts + self.board_parts
            else:
                # I no longer have a SKU-only list so try the whole database
                matches = self.try_sku_match(item_desc)
                if len(matches) > 0:
                    return self.sort_matches(item, matches)
                tried_skumatch = True

        # add any keyword parts to the list
        keywords = self._has_keyword(item_components)
        if keywords:
            for k in keywords:
                if k not in self.attrs:
                    parts += self.keyword_parts[k]

        # If this is hardware, don't look at any lumber.
        words = attrs if attrs is not None else [] + item_components["other"] if "other" in item_components else []
        is_hardware = words and fuzzy_match(self.hardware_terms, words, threshold=0.6)
        if is_hardware:
            cats = [c for c in self.lumber_categories if attrs is None or c not in attrs]
            parts = self._deselect_parts(parts, cats)

        matches = []
        confidence_scores = []
        parts = self._unique_parts(parts)
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
            if (self.debug_item == self.current_item and part_number.lower() in self.debug_part):
                print(f"item_components={item_components}")
                print(f"db_components={db_components}")
                print(f"match_score={match_score}")
                pdb.set_trace()
                match_score = self._calculate_lumber_match_score(item_components, db_components, sku=part_number, sell_by_foot=sell_by_foot, indent=3)
            # if the match was to a friend category, deduct points.
            if part['attr'] in friends:
                match_score += friends[part['attr']]

            length = ""
            if sell_by_foot and 'sell_by_foot' in db_components:
                length = db_components['sell_by_foot']
                del db_components['sell_by_foot']
                if 'sell_by_foot' in item_components:
                    del item_components['sell_by_foot']

            if match_score >= self.scoring["match-threshold"]:  # Threshold for considering it a match
                self._add_match(matches, item_desc, part, match_score, length)
                if self.debug:
                    print(f"    MATCH: {part_number} -> {part_desc} (score: {match_score:.2f})")

        matches = self.sort_matches(item, matches)
        if self.debug:
            print(f"  Found {len(sorted_matches)} matches")
        
        if use_original:
            # see if we get a better answer with the original text
            new_matches = self.match_lumber_item(item, True)
            if len(matches) == 0 or (len(new_matches) > 0 and matches[0].score <= new_matches[0].score):
                matches = new_matches

        # if we are doing variable length, set the quantity to the qty x length
        if len(matches) > 0 and self.merged_database[matches[0].part_number].get("Stocking Multiple") == "LF":
            lf = matches[0].lf
            qty = item.quantity
            found = []
            for o in item_components["other"] if "other" in item_components else []:
                if re.match(r"[0-9\"'-]*", o) and '/' in o:
                    found.append(re.sub(r'[^0-9/\.\-]', '', o))

            if re.match(r"[0-9][0-9]*/[0-9][0-9/\.\-]*", lf):
                found.append(lf)
                lf = ""
                qty = ""

            if '/' in qty:
                if not found:
                    # This shouldn't happen. set the quantity to *something*
                    qty = qty[:qty.find('/')]
                    qty = qty if qty != "" else "1"
                else:
                    lf = ""     # if this was not empty, we don't know what to do with it anyway
                    qty = ""    # we'll use the found list
            
            if qty != "":
                if lf == "": 
                    # no length was set so assume quantity is the length
                    found.insert(0, f"1/{qty}")
                else:
                    found.insert(0, f"{qty}/{re.sub(r'[^0-9/\.\-]', '', lf)}")

            item.quantity = ", ".join(found)

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

    def find_all_matches(self, scanned_items, output_dir: str = ".") -> None:
        """Find matches using original keyword-based matching (for comparison)"""
        if os.getenv("DEBUG_ITEM"):
            di = os.getenv("DEBUG_ITEM")
            self.debug_item = None
            self.debug_part = []
            while len(di) > 0:
                c = di.find(':')
                s = di[:c] if c > 0 else di
                di = di[c+1:] if c > 0 else ""
                if self.debug_item is None:
                    self.debug_item = int(s)
                else:
                    self.debug_part.append(s.lower())

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

            item.matches = self.match_lumber_item(item, use_original=False)

            if self.debug:
                if item.matches:
                    for match in item.matches:
                        print(f"  ✓ {match.confidence.upper():8s} | {match.part_number:15s} | {match.database_description[:80]}...")
                else:
                    print("  ✗ No matches found")
        
        print(f"\n✓ Detailed matching log saved to: {debug_log_file}")
