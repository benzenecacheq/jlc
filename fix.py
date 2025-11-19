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
import platform
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from util import *

'''
class ScannedItem:
    quantity: str
    description: str
    original_text: str
    matches: List[PartMatch]
'''

class ParsedItem:
    def __init__(self):
        self.words = None
        self.dim = None
        self.length = None
        self.attr = None
        self.grade = None
        self.keywords = []
        self.hardware_terms = []

    def print(self):
        print("Parsed Item:")
        if (self.dim):
            print(f"   Dim:      {self.dim}")
        if (self.length):
            print(f"   Length:   {self.length}")
        if (self.attr):
            print(f"   Attr:     {self.attr}")
        if (self.grade):
            print(f"   Grade:    {self.grade}")
        if (self.keywords):
            print(f"   Keywords: {self.keywords}")
        if (self.hardware_terms):
            print(f"   Hardware Terms: {self.hardware_terms}")

class Fixer:
    def __init__(self, matcher):
        self.matcher = matcher
        self.special_categories = matcher.get_setting("fix special categories")

    def _analyze_item(self, description):
        item = ParsedItem()
        item.words = self.matcher._cleanup(description, False)
        length2 = None
        for i,word in enumerate(item.words):
            if self.matcher._looks_like_dimension(word, requirex=True) and item.dim is None:
                item.dim = word
            elif self.matcher._looks_like_length(word, ok1and2=False):
                if item.length is not None:
                    length2 = word
                else:
                    item.length = word
            elif self.matcher._is_attr(word, False):
                item.attr = word
            elif len(word) > 1 and word.startswith('#') and word[1].isdigit():
                item.grade = word
            elif fuzzy_match(word, self.matcher.hardware_terms, threshold = 0.7):
                item.hardware_terms.append(word)
            elif self.matcher._is_keyword(word, False):
                item.keywords.append(word)

        if False: # length2 and item.dim is None:
            item.dim = item.length
            item.length = length2

        if item.attr is None:
            # try to get the category from the description
            if item.attr is None:
                for str, cat in self.matcher.category_indicators.items():
                    if (str in description or
                        fuzzy_match(str, item.words, threshold=0.7)):
                        item.attr = [cat]
                        break

        return item

    def _insert_after_dim(self, insert_this, description):
        words = self.matcher._cleanup(description, False)
        for i, word in enumerate(words):
            if self.matcher._looks_like_dimension(word, requirex=True):
                words.insert(i+1, insert_this)
                return " ".join(words)
        return description

    def _get_special(self, category):
        return self.special_categories.get(category)

    def _fix_okay(self, pitem, prev):
        if prev.attr:
            special = self._get_special(prev.attr)
            if special:
                for name,data in special.items():
                    if name in pitem.__dict__:
                        found = False
                        if len(data) == 0:
                            # this means we shouldn't carry forward
                            return False
                        for d in data:
                            if name == "dim" and self.matcher._dimensions_match(d, pitem.dim):
                                found = True
                            elif name == "length" and self.matcher._lengths_equal(d, pitem.length):
                                found = True
                            elif d == pitem.__dict__[name]:
                                found = True
                            if found:
                                break
                        if not found:
                            return False
        return True

    def _best_match(self, values, words):
        i = 0
        best = [None, 0.65, None] # [ value, score, offset ]
        while i < len(words):
            word = words[i]
            next = words[i+1] if i < len(words)-1 else ""
            # by putting word in a list, it will return strings from the values list rather than the word
            matches = fuzzy_match(values, [word], threshold=best[1]+0.00001)
            if matches:
                # pick best score from the list
                best = sorted(matches.keys(), key=lambda x: -matches[x])[0]
                best = [best, matches[best], i]
            if next != "":
                matches = fuzzy_match(values, [word+next], threshold=best[1]+0.00001)
                if matches:
                    # pick best score from the list
                    best = sorted(matches.keys(), key=lambda x: -matches[x])[0]
                    best = [best, matches[best], -1-i]    # neg index means delete two words
            i += 1

        if best[0] is not None:
            if best[2] < 0:
                best[2] = -1-best[2]
                words[best[2]+1] = "*****"
            words[best[2]] = "*****"

        return best[0]

    def fix_tji(self, item, prev):
        desc = do_subs("tji", item.description.lower()).strip()
        desc = self.matcher._cleanup(desc, False)
        
        # first look for a height.  It should be 9-1/2, 11-7/8, 14, 16, or 18
        heights = self.matcher.tji_heights
        height = self._best_match(heights, desc)
        if height is None and item.quantity is not None:
            height = self._best_match(heights, [item.quantity])
            if height is not None:
                item.quantity = ""
        
        # now look for a width.
        widths = self.matcher.tji_widths
        width = self._best_match(widths, desc)

        # look for a length, which would be something that's not the width or the height
        length = None
        found = None
        for i,word in enumerate(desc):
            if self.matcher._looks_like_length(word):
                length = word
            elif word == "tji" and not found:
                found = i
            if length is not None and found is not None:
                break
        if found is not None:
            del desc[found]

        pitem = ParsedItem()
        pitem.attr = "tji"
        if width is None:
            # see if prev looks similar
            if prev is not None and  prev.attr == pitem.attr:
                pitem.grade = prev.grade
        else:
            pitem.grade = "#" + width
        pitem.dim = height
        pitem.length = length

        item.description  = (pitem.dim + " tji ") if pitem.dim is not None else "tji "
        item.description += (pitem.grade + " ") if pitem.grade is not None else ""
        desc = [d for d in desc if d != "*****"]
        item.description += " ".join(desc)

        return pitem

    def fix_missing_specifications(self, scanned_items: List[ScannedItem], debug_item=None):
        fixed_items = []

        prev = None
        item_number = 0
        while item_number < len(scanned_items):
            item = scanned_items[item_number]
            item_number += 1
            if item_number == debug_item: 
                pdb.set_trace()

            # for some reason, with the latest prompt, I'm seeing the quantity being missed
            if False: # item.quantity == "" and item.description[0].isdigit():
                words = item.description.split()
                if words[0].isdigit():
                    item.quantity = words[0]
                    item.description = " ".join(words[1:])

            save_desc = item.description
            item.description = item.description.replace('|','').replace('@','')
            if "TJI" in item.description or " TJT " in item.description:
                pitem = self.fix_tji(item, prev)
            else:
                pitem = self._analyze_item(item.description)

            if pitem.attr in self.matcher.hardware_categories:
                prev = None
                continue
            if pitem.hardware_terms:
                # print(f"Item {item_number}: Found hardware terms {pitem.hardware_terms} in {item}")
                prev = None
                continue

            if prev is not None:
                changed = False
                if not pitem.dim:
                    if '*' in save_desc:
                        # we see this in specific forms (see 11.pdf) where they put a line through the x 
                        # for repeated dimensions.
                        pitem.dim = prev.dim
                        star = save_desc.find('*')
                        item.description = f"{save_desc[0:star]} {prev.dim} {save_desc[star+1:]}".strip()
                        changed = True

                if '|' in save_desc or '@' in save_desc:
                    # If there are | or @ it means there's something we're expected to get from the previous item
                    # in theory the locations of these
                    # for now, just copy everything that's missing from the previous answer because Claude is not
                    # that good at tell us where the missing pieces are
                    for n,data in prev.__dict__.items():
                        if pitem.__dict__[n] is None and prev.__dict__[n] is not None:
                            pitem.__dict__[n] = prev.__dict__[n]
                            if n == "dim": 
                                # put the dim in front
                                item.description = data + " " + item.description
                            elif n == "length":
                                # put it after the dim, if present
                                item.description = self._insert_after_dim(data, item.description)
                            elif n == "keywords":
                                for keyword in prev.keywords:
                                    item.description += " " + keyword
                            else:
                                # just put it at the end
                                item.description += " " + data
                            changed = True

                if pitem.dim and pitem.length and pitem.attr is None:
                    if self._fix_okay(pitem, prev):
                        # for now just put the attr at the end
                        item.description += " " + prev.attr
                        if prev.grade is not None:
                            item.description += " " + prev.grade
                        changed = True

                if pitem.dim is None and pitem.length and pitem.attr:
                    # it's a lumber item with a length and no dimensions.  Get those from the previous
                    if self._fix_okay(prev, pitem):     # reverse order of dimension check
                        pitem.dim = prev.dim
                        item.description = f"{pitem.dim} {item.description}"
                            
                if pitem.length is not None and pitem.dim is None and pitem.attr is None and pitem.grade is None:
                    # All we have is length.  This is rare but let's assume that everything else carries forward
                    pitem.dim = prev.dim
                    pitem.attr = prev.attr
                    pitem.grade = prev.grade
                    if not pitem.keywords:
                        pitem.keywords = prev.keywords
                    newdesc = f"{pitem.dim} {pitem.attr}"
                    if pitem.grade:
                        newdesc += " " + pitem.grade
                    for keyword in pitem.keywords:
                        newdesc += " " + keyword
                    item.description = f"{newdesc} {item.description}"

                if changed:
                    item.description = item.description.replace('|','').replace('@','').strip()
                    pitem = self._analyze_item(item.description)

            if pitem.dim and pitem.length and pitem.attr:
                prev = pitem
            else:
                prev = None

            if save_desc != item.description:
                print(f"Fixed item {item_number}: {save_desc} -> {item.description}")
