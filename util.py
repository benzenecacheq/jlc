from dataclasses import dataclass
from typing import List
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


