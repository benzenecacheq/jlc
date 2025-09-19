import os
import re

from dataclasses import dataclass
from typing import List
from pathlib import Path

###############################################################################
@dataclass
class PartMatch:
    """Represents a matched part"""
    description: str
    part_number: str
    database_name: str
    database_description: str
    score: float
    confidence: str     # "exact", "partial", "similar" or float
    part_idx: int = -1  # not set in AI match
    lf: str = ""        # linear feet if the match is for that kind of lumber

###############################################################################
@dataclass
class ScannedItem:
    """Represents an item from the scanned list"""
    quantity: str
    description: str
    original_text: str
    matches: List[PartMatch]

###############################################################################
def fuzzy_match(arg1, arg2, is_dimension=False, new_algo=True):
    """
    Perform fuzzy matching between two strings with length-based forgiveness.
    
    Args:
        arg1 is either a list or a string
        str1 is a string
    
    Returns:
        Closeness metric where 1 is equal
    """
    # if there is a list make it arg1; this works if arg1 is also a list
    if type(arg2) == type([]):
        save = arg2
        arg2 = arg1
        arg1 = save
    if type(arg1) == type([]):
        # if the first arg is a list, return a sorted list of matching words
        found = []
        for s in arg1:
            score = fuzzy_match(arg2, s) # arg2 could also be a list
            if score > 0:
                found.append([s,score])
        return sorted(found, key=lambda x: -x[1])
    str1 = arg1
    str2 = arg2

    # Handle exact matches and very short strings
    if str1 == str2:
        return 1.0
    
    # Convert to lowercase and remove non-alphanumeric for comparison
    if is_dimension:
        # keep fractions
        s1, s2 = re.sub(r'[^a-zA-Z0-9/\-]', '', str1).lower(), re.sub(r'[^a-zA-Z0-9/\-]', '', str2).lower()
    else:
        s1, s2 = re.sub(r'[^a-zA-Z0-9/\-]', '', str1).lower(), re.sub(r'[^a-zA-Z0-9/\-]', '', str2).lower()
    if len(s1) == 0 | len(s2) == 0:
        return 0
    
    # Calculate Levenshtein distance
    def damerau_levenshtein_distance(a, b):
        """Damerau-Levenshtein distance with transposition support"""
        len_a, len_b = len(a), len(b)
        
        # Create a dictionary of unique characters
        char_array = list(set(a + b))
        char_dict = {char: i for i, char in enumerate(char_array)}
        
        # Create the distance matrix
        max_dist = len_a + len_b
        H = {}
        H[-1, -1] = max_dist
        
        # Initialize first row and column
        for i in range(0, len_a + 1):
            H[i, -1] = max_dist
            H[i, 0] = i
        for j in range(0, len_b + 1):
            H[-1, j] = max_dist
            H[0, j] = j
        
        # Track last occurrence of each character
        last_match_col = {char: 0 for char in char_dict}
        
        for i in range(1, len_a + 1):
            last_match_row = 0
            for j in range(1, len_b + 1):
                i1 = last_match_col[b[j-1]]
                j1 = last_match_row
                cost = 1
                if a[i-1] == b[j-1]:
                    cost = 0
                    last_match_row = j
                
                H[i, j] = min(
                    H[i-1, j] + 1,      # deletion
                    H[i, j-1] + 1,      # insertion
                    H[i-1, j-1] + cost, # substitution
                    H[i1-1, j1-1] + (i-i1-1) + 1 + (j-j1-1)  # transposition
                )
            
            last_match_col[a[i-1]] = i
        
        return H[len_a, len_b]
    def levenshtein_distance(a, b):
        if len(a) < len(b):
            a, b = b, a
        
        if len(b) == 0:
            return len(a)
        
        previous_row = list(range(len(b) + 1))
        for i, c1 in enumerate(a):
            current_row = [i + 1]
            for j, c2 in enumerate(b):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    # Calculate similarity ratio
    if new_algo:
       distance = damerau_levenshtein_distance(s1, s2)
    else:
       distance = levenshtein_distance(s1, s2)

    max_len = max(len(s1), len(s2))
    similarity = 1 - (distance / max_len)
    return similarity
    
###############################################################################
submap = {}
def do_subs(subtype, word):
    global submap
    if subtype not in submap:
        exe_dir = str(Path(__file__).parent)
        fn = exe_dir + "/" + subtype + ".subs"
        if not os.path.exists(fn):
            print(f"Missing substitution file {fn}", file=sys.stderr)
            exit(1)
        subs = []
        with open(fn, "r", encoding='utf-8') as f:
            for line in f:
                if line[-1] == '\n':
                   line = line[:-1]
                if "`" not in line:     # comment line
                    continue
                subs.append(line.split("`", 1))
        submap[subtype] = subs
    subs = submap[subtype]
    for sub in subs:
       word = re.sub(sub[0], sub[1], word)
    
    return word

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
import anthropic
def call_ai_with_retry(client, messages, max_tokens=8000, max_retries=3, model=None):
     """Call AI API with retry logic (works with both Anthropic and OpenAI)"""
     import time
     
     # Set default model
     if model is None:
         model = "claude-sonnet-4-20250514"
     
     for attempt in range(max_retries):
         try:
             if model.startswith("claude"):
                 response = client.messages.create(
                     model=model,
                     max_tokens=max_tokens,
                     messages=messages
                 )
                 return response
             else: # assume openai for now
                 response = client.chat.completions.create(
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
 
###############################################################################
def count_tokens(client, text: str) -> int:
    """Count tokens in text using Anthropic's tokenizer"""
    try:
        return client.count_tokens(text)
    except Exception as e:
        # Fallback to rough estimate if counting fails
        return len(text) // 4
