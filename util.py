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
