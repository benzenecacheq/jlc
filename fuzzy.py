def fuzzy_match(str1, str2, threshold_ratio=0.6):
    """
    Perform fuzzy matching between two strings with length-based forgiveness.
    
    Args:
        str1, str2: Strings to compare
        threshold_ratio: Base similarity ratio (0.0 to 1.0)
    
    Returns:
        bool: True if strings are considered a match
    """
    # Handle exact matches and very short strings
    if str1 == str2:
        return True
    
    # Convert to lowercase for comparison
    s1, s2 = str1.lower(), str2.lower()
    
    # Calculate Levenshtein distance
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
    max_len = max(len(s1), len(s2))
    distance = levenshtein_distance(s1, s2)
    similarity = 1 - (distance / max_len)
    
    # Make threshold more forgiving for longer strings
    avg_len = (len(s1) + len(s2)) / 2
    if avg_len > 10:
        adjusted_threshold = threshold_ratio * 0.5   # More forgiving
    elif avg_len > 6:
        adjusted_threshold = threshold_ratio * 0.7   # Slightly more forgiving
    else:
        adjusted_threshold = threshold_ratio         # Standard threshold
    
    return similarity >= threshold_ratio

# Example usage and test cases
if __name__ == "__main__":
    test_cases = [
        ("durastrand", "durastrond", True),   # Close match
        ("durastrand", "duraslvrd", True),    # More different but still similar
        ("ab", "ac", False),                  # Short strings, not identical
        ("ab", "abc", True),                   # Short strings, identical
        ("ab", "acb", True),                   # Short strings, identical
        ("ab", "cba", False),                   # Short strings, identical
        ("a", "b", False),                    # Very short, different
        ("hello", "helo", True),              # Missing letter
        ("programming", "programing", True),   # Common typo
        ("cat", "dog", False),                # Completely different
        ("allthreaded","ALLTHRD", True),      # I'd like this to be true
    ]
    
    for str1, str2, expected in test_cases:
        result = fuzzy_match(str1, str2)
        status = "✓" if result == expected else "✗"
        print(f"{status} fuzzy_match('{str1}', '{str2}') = {result} (expected {expected})")
