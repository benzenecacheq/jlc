def fuzzy_match(str1, str2):
    """
    Perform fuzzy matching between two strings with length-based forgiveness.
    
    Args:
        str1, str2: Strings to compare
        threshold_ratio: Base similarity ratio (0.0 to 1.0)
    
    Returns:
        bool: True if strings are considered a match
    """

    new_algo = True

    # Handle exact matches and very short strings
    if str1 == str2:
        return 1.0
    
    # Convert to lowercase for comparison
    s1, s2 = str1.lower(), str2.lower()
    
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
    max_len = max(len(s1), len(s2))
    if new_algo:
       distance = damerau_levenshtein_distance(s1, s2)
    else:
       distance = levenshtein_distance(s1, s2)

    similarity = 1 - (distance / max_len)
    return similarity

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
        ("hello", "hello", True),              # Missing letter
        ("hello", "helo", True),              # Missing letter
        ("programming", "programing", True),   # Common typo
        ("cat", "dog", False),                # Completely different
        ("allthreaded","ALLTHRD", True),      # I'd like this to be true
        ("pcz4","pc4z", True),      # I'd like this to be true
        ("tying","titen", True),      # I'd like this to be true
    ]
    
    for str1, str2, expected in test_cases:
        result = fuzzy_match(str1, str2)
        status = "✓" if (result >= 0.6)  == expected else "✗"
        print(f"{status} fuzzy_match('{str1}', '{str2}') = {result} (expected {expected})")
