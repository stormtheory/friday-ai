# utils/fuzzy.py

import difflib

def match_command(user_input, keywords, cutoff=0.7): # was 0.8
    matches = difflib.get_close_matches(user_input.lower(), keywords, n=1, cutoff=cutoff)
    return matches[0] if matches else None
