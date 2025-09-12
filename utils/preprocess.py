import spacy
from collections import OrderedDict

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def filter_similar_commentary(comments, threshold=0.8):
    """
    Filters out redundant and similar commentary using NLP similarity.
    
    Args:
        comments (list): List of commentary sentences.
        threshold (float): Similarity threshold (default 0.8).
    
    Returns:
        str: Filtered commentary as a single string.
    """
    seen = OrderedDict()
    final_commentary = []

    for comment in comments:
        comment_cleaned = comment.strip().lower()
        if len(comment_cleaned.split()) < 3:  # Ignore very short comments
            continue  

        # Check if a similar comment already exists
        is_similar = False
        for existing in seen:
            if nlp(existing).similarity(nlp(comment_cleaned)) > threshold:
                is_similar = True
                break

        if not is_similar:
            seen[comment_cleaned] = True
            final_commentary.append(comment)

    return " ".join(final_commentary)  # Return cleaned commentary
