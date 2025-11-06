# Corpus cleaning utilities
import html
import unicodedata
import re

def normalize_text_passage(text: str) -> str:
    """
    Simple normalization for passages:
    - unescape HTML entities
    - normalize unicode
    - remove repeated whitespace, common headers/footers heuristics
    - trim and return
    """
    if not isinstance(text, str):
        return ""
    # Unescape & unicode normalize
    t = html.unescape(text)
    t = unicodedata.normalize("NFKC", t)

    # Remove very common header/footer markers (heuristic)
    t = re.sub(r'(?i)(page\s*\d+|confidential|internal use only|refer to the full policy)', ' ', t)

    # Remove sequences of repeated short lines (like ----- or ===)
    t = re.sub(r'[-=]{3,}', ' ', t)

    # Collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()

    return t