"""Vietnamese text normalization for ASR evaluation.

Handles Unicode normalization, lowercasing, punctuation removal,
while preserving Vietnamese diacritics.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_vietnamese(text: str) -> str:
    """Normalize Vietnamese text for WER/CER computation.

    Steps:
    1. Unicode NFC normalization (compose diacritics)
    2. Lowercase
    3. Remove punctuation (keep letters, digits, spaces, Vietnamese diacritics)
    4. Collapse whitespace
    5. Strip
    """
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Lowercase
    text = text.lower()

    # Remove punctuation but keep letters (including Vietnamese), digits, and spaces
    # Vietnamese characters are in Latin Extended Additional and Combining Diacritical Marks
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove underscores (matched by \w but not useful)
    text = text.replace("_", " ")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()
