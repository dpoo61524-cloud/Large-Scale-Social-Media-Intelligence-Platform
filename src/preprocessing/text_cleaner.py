"""Text Cleaning and Noise Removal Module (Company Step 2).

Provides modular, robust text cleaning for social media posts from Reddit and YouTube.
Strips HTML, URLs, invisible control characters, and social-media formatting artifacts
while strictly preserving meaningful words, sentence boundaries, punctuation, hashtags,
mentions, numbers, and technical terminology.
"""

import html
import re
from typing import Any

# Regular Expressions for social media noise
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_MARKDOWN_LINKS = re.compile(r"\[([^\]]+)\]\((?:https?|ftp)://[^\s)]+\)")
_RE_URLS = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\"']*[a-zA-Z0-9/]")
_RE_INVISIBLE_CHARS = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_RE_REPEATED_CHARS = re.compile(r"([a-zA-Z!?])\1{2,}")
_RE_EXCESS_SPACES = re.compile(r"[^\S\r\n]{2,}")
_RE_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: Any) -> str:
    """Cleans noisy social media text for downstream NLP pipelines.

    Operations:
    1. Safe handling of null, missing, or non-string inputs.
    2. HTML tag removal prior to entity unescaping to preserve literal entity text.
    3. Markdown link flattening ([text](url) -> text).
    4. URL and web link removal.
    5. HTML entity unescaping (&amp; -> &, &lt; -> <, etc.).
    6. Invisible noise and control character removal (zero-width spaces, BOM, etc.).
    7. Normalization of exaggerated letter/punctuation runs (e.g. 'sooooo' -> 'soo', '!!!!' -> '!!').
       Preserves numeric runs like '10,000'.
    8. Whitespace normalization (collapsing multiple spaces and runs of blank lines).

    Preserves:
    - Case sensitivity (for acronyms, named entities).
    - Sentence structure and punctuation (., !, ?, commas, apostrophes).
    - Numbers, currency, percentages, and technical terms (e.g. 'GPT-4', '$100', '99.5%').
    - Social tokens: Hashtags ('#AI') and Mentions ('@username', 'u/username').

    Returns:
        Cleaned, normalized string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return ""

    # 1. Strip raw HTML tags first
    cleaned = _RE_HTML_TAGS.sub(" ", text)

    # 2. Extract inner text from markdown links: [Display Text](https://link.com) -> Display Text
    cleaned = _RE_MARKDOWN_LINKS.sub(r"\1", cleaned)

    # 3. Strip raw URLs and web links
    cleaned = _RE_URLS.sub("", cleaned)

    # 4. Unescape HTML entities (&amp; -> &, &lt;tag&gt; -> <tag>)
    cleaned = html.unescape(cleaned)

    # 5. Remove invisible unicode control characters (zero-width spaces, null bytes)
    cleaned = _RE_INVISIBLE_CHARS.sub("", cleaned)

    # 6. Normalize elongated characters for letters and punctuation only (sooooo -> soo, !!!!! -> !!)
    cleaned = _RE_REPEATED_CHARS.sub(r"\1\1", cleaned)

    # 7. Normalize punctuation spacing (e.g. 'visit .' -> 'visit.')
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)

    # 8. Normalize whitespace while preserving line and sentence breaks
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _RE_EXCESS_NEWLINES.sub("\n\n", cleaned)
    cleaned = _RE_EXCESS_SPACES.sub(" ", cleaned)

    # Trim leading/trailing whitespace across individual lines
    lines = [line.strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines).strip()

    return cleaned
