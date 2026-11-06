"""Emoji and Emoticon Handling Module (Company Step 2).

Converts emojis and common ASCII emoticons into semantic textual representations
(e.g., '😍🔥' -> 'smiling_face_with_heart_eyes fire').
Preserves semantic meaning for downstream sentiment and topic modeling while
maintaining sentence structure, ordinary text, and punctuation.
"""

import re
from typing import Any
import emoji

# Common ASCII emoticons mapped to readable sentiment tokens
EMOTICON_MAP = {
    ":-)": "smiling_face",
    ":)": "smiling_face",
    ":-D": "grinning_face",
    ":D": "grinning_face",
    ";-)": "winking_face",
    ";)": "winking_face",
    ":-(": "frowning_face",
    ":(": "frowning_face",
    ":'(": "crying_face",
    ":'-(": "crying_face",
    "<3": "red_heart",
    "</3": "broken_heart",
    ":-P": "face_with_tongue",
    ":P": "face_with_tongue",
    ":-p": "face_with_tongue",
    ":p": "face_with_tongue",
    ":-O": "surprised_face",
    ":O": "surprised_face",
    ":-o": "surprised_face",
    ":o": "surprised_face",
    "xD": "laughing_face",
    "XD": "laughing_face",
}

# Compiled regex for emoji token extraction (alphanumerics, underscores, hyphens; no spaces)
_RE_EMOJI_TOKEN = re.compile(r":([a-zA-Z0-9_-]+):")
_RE_MULTIPLE_SPACES = re.compile(r"[^\S\r\n]+")


def handle_emojis(text: Any) -> str:
    """Detects emojis and emoticons and replaces them with descriptive semantic tokens.

    Requirements:
    1. Converts Unicode emojis into textual descriptions (e.g. '😍' -> 'smiling_face_with_heart_eyes').
    2. Converts common emoticons into sentiment tokens (e.g. ':-)' -> 'smiling_face', '<3' -> 'red_heart').
    3. Normalizes hyphens in emoji names to underscores.
    4. Handles multiple consecutive emojis and emoticons in the same sentence.
    5. Normalizes whitespace introduced by emoji translation without destroying line breaks.
    6. Preserves ordinary text, numbers, casing, and punctuation.

    Returns:
        String with emojis and emoticons replaced by readable tokens.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return ""

    result = text

    # 1. Translate ASCII emoticons when isolated by whitespace or boundaries
    # Sort by length descending to match longer patterns first (e.g., ':-)' before ':)')
    for emoticon, token in sorted(EMOTICON_MAP.items(), key=lambda item: -len(item[0])):
        pattern = rf"(?<!\S){re.escape(emoticon)}(?!\S)"
        result = re.sub(pattern, f" {token} ", result)

    # 2. Demojize Unicode emojis (e.g. '😍' -> ':smiling_face_with_heart-eyes:')
    demojized = emoji.demojize(result)

    # 3. Clean up colon wrappers and normalize hyphens to underscores:
    # ':smiling_face_with_heart-eyes:' -> ' smiling_face_with_heart_eyes '
    def _clean_token(match: re.Match) -> str:
        raw_name = match.group(1).replace("-", "_")
        return f" {raw_name} "

    result = _RE_EMOJI_TOKEN.sub(_clean_token, demojized)

    # 4. Normalize spaces introduced by demojization
    result = _RE_MULTIPLE_SPACES.sub(" ", result)

    # Clean leading and trailing spaces per line
    lines = [line.strip() for line in result.split("\n")]
    return "\n".join(lines).strip()
