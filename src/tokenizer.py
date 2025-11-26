import logging
import unicodedata

_CHARSET_STR = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜẞabcdefghijklmnopqrstuvwxyzäöüß"
_PUNCT_LIST = [
    "!",
    '"',
    "(",
    ")",
    ",",
    "-",
    ".",
    "/",
    ":",
    ";",
    "?",
    "[",
    "]",
    "{",
    "}",
    "«",
    "»",
    "‒",
    "–",
    "—",
    "'",
    "‚",
    '"',
    "„",
    "‹",
    "›",
]

def normalize_unicode_text(text: str) -> str:
    if not unicodedata.is_normalized("NFC", text):
        text = unicodedata.normalize("NFC", text)

    return text

def any_locale_text_preprocessing(text: str) -> str:
    res = []
    for c in normalize_unicode_text(text):
        if c in ['’']:
            res.append("'")
        else:
            res.append(c)

    return ''.join(res)

def tokenize_german(
    text: str, punct: bool = True, apostrophe: bool = True, pad_with_space: bool = True
) -> list[int]:
    """Tokenize German text into a list of integer token IDs.

    Args:
        text: Input text to tokenize
        punct: Whether to include punctuation tokens
        apostrophe: Whether to include apostrophe token
        pad_with_space: Whether to pad with spaces at start and end

    Returns:
        List of integer token IDs
    """

    tokens = []
    tokens.append(" ")  # Space at index 0
    tokens.extend(_CHARSET_STR)
    if apostrophe:
        tokens.append("'")
    if punct:
        tokens.extend(_PUNCT_LIST)

    tokens.append("<pad>")
    tokens.append("<blank>")
    tokens.append("<oov>")

    token2id = {token: i for i, token in enumerate(tokens)}
    space = " "

    text = any_locale_text_preprocessing(text)

    # Encode
    cs = []
    tokens_set = set(tokens)

    for c in text:
        if c == space and len(cs) > 0 and cs[-1] != space or (c.isalnum() or c == "'") and c in tokens_set or (c in _PUNCT_LIST) and punct:  # noqa: E501
            cs.append(c)
        elif c != space:
            logging.warning(
                f"Text: [{text}] contains unknown char: [{c}]. Symbol will be skipped."
            )

    if cs:
        while cs and cs[-1] == space:
            cs.pop()

    if pad_with_space:
        cs = [space] + cs + [space]

    return [token2id[p] for p in cs]
