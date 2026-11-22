"""Language-independent text comparison for speech-to-text checks."""

import re
import unicodedata

WORD = re.compile(r"\w+")
# Scripts written without spaces between words; there each character counts as a token.
UNSPACED = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯฀-๿]")
DIGIT = re.compile(r"\d")


def strip_accents(text: str) -> str:
    """Drop accents on Latin, Greek and Cyrillic letters only; in kana and Hangul the marks change the letter."""
    out = []
    for c in unicodedata.normalize("NFD", text):
        if unicodedata.combining(c) and out and out[-1] < "԰":
            continue
        out.append(c)
    return unicodedata.normalize("NFC", "".join(out))


def tokens(text: str) -> list[str]:
    """Casefolded words without accents, so "Всё" matches "все" and "é" matches "e"."""
    plain = strip_accents(unicodedata.normalize("NFKC", text).casefold())
    out = []
    for word in WORD.findall(plain):
        out.extend(word if UNSPACED.search(word) else [word])
    return out


def error_rate(ref: str, hyp: str) -> float:
    """Word error rate (character error rate for unspaced scripts)."""
    a, b = tokens(ref), tokens(hyp)
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1] / max(len(a), 1)


def has_digits(text: str) -> bool:
    return bool(DIGIT.search(text))


def max_new_tokens(text: str) -> int:
    """Generation cap with room for slow speech; the 12 Hz codec emits about 12.5 tokens per second."""
    return 4 * len(text) + 60
