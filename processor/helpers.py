import re
from typing import Optional


def normalize_whitespace(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def parse_float_from_text(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text.replace(",", ""))
    return float(m.group(1)) if m else None

