from __future__ import annotations

import re
from typing import Iterable


def mask_question_template(question: str, entities: Iterable[str] | None = None) -> str:
    masked = question or ""
    for ent in sorted({str(e).strip() for e in (entities or []) if str(e).strip()}, key=len, reverse=True):
        if len(ent) >= 3:
            masked = re.sub(re.escape(ent), " ENT ", masked, flags=re.IGNORECASE)

    masked = re.sub(r"'[^']+'|\"[^\"]+\"", " ENT ", masked)
    masked = re.sub(r"\b\d+(?:[.,:/-]\d+)*\b", " NUM ", masked)

    stop = {
        "What", "Which", "Who", "Whom", "Whose", "Where", "When", "How",
        "Did", "Does", "Do", "Is", "Are", "Was", "Were", "The", "A", "An",
        "In", "On", "Of", "For", "To", "By", "From", "With", "And", "Or", "Both",
    }

    def replace_title_span(match: re.Match) -> str:
        text = match.group(0)
        tokens = text.split()
        if len(tokens) == 1 and tokens[0] in stop:
            return text
        return " ENT "

    masked = re.sub(
        r"\b[A-Z][\w\-]*(?:\s+(?:of|the|and|de|da|du|del|la|le|[A-Z][\w\-]*))*\b",
        replace_title_span,
        masked,
    )
    masked = masked.lower()
    masked = re.sub(r"[^a-z0-9]+", " ", masked)
    return re.sub(r"\s+", " ", masked).strip()
