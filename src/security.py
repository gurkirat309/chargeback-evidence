"""
LLM input guard (defense-only). Evidence text can be attacker-influenced in
production (a fraudster's dispute reason, a support message), so a jailbreak in
that text could try to hijack the letter generator. This is a lightweight screen
that flags and withholds injected instructions BEFORE they reach the model.

It is a no-op on RokdaDaav's own templated evidence (which never contains these
patterns), so the normal letter output is unchanged — it only bites on attacker
text. Measured by src/llm_eval.py.
"""
from __future__ import annotations

import re

# Instruction-injection / jailbreak patterns (case-insensitive).
INJECTION_PATTERNS = [
    r"ignore\s+(all|the|any|previous|prior|above|earlier)",
    r"disregard\s+(all|the|any|previous|prior|above|instructions)",
    r"\b(system|assistant)\s*:",
    r"</?\s*(system|instructions?|prompt)\s*>",
    r"new\s+instructions?\b",
    r"\byou\s+(must|should|are required to|will now)\b",
    r"instead[, ]+(output|write|say|approve|claim|assert)",
    r"\boverride\b",
    r"do\s+not\s+(verify|check|strip|question)",
    r"reveal\s+(the\s+)?(system|prompt|instructions)",
    r"\bjailbreak\b",
    r"pay\s+the\s+merchant\s+in\s+full\s+regardless",
]
_RX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

WITHHELD = "[evidence text withheld — failed injection/safety screen]"


def scan(text: str) -> bool:
    """True if the text looks like an injected instruction."""
    return bool(_RX.search(text or ""))


def sanitize(text: str):
    """Return (clean_text, flagged). If injection-like, withhold the whole
    statement so the model never sees the payload."""
    if scan(text):
        return WITHHELD, True
    return text, False
