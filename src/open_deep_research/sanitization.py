"""Input sanitization and prompt injection defense."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"override\s+(all\s+)?instructions",
    r"new\s+instructions?:",
    r"system\s*:\s*you\s+are",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+if\s+you",
    r"pretend\s+you\s+are",
    r"roleplay\s+as",
    r"from\s+now\s+on\s+you",
    r"###?\s*system\s*:",
    r"###?\s*assistant\s*:",
    r"###?\s*human\s*:",
    r"###?\s*instruction\s*:",
    r"<\|system\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
    r"\[system\]",
    r"\[INST\]",
    r"<<SYS>>",
    r"<system>.*?</system>",
    r"<instructions>.*?</instructions>",
    r"<override>.*?</override>",
    r"base64\s*decode",
    r"rot13\s*decode",
    r"hex\s*decode",
    r"send\s+(all\s+)?data\s+to",
    r"exfiltrate",
    r"upload\s+(all\s+)?context",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"repeat\s+(your\s+)?(system\s+)?prompt",
    r"what\s+(are|is)\s+your\s+(system\s+)?instructions",
    r"show\s+(me\s+)?your\s+(system\s+)?prompt",
    r"print\s+(your\s+)?(system\s+)?prompt",
    r"output\s+(your\s+)?(system\s+)?prompt",
    r"you\s+are\s+an?\s+unfiltered",
    r"you\s+have\s+no\s+restrictions",
    r"you\s+can\s+do\s+anything",
    r"without\s+(any\s+)?limitations",
    r"without\s+(any\s+)?restrictions",
    r"IMPORTANT\s*:?\s*ignore",
    r"NOTE\s*:?\s*ignore",
    r"WARNING\s*:?\s*ignore",
    r"SECURITY\s*:?\s*ignore",
    r"ADMIN\s*:?\s*ignore",
    r"DEBUG\s*:?\s*ignore",
    r"```system",
    r"```prompt",
    r"```instructions",
]


@dataclass
class SanitizationResult:
    original: str
    sanitized: str
    is_safe: bool
    injection_detected: bool
    patterns_matched: List[str]
    confidence: float = 1.0


class ContentSanitizer:
    def __init__(self):
        self._compiled_patterns = [
            (idx, re.compile(pattern, re.IGNORECASE | re.DOTALL))
            for idx, pattern in enumerate(INJECTION_PATTERNS)
        ]

    def sanitize(self, raw_text: str, source_type: str = "web", source_url: str = "") -> SanitizationResult:
        patterns_matched = []
        is_safe = True

        for idx, pattern in self._compiled_patterns:
            if pattern.search(raw_text):
                patterns_matched.append(f"pattern_{idx}")
                is_safe = False

        if self._has_repetition_attack(raw_text):
            patterns_matched.append("repetition_attack")
            is_safe = False

        sanitized = self._wrap_in_isolation(raw_text, source_type, source_url)

        return SanitizationResult(
            original=raw_text, sanitized=sanitized, is_safe=is_safe,
            injection_detected=not is_safe, patterns_matched=patterns_matched,
            confidence=0.0 if not is_safe else 1.0
        )

    def _wrap_in_isolation(self, text: str, source_type: str, source_url: str) -> str:
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        return (
            f'<source_content type="{source_type}" '
            f'url="{source_url}" trusted="false">\n'
            f'{text}\n'
            f'</source_content>'
        )

    def _has_repetition_attack(self, text: str) -> bool:
        for length in range(5, min(50, len(text) // 3)):
            for i in range(len(text) - length * 3):
                chunk = text[i:i + length]
                if chunk * 3 in text:
                    return True
        return False

    def batch_sanitize(self, items: List[Dict]) -> List[SanitizationResult]:
        return [
            self.sanitize(
                raw_text=item["text"],
                source_type=item.get("source_type", "web"),
                source_url=item.get("source_url", "")
            )
            for item in items
        ]
