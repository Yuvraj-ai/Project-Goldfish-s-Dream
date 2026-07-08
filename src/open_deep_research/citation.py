"""Citation formatting engine — supports 6 academic/professional styles."""

import logging
from datetime import datetime
from typing import List, Literal

logger = logging.getLogger(__name__)


class CitationEntry:
    """Parsed citation metadata."""

    def __init__(
        self,
        authors: List[str],
        title: str,
        year: int | None,
        url: str,
        doi: str | None,
        source_type: str,
        publisher: str,
    ):
        """Initialize citation entry with metadata."""
        self.authors = authors
        self.title = title
        self.year = year
        self.url = url
        self.doi = doi
        self.source_type = source_type
        self.publisher = publisher


class CitationFormatter:
    """Format citations in various academic/professional styles."""

    def __init__(self, style: Literal["vanilla", "apa", "mla", "chicago", "harvard", "ieee"] = "vanilla"):
        """Initialize citation formatter with style."""
        self.style = style
        logger.debug("CitationFormatter initialized style=%s", style)

    def _parse_source(self, source: dict) -> CitationEntry:
        """Parse Source dict into CitationEntry."""
        date_str = source.get("date", "")
        year = None
        if date_str:
            try:
                year = int(date_str[:4])
            except (ValueError, IndexError):
                logger.warning(
                    "could not parse year from date=%r url=%s",
                    date_str, source.get("url", "")
                )
        if not source.get("title"):
            logger.warning(
                "source missing title url=%s", source.get("url", "")
            )
        return CitationEntry(
            authors=[],
            title=source.get("title", "Untitled"),
            year=year,
            url=source.get("url", ""),
            doi=source.get("doi"),
            source_type=source.get("source_type", "web"),
            publisher=source.get("publisher", ""),
        )

    def format_inline(self, source: dict) -> str:
        """Format inline citation."""
        logger.debug("format_inline style=%s url=%s", self.style, source.get("url", ""))
        entry = self._parse_source(source)
        if self.style == "vanilla":
            return f"({entry.publisher or 'Unknown'}, {entry.year or 'n.d.'})"
        elif self.style == "apa":
            author = entry.authors[0] if entry.authors else "Unknown"
            return f"({author}, {entry.year or 'n.d.'})"
        elif self.style == "mla":
            author = entry.authors[0] if entry.authors else "Unknown"
            return f"({author} {entry.year or 'n.d.'})"
        elif self.style == "chicago":
            author = entry.authors[0] if entry.authors else "Unknown"
            return f"({author} {entry.year or 'n.d.'}, {entry.url})"
        elif self.style == "harvard":
            author = entry.authors[0] if entry.authors else "Unknown"
            return f"({author} {entry.year or 'n.d.'})"
        elif self.style == "ieee":
            return f"[{source.get('_index', '?')}]"
        else:
            logger.warning(
                "unknown style=%s, using vanilla fallback for inline", self.style
            )
            return f"({entry.publisher or 'Unknown'}, {entry.year or 'n.d.'})"

    def format_bibliography(self, sources: List[dict]) -> str:
        """Format full bibliography."""
        logger.info(
            "format_bibliography entry style=%s sources=%d", self.style, len(sources)
        )
        if not sources:
            logger.warning("format_bibliography received empty sources")
        if self.style == "apa":
            return self._format_apa_bibliography(sources)
        elif self.style == "ieee":
            return self._format_ieee_bibliography(sources)
        else:
            if self.style not in ("vanilla", "mla", "chicago", "harvard"):
                logger.warning(
                    "unknown style=%s, using vanilla bibliography fallback",
                    self.style
                )
            return self._format_vanilla_bibliography(sources)

    def _format_vanilla_bibliography(self, sources: List[dict]) -> str:
        """Format a simple numbered bibliography."""
        logger.debug("formatting vanilla bibliography sources=%d", len(sources))
        lines = []
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            publisher = source.get("publisher", "")
            date = source.get("date", "n.d.")
            lines.append(f"{i}. {title}. {publisher}, {date}. {url}")
        return "\n".join(lines)

    def _format_apa_bibliography(self, sources: List[dict]) -> str:
        """APA-style bibliography."""
        logger.debug("formatting APA bibliography sources=%d", len(sources))
        lines = []
        for source in sources:
            entry = self._parse_source(source)
            author = entry.authors[0] if entry.authors else "Unknown"
            year = entry.year or "n.d."
            lines.append(f"{author} ({year}). {entry.title}. {entry.publisher}. {entry.url}")
        return "\n".join(lines)

    def _format_ieee_bibliography(self, sources: List[dict]) -> str:
        """IEEE numbered bibliography."""
        logger.debug("formatting IEEE bibliography sources=%d", len(sources))
        lines = []
        for i, source in enumerate(sources, 1):
            entry = self._parse_source(source)
            lines.append(f"[{i}] {entry.title}. {entry.publisher}, {entry.year or 'n.d.'}. {entry.url}")
        return "\n".join(lines)

    def export_bibtex(self, sources: List[dict]) -> str:
        """Export sources as BibTeX entries."""
        logger.info("export_bibtex entry sources=%d", len(sources))
        if not sources:
            logger.warning("export_bibtex received empty sources")
        entries = []
        for i, source in enumerate(sources, 1):
            entry = self._parse_source(source)
            key = f"source{i}"
            authors = " and ".join(entry.authors) if entry.authors else "Unknown"
            entries.append(
                f"@misc{{{key},\n"
                f"  title = {{{entry.title}}},\n"
                f"  author = {{{authors}}},\n"
                f"  year = {{{entry.year or 'n.d.'}}},\n"
                f"  url = {{{entry.url}}},\n"
                f"  note = {{Accessed: {datetime.now().strftime('%Y-%m-%d')}}}\n"
                f"}}"
            )
        logger.info("export_bibtex complete: %d entries", len(entries))
        return "\n\n".join(entries)
