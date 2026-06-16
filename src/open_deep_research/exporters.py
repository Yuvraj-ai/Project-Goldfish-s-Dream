"""Multi-format export pipeline — Markdown, HTML, PDF, DOCX, JSON, BibTeX."""

import json
from abc import ABC, abstractmethod
from pathlib import Path


class BaseExporter(ABC):
    """Base class for report exporters."""

    @abstractmethod
    async def export(self, markdown: str, metadata: dict, output_path: Path) -> Path:
        """Export report to file. Returns output path."""
        pass


class MarkdownExporter(BaseExporter):
    """Passthrough markdown exporter."""

    async def export(self, markdown: str, metadata: dict, output_path: Path) -> Path:
        """Export report as markdown file."""
        md_path = output_path.with_suffix(".md")
        md_path.write_text(markdown, encoding="utf-8")
        return md_path


class HTMLExporter(BaseExporter):
    """Export to styled HTML."""

    async def export(self, markdown: str, metadata: dict, output_path: Path) -> Path:
        """Export report as styled HTML file."""
        import markdown as md_lib

        html_content = md_lib.markdown(markdown, extensions=["tables", "fenced_code", "toc"])

        styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{metadata.get('title', 'Research Report')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 800px; margin: 0 auto; padding: 2rem; line-height: 1.6; color: #333; }}
        h1 {{ color: #1a1a1a; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.5rem; }}
        h2 {{ color: #2c3e50; margin-top: 2rem; }}
        blockquote {{ border-left: 4px solid #3498db; margin: 1rem 0; padding: 0.5rem 1rem; background: #f8f9fa; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f8f9fa; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

        html_path = output_path.with_suffix(".html")
        html_path.write_text(styled_html, encoding="utf-8")
        return html_path


class PDFExporter(BaseExporter):
    """Export to PDF via weasyprint."""

    async def export(self, markdown: str, metadata: dict, output_path: Path) -> Path:
        """Export report as PDF via weasyprint."""
        html_exporter = HTMLExporter()
        html_path = await html_exporter.export(markdown, metadata, output_path.with_suffix(".tmp"))

        try:
            from weasyprint import HTML

            pdf_path = output_path.with_suffix(".pdf")
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))
            return pdf_path
        finally:
            html_path.unlink(missing_ok=True)


class DOCXExporter(BaseExporter):
    """Export to DOCX via python-docx."""

    async def export(self, markdown: str, metadata: dict, output_path: Path) -> Path:
        """Export report as DOCX via python-docx."""
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc = Document()
        title = doc.add_heading(metadata.get("title", "Research Report"), 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        lines = markdown.split("\n")
        for line in lines:
            if line.startswith("# "):
                doc.add_heading(line[2:], level=1)
            elif line.startswith("## "):
                doc.add_heading(line[3:], level=2)
            elif line.startswith("### "):
                doc.add_heading(line[4:], level=3)
            elif line.startswith("- "):
                doc.add_paragraph(line[2:], style="List Bullet")
            elif line.strip():
                doc.add_paragraph(line)

        docx_path = output_path.with_suffix(".docx")
        doc.save(str(docx_path))
        return docx_path


class JSONExporter(BaseExporter):
    """Export structured JSON with all metadata."""

    async def export(self, markdown: str, metadata: dict, output_path: Path) -> Path:
        """Export report as structured JSON file."""
        structured = {
            "report": {
                "markdown": markdown,
                "title": metadata.get("title", ""),
                "generated_at": metadata.get("generated_at", ""),
                "profile": metadata.get("profile", ""),
                "word_count": len(markdown.split()),
            },
            "metadata": metadata,
        }

        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(structured, indent=2), encoding="utf-8")
        return json_path


class BibTeXExporter(BaseExporter):
    """Export bibliography as BibTeX file."""

    async def export(self, markdown: str, metadata: dict, output_path: Path) -> Path:
        """Export bibliography as BibTeX file."""
        from open_deep_research.citation import CitationFormatter

        sources = metadata.get("sources", [])
        formatter = CitationFormatter("vanilla")
        bibtex_content = formatter.export_bibtex(sources)

        bib_path = output_path.with_suffix(".bib")
        bib_path.write_text(bibtex_content, encoding="utf-8")
        return bib_path


EXPORTER_MAP = {
    "markdown": MarkdownExporter,
    "html": HTMLExporter,
    "pdf": PDFExporter,
    "docx": DOCXExporter,
    "json": JSONExporter,
    "bibtex": BibTeXExporter,
}


def get_exporter(format_name: str) -> BaseExporter:
    """Get exporter instance by format name."""
    cls = EXPORTER_MAP.get(format_name)
    if cls is None:
        raise ValueError(f"Unknown export format: {format_name}")
    return cls()
