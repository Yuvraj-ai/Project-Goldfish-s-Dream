"""Document parsing and agentic reading for uploaded research documents."""
import logging
import os
from typing import List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Metadata extracted from a document."""
    title: str = ""
    author: str = ""
    creation_date: str | None = None
    page_count: int = 0
    file_size: int = 0
    mime_type: str = "application/pdf"


class DocumentSection(BaseModel):
    """A section extracted from a document."""
    id: str = ""
    heading: str = ""
    level: int = 1
    content: str = ""
    page_start: int = 0
    page_end: int = 0


class DocumentArtifact(BaseModel):
    """A fully parsed document with sections, tables, and metadata."""
    file_path: str = ""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    sections: List[DocumentSection] = Field(default_factory=list)
    tables: List[dict] = Field(default_factory=list)
    images: List[dict] = Field(default_factory=list)
    raw_text: str = ""
    parse_confidence: float = 1.0


class DocumentReader:
    """Reads and parses PDF documents into structured artifacts."""

    def __init__(self, use_grobid: bool = False):
        self.use_grobid = use_grobid

    async def parse_pdf(self, file_path: str) -> DocumentArtifact:
        """Parse a PDF file into a DocumentArtifact."""
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return DocumentArtifact(file_path=file_path, parse_confidence=0.0)

        try:
            import pymupdf
            doc = pymupdf.open(file_path)
            sections = []
            raw_text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                raw_text_parts.append(text)

                # Simple section extraction based on font size
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block["type"] == 0:  # text block
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                if span["size"] > 14 and span["flags"] & 2**4:
                                    # Likely a heading (large font, bold)
                                    sections.append(DocumentSection(
                                        id=f"p{page_num}_s{len(sections)}",
                                        heading=span["text"].strip(),
                                        level=1 if span["size"] > 18 else 2,
                                        content="",
                                        page_start=page_num + 1,
                                        page_end=page_num + 1,
                                    ))

            metadata = DocumentMetadata(
                title=os.path.basename(file_path),
                page_count=len(doc),
                file_size=os.path.getsize(file_path),
            )

            doc.close()

            return DocumentArtifact(
                file_path=file_path,
                metadata=metadata,
                sections=sections,
                raw_text="\n".join(raw_text_parts),
                parse_confidence=1.0,
            )

        except ImportError:
            logger.warning("pymupdf not installed, cannot parse PDF")
            return DocumentArtifact(file_path=file_path, parse_confidence=0.0)
        except Exception as e:
            logger.warning(f"Failed to parse PDF {file_path}: {e}")
            return DocumentArtifact(file_path=file_path, parse_confidence=0.0)

    async def extract_evidence_cards(self, artifact: DocumentArtifact) -> List[dict]:
        """Extract evidence cards from a parsed document artifact."""
        if artifact.parse_confidence < 0.5 or not artifact.raw_text:
            return []

        cards = []
        # Split text into chunks and create evidence cards
        text = artifact.raw_text[:50000]  # Limit to prevent token overflow
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for i, para in enumerate(paragraphs[:20]):  # Limit to 20 cards
            if len(para) > 50:  # Skip very short paragraphs
                cards.append({
                    "id": f"doc_{artifact.metadata.title}_{i}",
                    "claim": para[:500],
                    "confidence": 0.6,
                    "supporting_source_ids": [],
                    "conflicting_source_ids": [],
                    "exact_excerpts": [para[:200]],
                    "subquestion_id": "document",
                    "researcher_id": "document_reader",
                    "deduplicated_from": [],
                })

        return cards
