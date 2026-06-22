# ADR 005: Agentic Document Reading (No Vector-RAG)

**Status:** Accepted

**Date:** 2025-07-20

## Context

The platform needed to support uploaded PDF documents as research sources, enabling the deep research graph to extract structured evidence from external documents beyond web search results. Two approaches were considered:

1. **Vector-RAG**: Chunk documents → embed → store in vector DB → semantic search over chunks
2. **Agentic reading**: Parse document → extract sections → LLM reads section-by-section → extract structured evidence

The vector-RAG approach is common in RAG pipelines but introduces several problems for a production research platform:

- **Chunking breaks logical document structure** — tables, figures, cross-references, and multi-paragraph arguments are split across arbitrary chunk boundaries, losing the semantic units a researcher needs
- **Embedding quality varies dramatically by domain** — legal, medical, and academic PDFs use domain-specific terminology that off-the-shelf embedding models handle poorly, requiring fine-tuning per domain
- **Vector DB infrastructure burden** — requires a running vector database, embedding service, and index management — all stateful infrastructure that complicates deployment
- **Chunking introduces opaque information loss** — the retrieval step is a black box: you get back "relevant" chunks but cannot audit why specific content was included or excluded

The agentic reading approach avoids these problems by having the LLM directly read the document in its original structure, section by section.

## Decision

Implement **agentic document reading** with no vector database. The approach has four stages:

### 1. Parse — `DocumentReader.parse_pdf()`

Uses PyMuPDF (`pymupdf`) to extract text from PDF files. Heading detection uses font heuristics: large font + bold text → section heading. Output is a structured `DocumentArtifact` containing all parsed sections, metadata, and raw text.

### 2. Structure — `DocumentSection` objects

Each document is split into `DocumentSection` objects with structured metadata:

- **heading**: Section title text (e.g., "Introduction", "Methodology")
- **level**: Heading depth (1 = top-level, 2 = subsection, etc.)
- **content**: Full section text content
- **page_start / page_end**: Page range for the section

### 3. Extract — `DocumentReader.extract_evidence_cards()`

The reader creates evidence cards from document paragraphs, capped at 20 cards per document with a 50,000 character limit on total extracted content. Cards are passed to downstream reasoning in the same `EvidenceCard` format used by web research.

### 4. Config — Feature flags

Controlled via `Configuration`:

- `enable_document_reading: bool = True` — master switch
- `max_document_pages: int = 500` — skip documents exceeding this page count

### Data Models

```
DocumentMetadata:
  title: str | None
  author: str | None
  creation_date: str | None
  page_count: int
  file_size: int
  mime_type: str

DocumentSection:
  id: str
  heading: str | None
  level: int
  content: str
  page_start: int
  page_end: int

DocumentArtifact:
  file_path: str
  metadata: DocumentMetadata
  sections: list[DocumentSection]
  tables: list[dict]
  images: list[dict]
  raw_text: str
  parse_confidence: float
```

### Reading Pipeline

1. User uploads document paths → stored in `AgentInputState.document_paths`
2. Graph node reads documents → calls `DocumentReader.parse_pdf()` per file
3. Results stored as `document_artifacts` in `AgentState`
4. `extract_evidence_cards()` creates evidence cards for downstream reasoning

### Source Code

| File | Lines | Purpose |
|------|-------|---------|
| `src/open_deep_research/document_reader.py` | 129 | 4 classes: `DocumentMetadata`, `DocumentSection`, `DocumentArtifact`, `DocumentReader` |
| `src/open_deep_research/configuration.py` | 402-412 | `enable_document_reading`, `max_document_pages` |
| `src/open_deep_research/state.py` | 194 | `document_paths` in `AgentInputState` |
| `src/open_deep_research/state.py` | 220 | `document_artifacts` in `AgentState` |

## Consequences

### Positive

- **No vector DB dependency** — zero infrastructure overhead. The entire document reading pipeline runs in-process with a single `pip install pymupdf` dependency.
- **Document structure preserved** — sections, headings, and page context remain intact. The LLM reads content in its logical order, not as isolated chunks.
- **Simple deployment** — no vector database, embedding service, or index management. The platform remains a single Python process.
- **Flexible extraction** — different extraction strategies per document type via LLM instructions. Legal contracts, scientific papers, and financial reports can all use the same pipeline but extract different evidence schemas through system prompt variation.

### Trade-offs

- **Full PDF text must fit in LLM context** — the 50,000 character extraction limit and 20-card cap mean very large documents are truncated. Documents exceeding the limit have their content sampled rather than fully extracted.
- **PDF-only support** — PyMuPDF handles only PDF files. DOCX, HTML, Markdown, and other formats are not supported. Each new format requires a new parser implementation.
- **Font-size-based heading detection is heuristic** — PDFs with unusual layouts, embedded fonts, or non-standard heading styling may produce incorrect section structure. The heuristic works well for standard academic and business PDFs but has no guarantees for irregular layouts.
- **Parse errors set `parse_confidence=0.0`** — encrypted, corrupted, or font-embedded PDFs fail hard and skip extraction entirely. There is no fallback OCR or degraded-mode parsing.
