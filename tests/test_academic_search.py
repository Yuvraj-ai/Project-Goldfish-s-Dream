"""Tests for academic search wrappers — all HTTP calls mocked with respx."""
import httpx
import pytest
import respx


class TestAcademicConfig:
    def test_academic_search_enabled_by_default(self):
        from open_deep_research.configuration import Configuration
        config = Configuration(research_model="test", research_model_max_tokens=1000)
        assert config.enable_academic_search is True

    def test_arxiv_enabled_by_default(self):
        from open_deep_research.configuration import Configuration
        config = Configuration(research_model="test", research_model_max_tokens=1000)
        assert config.arxiv_enabled is True

    def test_semantic_scholar_enabled_by_default(self):
        from open_deep_research.configuration import Configuration
        config = Configuration(research_model="test", research_model_max_tokens=1000)
        assert config.semantic_scholar_enabled is True

    def test_pubmed_enabled_by_default(self):
        from open_deep_research.configuration import Configuration
        config = Configuration(research_model="test", research_model_max_tokens=1000)
        assert config.pubmed_enabled is True

    def test_crossref_enabled_by_default(self):
        from open_deep_research.configuration import Configuration
        config = Configuration(research_model="test", research_model_max_tokens=1000)
        assert config.crossref_enabled is True


# Mock XML response for arXiv
ARXIV_MOCK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Test Paper</title>
    <id>http://arxiv.org/abs/2401.00001</id>
    <summary>Test abstract</summary>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>Test Author</name></author>
    <link href="http://arxiv.org/pdf/2401.00001" title="pdf"/>
  </entry>
</feed>"""


class TestArxivSearch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_results(self):
        respx.get("http://export.arxiv.org/api/query").respond(200, text=ARXIV_MOCK_XML)
        from open_deep_research.utils import arxiv_search
        results = await arxiv_search("quantum computing", max_results=3)
        assert len(results) > 0
        assert results[0]["source_type"] == "academic"
        assert "title" in results[0]

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_empty_response(self):
        respx.get("http://export.arxiv.org/api/query").respond(200, text="<feed></feed>")
        from open_deep_research.utils import arxiv_search
        results = await arxiv_search("nonexistent topic")
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_network_error(self):
        respx.get("http://export.arxiv.org/api/query").mock(side_effect=httpx.ConnectError("fail"))
        from open_deep_research.utils import arxiv_search
        results = await arxiv_search("test")
        assert results == []


# Mock JSON for Semantic Scholar
S2_MOCK = {"data": [{"title": "ML Paper", "abstract": "Abstract", "citationCount": 10, "authors": [{"name": "Author"}], "year": 2024, "url": "https://example.com", "externalIds": {"DOI": "10.1234/test"}}]}


class TestSemanticScholarSearch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_results(self):
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").respond(200, json=S2_MOCK)
        from open_deep_research.utils import semantic_scholar_search
        results = await semantic_scholar_search("machine learning")
        assert len(results) > 0
        assert results[0]["source_type"] == "academic"

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_empty_response(self):
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").respond(200, json={"data": []})
        from open_deep_research.utils import semantic_scholar_search
        results = await semantic_scholar_search("nonexistent")
        assert results == []


# Mock JSON for PubMed
PUBMED_SEARCH_MOCK = {"esearchresult": {"idlist": ["12345"]}}
PUBMED_DETAIL_MOCK = {"result": {"12345": {"title": "Cancer Paper", "authors": [{"name": "Dr. Smith"}], "pubdate": "2024 Jan"}}}


class TestPubMedSearch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_results(self):
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").respond(200, json=PUBMED_SEARCH_MOCK)
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi").respond(200, json=PUBMED_DETAIL_MOCK)
        from open_deep_research.utils import pubmed_search
        results = await pubmed_search("cancer treatment")
        assert len(results) > 0
        assert "pubmed" in results[0]["url"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_empty_ids(self):
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").respond(200, json={"esearchresult": {"idlist": []}})
        from open_deep_research.utils import pubmed_search
        results = await pubmed_search("nonexistent")
        assert results == []


# Mock JSON for Crossref
CROSSREF_MOCK = {"message": {"items": [{"title": ["Deep Learning Paper"], "URL": "https://doi.org/10.1234/test", "DOI": "10.1234/test", "author": [{"given": "Jane", "family": "Doe"}], "is-referenced-by-count": 5, "published-print": {"date-parts": [[2024, 6, 15]]}}]}}


class TestCrossrefSearch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_results(self):
        respx.get("https://api.crossref.org/works").respond(200, json=CROSSREF_MOCK)
        from open_deep_research.utils import crossref_search
        results = await crossref_search("deep learning")
        assert len(results) > 0
        assert results[0]["doi"] == "10.1234/test"

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_empty_response(self):
        respx.get("https://api.crossref.org/works").respond(200, json={"message": {"items": []}})
        from open_deep_research.utils import crossref_search
        results = await crossref_search("nonexistent")
        assert results == []
