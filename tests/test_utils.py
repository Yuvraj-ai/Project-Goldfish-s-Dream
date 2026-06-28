"""Tests for utility functions in utils.py — string utils, token checks, API key
resolution, message manipulation, websearch detection, and academic search error paths."""
import asyncio
from datetime import datetime
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from open_deep_research.utils import (
    _check_anthropic_token_limit,
    _check_gemini_token_limit,
    _check_openai_token_limit,
    _parse_date,
    anthropic_websearch_called,
    arxiv_search,
    crossref_search,
    enforce_source_diversity,
    get_api_key_for_model,
    get_config_value,
    get_domain,
    get_domain_histogram,
    get_mcp_access_token,
    get_model_token_limit,
    get_all_tools,
    get_notes_from_tool_calls,
    get_search_aggregator,
    get_search_tool,
    get_tavily_api_key,
    get_today_str,
    get_tokens,
    is_token_limit_exceeded,
    fetch_tokens,
    set_tokens,
    openai_websearch_called,
    pubmed_search,
    remove_up_to_last_ai_message,
    semantic_scholar_search,
    summarize_webpage,
    tavily_search_async,
    temporal_relevance_boost,
    think_tool,
)


class TestGetTodayStr:
    def test_returns_formatted_date(self):
        result = get_today_str()
        now = datetime.now()
        expected = f"{now:%a} {now:%b} {now.day}, {now:%Y}"
        assert result == expected


class TestGetConfigValue:
    def test_none(self):
        assert get_config_value(None) is None

    def test_str(self):
        assert get_config_value("hello") == "hello"

    def test_dict(self):
        assert get_config_value({"key": "val"}) == {"key": "val"}

    def test_enum(self):
        class Color(Enum):
            RED = "red"
        assert get_config_value(Color.RED) == "red"


class TestThinkTool:
    def test_invoke_returns_recorded_message(self):
        result = think_tool.invoke({"reflection": "test reflection"})
        assert "Reflection recorded" in result
        assert "test reflection" in result


class TestGetModelTokenLimit:
    def test_known_model_openai(self):
        assert get_model_token_limit("openai:gpt-4o") == 128000

    def test_known_model_anthropic(self):
        assert get_model_token_limit("anthropic:claude-sonnet-4") == 200000

    def test_known_model_google(self):
        assert get_model_token_limit("google:gemini-pro") == 32768

    def test_known_model_substring(self):
        assert get_model_token_limit("openai:gpt-4o-mini") == 128000

    def test_unknown_model(self):
        assert get_model_token_limit("unknown:model") is None


class TestRemoveUpToLastAiMessage:
    def test_removes_up_to_last_ai(self):
        from langchain_core.messages import AIMessage, HumanMessage
        messages = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
        ]
        result = remove_up_to_last_ai_message(messages)
        assert len(result) == 1
        assert result[0].content == "q1"

    def test_no_ai_message(self):
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content="a"), HumanMessage(content="b")]
        result = remove_up_to_last_ai_message(messages)
        assert len(result) == 2

    def test_empty_list(self):
        assert remove_up_to_last_ai_message([]) == []

    def test_multiple_ai_messages(self):
        from langchain_core.messages import AIMessage, HumanMessage
        messages = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
            HumanMessage(content="q3"),
        ]
        result = remove_up_to_last_ai_message(messages)
        assert len(result) == 3
        assert result[-1].content == "q2"

    def test_single_ai_message(self):
        from langchain_core.messages import AIMessage
        messages = [AIMessage(content="only")]
        result = remove_up_to_last_ai_message(messages)
        assert result == []


class TestGetNotesFromToolCalls:
    def test_extracts_tool_messages(self):
        from langchain_core.messages import HumanMessage, ToolMessage
        messages = [
            ToolMessage(content="note1", tool_call_id="1"),
            HumanMessage(content="hello"),
            ToolMessage(content="note2", tool_call_id="2"),
        ]
        result = get_notes_from_tool_calls(messages)
        assert result == ["note1", "note2"]

    def test_empty_list(self):
        assert get_notes_from_tool_calls([]) == []

    def test_no_tool_messages(self):
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content="hello")]
        result = get_notes_from_tool_calls(messages)
        assert result == []


class TestAnthropicWebsearchCalled:
    def test_web_search_was_called(self):
        response = MagicMock()
        response.response_metadata = {
            "usage": {
                "server_tool_use": {
                    "web_search_requests": 3
                }
            }
        }
        assert anthropic_websearch_called(response) is True

    def test_web_search_not_called_zero(self):
        response = MagicMock()
        response.response_metadata = {
            "usage": {
                "server_tool_use": {
                    "web_search_requests": 0
                }
            }
        }
        assert anthropic_websearch_called(response) is False

    def test_no_server_tool_use(self):
        response = MagicMock()
        response.response_metadata = {
            "usage": {}
        }
        assert anthropic_websearch_called(response) is False

    def test_no_server_tool_use_plain_object(self):
        class MockResponse:
            response_metadata = {"usage": {}}
        assert anthropic_websearch_called(MockResponse()) is False

    def test_no_usage_key(self):
        response = MagicMock()
        response.response_metadata = {}
        assert anthropic_websearch_called(response) is False

    def test_no_response_metadata(self):
        response = object()
        assert anthropic_websearch_called(response) is False

    def test_server_tool_use_is_empty_dict(self):
        response = MagicMock()
        response.response_metadata = {
            "usage": {"server_tool_use": {}}
        }
        assert anthropic_websearch_called(response) is False

    def test_web_search_requests_none(self):
        response = MagicMock()
        response.response_metadata = {
            "usage": {
                "server_tool_use": {
                    "web_search_requests": None
                }
            }
        }
        assert anthropic_websearch_called(response) is False


class TestOpenaiWebsearchCalled:
    def test_web_search_was_called(self):
        response = MagicMock()
        response.additional_kwargs = {
            "tool_outputs": [
                {"type": "web_search_call"},
                {"type": "code_interpreter"},
            ]
        }
        assert openai_websearch_called(response) is True

    def test_web_search_not_called(self):
        response = MagicMock()
        response.additional_kwargs = {
            "tool_outputs": [
                {"type": "code_interpreter"}
            ]
        }
        assert openai_websearch_called(response) is False

    def test_no_tool_outputs(self):
        response = MagicMock()
        response.additional_kwargs = {}
        assert openai_websearch_called(response) is False


class TestIsTokenLimitExceeded:
    def test_openai_by_model_name(self):
        class BadRequestError(Exception):
            pass
        BadRequestError.__module__ = 'openai'
        exc = BadRequestError("token limit exceeded")
        assert is_token_limit_exceeded(exc, "openai:gpt-4o") is True

    def test_openai_without_model_name(self):
        class BadRequestError(Exception):
            pass
        BadRequestError.__module__ = 'openai'
        exc = BadRequestError("context length exceeded")
        assert is_token_limit_exceeded(exc) is True

    def test_anthropic_by_model_name(self):
        class BadRequestError(Exception):
            pass
        BadRequestError.__module__ = 'anthropic'
        exc = BadRequestError("prompt is too long")
        assert is_token_limit_exceeded(exc, "anthropic:claude-sonnet-4") is True

    def test_gemini_by_model_name(self):
        class ResourceExhausted(Exception):
            pass
        ResourceExhausted.__module__ = 'google.api'
        exc = ResourceExhausted("quota exceeded")
        assert is_token_limit_exceeded(exc, "google:gemini-pro") is True

    def test_not_token_error(self):
        class BadRequestError(Exception):
            pass
        BadRequestError.__module__ = 'openai'
        exc = BadRequestError("invalid api key")
        assert is_token_limit_exceeded(exc) is False

    def test_non_matching_exception(self):
        exc = ValueError("some error")
        assert is_token_limit_exceeded(exc) is False

    def test_gemini_prefix_model_name(self):
        class ResourceExhausted(Exception):
            pass
        ResourceExhausted.__module__ = 'google.api'
        exc = ResourceExhausted("quota exceeded")
        assert is_token_limit_exceeded(exc, "gemini:gemini-pro") is True

    def test_anthropic_model_name_no_match(self):
        class BadRequestError(Exception):
            pass
        BadRequestError.__module__ = 'anthropic'
        exc = BadRequestError("invalid request")
        assert is_token_limit_exceeded(exc, "anthropic:claude-sonnet-4") is False

    def test_unknown_model_name_falls_through(self):
        class BadRequestError(Exception):
            pass
        BadRequestError.__module__ = 'openai'
        exc = BadRequestError("token limit exceeded")
        assert is_token_limit_exceeded(exc, "unknown:model") is True


class TestCheckOpenaiTokenLimit:
    def test_token_keywords_match(self):
        class BadRequestError(Exception):
            pass
        BadRequestError.__module__ = 'openai'
        exc = BadRequestError("maximum context length exceeded")
        assert _check_openai_token_limit(exc, "maximum context length exceeded") is True

    def test_code_attribute_match(self):
        exc = Exception("some error")
        exc.code = "context_length_exceeded"
        exc.type = "invalid_request_error"
        assert _check_openai_token_limit(exc, "some error") is True

    def test_no_match(self):
        class BadRequestError(Exception):
            pass
        BadRequestError.__module__ = 'openai'
        exc = BadRequestError("invalid request")
        assert _check_openai_token_limit(exc, "invalid request") is False


class TestCheckAnthropicTokenLimit:
    def test_prompt_too_long_match(self):
        class BadRequestError(Exception):
            pass
        BadRequestError.__module__ = 'anthropic'
        exc = BadRequestError("prompt is too long")
        assert _check_anthropic_token_limit(exc, "prompt is too long") is True

    def test_not_anthropic_exception(self):
        exc = ValueError("prompt is too long")
        assert _check_anthropic_token_limit(exc, "prompt is too long") is False

    def test_not_bad_request_class(self):
        class OtherError(Exception):
            pass
        OtherError.__module__ = 'anthropic'
        exc = OtherError("some error")
        assert _check_anthropic_token_limit(exc, "some error") is False

    def test_no_match(self):
        class BadRequestError(Exception):
            pass
        BadRequestError.__module__ = 'anthropic'
        exc = BadRequestError("invalid request")
        assert _check_anthropic_token_limit(exc, "invalid request") is False


class TestCheckGeminiTokenLimit:
    def test_resource_exhausted_match(self):
        class ResourceExhausted(Exception):
            pass
        ResourceExhausted.__module__ = 'google.api'
        exc = ResourceExhausted("quota exceeded")
        assert _check_gemini_token_limit(exc, "quota exceeded") is True

    def test_google_generative_ai_fetch_error(self):
        class GoogleGenerativeAIFetchError(Exception):
            pass
        GoogleGenerativeAIFetchError.__module__ = 'google'
        exc = GoogleGenerativeAIFetchError("fetch error")
        assert _check_gemini_token_limit(exc, "fetch error") is True

    def test_type_string_fallback(self):
        class _MockGoogleResourceExhausted(Exception):
            pass
        _MockGoogleResourceExhausted.__module__ = \
            'google.api_core.exceptions.resourceexhausted'
        exc = _MockGoogleResourceExhausted("test")
        assert _check_gemini_token_limit(exc, "test") is True

    def test_no_match(self):
        exc = ValueError("some error")
        assert _check_gemini_token_limit(exc, "some error") is False


class TestGetApiKeyForModel:
    @patch("open_deep_research.utils.os.getenv")
    def test_from_env_openai(self, mock_getenv):
        mock_getenv.side_effect = lambda k, d=None: {
            "GET_API_KEYS_FROM_CONFIG": "false",
            "OPENAI_API_KEY": "sk-openai-env",
        }.get(k, d)
        assert get_api_key_for_model("openai:gpt-4o", {}) == "sk-openai-env"

    @patch("open_deep_research.utils.os.getenv")
    def test_from_env_anthropic(self, mock_getenv):
        mock_getenv.side_effect = lambda k, d=None: {
            "GET_API_KEYS_FROM_CONFIG": "false",
            "ANTHROPIC_API_KEY": "sk-anthropic-env",
        }.get(k, d)
        assert get_api_key_for_model("anthropic:claude-sonnet-4", {}) == "sk-anthropic-env"

    @patch("open_deep_research.utils.os.getenv")
    def test_from_env_google(self, mock_getenv):
        mock_getenv.side_effect = lambda k, d=None: {
            "GET_API_KEYS_FROM_CONFIG": "false",
            "GOOGLE_API_KEY": "sk-google-env",
        }.get(k, d)
        assert get_api_key_for_model("google:gemini-pro", {}) == "sk-google-env"

    @patch("open_deep_research.utils.os.getenv")
    def test_from_env_unknown_model(self, mock_getenv):
        mock_getenv.side_effect = lambda k, d=None: {
            "GET_API_KEYS_FROM_CONFIG": "false",
        }.get(k, d)
        assert get_api_key_for_model("unknown:model", {}) is None

    @patch("open_deep_research.utils.os.getenv")
    def test_from_config(self, mock_getenv):
        mock_getenv.return_value = "true"
        config = {
            "configurable": {
                "apiKeys": {
                    "OPENAI_API_KEY": "sk-openai-config",
                }
            }
        }
        assert get_api_key_for_model("openai:gpt-4o", config) == "sk-openai-config"

    @patch("open_deep_research.utils.os.getenv")
    def test_from_config_no_api_keys(self, mock_getenv):
        mock_getenv.return_value = "true"
        assert get_api_key_for_model("openai:gpt-4o", {}) is None

    @patch("open_deep_research.utils.os.getenv")
    def test_from_config_anthropic(self, mock_getenv):
        mock_getenv.return_value = "true"
        config = {
            "configurable": {
                "apiKeys": {
                    "ANTHROPIC_API_KEY": "sk-anthropic-config",
                }
            }
        }
        assert get_api_key_for_model("anthropic:claude-sonnet-4", config) == "sk-anthropic-config"

    @patch("open_deep_research.utils.os.getenv")
    def test_from_config_google(self, mock_getenv):
        mock_getenv.return_value = "true"
        config = {
            "configurable": {
                "apiKeys": {
                    "GOOGLE_API_KEY": "sk-google-config",
                }
            }
        }
        assert get_api_key_for_model("google:gemini-pro", config) == "sk-google-config"

    @patch("open_deep_research.utils.os.getenv")
    def test_from_config_unknown_model(self, mock_getenv):
        mock_getenv.return_value = "true"
        config = {
            "configurable": {
                "apiKeys": {
                    "SOME_KEY": "some-value",
                }
            }
        }
        assert get_api_key_for_model("unknown:model", config) is None


class TestGetTavilyApiKey:
    @patch("open_deep_research.utils.os.getenv")
    def test_from_env(self, mock_getenv):
        mock_getenv.side_effect = lambda k, d=None: {
            "GET_API_KEYS_FROM_CONFIG": "false",
            "TAVILY_API_KEY": "tvly-env",
        }.get(k, d)
        assert get_tavily_api_key({}) == "tvly-env"

    @patch("open_deep_research.utils.os.getenv")
    def test_from_env_not_set(self, mock_getenv):
        mock_getenv.side_effect = lambda k, d=None: {
            "GET_API_KEYS_FROM_CONFIG": "false",
        }.get(k, d)
        assert get_tavily_api_key({}) is None

    @patch("open_deep_research.utils.os.getenv")
    def test_from_config(self, mock_getenv):
        mock_getenv.return_value = "true"
        config = {"configurable": {"apiKeys": {"TAVILY_API_KEY": "tvly-config"}}}
        assert get_tavily_api_key(config) == "tvly-config"

    @patch("open_deep_research.utils.os.getenv")
    def test_from_config_no_keys(self, mock_getenv):
        mock_getenv.return_value = "true"
        assert get_tavily_api_key({}) is None


class TestArxivSearch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_success(self):
        xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2401.00001</id>
            <title>Quantum Computing Advances</title>
            <summary>Recent advances in quantum computing have shown promise.</summary>
            <published>2024-01-15T00:00:00Z</published>
            <author><name>Alice Smith</name></author>
            <link title="pdf" href="http://arxiv.org/pdf/2401.00001"/>
          </entry>
        </feed>"""
        respx.get("http://export.arxiv.org/api/query").respond(200, text=xml)
        results = await arxiv_search("quantum", max_results=1)
        assert len(results) == 1
        assert results[0]["title"] == "Quantum Computing Advances"
        assert results[0]["provider"] == "arxiv"
        assert "arxiv" in results[0]["url"]
        assert "Alice" in results[0]["authors"][0]
        assert results[0]["pdf_url"] == "http://arxiv.org/pdf/2401.00001"

    @pytest.mark.asyncio
    @respx.mock
    async def test_empty_response(self):
        xml = """<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""
        respx.get("http://export.arxiv.org/api/query").respond(200, text=xml)
        results = await arxiv_search("nothing", max_results=5)
        assert results == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_http_error_returns_empty(self):
        respx.get("http://export.arxiv.org/api/query").respond(500)
        results = await arxiv_search("test", max_results=3)
        assert results == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_network_error(self):
        respx.get("http://export.arxiv.org/api/query").mock(
            side_effect=httpx.ConnectError("connection failed")
        )
        results = await arxiv_search("quantum", max_results=3)
        assert results == []


class TestSemanticScholarSearch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_success(self):
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").respond(
            200,
            json={
                "data": [{
                    "title": "Deep Learning Advances",
                    "url": "https://semanticscholar.org/paper/abc123",
                    "abstract": "Recent advances in deep learning...",
                    "year": 2024,
                    "citationCount": 42,
                    "externalIds": {"DOI": "10.1234/test"},
                    "authors": [{"name": "Bob Johnson"}],
                }]
            },
        )
        results = await semantic_scholar_search("deep learning", max_results=1)
        assert len(results) == 1
        assert results[0]["title"] == "Deep Learning Advances"
        assert results[0]["provider"] == "semantic_scholar"
        assert results[0]["citation_count"] == 42
        assert results[0]["doi"] == "10.1234/test"

    @pytest.mark.asyncio
    @respx.mock
    async def test_empty_data(self):
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").respond(
            200, json={"data": []}
        )
        results = await semantic_scholar_search("test", max_results=5)
        assert results == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_http_error_returns_empty(self):
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").respond(500)
        results = await semantic_scholar_search("test", max_results=3)
        assert results == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_network_error(self):
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            side_effect=httpx.ConnectError("connection failed")
        )
        results = await semantic_scholar_search("ml", max_results=3)
        assert results == []


class TestPubMedSearch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_success(self):
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").respond(
            200, json={"esearchresult": {"idlist": ["12345"]}}
        )
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi").respond(
            200,
            json={
                "result": {
                    "12345": {
                        "title": "Cancer Research Breakthrough",
                        "pubdate": "2024 Jan",
                        "authors": [{"name": "Dr. Smith"}, {"name": "Dr. Jones"}],
                    }
                }
            },
        )
        results = await pubmed_search("cancer", max_results=1)
        assert len(results) == 1
        assert results[0]["title"] == "Cancer Research Breakthrough"
        assert results[0]["provider"] == "pubmed"
        assert "12345" in results[0]["url"]
        assert len(results[0]["authors"]) == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_empty_id_list(self):
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").respond(
            200, json={"esearchresult": {"idlist": []}}
        )
        results = await pubmed_search("unknown", max_results=5)
        assert results == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_http_error_returns_empty(self):
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").respond(500)
        results = await pubmed_search("test", max_results=3)
        assert results == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_network_error(self):
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
            side_effect=httpx.ConnectError("connection failed")
        )
        results = await pubmed_search("cancer", max_results=3)
        assert results == []


class TestSemanticScholarErrors:
    @pytest.mark.asyncio
    @respx.mock
    async def test_network_error(self):
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            side_effect=httpx.ConnectError("connection failed")
        )
        results = await semantic_scholar_search("ml", max_results=3)
        assert results == []


class TestPubMedErrors:
    @pytest.mark.asyncio
    @respx.mock
    async def test_network_error(self):
        respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
            side_effect=httpx.ConnectError("connection failed")
        )
        results = await pubmed_search("cancer", max_results=3)
        assert results == []


class TestCrossrefSearch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_http_error_returns_empty(self):
        respx.get("https://api.crossref.org/works").respond(500)
        results = await crossref_search("test", max_results=3)
        assert results == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_network_error(self):
        respx.get("https://api.crossref.org/works").mock(
            side_effect=httpx.ConnectError("connection failed")
        )
        results = await crossref_search("deep learning", max_results=3)
        assert results == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_empty_items(self):
        respx.get("https://api.crossref.org/works").respond(
            200, json={"message": {"items": []}}
        )
        results = await crossref_search("test", max_results=5)
        assert results == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_missing_message_key(self):
        respx.get("https://api.crossref.org/works").respond(200, json={})
        results = await crossref_search("test", max_results=5)
        assert results == []


class TestCrossrefDateParts:
    @pytest.mark.asyncio
    @respx.mock
    async def test_two_part_date(self):
        respx.get("https://api.crossref.org/works").respond(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "title": ["Paper"],
                            "URL": "https://doi.org/10.1234/test",
                            "DOI": "10.1234/test",
                            "author": [],
                            "is-referenced-by-count": 0,
                            "published-print": {"date-parts": [[2024, 6]]},
                        }
                    ]
                }
            },
        )
        results = await crossref_search("test", max_results=5)
        assert results[0]["date"] == "2024-06"

    @pytest.mark.asyncio
    @respx.mock
    async def test_one_part_date(self):
        respx.get("https://api.crossref.org/works").respond(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "title": ["Paper"],
                            "URL": "https://doi.org/10.1234/test",
                            "DOI": "10.1234/test",
                            "author": [],
                            "is-referenced-by-count": 0,
                            "published-print": {"date-parts": [[2024]]},
                        }
                    ]
                }
            },
        )
        results = await crossref_search("test", max_results=5)
        assert results[0]["date"] == "2024"


class TestParseDateEdgeCases:
    def test_invalid_month_in_three_part_list(self):
        assert _parse_date([2024, 13, 1]) is None

    def test_invalid_month_in_two_part_list(self):
        assert _parse_date([2024, 13]) is None

    def test_single_element_list(self):
        assert _parse_date([2024]) is None

    def test_invalid_types_in_list(self):
        assert _parse_date(["bad", "data", "here"]) is None


class TestGetSearchTool:
    @pytest.mark.asyncio
    async def test_anthropic(self):
        from open_deep_research.configuration import SearchAPI
        result = await get_search_tool(SearchAPI.ANTHROPIC)
        assert result == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]

    @pytest.mark.asyncio
    async def test_openai(self):
        from open_deep_research.configuration import SearchAPI
        result = await get_search_tool(SearchAPI.OPENAI)
        assert result == [{"type": "web_search_preview"}]

    @pytest.mark.asyncio
    async def test_none(self):
        from open_deep_research.configuration import SearchAPI
        result = await get_search_tool(SearchAPI.NONE)
        assert result == []

    @pytest.mark.asyncio
    async def test_tavily(self):
        from open_deep_research.configuration import SearchAPI
        result = await get_search_tool(SearchAPI.TAVILY)
        assert len(result) == 1
        assert result[0].metadata.get("type") == "search"
        assert result[0].metadata.get("name") == "web_search"

    @pytest.mark.asyncio
    async def test_unknown_fallback(self):
        result = await get_search_tool(None)
        assert result == []


class TestGetAllTools:
    @pytest.mark.asyncio
    async def test_basic_config(self):
        config = {
            "configurable": {
                "research_model": "test",
                "research_model_max_tokens": 1000,
            }
        }
        tools = await get_all_tools(config)
        assert len(tools) >= 2
        assert any(t.name == "think_tool" if hasattr(t, "name") else False for t in tools)


class TestGetSearchAggregator:
    def test_empty_config(self):
        class MockConfig:
            search_providers = []
            enable_academic_search = False
            arxiv_enabled = False
            semantic_scholar_enabled = False
            pubmed_enabled = False
            crossref_enabled = False

        from open_deep_research.search_aggregator import SearchAggregator as SA
        result = get_search_aggregator(MockConfig())
        assert isinstance(result, SA)

    def test_with_academic_sources(self):
        class MockConfig:
            search_providers = []
            enable_academic_search = True
            arxiv_enabled = True
            semantic_scholar_enabled = True
            pubmed_enabled = True
            crossref_enabled = True

        from open_deep_research.search_aggregator import SearchAggregator as SA
        result = get_search_aggregator(MockConfig())
        assert isinstance(result, SA)

    def test_with_web_providers(self):
        class MockConfig:
            search_providers = ["tavily", "duckduckgo"]
            enable_academic_search = False
            arxiv_enabled = False
            semantic_scholar_enabled = False
            pubmed_enabled = False
            crossref_enabled = False

        from open_deep_research.search_aggregator import SearchAggregator as SA
        result = get_search_aggregator(MockConfig())
        assert isinstance(result, SA)


class TestGetDomain:
    def test_extracts_domain_strips_www(self):
        assert get_domain("https://www.example.com/path") == "example.com"

    def test_lowercases_domain(self):
        assert get_domain("http://EXAMPLE.COM/Path") == "example.com"

    def test_handles_subdomain(self):
        assert get_domain("https://sub.example.com/page") == "sub.example.com"

    def test_no_www(self):
        assert get_domain("https://example.com") == "example.com"


class TestGetDomainHistogram:
    def test_counts_per_domain(self):
        results = [
            {"url": "https://a.com/x"},
            {"url": "https://a.com/y"},
            {"url": "https://b.com/z"},
        ]
        assert get_domain_histogram(results) == {"a.com": 2, "b.com": 1}

    def test_empty_results(self):
        assert get_domain_histogram([]) == {}

    def test_skips_empty_urls(self):
        results = [{"url": ""}, {"url": "https://a.com/x"}]
        assert get_domain_histogram(results) == {"a.com": 1}


class TestEnforceSourceDiversity:
    def test_empty_results_passes(self):
        result = enforce_source_diversity([])
        assert result["passed"] is True
        assert result["needs_supplement"] is False
        assert result["unique_domains"] == 0

    def test_passing_diversity(self):
        results = [
            {"url": "https://a.com/1"},
            {"url": "https://b.com/1"},
            {"url": "https://c.com/1"},
        ]
        result = enforce_source_diversity(results, min_unique_domains=3, max_same_domain_ratio=0.4)
        assert result["passed"] is True
        assert result["needs_supplement"] is False
        assert result["unique_domains"] == 3

    def test_failing_diversity_too_few_domains(self):
        results = [
            {"url": "https://a.com/1"},
            {"url": "https://a.com/2"},
            {"url": "https://b.com/3"},
        ]
        result = enforce_source_diversity(results, min_unique_domains=3, max_same_domain_ratio=0.4)
        assert result["passed"] is False
        assert result["needs_supplement"] is True
        assert result["unique_domains"] == 2

    def test_failing_diversity_dominant_ratio_too_high(self):
        results = [
            {"url": "https://a.com/1"},
            {"url": "https://a.com/2"},
            {"url": "https://a.com/3"},
            {"url": "https://b.com/4"},
        ]
        result = enforce_source_diversity(results, min_unique_domains=2, max_same_domain_ratio=0.4)
        assert result["passed"] is False
        assert result["needs_supplement"] is True
        assert result["dominant_domain"] == "a.com"
        assert result["dominant_ratio"] == 0.75


class TestParseDate:
    def test_iso_format(self):
        result = _parse_date("2024-06-15")
        assert result == datetime(2024, 6, 15)

    def test_crossref_three_element_list(self):
        result = _parse_date([2024, 6, 15])
        assert result == datetime(2024, 6, 15)

    def test_crossref_two_element_list(self):
        result = _parse_date([2024, 6])
        assert result == datetime(2024, 6, 1)

    def test_none_returns_none(self):
        assert _parse_date(None) is None

    def test_invalid_string_returns_none(self):
        assert _parse_date("not-a-date") is None


class TestTemporalRelevanceBoost:
    def test_recent_date_gets_high_score(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        results = [{"url": "https://a.com", "date": today_str}]
        boosted = temporal_relevance_boost(results, max_age_days=30)
        assert boosted[0]["recency_score"] >= 0.95

    def test_no_date_gets_default_score(self):
        results = [{"url": "https://a.com"}]
        boosted = temporal_relevance_boost(results, max_age_days=30)
        assert boosted[0]["recency_score"] == 0.5

    def test_older_date_gets_lower_score(self):
        results = [{"url": "https://a.com", "date": "2024-01-01"}]
        boosted = temporal_relevance_boost(results, max_age_days=30)
        assert boosted[0]["recency_score"] < 0.5

    def test_sorts_by_recency_descending(self):
        results = [
            {"url": "https://old.com", "date": "2024-01-01"},
            {"url": "https://new.com", "date": datetime.now().strftime("%Y-%m-%d")},
        ]
        boosted = temporal_relevance_boost(results, max_age_days=30)
        assert boosted[0]["url"] == "https://new.com"
        assert boosted[1]["url"] == "https://old.com"

    def test_empty_results(self):
        assert temporal_relevance_boost([]) == []

    def test_uses_publication_date_fallback(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        results = [{"url": "https://a.com", "publication_date": today_str}]
        boosted = temporal_relevance_boost(results, max_age_days=30)
        assert boosted[0]["recency_score"] >= 0.95


class TestOpenaiWebsearchCalledMissingAttribute:
    def test_missing_additional_kwargs(self):
        response = object()
        with pytest.raises(AttributeError):
            openai_websearch_called(response)


class TestSummarizeWebpage:
    @pytest.mark.asyncio
    async def test_timeout_returns_content(self):
        from unittest.mock import AsyncMock
        mock_model = AsyncMock()
        mock_model.ainvoke.side_effect = asyncio.TimeoutError()
        result = await summarize_webpage(mock_model, "test content")
        assert result == "test content"

    @pytest.mark.asyncio
    async def test_generic_exception_returns_content(self):
        from unittest.mock import AsyncMock
        mock_model = AsyncMock()
        mock_model.ainvoke.side_effect = Exception("generic error")
        result = await summarize_webpage(mock_model, "test content")
        assert result == "test content"


class TestTavilySearchAsync:
    @pytest.mark.asyncio
    async def test_returns_search_results(self):
        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value={"results": [{"title": "Test", "url": "https://a.com"}]})
        with patch("open_deep_research.utils.AsyncTavilyClient", return_value=mock_client):
            with patch("open_deep_research.utils.get_tavily_api_key", return_value="tvly-test"):
                results = await tavily_search_async(["test query"])
        assert len(results) == 1


class TestGetMcpAccessToken:
    @pytest.mark.asyncio
    async def test_successful_token_exchange(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"access_token": "mcp-token"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_cm):
            result = await get_mcp_access_token("supabase-token", "https://mcp.example.com")
        assert result == {"access_token": "mcp-token"}

    @pytest.mark.asyncio
    async def test_failed_exchange_returns_none(self):
        mock_resp = AsyncMock()
        mock_resp.status = 401
        mock_resp.text = AsyncMock(return_value="unauthorized")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_cm):
            result = await get_mcp_access_token("bad-token", "https://mcp.example.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=Exception("connection error"))

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_cm):
            result = await get_mcp_access_token("token", "https://mcp.example.com")
        assert result is None


class TestSummarizeWebpageHappyPath:
    @pytest.mark.asyncio
    async def test_successful_summarization(self):
        from unittest.mock import AsyncMock, MagicMock

        mock_summary = MagicMock()
        mock_summary.summary = "This is a summary"
        mock_summary.key_excerpts = "Key excerpt"
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_summary)
        result = await summarize_webpage(mock_model, "test content")
        assert "<summary>" in result
        assert "This is a summary" in result
        assert "Key excerpt" in result


class TestGetSearchAggregator:
    def test_tavily_provider_registered(self):
        from unittest.mock import MagicMock
        from open_deep_research.search_aggregator import SearchAggregator

        config = MagicMock()
        config.search_providers = ["tavily"]
        config.enable_academic_search = False

        result = get_search_aggregator(config)
        assert isinstance(result, SearchAggregator)
        assert "tavily" in result.search_functions

    def test_duckduckgo_provider_registered(self):
        from unittest.mock import MagicMock
        from open_deep_research.search_aggregator import SearchAggregator

        config = MagicMock()
        config.search_providers = ["duckduckgo"]
        config.enable_academic_search = False

        result = get_search_aggregator(config)
        assert isinstance(result, SearchAggregator)
        assert "duckduckgo" in result.search_functions

    def test_both_providers_registered(self):
        from unittest.mock import MagicMock
        from open_deep_research.search_aggregator import SearchAggregator

        config = MagicMock()
        config.search_providers = ["tavily", "duckduckgo"]
        config.enable_academic_search = False

        result = get_search_aggregator(config)
        assert isinstance(result, SearchAggregator)
        assert "tavily" in result.search_functions
        assert "duckduckgo" in result.search_functions

    def test_academic_providers_registered(self):
        from unittest.mock import MagicMock
        from open_deep_research.search_aggregator import SearchProviderConfig

        config = MagicMock()
        config.search_providers = []
        config.enable_academic_search = True
        config.arxiv_enabled = True
        config.semantic_scholar_enabled = True
        config.pubmed_enabled = True
        config.crossref_enabled = True

        result = get_search_aggregator(config)
        provider_names = [p.name for p in result.providers]
        assert "arxiv" in provider_names
        assert "semantic_scholar" in provider_names
        assert "pubmed" in provider_names
        assert "crossref" in provider_names

    @pytest.mark.asyncio
    async def test_tavily_inner_function_no_key(self):
        from unittest.mock import MagicMock

        config = MagicMock()
        config.search_providers = ["tavily"]
        config.enable_academic_search = False

        aggregator = get_search_aggregator(config)
        fn = aggregator.search_functions["tavily"]
        result = await fn("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_duckduckgo_inner_function_fallback(self):
        from unittest.mock import MagicMock, patch

        config = MagicMock()
        config.search_providers = ["duckduckgo"]
        config.enable_academic_search = False

        aggregator = get_search_aggregator(config)
        fn = aggregator.search_functions["duckduckgo"]
        with patch("duckduckgo_search.DDGS", side_effect=Exception("no net")):
            result = await fn("test query")
        assert result == []


class TestGetTokens:
    @pytest.mark.asyncio
    async def test_get_tokens_missing_thread_id(self):
        with patch("open_deep_research.utils.get_store") as mock_get_store:
            mock_store = AsyncMock()
            mock_store.aget = AsyncMock()
            mock_get_store.return_value = mock_store
            result = await get_tokens({"configurable": {}})
        assert result is None

    @pytest.mark.asyncio
    async def test_get_tokens_missing_user_id(self):
        with patch("open_deep_research.utils.get_store") as mock_get_store:
            mock_store = AsyncMock()
            mock_store.aget = AsyncMock()
            mock_get_store.return_value = mock_store
            result = await get_tokens(
                {"configurable": {"thread_id": "t1"}, "metadata": {}}
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_tokens_no_stored_tokens(self):
        config = {
            "configurable": {"thread_id": "t1"},
            "metadata": {"owner": "user1"},
        }
        with patch("open_deep_research.utils.get_store") as mock_get_store:
            mock_store = AsyncMock()
            mock_store.aget = AsyncMock(return_value=None)
            mock_get_store.return_value = mock_store
            result = await get_tokens(config)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_tokens_expired(self):
        from datetime import datetime, timedelta, timezone

        config = {
            "configurable": {"thread_id": "t1"},
            "metadata": {"owner": "user1"},
        }
        mock_token_value = MagicMock()
        mock_token_value.value = {"expires_in": 0}
        mock_token_value.created_at = datetime.now(timezone.utc) - timedelta(hours=1)

        with patch("open_deep_research.utils.get_store") as mock_get_store:
            mock_store = AsyncMock()
            mock_store.aget = AsyncMock(return_value=mock_token_value)
            mock_store.adelete = AsyncMock()
            mock_get_store.return_value = mock_store
            result = await get_tokens(config)
        assert result is None
        mock_store.adelete.assert_awaited_once()


class TestSetTokens:
    @pytest.mark.asyncio
    async def test_set_tokens_missing_thread_id(self):
        with patch("open_deep_research.utils.get_store") as mock_get_store:
            mock_store = AsyncMock()
            mock_store.aput = AsyncMock()
            mock_get_store.return_value = mock_store
            result = await set_tokens({"configurable": {}}, {"token": "val"})
        assert result is None

    @pytest.mark.asyncio
    async def test_set_tokens_missing_user_id(self):
        with patch("open_deep_research.utils.get_store") as mock_get_store:
            mock_store = AsyncMock()
            mock_store.aput = AsyncMock()
            mock_get_store.return_value = mock_store
            result = await set_tokens(
                {"configurable": {"thread_id": "t1"}, "metadata": {}}, {"token": "val"}
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_set_tokens_stores(self):
        config = {
            "configurable": {"thread_id": "t1"},
            "metadata": {"owner": "user1"},
        }
        tokens = {"access_token": "mcp-token"}
        with patch("open_deep_research.utils.get_store") as mock_get_store:
            mock_store = AsyncMock()
            mock_store.aput = AsyncMock()
            mock_get_store.return_value = mock_store
            result = await set_tokens(config, tokens)
        assert result is None
        mock_store.aput.assert_awaited_once_with(
            ("user1", "tokens"), "data", tokens
        )


class TestFetchTokens:
    @pytest.mark.asyncio
    async def test_fetch_tokens_no_supabase_token(self):
        config = {
            "configurable": {"thread_id": "t1"},
            "metadata": {"owner": "user1"},
        }
        with patch(
            "open_deep_research.utils.get_tokens", new_callable=AsyncMock
        ) as mock_get_tokens:
            mock_get_tokens.return_value = None
            result = await fetch_tokens(config)
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_tokens_no_mcp_config(self):
        config = {
            "configurable": {
                "thread_id": "t1",
                "x-supabase-access-token": "sb-token",
            },
            "metadata": {"owner": "user1"},
        }
        with patch(
            "open_deep_research.utils.get_tokens", new_callable=AsyncMock
        ) as mock_get_tokens:
            mock_get_tokens.return_value = None
            result = await fetch_tokens(config)
        assert result is None
