"""Utility functions and helpers for the Deep Research agent."""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from open_deep_research.configuration import (
    Configuration,
    SearchAPI,
    build_model_config,
)
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.state import ResearchComplete, Summary

logger = logging.getLogger(__name__)

##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from Tavily search API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        topic: Topic filter for search results (general, news, or finance)
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing summarized search results
    """
    logger.info(
        "Tavily search starting: %d queries (max_results=%d, topic=%s)",
        len(queries), max_results, topic,
    )
    # Step 1: Execute search queries asynchronously
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    
    # Step 2: Deduplicate results by URL to avoid processing the same content multiple times
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    logger.debug("Tavily search deduplicated to %d unique URLs", len(unique_results))

    # Step 3: Set up the summarization model with configuration
    configurable = Configuration.from_runnable_config(config)
    
    # Character limit to stay within model token limits (configurable)
    max_char_to_include = configurable.max_content_length
    
    # Initialize summarization model with retry logic
    model_config = build_model_config(
        configurable, configurable.summarization_model,
        configurable.summarization_model_max_tokens, config,
    )
    summarization_model = init_chat_model(
        model=model_config["model"],
        max_tokens=model_config["max_tokens"],
        api_key=model_config["api_key"],
        tags=model_config["tags"],
    ).with_structured_output(Summary, method="function_calling").with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    
    # Step 4: Create summarization tasks (skip empty content)
    async def noop():
        """No-op function for results without raw content."""
        return None
    
    summarization_tasks = [
        noop() if not result.get("raw_content") 
        else summarize_webpage(
            summarization_model, 
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]
    
    # Step 5: Execute all summarization tasks in parallel
    logger.debug(
        "Summarizing %d webpages with model %s",
        len(summarization_tasks), configurable.summarization_model,
    )
    summaries = await asyncio.gather(*summarization_tasks)
    
    # Step 6: Combine results with their summaries
    summarized_results = {
        url: {
            'title': result['title'], 
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(), 
            unique_results.values(), 
            summaries
        )
    }
    
    # Step 7: Format the final output
    if not summarized_results:
        logger.warning("Tavily search returned no valid results for %d queries", len(queries))
        return "No valid search results found. Please try different search queries or use a different search API."
    logger.info("Tavily search completed: %d summarized results", len(summarized_results))
    
    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    return formatted_output

async def tavily_search_async(
    search_queries, 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
    config: RunnableConfig = None
):
    """Execute multiple Tavily search queries asynchronously.
    
    Args:
        search_queries: List of search query strings to execute
        max_results: Maximum number of results per query
        topic: Topic category for filtering results
        include_raw_content: Whether to include full webpage content
        config: Runtime configuration for API key access
        
    Returns:
        List of search result dictionaries from Tavily API
    """
    # Initialize the Tavily client with API key from config
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    
    # Create search tasks for parallel execution
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]
    
    # Execute all search queries in parallel and return results
    logger.debug("Dispatching %d Tavily API queries (topic=%s)", len(search_tasks), topic)
    search_results = await asyncio.gather(*search_tasks)
    return search_results

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """Summarize webpage content using AI model with timeout protection.
    
    Args:
        model: The chat model configured for summarization
        webpage_content: Raw webpage content to be summarized
        
    Returns:
        Formatted summary with key excerpts, or original content if summarization fails
    """
    try:
        # Create prompt with current date context
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content, 
            date=get_today_str()
        )
        
        # Execute summarization with timeout to prevent hanging
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0  # 60 second timeout for summarization
        )
        
        # Format the summary with structured sections
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        
        return formatted_summary
        
    except asyncio.TimeoutError:
        # Timeout during summarization - return original content
        logger.warning("Summarization timed out after 60 seconds, returning original content")
        return webpage_content
    except Exception:
        # Other errors during summarization - log and return original content
        logger.exception("Summarization failed, returning original content")
        return webpage_content

##########################
# Reflection Tool Utils
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

##########################
# MCP Utils
##########################

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Dict[str, Any] | None:
    """Exchange Supabase token for MCP access token using OAuth token exchange.
    
    Args:
        supabase_token: Valid Supabase authentication token
        base_mcp_url: Base URL of the MCP server
        
    Returns:
        Token data dictionary if successful, None if failed
    """
    try:
        # Prepare OAuth token exchange request data
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        
        # Execute token exchange request
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            logger.info("Requesting MCP access token via OAuth token exchange")
            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # Successfully obtained token
                    token_data = await response.json()
                    logger.info("MCP access token obtained successfully")
                    return token_data
                else:
                    # Log error details for debugging
                    response_text = await response.text()
                    logger.error(
                        "Token exchange failed with status %s: %s",
                        response.status, response_text,
                    )

    except Exception:
        logger.exception("Error during MCP token exchange")

    return None

async def get_tokens(config: RunnableConfig):
    """Retrieve stored authentication tokens with expiration validation.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        
    Returns:
        Token dictionary if valid and not expired, None otherwise
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    
    # Retrieve stored tokens
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    
    # Check token expiration
    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    
    if current_time > expiration_time:
        # Token expired, clean up and return None
        logger.info("Stored MCP tokens expired; deleting cached tokens")
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """Store authentication tokens in the configuration store.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        tokens: Token dictionary to store
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    
    # Store the tokens
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """Fetch and refresh MCP tokens, obtaining new ones if needed.
    
    Args:
        config: Runtime configuration with authentication details
        
    Returns:
        Valid token dictionary, or None if unable to obtain tokens
    """
    # Try to get existing valid tokens first
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    
    # Extract Supabase token for new token exchange
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        logger.debug("No Supabase access token in config; cannot fetch MCP tokens")
        return None

    # Extract MCP configuration
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        logger.debug("No MCP config/url present; cannot fetch MCP tokens")
        return None

    # Exchange Supabase token for MCP tokens
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        logger.warning("MCP token exchange returned no tokens")
        return None

    # Store the new tokens and return them
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Wrap MCP tool with comprehensive authentication and error handling.
    
    Args:
        tool: The MCP structured tool to wrap
        
    Returns:
        Enhanced tool with authentication error handling
    """
    original_coroutine = tool.coroutine
    
    async def authentication_wrapper(**kwargs):
        """Enhanced coroutine with MCP error handling and user-friendly messages."""
        
        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """Recursively search for MCP errors in exception chains."""
            if isinstance(exc, McpError):
                return exc
            
            # Handle ExceptionGroup (Python 3.11+) by checking attributes
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None
        
        try:
            # Execute the original tool functionality
            return await original_coroutine(**kwargs)
            
        except BaseException as original_error:
            # Search for MCP-specific errors in the exception chain
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # Not an MCP error, re-raise the original exception
                logger.debug("MCP tool raised non-MCP error; re-raising")
                raise original_error
            
            # Handle MCP-specific error cases
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            
            # Check for authentication/interaction required error
            if error_code == -32003:  # Interaction required error code
                logger.warning("MCP tool requires interaction/authentication (code -32003)")
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"
                
                # Extract user-friendly message if available
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                
                # Append URL if provided for user reference
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                
                raise ToolException(error_message) from original_error
            
            # For other MCP errors, re-raise the original
            raise original_error
    
    # Replace the tool's coroutine with our enhanced version
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """Load and configure MCP (Model Context Protocol) tools with authentication.
    
    Args:
        config: Runtime configuration containing MCP server details
        existing_tool_names: Set of tool names already in use to avoid conflicts
        
    Returns:
        List of configured MCP tools ready for use
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Step 1: Handle authentication if required
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    
    # Step 2: Validate configuration requirements
    config_valid = (
        configurable.mcp_config and 
        configurable.mcp_config.url and 
        configurable.mcp_config.tools and 
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )
    
    if not config_valid:
        logger.debug("MCP config invalid or incomplete; loading no MCP tools")
        return []

    # Step 3: Set up MCP server connection
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    
    # Configure authentication headers if tokens are available
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
    
    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO: When Multi-MCP Server support is merged in OAP, update this code
    
    # Step 4: Load tools from MCP server
    try:
        logger.info("Connecting to MCP server at %s", server_url)
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
        logger.info("MCP server returned %d available tools", len(available_mcp_tools))
    except Exception:
        # If MCP server connection fails, return empty list
        logger.exception("Failed to connect to MCP server at %s; returning no tools", server_url)
        return []
    
    # Step 5: Filter and configure tools
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # Skip tools with conflicting names
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue
        
        # Only include tools specified in configuration
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue
        
        # Wrap tool with authentication handling and add to list
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)

    logger.info("Configured %d MCP tools after filtering", len(configured_tools))
    return configured_tools


##########################
# Tool Utils
##########################

async def get_search_tool(search_api: SearchAPI):
    """Configure and return search tools based on the specified API provider.
    
    Args:
        search_api: The search API provider to use (Anthropic, OpenAI, Tavily, or None)
        
    Returns:
        List of configured search tool objects for the specified provider
    """
    if search_api == SearchAPI.ANTHROPIC:
        # Anthropic's native web search with usage limits
        return [{
            "type": "web_search_20250305", 
            "name": "web_search", 
            "max_uses": 5
        }]
        
    elif search_api == SearchAPI.OPENAI:
        # OpenAI's web search preview functionality
        return [{"type": "web_search_preview"}]
        
    elif search_api == SearchAPI.TAVILY:
        # Configure Tavily search tool with metadata
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}), 
            "type": "search", 
            "name": "web_search"
        }
        return [search_tool]

    elif search_api == SearchAPI.NONE:
        # No search functionality configured
        return []

    # Default fallback for unknown search API types
    logger.warning("Unknown search API %r; returning no search tools", search_api)
    return []
    
async def get_all_tools(config: RunnableConfig):
    """Assemble complete toolkit including research, search, and MCP tools.
    
    Args:
        config: Runtime configuration specifying search API and MCP settings
        
    Returns:
        List of all configured and available tools for research operations
    """
    # Start with core research tools
    tools = [tool(ResearchComplete), think_tool]
    
    # Add configured search tools
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    logger.debug("Assembling tools with search API: %s", search_api)
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    
    # Track existing tool names to prevent conflicts
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    
    # Add MCP tools if configured
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)

    logger.info("Assembled %d total tools for research operations", len(tools))
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# Model Provider Native Websearch Utils
##########################

def anthropic_websearch_called(response):
    """Detect if Anthropic's native web search was used in the response.
    
    Args:
        response: The response object from Anthropic's API
        
    Returns:
        True if web search was called, False otherwise
    """
    try:
        # Navigate through the response metadata structure
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        
        # Check for server-side tool usage information
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        
        # Look for web search request count
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        
        # Return True if any web search requests were made
        return web_search_requests > 0
        
    except (AttributeError, TypeError):
        # Handle cases where response structure is unexpected
        logger.debug("Could not inspect Anthropic response for web search usage", exc_info=True)
        return False

def openai_websearch_called(response):
    """Detect if OpenAI's web search functionality was used in the response.
    
    Args:
        response: The response object from OpenAI's API
        
    Returns:
        True if web search was called, False otherwise
    """
    # Check for tool outputs in the response metadata
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False
    
    # Look for web search calls in the tool outputs
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True
    
    return False


##########################
# Token Limit Exceeded Utils
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded.
    
    Args:
        exception: The exception to analyze
        model_name: Optional model name to optimize provider detection
        
    Returns:
        True if the exception indicates a token limit was exceeded, False otherwise
    """
    error_str = str(exception).lower()
    
    # Step 1: Determine provider from model name if available
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    
    logger.debug(
        "Checking token-limit exception (provider=%s, type=%s)",
        provider or "unknown", type(exception).__name__,
    )

    # Step 2: Check provider-specific token limit patterns
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)

    # Step 3: If provider unknown, check all providers
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates OpenAI token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is an OpenAI exception
    is_openai_exception = (
        'openai' in exception_type.lower() or 
        'openai' in module_name.lower()
    )
    
    # Check for typical OpenAI token limit error types
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']
    
    if is_openai_exception and is_request_error:
        # Look for token-related keywords in error message
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    
    # Check for specific OpenAI error codes
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')
        
        if (error_code == 'context_length_exceeded' or
            error_type == 'invalid_request_error'):
            return True
    
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Anthropic token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is an Anthropic exception
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or 
        'anthropic' in module_name.lower()
    )
    
    # Check for Anthropic-specific error patterns
    is_bad_request = class_name == 'BadRequestError'
    
    if is_anthropic_exception and is_bad_request:
        # Anthropic uses specific error messages for token limits
        if 'prompt is too long' in error_str:
            return True
    
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Google/Gemini token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # Check if this is a Google/Gemini exception
    is_google_exception = (
        'google' in exception_type.lower() or 
        'google' in module_name.lower()
    )
    
    # Check for Google-specific resource exhaustion errors
    is_resource_exhausted = class_name in [
        'ResourceExhausted', 
        'GoogleGenerativeAIFetchError'
    ]
    
    if is_google_exception and is_resource_exhausted:
        return True
    
    # Check for specific Google API resource exhaustion patterns
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """Truncate message history by removing up to the last AI message.
    
    This is useful for handling token limit exceeded errors by removing recent context.
    
    Args:
        messages: List of message objects to truncate
        
    Returns:
        Truncated message list up to (but not including) the last AI message
    """
    # Search backwards through messages to find the last AI message
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # Return everything up to (but not including) the last AI message
            logger.debug("Truncating message history to %d messages (removed from last AI message)", i)
            return messages[:i]

    # No AI messages found, return original list
    logger.debug("No AI message found; returning message history unchanged")
    return messages

##########################
# Misc Utils
##########################

def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.
    
    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """Extract value from configuration, handling enums and None values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_tavily_api_key(config: RunnableConfig):
    """Get Tavily API key from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            logger.warning("GET_API_KEYS_FROM_CONFIG set but no apiKeys in config for Tavily")
            return None
        if not api_keys.get("TAVILY_API_KEY"):
            logger.warning("No TAVILY_API_KEY available in config")
        return api_keys.get("TAVILY_API_KEY")
    else:
        if not os.getenv("TAVILY_API_KEY"):
            logger.warning("No TAVILY_API_KEY available in environment")
        return os.getenv("TAVILY_API_KEY")


# ──────────────────────────────────────────────
# Phase 2: Academic Search Wrappers
# ──────────────────────────────────────────────

async def arxiv_search(query: str, max_results: int = 5) -> list:
    """Search arXiv for academic papers."""
    import httpx
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "max_results": max_results, "sortBy": "relevance"}
    logger.info("arXiv search starting: query=%.80r (max_results=%d)", query, max_results)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
    except Exception:
        logger.exception("arXiv search failed for query=%.80r", query)
        return []

    import xml.etree.ElementTree as ET
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        published_el = entry.find("atom:published", ns)
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None]
        pdf_link = ""
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_link = link.get("href", "")
                break
        results.append({
            "title": (title_el.text or "").strip() if title_el is not None else "",
            "url": (id_el.text or "").strip() if id_el is not None else "",
            "snippet": (summary_el.text or "").strip()[:500] if summary_el is not None else "",
            "source_type": "academic",
            "date": (published_el.text or "")[:10] if published_el is not None else None,
            "provider": "arxiv",
            "authors": authors,
            "pdf_url": pdf_link,
        })
    if not results:
        logger.warning("arXiv search returned no results for query=%.80r", query)
    else:
        logger.info("arXiv search completed: %d results", len(results))
    return results


async def semantic_scholar_search(query: str, max_results: int = 5) -> list:
    """Search Semantic Scholar for academic papers."""
    import httpx
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": max_results, "fields": "title,abstract,citationCount,authors,year,url,externalIds"}
    logger.info("Semantic Scholar search starting: query=%.80r (max_results=%d)", query, max_results)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        logger.exception("Semantic Scholar search failed for query=%.80r", query)
        return []

    results = []
    for paper in data.get("data", []):
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        ext_ids = paper.get("externalIds", {})
        results.append({
            "title": paper.get("title", ""),
            "url": paper.get("url", ""),
            "snippet": (paper.get("abstract") or "")[:500],
            "source_type": "academic",
            "date": f"{paper.get('year', '')}" if paper.get("year") else None,
            "provider": "semantic_scholar",
            "citation_count": paper.get("citationCount"),
            "doi": ext_ids.get("DOI"),
            "authors": authors,
        })
    if not results:
        logger.warning("Semantic Scholar search returned no results for query=%.80r", query)
    else:
        logger.info("Semantic Scholar search completed: %d results", len(results))
    return results


async def pubmed_search(query: str, max_results: int = 5) -> list:
    """Search PubMed for biomedical papers."""
    import httpx
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
    logger.info("PubMed search starting: query=%.80r (max_results=%d)", query, max_results)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            search_resp = await client.get(search_url, params=search_params)
            search_resp.raise_for_status()
            ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                logger.warning("PubMed search returned no IDs for query=%.80r", query)
                return []
            summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            summary_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
            summary_resp = await client.get(summary_url, params=summary_params)
            summary_resp.raise_for_status()
            results_data = summary_resp.json().get("result", {})
    except Exception:
        logger.exception("PubMed search failed for query=%.80r", query)
        return []

    results = []
    for pid in ids:
        paper = results_data.get(pid, {})
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        pubdate = paper.get("pubdate", "")
        results.append({
            "title": paper.get("title", ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
            "snippet": "",
            "source_type": "academic",
            "date": pubdate[:10] if pubdate else None,
            "provider": "pubmed",
            "authors": authors,
        })
    logger.info("PubMed search completed: %d results", len(results))
    return results


async def crossref_search(query: str, max_results: int = 5) -> list:
    """Search Crossref for academic papers."""
    import httpx
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": max_results}
    logger.info("Crossref search starting: query=%.80r (max_results=%d)", query, max_results)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])
    except Exception:
        logger.exception("Crossref search failed for query=%.80r", query)
        return []

    results = []
    for item in items:
        title_list = item.get("title", [])
        title = title_list[0] if title_list else ""
        authors = []
        for a in item.get("author", []):
            name = f"{a.get('given', '')} {a.get('family', '')}".strip()
            if name:
                authors.append(name)
        doi = item.get("DOI", "")
        pub_date_parts = item.get("published-print", {}).get("date-parts", [[]])
        date_str = None
        if pub_date_parts and pub_date_parts[0]:
            parts = pub_date_parts[0]
            if len(parts) >= 3:
                date_str = f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}"
            elif len(parts) >= 2:
                date_str = f"{parts[0]:04d}-{parts[1]:02d}"
            elif len(parts) >= 1:
                date_str = f"{parts[0]:04d}"
        results.append({
            "title": title,
            "url": item.get("URL", f"https://doi.org/{doi}"),
            "snippet": "",
            "source_type": "academic",
            "date": date_str,
            "provider": "crossref",
            "doi": doi,
            "citation_count": item.get("is-referenced-by-count"),
            "authors": authors,
        })
    if not results:
        logger.warning("Crossref search returned no results for query=%.80r", query)
    else:
        logger.info("Crossref search completed: %d results", len(results))
    return results


# ──────────────────────────────────────────────
# Phase 2: Source Diversity & Temporal Relevance
# ──────────────────────────────────────────────

def get_domain(url: str) -> str:
    """Extract domain from URL."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.netloc.lower().replace("www.", "")


def get_domain_histogram(results: List[dict]) -> Dict[str, int]:
    """Count results per domain."""
    histogram = {}
    for result in results:
        domain = get_domain(result.get("url", ""))
        if domain:
            histogram[domain] = histogram.get(domain, 0) + 1
    return histogram


def enforce_source_diversity(
    results: List[dict],
    min_unique_domains: int = 3,
    max_same_domain_ratio: float = 0.4,
) -> dict:
    """Check if source diversity meets thresholds."""
    if not results:
        return {"passed": True, "histogram": {}, "dominant_domain": None, "dominant_ratio": 0.0, "unique_domains": 0, "needs_supplement": False}
    histogram = get_domain_histogram(results)
    total = len(results)
    unique_domains = len(histogram)
    dominant_domain, dominant_ratio = None, 0.0
    for domain, count in histogram.items():
        ratio = count / total
        if ratio > dominant_ratio:
            dominant_ratio = ratio
            dominant_domain = domain
    passed = unique_domains >= min_unique_domains and dominant_ratio <= max_same_domain_ratio
    if not passed:
        logger.warning(
            "Source diversity check failed: %d unique domains (min=%d), dominant=%s ratio=%.2f (max=%.2f)",
            unique_domains, min_unique_domains, dominant_domain, dominant_ratio, max_same_domain_ratio,
        )
    else:
        logger.debug("Source diversity check passed: %d unique domains across %d results", unique_domains, total)
    return {
        "passed": passed,
        "histogram": histogram,
        "dominant_domain": dominant_domain,
        "dominant_ratio": round(dominant_ratio, 2),
        "unique_domains": unique_domains,
        "needs_supplement": not passed,
    }


def _parse_date(date_str) -> datetime | None:
    """Parse date string using fromisoformat with fallbacks."""
    from datetime import datetime
    if not date_str:
        return None
    if isinstance(date_str, list):
        # Handle Crossref date-parts: [2024, 6, 15] or [2024, 6]
        if len(date_str) >= 3:
            try:
                return datetime(date_str[0], date_str[1], date_str[2])
            except (ValueError, TypeError):
                pass
        elif len(date_str) == 2:
            try:
                return datetime(date_str[0], date_str[1], 1)
            except (ValueError, TypeError):
                pass
        return None
    try:
        return datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def temporal_relevance_boost(results: List[dict], max_age_days: int = 30) -> List[dict]:
    """Boost results by recency, returning sorted by recency_score."""
    from datetime import datetime
    now = datetime.now()
    boosted = []
    for result in results:
        date_str = result.get("date") or result.get("publication_date")
        date_obj = _parse_date(date_str)
        if date_obj:
            days_ago = max((now - date_obj).days, 0)
            recency_score = 1 / (1 + days_ago / 30)
        else:
            recency_score = 0.5
        result["recency_score"] = round(recency_score, 3)
        boosted.append(result)
    boosted.sort(key=lambda r: r.get("recency_score", 0), reverse=True)
    return boosted


# ──────────────────────────────────────────────
# Phase 2: Search Aggregator Factory
# ──────────────────────────────────────────────

def get_search_aggregator(config):
    """Create a SearchAggregator instance from configuration with real search functions."""
    from open_deep_research.search_aggregator import (
        SearchAggregator,
        SearchProviderConfig,
    )

    providers = []
    search_functions = {}

    # Tavily — wraps existing tavily_search tool if available
    if "tavily" in config.search_providers:
        providers.append(SearchProviderConfig(name="tavily", priority=1))

        async def tavily_search(query: str) -> list:
            import os

            from tavily import AsyncTavilyClient
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                logger.warning("Tavily provider selected but no TAVILY_API_KEY available")
                return []
            client = AsyncTavilyClient(api_key=api_key)
            logger.info("Tavily aggregator search: query=%.80r", query)
            response = await client.search(query, max_results=10)
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "provider": "tavily",
                    "source_type": "web",
                }
                for r in response.get("results", [])
            ]
        search_functions["tavily"] = tavily_search

    # DuckDuckGo
    if "duckduckgo" in config.search_providers:
        providers.append(SearchProviderConfig(name="duckduckgo", priority=2))

        async def duckduckgo_search(query: str) -> list:
            try:
                from duckduckgo_search import DDGS
                logger.info("DuckDuckGo aggregator search: query=%.80r", query)
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=10))
                    logger.info("DuckDuckGo search completed: %d results", len(results))
                    return [
                        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", ""), "provider": "duckduckgo"}
                        for r in results
                    ]
            except Exception:
                logger.exception("DuckDuckGo search failed for query=%.80r", query)
                return []
        search_functions["duckduckgo"] = duckduckgo_search

    # Academic providers
    if config.enable_academic_search:
        if config.arxiv_enabled:
            providers.append(SearchProviderConfig(name="arxiv", priority=5, query_pattern=r"(?:arxiv|paper|preprint|research)"))
            search_functions["arxiv"] = arxiv_search
        if config.semantic_scholar_enabled:
            providers.append(SearchProviderConfig(name="semantic_scholar", priority=6))
            search_functions["semantic_scholar"] = semantic_scholar_search
        if config.pubmed_enabled:
            providers.append(SearchProviderConfig(name="pubmed", priority=7, query_pattern=r"(?:pubmed|biomedical|medical)"))
            search_functions["pubmed"] = pubmed_search
        if config.crossref_enabled:
            providers.append(SearchProviderConfig(name="crossref", priority=8))
            search_functions["crossref"] = crossref_search

    logger.info(
        "Search aggregator built with %d providers: %s",
        len(providers), sorted(search_functions.keys()),
    )
    return SearchAggregator(providers, search_functions)
