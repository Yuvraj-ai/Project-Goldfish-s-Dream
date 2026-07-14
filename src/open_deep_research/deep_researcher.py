"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import logging
import re
import traceback
from datetime import datetime, timezone
from typing import List, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from open_deep_research.api.model_router import ModelRouter, ModelTier, TaskType
from open_deep_research.api.models import ProgressEvent
from open_deep_research.citation_verifier import CitationVerifier
from open_deep_research.configuration import (
    Configuration,
    ResolvedModel,
    build_model_config,
)
from open_deep_research.evidence import (
    compress_evidence,
    deduplicate_claims,
    detect_conflicts,
    extract_evidence,
)
from open_deep_research.exceptions import (
    ModelError,
    ToolPermanentError,
    ToolTransientError,
)
from open_deep_research.exporters import get_exporter
from open_deep_research.logging_config import setup_logging
from open_deep_research.prompts import (
    CLASSIFIER_HUMAN_PROMPT,
    CLASSIFIER_SYSTEM_PROMPT,
    PLANNER_HUMAN_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.report_profiles import MODE_TO_PROFILE, get_profile
from open_deep_research.reviewers import (
    CompletenessReviewer,
    ContradictionReviewer,
    CoverageReviewer,
    EvidenceReviewer,
    StyleReviewer,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    CitationCheck,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchPlanExtended,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# Configure package-wide logging once for library / LangGraph usage (the API
# entry point calls this too; it is idempotent).
setup_logging()

logger = logging.getLogger(__name__)


async def _emit_progress(config, event_type: str, phase: str, message: str) -> None:
    """Emit a progress event if repo and run_id are available in config."""
    repo = config.get("configurable", {}).get("repo")
    run_id = config.get("configurable", {}).get("run_id")
    if repo and run_id:
        try:
            seq = await repo.next_seq(run_id)
            await repo.append_progress(run_id, ProgressEvent(
                seq=seq, event_type=event_type, phase=phase,
                message=message,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
        except Exception:
            logger.exception("failed to emit progress event type=%s phase=%s", event_type, phase)

_router: ModelRouter | None = None

def get_model_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router

def _resolve_model_via_router(
    configurable: Configuration,
    config: RunnableConfig,
    task_type: TaskType,
    current_model: str,
    input_length: int = 0,
    complexity_hint: ModelTier | None = None,
) -> "ResolvedModel | None":
    """Resolve model via router. Returns ResolvedModel or None (fallback to current)."""
    if not configurable.enable_model_routing:
        return None

    router = get_model_router()
    result = router.select_with_tier_if_enabled(
        task_type=task_type,
        input_length=input_length,
        complexity_hint=complexity_hint,
        enabled=True,
    )
    if result is None:
        logger.debug("model router returned no selection for task=%s, using current=%s", task_type, current_model)
        return None

    routed_model_string, _ = result

    try:
        resolved = configurable.resolve_model(routed_model_string, config)
        return resolved
    except ValueError:
        logger.debug("could not resolve routed model=%s, falling back to current=%s", routed_model_string, current_model)
        return None

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "base_url"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["parse_document", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to document parsing
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    logger.info("clarify_with_user: entry, #messages=%d", len(state.get("messages", [])))
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to document parsing
        logger.debug("clarify_with_user: clarification disabled, skipping to parse_document")
        return Command(goto="parse_document")

    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    resolved = _resolve_model_via_router(
        configurable, config, TaskType.CLASSIFICATION,
        configurable.research_model,
    )
    if resolved is not None:
        model_config = build_model_config(
            configurable, resolved.model_string,
            configurable.research_model_max_tokens, config,
        )
    else:
        model_config = build_model_config(
            configurable, configurable.research_model,
            configurable.research_model_max_tokens, config,
        )
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser, method="function_calling")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    logger.info("clarify_with_user: invoking clarification model=%s", model_config["model"])
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    # Step 4: Route based on clarification analysis
    logger.info("clarify_with_user: exit, need_clarification=%s", response.need_clarification)
    if response.need_clarification:
        # End with clarifying question for user
        logger.debug("clarify_with_user: ending with clarifying question (%d chars)", len(response.question or ""))
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to document parsing with verification message
        logger.debug("clarify_with_user: proceeding to parse_document")
        return Command(
            goto="parse_document",
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    logger.info("write_research_brief: entry, #messages=%d", len(state.get("messages", [])))
    resolved = _resolve_model_via_router(
        configurable, config, TaskType.PLANNING,
        configurable.research_model,
    )
    if resolved is not None:
        research_model_config = build_model_config(
            configurable, resolved.model_string,
            configurable.research_model_max_tokens, config,
        )
    else:
        research_model_config = build_model_config(
            configurable, configurable.research_model,
            configurable.research_model_max_tokens, config,
        )

    # Configure model for structured research question generation
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion, method="function_calling")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    logger.info("write_research_brief: invoking research model=%s", research_model_config["model"])
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    logger.info("write_research_brief: exit, brief=%d chars, preview=%.80s",
                len(response.research_brief or ""), response.research_brief or "")
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="classify_research_request",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def classify_research_request(state: AgentState, config: RunnableConfig) -> Command[Literal["generate_research_plan"]]:
    """Classify the research request into a mode and generate report profile."""
    configurable = Configuration.from_runnable_config(config)
    logger.info("classify_research_request: entry")

    if not configurable.enable_mode_classification:
        logger.warning("classify_research_request: mode classification disabled, using default mode=%s",
                       configurable.default_research_mode.value)
        return Command(
            goto="generate_research_plan",
            update={"research_mode": configurable.default_research_mode.value}
        )

    class ClassificationResult(BaseModel):
        mode: str
        confidence: float
        reasoning: str
        suggested_report_profile: dict

    classifier_model_string = configurable.classifier_model or configurable.research_model
    resolved = _resolve_model_via_router(
        configurable, config, TaskType.CLASSIFICATION,
        classifier_model_string,
    )
    if resolved is not None:
        classifier_model_config = build_model_config(
            configurable, resolved.model_string, 1024, config,
        )
    else:
        classifier_model_config = build_model_config(
            configurable, classifier_model_string, 1024, config,
        )

    classifier_model = (
        configurable_model
        .with_structured_output(ClassificationResult, method="function_calling")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(classifier_model_config)
    )

    research_brief = state.get("research_brief", "")
    prompt = CLASSIFIER_HUMAN_PROMPT.format(query=research_brief, date=get_today_str())
    messages = [SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    logger.info("classify_research_request: invoking classifier model=%s", classifier_model_config["model"])
    response = await classifier_model.ainvoke(messages)

    # Confidence fallback: if below threshold, use CUSTOM mode
    mode = response.mode
    logger.info("classify_research_request: classified mode=%s confidence=%.2f", response.mode, response.confidence)
    if response.confidence < configurable.classifier_confidence_threshold:
        logger.info(f"Classifier confidence {response.confidence:.2f} < {configurable.classifier_confidence_threshold}, falling back to CUSTOM")
        mode = "custom"

    logger.info("classify_research_request: exit, mode=%s", mode)
    return Command(
        goto="generate_research_plan",
        update={
            "research_mode": mode,
            "report_profile": response.suggested_report_profile,
        }
    )


async def parse_document(state: AgentInputState, config: RunnableConfig):
    """Parse uploaded documents into structured artifacts and extract evidence."""
    configurable = Configuration.from_runnable_config(config)
    logger.info("parse_document: entry")

    if not configurable.enable_document_reading:
        logger.debug("parse_document: document reading disabled, skipping")
        return {}

    document_paths = state.get("document_paths", [])
    if not document_paths:
        logger.debug("parse_document: no document paths provided, skipping")
        return {}

    from open_deep_research.document_reader import DocumentReader

    logger.info("parse_document: parsing %d document(s)", len(document_paths))
    reader = DocumentReader()
    artifacts = []
    all_evidence = []

    for path in document_paths:
        artifact = await reader.parse_pdf(path)
        artifacts.append(artifact.model_dump())
        logger.debug("parse_document: parsed doc, parse_confidence=%.2f", artifact.parse_confidence)

        if artifact.parse_confidence > 0.5:
            cards = await reader.extract_evidence_cards(artifact)
            logger.debug("parse_document: extracted %d evidence cards from document", len(cards))
            all_evidence.extend(cards)
        else:
            logger.warning("parse_document: low parse_confidence=%.2f, skipping evidence extraction",
                           artifact.parse_confidence)

    logger.info("parse_document: exit, %d artifacts, %d evidence cards",
                len(artifacts), len(all_evidence))
    return {
        "document_artifacts": artifacts,
        "evidence_cards": all_evidence,
    }


async def generate_research_plan(state: AgentState, config: RunnableConfig) -> Command[Literal["optional_plan_review"]]:
    """Generate a detailed research plan based on the classified mode."""
    configurable = Configuration.from_runnable_config(config)
    logger.info("generate_research_plan: entry, mode=%s", state.get("research_mode", "custom"))

    class PlanResult(BaseModel):
        objective: str
        subquestions: list[str]
        search_strategy: dict
        expected_source_types: list[str]
        proposed_sections: list[str]
        stop_conditions: list[str]
        risks: list[str]

        @field_validator("subquestions", "expected_source_types", "proposed_sections", "stop_conditions", "risks", mode="before")
        @classmethod
        def coerce_string_to_list(cls, v):
            if isinstance(v, str):
                return [v]
            return v

        @field_validator("search_strategy", mode="before")
        @classmethod
        def flatten_search_strategy(cls, v):
            if not isinstance(v, dict):
                return v
            # Check if nested per-subquestion format (values are dicts)
            if v and all(isinstance(val, dict) for val in v.values()):
                flat: dict[str, list[str]] = {}
                for subq_val in v.values():
                    for provider, sources in subq_val.items():
                        if isinstance(sources, list):
                            flat.setdefault(provider, []).extend(sources)
                        elif isinstance(sources, str):
                            flat.setdefault(provider, []).append(sources)
                return flat
            # Coerce string values to single-element lists
            if v and all(isinstance(val, str) for val in v.values()):
                return {k: [v] for k, v in v.items()}
            return v

    resolved = _resolve_model_via_router(
        configurable, config, TaskType.PLANNING,
        configurable.research_model,
    )
    if resolved is not None:
        planner_model_config = build_model_config(
            configurable, resolved.model_string, 2048, config,
        )
    else:
        planner_model_config = build_model_config(
            configurable, configurable.research_model, 2048, config,
        )

    planner_model = (
        configurable_model
        .with_structured_output(PlanResult, method="function_calling")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(planner_model_config)
    )

    research_brief = state.get("research_brief", "")
    research_mode = state.get("research_mode", "custom")

    prompt = PLANNER_HUMAN_PROMPT.format(brief=research_brief, mode=research_mode, date=get_today_str())
    messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    logger.info("generate_research_plan: invoking planner model=%s", planner_model_config["model"])
    response = await planner_model.ainvoke(messages)
    logger.info("generate_research_plan: exit, %d subquestions, %d proposed sections",
                len(response.subquestions), len(response.proposed_sections))

    plan = ResearchPlanExtended(
        objective=response.objective,
        subquestions=response.subquestions,
        search_strategy=response.search_strategy,
        expected_source_types=response.expected_source_types,
        proposed_sections=response.proposed_sections,
        stop_conditions=response.stop_conditions,
        risks=response.risks,
    )

    return Command(
        goto="optional_plan_review",
        update={"research_plan": plan.model_dump()}
    )


async def optional_plan_review(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor", "generate_research_plan"]]:
    """Review plan with rejection loop (max 2 revisions)."""
    configurable = Configuration.from_runnable_config(config)
    logger.info("optional_plan_review: entry, mode=%s", configurable.plan_review_mode)

    if configurable.plan_review_mode == "none":
        logger.debug("optional_plan_review: review mode 'none', proceeding to supervisor")
        return Command(goto="research_supervisor")

    elif configurable.plan_review_mode == "interrupt":
        from langgraph.types import interrupt
        plan = state.get("research_plan", {})
        review_result = interrupt({
            "plan": plan,
            "message": "Please review the research plan. Approve or provide feedback."
        })

        if review_result.get("approved", False):
            logger.info("optional_plan_review: plan approved, proceeding to supervisor")
            return Command(goto="research_supervisor")
        else:
            # Check revision count
            revision_count = state.get("plan_revision_count", 0)
            if revision_count >= configurable.max_plan_revisions:
                logger.warning(f"Max plan revisions ({configurable.max_plan_revisions}) reached, proceeding")
                return Command(goto="research_supervisor")

            # Loop back with feedback
            logger.info("optional_plan_review: plan rejected, revising (revision %d)", revision_count + 1)
            return Command(
                goto="generate_research_plan",
                update={"plan_revision_count": revision_count + 1}
            )

    elif configurable.plan_review_mode == "auto_review":
        logger.debug("optional_plan_review: auto_review mode, proceeding to supervisor")
        return Command(goto="research_supervisor")

    logger.debug("optional_plan_review: default path, proceeding to supervisor")
    return Command(goto="research_supervisor")


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    logger.info("supervisor: entry, iteration=%d", state.get("research_iterations", 0))
    resolved = _resolve_model_via_router(
        configurable, config, TaskType.REASONING,
        configurable.research_model,
    )
    if resolved is not None:
        research_model_config = build_model_config(
            configurable, resolved.model_string,
            configurable.research_model_max_tokens, config,
        )
    else:
        research_model_config = build_model_config(
            configurable, configurable.research_model,
            configurable.research_model_max_tokens, config,
        )

    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    logger.info("supervisor: invoking model=%s with %d messages",
                research_model_config["model"], len(supervisor_messages))
    response = await research_model.ainvoke(supervisor_messages)
    logger.info("supervisor: exit, %d tool call(s) requested", len(response.tool_calls or []))

    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    logger.info("supervisor_tools: entry, iteration=%d", research_iterations)

    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        logger.info("supervisor_tools: research phase ending (exceeded_iters=%s, no_tool_calls=%s, research_complete=%s)",
                    exceeded_allowed_iterations, no_tool_calls, research_complete_tool_call)
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    if think_tool_calls:
        logger.debug("supervisor_tools: recording %d think_tool reflection(s)", len(think_tool_calls))
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            logger.info("supervisor_tools: delegating %d research unit(s) (%d overflowed, cap=%d)",
                        len(allowed_conduct_research_calls), len(overflow_conduct_research_calls),
                        configurable.max_concurrent_research_units)
            if overflow_conduct_research_calls:
                logger.warning("supervisor_tools: %d research call(s) exceeded concurrency cap and were skipped",
                               len(overflow_conduct_research_calls))

            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            logger.info("supervisor_tools: %d research unit(s) completed", len(tool_results))

            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
            
            # Collect sources and evidence cards from all research results
            all_sources = []
            all_evidence = []
            for observation in tool_results:
                all_sources.extend(observation.get("sources", []))
                all_evidence.extend(observation.get("evidence_cards", []))
            logger.debug("supervisor_tools: aggregated %d source(s), %d evidence card(s)",
                         len(all_sources), len(all_evidence))
            if all_sources:
                update_payload["sources"] = all_sources
            if all_evidence:
                update_payload["evidence_cards"] = all_evidence

        except Exception as e:
            # Handle research execution errors with specific types
            error_msg = f"Research error: {type(e).__name__}: {e}"
            error_type = type(e).__name__

            if is_token_limit_exceeded(e, configurable.research_model):
                error_type = "TokenLimitError"
                error_msg = f"Token limit exceeded: {e}"
            elif isinstance(e, ToolTransientError):
                # Retryable — log and re-raise
                logger.warning(f"Transient tool error (retryable): {e}")
                raise
            elif isinstance(e, (ToolPermanentError, ModelError)):
                error_msg = f"Tool/model failure: {e}"

            logger.exception("supervisor_tools: research delegation failed, error_type=%s", error_type)
            return Command(
                goto=END,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "research_brief": state.get("research_brief", ""),
                    "error_artifact": {
                        "node": "supervisor_tools",
                        "error_type": error_type,
                        "error_message": error_msg,
                        "stack_trace": traceback.format_exc(),
                    }
                }
            )
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    logger.info("supervisor_tools: exit, returning %d tool message(s) to supervisor",
                len(all_tool_messages))
    return Command(
        goto="supervisor",
        update=update_payload
    )

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    logger.info("researcher: entry, tool_call_iteration=%d, topic=%.80s",
                state.get("tool_call_iterations", 0), state.get("research_topic", ""))

    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    logger.debug("researcher: %d tool(s) available", len(tools))
    if len(tools) == 0:
        logger.error("researcher: no tools available for research")
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    resolved = _resolve_model_via_router(
        configurable, config, TaskType.EXTRACTION,
        configurable.research_model,
    )
    if resolved is not None:
        research_model_config = build_model_config(
            configurable, resolved.model_string,
            configurable.research_model_max_tokens, config,
        )
    else:
        research_model_config = build_model_config(
            configurable, configurable.research_model,
            configurable.research_model_max_tokens, config,
        )
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    logger.info("researcher: invoking model=%s with %d tool(s) bound",
                research_model_config["model"], len(tools))
    response = await research_model.ainvoke(messages)
    logger.info("researcher: exit, %d tool call(s) requested", len(response.tool_calls or []))

    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    tool_name = getattr(tool, "name", "unknown")
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        logger.exception("execute_tool_safely: tool=%s raised, returning error string", tool_name)
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    logger.info("researcher_tools: entry, tool_call_iteration=%d",
                state.get("tool_call_iterations", 0))

    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or
        anthropic_websearch_called(most_recent_message)
    )

    if not has_tool_calls and not has_native_search:
        logger.info("researcher_tools: no tool calls, proceeding to compress_research")
        return Command(goto="compress_research")

    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    logger.info("researcher_tools: executing %d tool call(s) in parallel", len(tool_calls))
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    logger.debug("researcher_tools: %d tool observation(s) collected", len(observations))
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        logger.info("researcher_tools: exit to compress_research (exceeded_iters=%s, research_complete=%s)",
                    exceeded_iterations, research_complete_called)
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )

    # Continue research loop with tool results
    logger.info("researcher_tools: exit, continuing research loop")
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def extract_structured_evidence(state: ResearcherState, config: RunnableConfig):
    """Extract structured evidence cards from researcher outputs.

    This node runs when enable_evidence_first is True, extracting structured
    EvidenceCards from raw research findings. When disabled, falls back to
    legacy flat-text compression.
    """
    configurable = Configuration.from_runnable_config(config)
    logger.info("extract_structured_evidence: entry")

    if not configurable.enable_evidence_first:
        logger.debug("extract_structured_evidence: evidence-first disabled, using legacy compression")
        return {}

    # Parse search tool results from tool messages to extract URLs
    # Search tools format results as: "--- SOURCE N: <title> ---\nURL: <url>\n\nSUMMARY:\n<content>"
    raw_results = []
    seen_urls = set()
    researcher_messages = state.get("researcher_messages", [])
    for msg in researcher_messages:
        if hasattr(msg, "type") and msg.type == "tool":
            content = str(msg.content)
            # Extract structured results from formatted search output
            import re
            for match in re.finditer(
                r"--- SOURCE \d+: (?P<title>.+?) ---\s*URL: (?P<url>https?://\S+)\s*SUMMARY:\s*(?P<content>.+?)(?:\n---|\Z)",
                content, re.DOTALL
            ):
                url = match.group("url").rstrip("/")
                if url not in seen_urls:
                    seen_urls.add(url)
                    raw_results.append({
                        "url": url,
                        "title": match.group("title").strip(),
                        "content": match.group("content").strip()[:2000],
                    })

    research_topic = state.get("research_topic", "")
    raw_count = len(raw_results)
    raw_results = [r for r in raw_results if score_source_relevance(research_topic, r) >= 0.2]
    logger.debug("extract_structured_evidence: %d raw results, %d after relevance filter",
                 raw_count, len(raw_results))

    if not raw_results:
        logger.warning("extract_structured_evidence: no relevant results after filtering, returning empty")
        return {}

    cards, sources = extract_evidence(
        raw_results=raw_results,
        subquestion_id="main",
        researcher_id="supervisor",
    )
    logger.info("extract_structured_evidence: extracted %d card(s) from %d source(s)",
                len(cards), len(sources))

    unique_cards, _ = deduplicate_claims(cards)
    detect_conflicts(unique_cards)
    compressed = compress_evidence(unique_cards)
    logger.info("extract_structured_evidence: exit, %d unique -> %d compressed card(s)",
                len(unique_cards), len(compressed))

    return {
        "evidence_cards": [card.model_dump() for card in compressed],
        "sources": [src.model_dump() for src in sources],
    }


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    logger.info("compress_research: entry, #researcher_messages=%d",
                len(state.get("researcher_messages", [])))
    synthesizer_model = configurable_model.with_config(
        build_model_config(
            configurable, configurable.compression_model,
            configurable.compression_model_max_tokens, config,
        )
    )
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages

            # Execute compression
            logger.info("compress_research: invoking synthesizer model=%s (attempt %d/%d)",
                        configurable.compression_model, synthesis_attempts + 1, max_attempts)
            response = await synthesizer_model.ainvoke(messages)

            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content)
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])

            # Return successful compression result
            logger.info("compress_research: exit, compressed=%d chars, raw_notes=%d chars",
                        len(str(response.content)), len(raw_notes_content))
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }

        except Exception as e:
            synthesis_attempts += 1

            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                logger.warning("compress_research: token limit exceeded on attempt %d, truncating messages",
                               synthesis_attempts)
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue

            # For other errors, continue retrying
            logger.exception("compress_research: synthesis attempt %d failed, retrying", synthesis_attempts)
            continue

    # Step 4: Return error result if all attempts failed
    logger.error("compress_research: all %d synthesis attempts failed, returning error result", max_attempts)
    raw_notes_content = "\n".join([
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])

    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression
researcher_builder.add_node("extract_structured_evidence", extract_structured_evidence)  # Evidence extraction

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", "extract_structured_evidence")  # Compression to evidence
researcher_builder.add_edge("extract_structured_evidence", END)  # Evidence extraction to exit

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def verify_citations(state: AgentState, config: RunnableConfig) -> dict:
    """Node: Verify all citations in evidence cards and sources."""
    await _emit_progress(config, "phase_start", "verify_citations", "Starting citation verification...")
    configurable = Configuration.from_runnable_config(config)
    logger.info("verify_citations: entry")

    if not configurable.enable_citation_verification:
        logger.debug("verify_citations: citation verification disabled, skipping")
        return {"citation_checks": []}

    sources = state.get("sources", [])
    urls = list({s.get("url", "") for s in sources if s.get("url")})

    if not urls:
        logger.warning("verify_citations: no source URLs to verify, returning empty")
        return {"citation_checks": []}

    logger.info("verify_citations: verifying %d unique URL(s) across %d source(s)", len(urls), len(sources))
    verifier = CitationVerifier()
    try:
        verification_results = await verifier.verify_batch(urls)

        citation_checks = []
        alive_count = 0
        for source in sources:
            url = source.get("url", "")
            result = verification_results.get(url, {"status": "unverified"})

            raw_status = result.get("status", "unverified")
            mapped_status = "verified" if raw_status == "alive" else raw_status
            if raw_status == "alive":
                alive_count += 1
            check = CitationCheck(
                claim="",
                url=url,
                status=mapped_status,
                problem="" if raw_status == "alive" else f"URL {raw_status}",
            )
            citation_checks.append(check.model_dump())

        logger.info("verify_citations: exit, %d check(s), %d alive/verified",
                    len(citation_checks), alive_count)
        return {"citation_checks": citation_checks}
    finally:
        await verifier.close()


def _allocate_evidence(subquestions: list, evidence_cards: list) -> dict:
    """Allocate evidence cards to sections by subquestion_id.

    Uses actual subquestion text as keys (matching what's on evidence cards).
    Falls back to "main" for cards with unaligned subquestion_id.
    """
    allocation = {}
    if not subquestions:
        # No subquestions — put all evidence under "main"
        allocation["main"] = [card.get("id", "") for card in evidence_cards]
        logger.debug("_allocate_evidence: no subquestions, %d card(s) -> 'main' bucket",
                     len(evidence_cards))
        return allocation

    # Initialize allocation per subquestion (using actual text as key)
    for sq in subquestions:
        allocation[sq] = []

    # Also create a "main" bucket for unaligned cards
    allocation["main"] = []

    # Assign cards by subquestion_id
    for card in evidence_cards:
        sq_id = card.get("subquestion_id", "")
        if sq_id in allocation:
            allocation[sq_id].append(card.get("id", ""))
        elif sq_id == "main":
            allocation["main"].append(card.get("id", ""))
        else:
            # Unknown subquestion_id — try keyword matching
            matched = False
            for sq in subquestions:
                sq_words = set(sq.lower().split())
                card_words = set(card.get("claim", "").lower().split())
                if len(sq_words & card_words) >= 2:  # At least 2 words in common
                    allocation[sq].append(card.get("id", ""))
                    matched = True
                    break
            if not matched:
                allocation["main"].append(card.get("id", ""))

    main_count = len(allocation.get("main", []))
    if main_count:
        logger.debug("_allocate_evidence: %d card(s) fell back to 'main' bucket", main_count)
    logger.debug("_allocate_evidence: allocated %d card(s) across %d bucket(s)",
                 len(evidence_cards), len(allocation))
    return allocation


def _map_section_to_subquestion(section: dict, subquestions: list) -> str:
    """Map a section to the best-matching subquestion using keyword overlap."""
    section_title = section.get("title", "").lower()
    section_desc = section.get("description", "").lower()
    section_words = set(section_title.split()) | set(section_desc.split())

    best_sq = ""
    best_score = 0

    for sq in subquestions:
        sq_words = set(sq.lower().split())
        overlap = len(section_words & sq_words)
        if overlap > best_score:
            best_score = overlap
            best_sq = sq

    return best_sq if best_score >= 1 else ""


def _format_evidence_cards(cards: list, citation_index: dict | None = None) -> str:
    """Format evidence cards for section writer prompt.

    When citation_index is provided, uses sequential [N] identifiers
    instead of raw internal IDs.
    """
    if not cards:
        return "No specific evidence cards allocated."
    lines = []
    for card in cards:
        card_id = card.get("id", "unknown")
        if citation_index and card_id in citation_index:
            ref = f"[{citation_index[card_id]['number']}]"
        else:
            ref = f"[{card_id}]"
        lines.append(
            f"- {ref} {card.get('claim', '')} "
            f"(confidence: {card.get('confidence', 0):.2f})"
        )
        for excerpt in card.get("exact_excerpts", [])[:2]:
            lines.append(f"  Excerpt: {excerpt[:200]}")
    return "\n".join(lines)


def score_source_relevance(query: str, source: dict) -> float:
    query_words = {w.lower() for w in query.split() if len(w) > 3}
    if not query_words:
        return 1.0
    title = source.get("title", "")
    content = source.get("content", "")[:500]
    source_text = (title + " " + content).lower()
    source_words = {w.lower() for w in source_text.split() if len(w) > 3}
    if not source_words:
        return 0.0
    return len(query_words & source_words) / len(query_words)


def _remove_placeholder_sentences(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered = [s for s in sentences if not PLACEHOLDER_PATTERNS.search(s)]
    return " ".join(filtered)


def _replace_raw_ids_in_text(text: str, citation_index: dict) -> str:
    """Replace any remaining raw evidence card IDs with sequential [N] refs.

    Matches bracket-wrapped IDs like [researcher_1_0] or [functions.supervisor_1_0]
    that appear in citation_index, and replaces them with [N].
    Does not touch already-correct [N] references.
    """
    for match in re.finditer(r'\[([^\]]+)\]', text):
        potential_id = match.group(1)
        if potential_id in citation_index:
            text = text.replace(
                f"[{potential_id}]",
                f"[{citation_index[potential_id]['number']}]",
            )
    return text


def _format_bibliography_from_index(citation_index: dict) -> str:
    """Format a numbered bibliography from citation_index ordering.

    Each citation_index entry should have 'number', 'title', and 'url' keys.
    Entries without a 'number' key are skipped.
    """
    sorted_entries = sorted(
        (e for e in citation_index.values() if "number" in e),
        key=lambda e: e["number"],
    )
    lines = []
    for entry in sorted_entries:
        title = entry.get("title", "Untitled")
        url = entry.get("url", "")
        lines.append(f"[{entry['number']}] {title}. {url}")
    return "\n".join(lines)


def _build_citation_index(
    evidence_allocation: dict,
    evidence_cards: list,
    sources: list,
) -> dict:
    """Build sequential citation index from allocated evidence cards.

    Maps each card ID to a sequential number and its source metadata.
    Ensures each card appears once even if allocated to multiple buckets.
    """
    seen = set()
    ordered_ids = []
    for bucket in evidence_allocation.values():
        for cid in bucket:
            if cid not in seen:
                seen.add(cid)
                ordered_ids.append(cid)

    source_by_url = {s.get("url", ""): s for s in sources}
    card_by_id = {c.get("id", ""): c for c in evidence_cards}

    citation_index = {}
    for idx, card_id in enumerate(ordered_ids, 1):
        card = card_by_id.get(card_id)
        if not card:
            continue
        source_urls = card.get("supporting_source_ids", [])
        source = source_by_url.get(source_urls[0]) if source_urls else None
        citation_index[card_id] = {
            "number": idx,
            "title": source.get("title", "Untitled") if source else "Untitled",
            "url": source_urls[0] if source_urls else "",
        }
    logger.debug("_build_citation_index: indexed %d citation(s) from %d ordered id(s)",
                 len(citation_index), len(ordered_ids))
    return citation_index


def _build_section_prompt(
    section: dict,
    profile,
    evidence_cards: list,
    research_topic: str,
    citation_index: dict | None = None,
) -> str:
    """Build the section writer prompt with optional premise cross-check.

    For fact_check profiles, injects the actual research topic and a
    premise cross-check instruction that the model can act on.
    When citation_index is provided, uses [N] identifiers for evidence refs.
    """
    card_text = _format_evidence_cards(evidence_cards, citation_index)
    prompt = (
        f'Write the "{section["title"]}" section of a {profile.tone} research report.\n\n'
        f'Section description: {section["description"]}\n\n'
        f"Evidence cards to incorporate:\n{card_text}\n\n"
        f"Tone: {profile.tone}\n\n"
        "CRITICAL INSTRUCTION ON CONFLICTS:\n"
        "When sources disagree, present the disagreement explicitly rather than synthesizing "
        "a false consensus. Never average contradictory claims into a middle position.\n\n"
        "CRITICAL INSTRUCTION ON CITATIONS:\n"
        "Reference evidence cards by their [N] identifier. Do not generate new citations or "
        "a References section — the system handles bibliography formatting.\n\n"
    )
    if profile.name == "fact_check":
        prompt += (
            f"You are evaluating the following claim: {research_topic}\n\n"
            "CRITICAL: Before writing, determine if this claim's premise is true or false. "
            "If evidence cards contradict the premise, prioritize the evidence and "
            "explicitly state the contradiction. Never validate a false premise.\n\n"
        )
    prompt += "Write a well-structured section with inline citations using [N] identifiers."
    return prompt


def _resolve_profile_name(state: AgentState, config: Configuration) -> str:
    """Resolve report profile name from config override, research mode, or default."""
    if config.report_profile_override:
        logger.debug("_resolve_profile_name: using override profile=%s", config.report_profile_override)
        return config.report_profile_override
    research_mode = state.get("research_mode", "custom")
    if research_mode in MODE_TO_PROFILE:
        logger.debug("_resolve_profile_name: mode=%s -> profile=%s",
                     research_mode, MODE_TO_PROFILE[research_mode])
        return MODE_TO_PROFILE[research_mode]
    logger.warning("_resolve_profile_name: mode=%s has no profile mapping, falling back to deep_research_report",
                   research_mode)
    return "deep_research_report"


async def generate_report_outline(state: AgentState, config: RunnableConfig) -> dict:
    """Generate structured report outline based on profile and evidence."""
    await _emit_progress(config, "phase_start", "generate_report_outline", "Generating report outline...")
    configurable = Configuration.from_runnable_config(config)
    logger.info("generate_report_outline: entry")
    if not configurable.enable_section_writers:
        logger.warning("generate_report_outline: section writers disabled, skipping outline generation")
        return {}
    profile_name = _resolve_profile_name(state, configurable)
    profile = get_profile(profile_name) or get_profile("deep_research_report")
    outline = []
    for section in profile.required_sections:
        outline.append({"title": section.name, "description": section.description, "required": True})
    for section in profile.optional_sections:
        outline.append({"title": section.name, "description": section.description, "required": False})
    subquestions = state.get("research_plan", {}).get("subquestions", [])
    evidence_cards = state.get("evidence_cards", [])
    evidence_allocation = _allocate_evidence(subquestions, evidence_cards)
    sources = state.get("sources", [])
    citation_index = _build_citation_index(evidence_allocation, evidence_cards, sources)
    logger.info("generate_report_outline: exit, profile=%s, %d section(s), %d subquestion(s), %d evidence card(s)",
                profile_name, len(outline), len(subquestions), len(evidence_cards))
    return {
        "report_outline": {
            "profile": profile_name,
            "sections": outline,
            "evidence_allocation": evidence_allocation,
            "subquestions": subquestions,
            "citation_index": citation_index,
        }
    }


class SectionOutput(BaseModel):
    """Structured output for section writer."""

    section_title: str
    content: str
    citation_ids: List[str] = Field(default_factory=list)
    acknowledged_conflicts: List[str] = Field(default_factory=list)

    @field_validator("citation_ids", "acknowledged_conflicts", mode="before")
    @classmethod
    def coerce_none_to_empty_list(cls, v):
        if v is None:
            return []
        return v


async def write_section(
    section: dict, evidence_cards: list, profile, subquestions: list,
    research_topic: str, config: RunnableConfig,
    citation_index: dict | None = None,
) -> dict:
    """Write a single report section with evidence and citations using structured output."""
    configurable = Configuration.from_runnable_config(config)
    logger.info("write_section: entry, section=%.80s", section.get("title", ""))
    resolved = _resolve_model_via_router(
        configurable, config, TaskType.REPORT_WRITING,
        configurable.final_report_model,
        complexity_hint=ModelTier.QUALITY,
    )
    if resolved is not None:
        writer_model_config = build_model_config(
            configurable, resolved.model_string,
            configurable.final_report_model_max_tokens, config,
        )
    else:
        writer_model_config = build_model_config(
            configurable, configurable.final_report_model,
            configurable.final_report_model_max_tokens, config,
        )
    # Map section to best-matching subquestion using keyword overlap
    matched_sq = _map_section_to_subquestion(section, subquestions)
    evidence_allocation = section.get("evidence_allocation", {})
    card_ids = evidence_allocation.get(matched_sq, []) if matched_sq else []
    # If no match, try "main" bucket
    if not card_ids:
        logger.debug("write_section: section=%.60s no matched subquestion, using 'main' bucket",
                     section.get("title", ""))
        card_ids = evidence_allocation.get("main", [])
    relevant_cards = [c for c in evidence_cards if c.get("id") in card_ids]
    if not relevant_cards:
        logger.warning("write_section: section=%.60s has no allocated evidence cards",
                       section.get("title", ""))
    section_prompt = _build_section_prompt(section, profile, relevant_cards, research_topic, citation_index)
    writer_model = (
        configurable_model.with_structured_output(SectionOutput, method="function_calling")
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(writer_model_config)
    )
    logger.info("write_section: invoking writer model=%s for section=%.60s with %d card(s)",
                writer_model_config["model"], section.get("title", ""), len(relevant_cards))
    response = await writer_model.ainvoke([HumanMessage(content=section_prompt)])
    logger.info("write_section: exit, section=%.60s, content=%d chars, %d citation(s)",
                response.section_title, len(response.content or ""), len(response.citation_ids))
    return {
        "section_title": response.section_title,
        "content": response.content,
        "citation_ids": response.citation_ids,
        "acknowledged_conflicts": response.acknowledged_conflicts,
    }


async def write_sections_parallel(state: AgentState, config: RunnableConfig) -> dict:
    """Write all report sections in parallel. Returns replaced list (not appended)."""
    await _emit_progress(config, "phase_start", "write_sections_parallel", "Writing report sections...")
    configurable = Configuration.from_runnable_config(config)
    logger.info("write_sections_parallel: entry")
    if not configurable.enable_section_writers:
        logger.warning("write_sections_parallel: section writers disabled, returning None")
        return {"written_sections": None}
    outline = state.get("report_outline", {})
    sections = outline.get("sections", [])
    evidence_cards = state.get("evidence_cards", [])
    evidence_allocation = outline.get("evidence_allocation", {})
    subquestions = outline.get("subquestions", [])
    profile_name = outline.get("profile", "deep_research_report")
    profile = get_profile(profile_name) or get_profile("deep_research_report")

    # Pass evidence allocation to each section
    for section in sections:
        section["evidence_allocation"] = evidence_allocation

    research_topic = state.get("research_brief", "")
    citation_index = outline.get("citation_index")
    logger.info("write_sections_parallel: writing %d section(s) in parallel with profile=%s",
                len(sections), profile_name)
    tasks = [
        write_section(section, evidence_cards, profile, subquestions,
                      research_topic, config, citation_index)
        for section in sections
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    written_sections = []
    failed = 0
    for result in results:
        if isinstance(result, Exception):
            failed += 1
            logger.warning(f"Section writer failed: {result}")
            continue
        written_sections.append(result)
    logger.info("write_sections_parallel: exit, %d section(s) written, %d failed",
                len(written_sections), failed)
    return {"written_sections": written_sections}


_ET_AL_RE = re.compile(r'\([A-Za-z]+ et al\., \d{4}\)')
PLACEHOLDER_PATTERNS = re.compile(
    r"\(source required\)|\[citation needed\]|\[insert source\]|"
    r"\(TODO\)|\[TODO\]|<TODO>|\(insert .+?\)",
    re.IGNORECASE,
)


async def compile_report(state: AgentState, config: RunnableConfig) -> dict:
    """Compile written sections into final report with TOC and bibliography."""
    await _emit_progress(config, "phase_start", "compile_report", "Compiling final report...")
    outline = state.get("report_outline", {})
    written_sections = state.get("written_sections") or []
    sources = state.get("sources", [])
    profile_name = outline.get("profile", "deep_research_report")
    profile = get_profile(profile_name) or get_profile("deep_research_report")
    citation_index = outline.get("citation_index")
    logger.info("compile_report: entry, %d written section(s), %d source(s), profile=%s",
                len(written_sections), len(sources), profile_name)

    if citation_index:
        logger.debug("compile_report: formatting bibliography from citation_index")
        bibliography = _format_bibliography_from_index(citation_index)
    else:
        logger.debug("compile_report: no citation_index, using CitationFormatter fallback")
        from open_deep_research.citation import CitationFormatter
        formatter = CitationFormatter(profile.citation_style)
        bibliography = formatter.format_bibliography(sources)

    toc_lines = []
    for i, section in enumerate(written_sections, 1):
        title = section.get("section_title", f"Section {i}")
        toc_lines.append(f"{i}. {title}")
    report_parts = []
    if profile.include_table_of_contents:
        report_parts.append("# Table of Contents\n\n" + "\n".join(toc_lines) + "\n\n---\n")
    for section in written_sections:
        title = section.get("section_title", "")
        content = section.get("content", "")
        content = _ET_AL_RE.sub("", content)
        if citation_index:
            content = _replace_raw_ids_in_text(content, citation_index)
        content = _remove_placeholder_sentences(content)
        report_parts.append(f"## {title}\n\n{content}\n")
    report_parts.append(f"## References\n\n{bibliography}\n")
    final_report = "\n".join(report_parts)
    logger.info("compile_report: exit, compiled report: %d section(s), %d chars",
                len(written_sections), len(final_report))
    return {"final_report": final_report, "messages": [AIMessage(content=final_report)]}


async def export_report(state: AgentState, config: RunnableConfig) -> dict:
    """Export the compiled report to configured formats."""
    await _emit_progress(config, "phase_start", "export_report", "Exporting report...")
    import uuid
    from pathlib import Path

    configurable = Configuration.from_runnable_config(config)
    logger.info("export_report: entry")
    final_report = state.get("final_report", "")
    if not final_report:
        logger.warning("export_report: no final report to export, skipping")
        return {}

    sources = state.get("sources", [])
    outline = state.get("report_outline", {})
    profile_name = outline.get("profile", "deep_research_report")

    metadata = {
        "title": state.get("research_brief", "Research Report"),
        "generated_at": get_today_str(),
        "profile": profile_name,
        "sources": sources,
    }

    run_id = uuid.uuid4().hex[:8]
    output_dir = Path("research_output") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("export_report: exporting to %d format(s): %s",
                len(configurable.export_formats), configurable.export_formats)
    exported_files = {}
    for fmt in configurable.export_formats:
        try:
            exporter = get_exporter(fmt)
            result_path = await exporter.export(final_report, metadata, output_dir / "report")
            exported_files[fmt] = str(result_path)
            logger.debug("export_report: exported format=%s to %s", fmt, result_path)
        except Exception as e:
            logger.exception("export_report: export to format=%s failed", fmt)
            logger.warning(f"Export to {fmt} failed: {e}")

    logger.info("export_report: exit, %d/%d format(s) exported successfully",
                len(exported_files), len(configurable.export_formats))
    return {"exported_files": exported_files}


async def final_review(state: AgentState, config: RunnableConfig) -> dict:
    """Run reviewer agents and decide whether to rewrite or export."""
    await _emit_progress(config, "phase_start", "final_review", "Running quality review...")
    configurable = Configuration.from_runnable_config(config)
    logger.info("final_review: entry, iteration=%d", state.get("review_iteration_count", 0))

    if not configurable.enable_reviewer_loop:
        logger.debug("final_review: reviewer loop disabled, skipping")
        return {}

    final_report = state.get("final_report", "")
    if not final_report:
        logger.warning("final_review: no final report to review, skipping")
        return {}

    outline = state.get("report_outline", {})
    profile_name = outline.get("profile", "deep_research_report")
    profile = get_profile(profile_name) or get_profile("deep_research_report")
    research_plan = state.get("research_plan", {})
    evidence_cards = state.get("evidence_cards", [])
    citation_checks = state.get("citation_checks", [])
    iteration = state.get("review_iteration_count", 0)

    coverage_reviewer = CoverageReviewer()
    evidence_reviewer = EvidenceReviewer()
    contradiction_reviewer = ContradictionReviewer()
    style_reviewer = StyleReviewer()
    completeness_reviewer = CompletenessReviewer()

    review_tasks = [
        coverage_reviewer.review(final_report, research_plan),
        evidence_reviewer.review(final_report, evidence_cards, citation_checks),
        contradiction_reviewer.review(final_report),
        style_reviewer.review(final_report, profile),
        completeness_reviewer.review(final_report),
    ]
    logger.info("final_review: running %d reviewer(s) on report (%d chars, %d evidence card(s))",
                len(review_tasks), len(final_report), len(evidence_cards))
    review_results = await asyncio.gather(*review_tasks, return_exceptions=True)

    failed_reviewers = sum(1 for r in review_results if isinstance(r, Exception))
    if failed_reviewers:
        logger.warning("final_review: %d/%d reviewer(s) raised and defaulted to score 0.0",
                       failed_reviewers, len(review_results))
        for r in review_results:
            if isinstance(r, Exception):
                logger.error("final_review: reviewer failed: %s: %s", type(r).__name__, r)

    coverage_feedback, evidence_feedback, contradiction_feedback, style_feedback, completeness_feedback = (
        r if not isinstance(r, Exception) else None for r in review_results
    )
    if coverage_feedback is None:
        coverage_feedback = type("ReviewFeedback", (), {"score": 0.0, "issues": [], "rewrite_instructions": [], "reviewer_name": "coverage"})()
    if evidence_feedback is None:
        evidence_feedback = type("ReviewFeedback", (), {"score": 0.0, "issues": [], "rewrite_instructions": [], "reviewer_name": "evidence"})()
    if contradiction_feedback is None:
        contradiction_feedback = type("ReviewFeedback", (), {"score": 0.0, "issues": [], "rewrite_instructions": [], "reviewer_name": "contradiction"})()
    if style_feedback is None:
        style_feedback = type("ReviewFeedback", (), {"score": 0.0, "issues": [], "rewrite_instructions": [], "reviewer_name": "style"})()
    if completeness_feedback is None:
        completeness_feedback = type("ReviewFeedback", (), {"score": 0.0, "issues": [], "rewrite_instructions": [], "reviewer_name": "completeness"})()

    all_feedback = [f for f in [coverage_feedback, evidence_feedback, contradiction_feedback, style_feedback, completeness_feedback] if f.score is not None]
    avg_score = sum(f.score for f in all_feedback) / len(all_feedback)

    review_result = {
        "coverage_score": coverage_feedback.score,
        "evidence_score": evidence_feedback.score,
        "style_score": style_feedback.score,
        "contradiction_flags": contradiction_feedback.issues,
        "completeness_score": completeness_feedback.score,
        "iteration_count": iteration,
        "feedback": "; ".join(f"{f.reviewer_name}: {f.score:.2f}" for f in all_feedback),
    }

    all_instructions = []
    for f in all_feedback:
        all_instructions.extend(f.rewrite_instructions)

    logger.info("final_review: avg_score=%.2f (threshold=%.2f), iteration=%d/%d",
                avg_score, configurable.review_score_threshold, iteration,
                configurable.max_review_iterations)
    if avg_score >= configurable.review_score_threshold or iteration >= configurable.max_review_iterations:
        logger.info("final_review: exit, review passed/exhausted -> export (no rewrite)")
        return {
            "review_results": [review_result],
            "review_iteration_count": iteration + 1,
        }
    else:
        logger.info("final_review: exit, triggering rewrite with %d instruction(s)",
                    len(all_instructions))
        return {
            "review_results": [review_result],
            "review_iteration_count": iteration + 1,
            "rewrite_instructions": all_instructions,
        }


async def rewrite_sections(state: AgentState, config: RunnableConfig) -> dict:
    """Rewrite sections based on reviewer feedback."""
    configurable = Configuration.from_runnable_config(config)
    logger.info("rewrite_sections: entry")

    if not configurable.enable_section_writers:
        logger.warning("rewrite_sections: section writers disabled, returning None")
        return {"written_sections": None}

    rewrite_instructions = state.get("rewrite_instructions", [])
    if not rewrite_instructions:
        logger.debug("rewrite_sections: no rewrite instructions, skipping")
        return {}
    logger.info("rewrite_sections: rewriting with %d instruction(s)", len(rewrite_instructions))

    outline = state.get("report_outline", {})
    sections = outline.get("sections", [])
    evidence_cards = state.get("evidence_cards", [])
    subquestions = outline.get("subquestions", [])
    profile_name = outline.get("profile", "deep_research_report")
    profile = get_profile(profile_name) or get_profile("deep_research_report")

    feedback_text = "\n".join(f"- {inst}" for inst in rewrite_instructions)

    for section in sections:
        section["rewrite_feedback"] = feedback_text
        section["evidence_allocation"] = outline.get("evidence_allocation", {})

    research_topic = state.get("research_brief", "")
    citation_index = outline.get("citation_index")
    tasks = [
        write_section(section, evidence_cards, profile, subquestions,
                      research_topic, config, citation_index)
        for section in sections
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    written_sections = []
    failed = 0
    for result in results:
        if isinstance(result, Exception):
            failed += 1
            logger.warning(f"Section rewrite failed: {result}")
            continue
        written_sections.append(result)

    logger.info("rewrite_sections: exit, %d section(s) rewritten, %d failed",
                len(written_sections), failed)
    return {"written_sections": written_sections, "rewrite_instructions": []}


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    logger.info("final_report_generation: entry, %d note(s), findings=%d chars",
                len(notes), len(findings))
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    resolved = _resolve_model_via_router(
        configurable, config, TaskType.REPORT_WRITING,
        configurable.final_report_model,
        complexity_hint=ModelTier.QUALITY,
    )
    if resolved is not None:
        writer_model_config = build_model_config(
            configurable, resolved.model_string,
            configurable.final_report_model_max_tokens, config,
        )
    else:
        writer_model_config = build_model_config(
            configurable, configurable.final_report_model,
            configurable.final_report_model_max_tokens, config,
        )
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            logger.info("final_report_generation: invoking writer model=%s (attempt %d/%d)",
                        writer_model_config["model"], current_retry + 1, max_retries + 1)
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])

            # Return successful report generation
            logger.info("final_report_generation: exit, report=%d chars",
                        len(str(final_report.content)))
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state
            }

        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                logger.warning("final_report_generation: token limit exceeded, retry %d/%d",
                               current_retry, max_retries)

                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        logger.error("final_report_generation: unknown token limit for model=%s, aborting",
                                     configurable.final_report_model)
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)

                # Truncate findings and retry
                logger.debug("final_report_generation: truncating findings to %d chars", findings_token_limit)
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                logger.exception("final_report_generation: non-token-limit error, aborting report generation")
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }

    # Step 4: Return failure result if all retries exhausted
    logger.critical("final_report_generation: max retries (%d) exhausted, report generation failed", max_retries)
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("parse_document", parse_document)                 # Document parsing phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("classify_research_request", classify_research_request)  # Mode classification
deep_researcher_builder.add_node("generate_research_plan", generate_research_plan)  # Research planning
deep_researcher_builder.add_node("optional_plan_review", optional_plan_review)      # Plan review/HITL
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("verify_citations", verify_citations)            # Citation verification
deep_researcher_builder.add_node("generate_report_outline", generate_report_outline)  # Outline generation
deep_researcher_builder.add_node("write_sections_parallel", write_sections_parallel)  # Section writing
deep_researcher_builder.add_node("compile_report", compile_report)                # Report compilation
deep_researcher_builder.add_node("export_report", export_report)                # Report export
deep_researcher_builder.add_node("final_review", final_review)                  # Quality review
deep_researcher_builder.add_node("rewrite_sections", rewrite_sections)          # Section rewriting
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Legacy report generation

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("parse_document", "write_research_brief")         # Parse to brief
deep_researcher_builder.add_edge("classify_research_request", "generate_research_plan")  # Classify to plan
deep_researcher_builder.add_edge("generate_research_plan", "optional_plan_review")  # Plan to review
deep_researcher_builder.add_edge("optional_plan_review", "research_supervisor")     # Review to supervisor
deep_researcher_builder.add_edge("research_supervisor", "verify_citations")       # Research to citation verification
deep_researcher_builder.add_edge("verify_citations", "generate_report_outline")   # Citations to outline
deep_researcher_builder.add_edge("generate_report_outline", "write_sections_parallel")  # Outline to sections
deep_researcher_builder.add_edge("write_sections_parallel", "compile_report")     # Sections to compilation
deep_researcher_builder.add_edge("compile_report", "final_review")               # Compilation to review
deep_researcher_builder.add_conditional_edges(
    "final_review",
    lambda state: "rewrite_sections" if state.get("rewrite_instructions") else "export_report",
    {"rewrite_sections": "rewrite_sections", "export_report": "export_report"},
)
deep_researcher_builder.add_edge("rewrite_sections", "compile_report")           # Rewrite loops back to compilation
deep_researcher_builder.add_edge("export_report", END)                           # Export to END

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()