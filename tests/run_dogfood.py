#!/usr/bin/env python
"""Task 3-11: Dogfooding re-run (model_routing disabled — needs Google key)."""
import asyncio, json, os, time
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENROUTER_API_KEY",""))
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
from open_deep_research.deep_researcher import deep_researcher

CFG = {
    "research_model": "openai:openai/gpt-4o-mini",
    "compression_model": "openai:openai/gpt-4o-mini",
    "final_report_model": "openai:openai/gpt-4o-mini",
    "summarization_model": "openai:openai/gpt-4o-mini",
    "search_api": "tavily",
    "allow_clarification": False,
    "max_researcher_iterations": 3,
    "max_react_tool_calls": 4,
    "max_concurrent_research_units": 1,
    "max_subquestions": 2,
    "min_subquestions": 1,
    "research_model_max_tokens": 4096,
    "enable_evidence_first": True,
    "enable_mode_classification": True,
    "enable_citation_verification": True,
    "enable_section_writers": True,
    "enable_reviewer_loop": True,
    "enable_search_aggregation": True,
    "enable_academic_search": True,
    "enable_storm_research": True,
    "enable_model_routing": False,
}

async def main():
    query = "What are the environmental impacts of data centers and AI compute?"
    print(f"Query: {query}", flush=True)
    t0 = time.time()
    state = await deep_researcher.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": CFG},
    )
    elapsed = time.time() - t0
    report = state.get("final_report", "")
    sources = state.get("sources", [])
    n_src = len(sources) if isinstance(sources, list) else 0
    print(f"Time: {elapsed:.0f}s", flush=True)
    print(f"Sources: {n_src}", flush=True)
    print(f"Report: {len(report)} chars", flush=True)

    features_fired = {
        "evidence_first": bool(state.get("evidence_cards")),
        "section_writers": bool(state.get("written_sections")),
        "reviewer_loop": bool(state.get("review_results")),
        "citation_verification": bool(state.get("citation_checks")),
    }
    print(f"Features fired: {features_fired}", flush=True)

    out = {
        "task": "3-11",
        "query": query,
        "status": "ok",
        "elapsed": round(elapsed, 1),
        "sources": n_src,
        "report_chars": len(report),
        "features_fired": features_fired,
        "feature_flags": {k: CFG[k] for k in CFG if k.startswith("enable_")},
    }
    with open("docs/verification/task-3-11-dogfood.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved to docs/verification/task-3-11-dogfood.json", flush=True)

asyncio.run(main())
