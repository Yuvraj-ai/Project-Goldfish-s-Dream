#!/usr/bin/env python
"""Sequential feature smoke-test driver for Advanced Deep Research (Gemini).

Scratch tool — safe to delete. Runs deep_researcher.ainvoke with Gemini models,
toggling ONE feature flag at a time, then prints the state slice that feature is
supposed to populate plus a short report snippet.

Usage:
    .venv/bin/python scripts/smoke_features.py --list
    .venv/bin/python scripts/smoke_features.py base
    .venv/bin/python scripts/smoke_features.py evidence_first
    .venv/bin/python scripts/smoke_features.py all          # run every feature in order
    .venv/bin/python scripts/smoke_features.py base --query "What is LangGraph?"
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from open_deep_research.deep_researcher import deep_researcher  # noqa: E402

GEMINI = "google_genai:gemini-2.5-flash"

# Model is overridable via env so we can switch providers without editing code.
# Default to Gemini; we run on Groq today because Gemini free tier = 20 req/day.
MODEL = os.getenv("SMOKE_MODEL", GEMINI)
SUMMARY_MODEL = os.getenv("SMOKE_SUMMARY_MODEL", MODEL)

# Cheap, fast base config so smoke runs stay short and inexpensive.
BASE_CONFIG: dict = {
    "research_model": MODEL,
    "compression_model": MODEL,
    "final_report_model": MODEL,
    "summarization_model": SUMMARY_MODEL,
    "classifier_model": MODEL,
    "search_api": "tavily",
    "allow_clarification": False,        # don't halt to ask the user a question
    "max_researcher_iterations": 1,
    "max_react_tool_calls": 2,
    "max_concurrent_research_units": 1,
    "max_subquestions": 2,
    "min_subquestions": 1,
}

# feature name -> (extra config overrides, state keys to display after the run)
FEATURES: dict[str, tuple[dict, list[str]]] = {
    "base":                 ({}, ["final_report"]),
    "evidence_first":       ({"enable_evidence_first": True}, ["evidence_cards"]),
    "mode_classification":  ({"enable_mode_classification": True}, ["research_mode"]),
    "planner":              ({"enable_mode_classification": True}, ["research_plan"]),
    "search_aggregation":   ({"enable_search_aggregation": True}, ["final_report"]),
    "academic_search":      ({"enable_academic_search": True}, ["evidence_cards", "final_report"]),
    "storm":                ({"enable_storm_research": True}, ["perspectives"]),
    "citation_verification":({"enable_citation_verification": True}, ["citation_checks"]),
    "section_writers":      ({"enable_section_writers": True}, ["report_outline", "written_sections"]),
    "reviewer_loop":        ({"enable_reviewer_loop": True}, ["review_results", "review_iteration_count"]),
    "export":               ({"export_formats": ["markdown", "html", "json"],
                              "citation_style": "apa"}, ["exported_files"]),
    "model_routing":        ({"enable_model_routing": True}, ["final_report"]),
}

ORDER = list(FEATURES.keys())


def _show(state: dict, keys: list[str]) -> None:
    for k in keys:
        v = state.get(k)
        if k == "final_report":
            text = (v or "")[:600]
            print(f"  • final_report: {len(v or '')} chars")
            if text:
                print("    ---")
                for line in text.splitlines()[:12]:
                    print(f"    {line}")
                print("    ---")
        elif isinstance(v, list):
            print(f"  • {k}: {len(v)} item(s)")
            if v:
                first = v[0]
                preview = str(first)[:200]
                print(f"    [0] {preview}")
        elif isinstance(v, dict):
            print(f"  • {k}: {len(v)} key(s) -> {list(v.keys())}")
        else:
            print(f"  • {k}: {v!r}")


async def run_feature(name: str, query: str) -> None:
    extra, show_keys = FEATURES[name]
    cfg = {**BASE_CONFIG, **extra}
    flags = {k: v for k, v in extra.items()} or {"(legacy base path)": True}
    print(f"\n{'='*70}\nFEATURE: {name}\n  flags: {flags}\n  query: {query!r}\n{'='*70}")
    t0 = time.time()
    try:
        state = await deep_researcher.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": cfg},
        )
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ FAILED after {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
        return
    print(f"  ✅ completed in {time.time()-t0:.1f}s")
    _show(state, show_keys)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("feature", nargs="?", default="base")
    ap.add_argument("--query", default="What is LangGraph and what is it used for?")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("Available features (in run order):")
        for f in ORDER:
            print(f"  - {f}: {FEATURES[f][0] or 'legacy base path'}")
        return

    needed = "GROQ_API_KEY" if MODEL.startswith("groq:") else "GOOGLE_API_KEY"
    if not os.getenv(needed):
        print(f"❌ {needed} not set in .env — add it before running live (MODEL={MODEL}).")
        return
    print(f"[model={MODEL}, summary={SUMMARY_MODEL}]")

    targets = ORDER if args.feature == "all" else [args.feature]
    for f in targets:
        if f not in FEATURES:
            print(f"Unknown feature {f!r}. Use --list.")
            continue
        await run_feature(f, args.query)


if __name__ == "__main__":
    asyncio.run(main())
