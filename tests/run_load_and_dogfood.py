#!/usr/bin/env python
"""Tasks 3-9 (Load Test) and 3-11 (Dogfooding) combined.

Task 3-9: 3 concurrent graph runs → concurrency handling
Task 3-11: 1 run with ALL feature flags enabled → full-system smoke test

Total: ~4 graph runs @ ~4¢ each ≈ ~16¢
"""
import asyncio, json, os, sys, time
from dotenv import load_dotenv
load_dotenv()

OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
os.environ.setdefault("OPENAI_API_KEY", OR_KEY)
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

from open_deep_research.deep_researcher import deep_researcher

GRAPH_MODEL = "openai/gpt-4o-mini"

# Base config (same as gold-set runs)
BASE_CFG = {
    "research_model": f"openai:{GRAPH_MODEL}",
    "compression_model": f"openai:{GRAPH_MODEL}",
    "final_report_model": f"openai:{GRAPH_MODEL}",
    "summarization_model": f"openai:{GRAPH_MODEL}",
    "search_api": "tavily",
    "allow_clarification": False,
    "max_researcher_iterations": 3,
    "max_react_tool_calls": 4,
    "max_concurrent_research_units": 1,
    "max_subquestions": 2,
    "min_subquestions": 1,
    "research_model_max_tokens": 4096,
    "enable_evidence_first": True,
}

QUERIES = [
    "Compare React and Vue.js for building enterprise dashboards in 2026",
    "What are the latest advances in retrieval-augmented generation (RAG) since 2024?",
    "What are the major AI regulation developments in the EU and US in 2026?",
]

async def run_one(query: str, cfg: dict, label: str) -> dict:
    t0 = time.time()
    try:
        state = await deep_researcher.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": cfg},
        )
        elapsed = time.time() - t0
        report = state.get("final_report", "")
        sources = state.get("sources", [])
        n_sources = len(sources) if isinstance(sources, list) else 0
        return {
            "label": label,
            "query": query[:60],
            "status": "ok",
            "elapsed": round(elapsed, 1),
            "sources": n_sources,
            "report_chars": len(report or ""),
        }
    except Exception as e:
        return {
            "label": label,
            "query": query[:60],
            "status": "error",
            "elapsed": round(time.time() - t0, 1),
            "error": f"{type(e).__name__}: {e}",
        }

async def main():
    print("=" * 60)
    print("TASK 3-9: LOAD TEST (3 concurrent runs)")
    print("=" * 60)
    t0 = time.time()

    tasks = [run_one(q, BASE_CFG, f"load-{i+1}") for i, q in enumerate(QUERIES)]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - t0

    ok_runs = [r for r in results if r["status"] == "ok"]
    err_runs = [r for r in results if r["status"] == "error"]

    print(f"\n{'Label':<10} {'Query':<50} {'Time':<8} {'Sources':<8} {'Chars':<8}")
    print("-" * 84)
    for r in results:
        q = r["query"][:48] + ".." if len(r["query"]) > 50 else r["query"]
        if r["status"] == "ok":
            print(f"{r['label']:<10} {q:<50} {r['elapsed']:<8} {r['sources']:<8} {r['report_chars']:<8}")
        else:
            print(f"{r['label']:<10} {q:<50} {'ERR':<8} {'':<8} {'':<8}")
            print(f"{'':10} error: {r.get('error','')[:100]}")

    concurrency_overhead = total_time / max(r["elapsed"] for r in results if r["status"] == "ok") if ok_runs else 0
    print("-" * 84)
    print(f"Wall clock: {total_time:.0f}s")
    print(f"Concurrent runs: {len(QUERIES)}")
    print(f"Concurrency overhead: {concurrency_overhead:.2f}x (1.0 = perfect parallel)")
    print(f"Errors: {len(err_runs)}/{len(results)}")
    load_pass = concurrency_overhead <= 3.0 and len(err_runs) == 0
    print(f"Gate: overhead ≤3x, zero errors — {'PASS' if load_pass else 'FAIL'}")

    load_output = {
        "task": "3-9",
        "graph_model": GRAPH_MODEL,
        "queries": QUERIES,
        "concurrent": True,
        "wall_clock_s": round(total_time, 1),
        "concurrency_overhead": round(concurrency_overhead, 2),
        "runs": results,
        "gate": "overhead <=3x, zero errors",
        "result": "PASS" if load_pass else "FAIL",
    }
    with open("docs/verification/task-3-9-load.json", "w") as f:
        json.dump(load_output, f, indent=2)

    # === Task 3-11: Dogfooding ===
    print(f"\n{'='*60}")
    print("TASK 3-11: DOGFOODING (all feature flags)")
    print("=" * 60)

    DOG_CFG = {
        **BASE_CFG,
        "enable_mode_classification": True,
        "enable_citation_verification": True,
        "enable_section_writers": True,
        "enable_reviewer_loop": True,
        "enable_evidence_first": True,
        "enable_search_aggregation": True,
        "enable_academic_search": True,
        "enable_storm_research": True,
        "enable_model_routing": True,
        "export_formats": ["markdown", "json"],
    }

    dog_query = "What are the environmental impacts of data centers and AI compute?"
    print(f"\nQuery: {dog_query}")
    dog_result = await run_one(dog_query, DOG_CFG, "dogfood")
    print(f"Status: {dog_result['status']}")
    if dog_result["status"] == "ok":
        print(f"Time: {dog_result['elapsed']:.0f}s")
        print(f"Sources: {dog_result['sources']}")
        print(f"Report: {dog_result['report_chars']} chars")
    else:
        print(f"Error: {dog_result.get('error','')}")

    dogfood_output = {
        "task": "3-11",
        "graph_model": GRAPH_MODEL,
        "feature_flags": {k: DOG_CFG[k] for k in DOG_CFG if k.startswith("enable_")},
        "query": dog_query,
        "result": dog_result,
    }
    with open("docs/verification/task-3-11-dogfood.json", "w") as f:
        json.dump(dogfood_output, f, indent=2)

    print(f"\nResults saved to docs/verification/ (tasks 3-9 and 3-11)")
    print(f"\n{'='*60}")
    print(f"COST SUMMARY")
    print(f"{'='*60}")
    total_runs = len(results) + 1  # 3 load + 1 dogfood
    print(f"Total graph runs: {total_runs}")
    print(f"Est. cost: {total_runs * 4:.0f}¢ ({total_runs * 0.04:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
