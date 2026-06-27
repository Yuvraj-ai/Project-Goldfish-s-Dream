#!/usr/bin/env python
"""Task 3-3: Gold-set evaluation adapted for OpenRouter + gpt-4o-mini.

Usage:
    OPENAI_API_KEY=<openrouter-key> OPENAI_BASE_URL=https://openrouter.ai/api/v1 \\
        .venv/bin/python tests/run_golden_eval.py

Sets OPENAI_API_KEY/BASE_URL internally so both the graph and evaluator
route through OpenRouter.
"""
import asyncio, json, os, sys, time, uuid
from dotenv import load_dotenv
load_dotenv()

OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
os.environ.setdefault("OPENAI_API_KEY", OR_KEY)
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

GRAPH_MODEL = "openai/gpt-4o-mini"
EVAL_MODEL = "openai/gpt-4o"

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.utils import get_today_str
from tests.prompts import (
    OVERALL_QUALITY_PROMPT,
    RELEVANCE_PROMPT,
    STRUCTURE_PROMPT,
    CORRECTNESS_PROMPT,
    COMPLETENESS_PROMPT,
)

SEED_QUERIES = json.load(open("tests/golden_set/seed_queries.json"))
BASELINE = json.load(open("tests/golden_set/baseline_scores.json"))

eval_llm = ChatOpenAI(
    model=EVAL_MODEL,
    temperature=0,
)

# ---- Pydantic scorers (mirror evaluators.py) ----

class OverallQualityScore(BaseModel):
    research_depth: int = Field(ge=1, le=5)
    source_quality: int = Field(ge=1, le=5)
    analytical_rigor: int = Field(ge=1, le=5)
    practical_value: int = Field(ge=1, le=5)
    balance_and_objectivity: int = Field(ge=1, le=5)
    writing_quality: int = Field(ge=1, le=5)

class SingleScore(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)

def scale(v: int) -> float:
    return v / 5.0

GRAPH_CFG = {
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

async def run_query(q: dict) -> dict:
    qid = q["id"]
    query = q["query"]
    print(f"\n{'='*60}")
    print(f"[{qid}] {query[:80]}", flush=True)
    print(f"{'='*60}")
    t0 = time.time()
    try:
        state = await deep_researcher.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {**GRAPH_CFG}},
        )
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED after {elapsed:.0f}s: {type(e).__name__}: {e}", flush=True)
        return {"id": qid, "query": query, "status": "error", "error": str(e), "elapsed": elapsed}

    elapsed = time.time() - t0
    report = state.get("final_report", "")
    brief = state.get("research_brief", "")
    sources = state.get("sources", [])
    n_sources = len(sources) if isinstance(sources, list) else 0
    print(f"  Done: {elapsed:.0f}s, report={len(report)} chars, sources={n_sources}", flush=True)

    # ---- Evaluate with LLM judge ----
    scores = {}

    # Overall quality
    try:
        oq = await eval_llm.with_structured_output(OverallQualityScore, method="function_calling").ainvoke([
            {"role": "system", "content": OVERALL_QUALITY_PROMPT.format(today=get_today_str())},
            {"role": "user", "content": f"User input: {query}\n\nReport:\n\n{report}\n\nEvaluate."},
        ])
        oq_scores = {
            "research_depth": scale(oq.research_depth),
            "source_quality": scale(oq.source_quality),
            "analytical_rigor": scale(oq.analytical_rigor),
            "practical_value": scale(oq.practical_value),
            "balance_and_objectivity": scale(oq.balance_and_objectivity),
            "writing_quality": scale(oq.writing_quality),
        }
        scores["overall_quality"] = oq_scores
        scores["overall_avg"] = sum(oq_scores.values()) / len(oq_scores)
    except Exception as e:
        scores["overall_quality"] = {"error": str(e)}
        scores["overall_avg"] = 0.0
        print(f"  [WARN] OverallQuality eval failed: {e}", flush=True)

    # Relevance
    try:
        rel = await eval_llm.with_structured_output(SingleScore, method="function_calling").ainvoke([
            {"role": "system", "content": RELEVANCE_PROMPT.format(today=get_today_str())},
            {"role": "user", "content": f"User input: {query}\n\nReport:\n\n{report}\n\nEvaluate relevance."},
        ])
        scores["relevance"] = scale(rel.score)
    except Exception as e:
        scores["relevance"] = 0.0

    # Structure
    try:
        struct = await eval_llm.with_structured_output(SingleScore, method="function_calling").ainvoke([
            {"role": "user", "content": STRUCTURE_PROMPT.format(
                user_question=query, report=report, today=get_today_str()
            )},
        ])
        scores["structure"] = scale(struct.score)
    except Exception as e:
        scores["structure"] = 0.0

    # Correctness (has reference answer)
    ref = q.get("reference_answer", "")
    if ref:
        try:
            corr = await eval_llm.with_structured_output(SingleScore, method="function_calling").ainvoke([
                {"role": "user", "content": CORRECTNESS_PROMPT.format(
                    user_question=query, report=report, answer=ref, today=get_today_str()
                )},
            ])
            scores["correctness"] = scale(corr.score)
        except Exception:
            scores["correctness"] = 0.0

    # Completeness
    try:
        comp = await eval_llm.with_structured_output(SingleScore, method="function_calling").ainvoke([
            {"role": "user", "content": COMPLETENESS_PROMPT.format(
                user_question=query, research_brief=brief, report=report, today=get_today_str()
            )},
        ])
        scores["completeness"] = scale(comp.score)
    except Exception as e:
        scores["completeness"] = 0.0

    print(f"  SCORES: overall_avg={scores.get('overall_avg',0):.3f}, "
          f"relevance={scores.get('relevance',0):.3f}, "
          f"structure={scores.get('structure',0):.3f}, "
          f"correctness={scores.get('correctness',0):.3f}, "
          f"completeness={scores.get('completeness',0):.3f}", flush=True)

    return {
        "id": qid,
        "query": query,
        "status": "ok",
        "elapsed": elapsed,
        "report_chars": len(report),
        "n_sources": n_sources,
        "scores": scores,
    }

async def main():
    os.makedirs("docs/verification", exist_ok=True)
    # Run 3 representative queries (comparison, academic, news)
    SUBSET = [q for q in SEED_QUERIES if q["id"] in ("comp_001", "acad_001", "news_001")]
    all_results = []
    for q in SUBSET:
        r = await run_query(q)
        all_results.append(r)

    print(f"\n{'='*60}")
    print("TASK 3-3: GOLD-SET EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Graph model: {GRAPH_MODEL} | Eval model: {EVAL_MODEL} (via OpenRouter)")
    print(f"Baseline: avg={BASELINE['avg_score']}, min={BASELINE['min_score']}, max={BASELINE['max_score']}")
    print()

    ok_scores = [r for r in all_results if r["status"] == "ok"]
    if ok_scores:
        avg_scores = {}
        for r in ok_scores:
            for k, v in r.get("scores", {}).items():
                if isinstance(v, (int, float)):
                    avg_scores.setdefault(k, []).append(v)
        print("Average scores across completed runs (0-1 scale):")
        for k, vals in sorted(avg_scores.items()):
            mean = sum(vals) / len(vals)
            print(f"  {k}: {mean:.3f}")

        # Convert 0-1 → 1-10 for baseline comparison (multiply by 10)
        all_dims = [v for r in ok_scores for k, v in r.get("scores", {}).items()
                    if isinstance(v, (int, float)) and k != "overall_quality"]
        overall_mean_01 = sum(all_dims) / len(all_dims) if all_dims else 0
        overall_mean_10 = overall_mean_01 * 10
        print(f"\n  === Composite avg (1-10 scale): {overall_mean_10:.1f} (baseline: {BASELINE['avg_score']}) ===")
        delta = overall_mean_10 - BASELINE["avg_score"]
        print(f"  Delta vs baseline: {delta:+.1f}")
        if delta >= -0.3:
            print(f"  TASK 3-3: PASS (delta {delta:+.1f} >= -0.3)")
        else:
            print(f"  TASK 3-3: FAIL (delta {delta:+.1f} < -0.3)")

    errors = [r for r in all_results if r["status"] == "error"]
    if errors:
        print(f"\nErrors: {len(errors)}/{len(SEED_QUERIES)}")
        for r in errors:
            print(f"  [{r['id']}] {r.get('error','')[:120]}")

    output = {
        "graph_model": GRAPH_MODEL,
        "eval_model": EVAL_MODEL,
        "baseline": BASELINE,
        "results": all_results,
        "summary": {
            "n_total": len(SUBSET),
            "n_ok": len(ok_scores),
            "n_errors": len(errors),
            "composite_avg_01": overall_mean_01 if ok_scores else None,
            "composite_avg_10": overall_mean_10 if ok_scores else None,
            "baseline_avg": BASELINE["avg_score"],
            "delta": delta if ok_scores else None,
            "pass": (delta >= -0.3) if ok_scores else False,
        },
    }
    with open("docs/verification/task-3-3-gold-set.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to docs/verification/task-3-3-gold-set.json")

if __name__ == "__main__":
    asyncio.run(main())
