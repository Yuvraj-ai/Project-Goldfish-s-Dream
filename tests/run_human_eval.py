#!/usr/bin/env python
"""Task 3-6: Generate 20 reports for human evaluation + blank CSV templates."""
import asyncio, csv, json, os, sys, time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
from open_deep_research.deep_researcher import deep_researcher

OUT_DIR = "research_output/human_eval"
os.makedirs(OUT_DIR, exist_ok=True)

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
    "enable_citation_verification": False,
    "enable_section_writers": True,
    "enable_reviewer_loop": True,
    "enable_search_aggregation": True,
    "enable_academic_search": True,
    "enable_storm_research": True,
    "enable_model_routing": False,
    "fetch_full_page": False,
    "max_structured_output_retries": 2,
}

QUERIES = [
    # comparison (4)
    {"id": "comp_001", "mode": "comparison", "query": "Compare React and Vue.js for building enterprise dashboards in 2026"},
    {"id": "comp_002", "mode": "comparison", "query": "Compare PostgreSQL and MongoDB for a high-throughput logging system"},
    {"id": "comp_003", "mode": "comparison", "query": "Compare AWS Lambda and Google Cloud Functions for serverless data processing pipelines"},
    {"id": "comp_004", "mode": "comparison", "query": "Compare PyTorch and TensorFlow for production deep learning in 2026"},
    # academic_literature_review (4)
    {"id": "acad_001", "mode": "academic_literature_review", "query": "What are the latest advances in retrieval-augmented generation (RAG) since 2024?"},
    {"id": "acad_002", "mode": "academic_literature_review", "query": "Survey the research on prompt engineering techniques for improving LLM reasoning"},
    {"id": "acad_003", "mode": "academic_literature_review", "query": "What does recent research say about the effectiveness of AI-driven code generation tools?"},
    {"id": "acad_004", "mode": "academic_literature_review", "query": "Review the literature on multimodal AI models combining text, image, and audio understanding"},
    # news_or_current_events (4)
    {"id": "news_001", "mode": "news_or_current_events", "query": "What are the major AI regulation developments in the EU and US in 2026?"},
    {"id": "news_002", "mode": "news_or_current_events", "query": "What are the current trends in cloud computing costs and provider pricing in 2026?"},
    {"id": "news_003", "mode": "news_or_current_events", "query": "How are major tech companies approaching carbon neutrality targets in 2026?"},
    {"id": "news_004", "mode": "news_or_current_events", "query": "What is the state of the global semiconductor industry supply chain in 2026?"},
    # technical_implementation (4)
    {"id": "tech_001", "mode": "technical_implementation", "query": "How do I implement real-time streaming with WebSockets in a FastAPI application?"},
    {"id": "tech_002", "mode": "technical_implementation", "query": "What is the best approach to implement rate limiting in a Python async web application?"},
    {"id": "tech_003", "mode": "technical_implementation", "query": "How do I build a CI/CD pipeline for a Python monorepo using GitHub Actions?"},
    {"id": "tech_004", "mode": "technical_implementation", "query": "What is the recommended architecture for deploying LLM-based applications to production?"},
    # validation_or_fact_check (4)
    {"id": "fact_001", "mode": "validation_or_fact_check", "query": "Is it true that Python 3.12 removed the GIL?"},
    {"id": "fact_002", "mode": "validation_or_fact_check", "query": "Verify: Docker containers are always lighter than virtual machines"},
    {"id": "fact_003", "mode": "validation_or_fact_check", "query": "Is it true that quantum computers can break all current encryption within the next 5 years?"},
    {"id": "fact_004", "mode": "validation_or_fact_check", "query": "Verify: Edge computing will completely replace cloud computing for enterprise workloads"},
]

CSV_HEADERS = [
    "report_id", "query", "mode", "rater_name",
    "completeness", "completeness_notes",
    "citation_accuracy", "citation_accuracy_notes",
    "source_diversity", "source_diversity_notes",
    "structure", "structure_notes",
    "temporal_relevance", "temporal_relevance_notes",
    "overall_comment",
]

def create_csv_templates():
    for rater in ["Rater A", "Rater B", "Rater C"]:
        path = os.path.join(OUT_DIR, f"scores_{rater.lower().replace(' ','_')}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADERS)
            for q in QUERIES:
                w.writerow([q["id"], q["query"], q["mode"], rater, "", "", "", "", "", "", "", "", "", "", ""])
        print(f"Created {path}")

async def run_one(item, retries=3):
    qid = item["id"]
    query = item["query"]
    report_path = os.path.join(OUT_DIR, f"{qid}.md")
    meta_path = os.path.join(OUT_DIR, f"{qid}.json")

    print(f"\n{'='*60}", flush=True)
    print(f"[{qid}] {query}", flush=True)
    print(f"[{qid}] Starting...", flush=True)

    for attempt in range(retries):
        t0 = time.time()
        try:
            state = await deep_researcher.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                {"configurable": CFG},
            )
            report = state.get("final_report") or state.get("report") or ""
            if not report.strip() and attempt < retries - 1:
                print(f"[{qid}] Empty report, retrying...", flush=True)
                await asyncio.sleep(10)
                continue
            break
        except Exception as e:
            print(f"[{qid}] Error (attempt {attempt+1}/{retries}): {type(e).__name__}: {e}", flush=True)
            if attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"[{qid}] Waiting {wait}s before retry...", flush=True)
                await asyncio.sleep(wait)
                continue
            report = ""
            elapsed = time.time() - t0

    elapsed = time.time() - t0

    with open(report_path, "w") as f:
        f.write(f"# Query: {query}\n\n")
        f.write(f"**Mode:** {item['mode']}  \n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Time:** {elapsed:.0f}s  \n\n")
        f.write("---\n\n")
        f.write(report)

    meta = {
        "id": qid, "mode": item["mode"], "query": query,
        "elapsed_s": round(elapsed, 1),
        "report_chars": len(report),
        "status": "ok" if report.strip() else "empty",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{qid}] Done in {elapsed:.0f}s — {len(report)} chars", flush=True)
    return meta

async def main():
    results = []
    for i, item in enumerate(QUERIES):
        meta = await run_one(item)
        results.append(meta)
        if i < len(QUERIES) - 1:
            delay = 15
            print(f"Waiting {delay}s before next query to avoid rate limits...", flush=True)
            await asyncio.sleep(delay)

    summary_path = os.path.join(OUT_DIR, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"count": len(results), "results": results}, f, indent=2)

    total_time = sum(r["elapsed_s"] for r in results)
    total_chars = sum(r["report_chars"] for r in results)
    print(f"\n{'='*60}", flush=True)
    print(f"All done. {len(results)} reports in {total_time:.0f}s total.", flush=True)
    print(f"Total chars: {total_chars}, Avg: {total_chars//len(results)} per report", flush=True)
    print(f"Reports in: {OUT_DIR}/", flush=True)

    create_csv_templates()

asyncio.run(main())
