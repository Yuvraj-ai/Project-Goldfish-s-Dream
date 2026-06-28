#!/usr/bin/env python
"""Task 3-7: Latency Budget — compile timing data from prior runs. Zero LLM calls."""
import json, statistics

# Data from Task 3-3 gold-set eval (gpt-4o-mini, max_researcher_iterations=3, max_react_tool_calls=4)
GOLD_RUNS = [
    {"query": "Compare React and Vue.js for enterprise dashboards", "elapsed": 138, "sources": 41, "report_chars": 18659},
    {"query": "Latest advances in RAG since 2024",                   "elapsed": 135, "sources": 65, "report_chars": 22949},
    {"query": "AI regulation developments in EU and US 2026",        "elapsed": 121, "sources": 34, "report_chars": 19271},
]

# Data from Task 3-2 domain diversity (gpt-4o-mini, max_researcher_iterations=3, max_react_tool_calls=2)
DIVERSITY_RUNS = [
    {"query": "Solid-state battery technology for EVs",             "elapsed": 109, "sources": 22, "report_chars": 15911},
    {"query": "Health effects of microplastics in drinking water",   "elapsed": 163, "sources": 30, "report_chars": 16764},
    {"query": "Global semiconductor chip shortage 2026",             "elapsed": 112, "sources": 20, "report_chars": 16590},
]

all_runs = GOLD_RUNS + DIVERSITY_RUNS
elapsed_vals = [r["elapsed"] for r in all_runs]

print("=" * 60)
print("TASK 3-7: LATENCY BUDGET")
print("=" * 60)
print(f"\nTotal runs sampled: {len(all_runs)}")
print(f"Model: openai/gpt-4o-mini via OpenRouter + Tavily search")
print(f"Gate target: ≤10 min (from phase_wise_execution_plan.md)\n")

print(f"{'Query':<55} {'Time':<8} {'Sources':<8} {'Chars':<8}")
print("-" * 79)
for r in all_runs:
    q = r["query"][:53] + ".." if len(r["query"]) > 55 else r["query"]
    print(f"{q:<55} {r['elapsed']:<8} {r['sources']:<8} {r['report_chars']:<8}")

print("-" * 79)
print(f"{'Statistics':<55}")
mean_t = statistics.mean(elapsed_vals)
med_t = statistics.median(elapsed_vals)
min_t = min(elapsed_vals)
max_t = max(elapsed_vals)
print(f"  Mean:   {mean_t:.0f}s ({mean_t/60:.1f} min)")
print(f"  Median: {med_t:.0f}s ({med_t/60:.1f} min)")
print(f"  Min:    {min_t:.0f}s ({min_t/60:.1f} min)")
print(f"  Max:    {max_t:.0f}s ({max_t/60:.1f} min)")
print(f"  Gate:   ≤10 min (600s) — {'PASS' if max_t < 600 else 'FAIL'}")

output = {
    "task": "3-7",
    "n_runs": len(all_runs),
    "model": "openai/gpt-4o-mini",
    "gate": "<=600s (10 min)",
    "mean_s": round(mean_t, 1),
    "median_s": round(med_t, 1),
    "min_s": min_t,
    "max_s": max_t,
    "result": "PASS" if max_t < 600 else "FAIL",
    "runs": all_runs,
}
with open("docs/verification/task-3-7-latency.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to docs/verification/task-3-7-latency.json")
