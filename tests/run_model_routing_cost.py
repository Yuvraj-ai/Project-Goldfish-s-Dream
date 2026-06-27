#!/usr/bin/env python
"""Task 3-10: Model Routing Cost Savings — compare routed vs always-QUEUE costs.

Uses live pricing from OpenRouter API. Zero LLM calls.
"""
import asyncio, json, os, sys
from dotenv import load_dotenv
load_dotenv()

OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# ---- Task distribution for a typical research run ----
# Based on the 19-node graph with max_researcher_iterations=3, no review loop
# Each entry: (task_type, tier, calls, avg_input_tokens, avg_output_tokens)
TASKS = [
    # Classification
    ("classification", "fast",       1,   800,   200),
    # Summarization (research brief + search summaries)
    ("summarization", "fast",        2,  2000,   500),
    # Search query generation (per researcher iteration)
    ("search_query_generation", "fast", 4,  500,   100),
    # Planning (research plan + outline)
    ("planning", "balanced",         2,  3000,  1000),
    # Extraction (evidence extraction)
    ("extraction", "balanced",       1,  5000,  1000),
    # Research (supervisor + researcher reasoning)
    ("reasoning", "quality",        12,  4000,  2000),
    # Report writing (sections × 4)
    ("report_writing", "quality",    4,  3000,  3000),
    # Review (final review)
    ("review", "quality",            1,  8000,  1000),
]

async def get_pricing() -> dict:
    """Fetch live model pricing from OpenRouter."""
    import httpx
    models_of_interest = [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
    ]
    pricing = {}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OR_KEY}"},
        )
        data = resp.json()
        for m in data.get("data", []):
            mid = m["id"]
            if mid in models_of_interest or "gpt-4o-mini" in mid or "gpt-4o" in mid:
                p = m.get("pricing", {})
                pricing[mid] = {
                    "input": float(p.get("prompt", 0)),
                    "output": float(p.get("completion", 0)),
                }
    return pricing

def cost(pricing: dict, model: str, in_tok: int, out_tok: int) -> float:
    """Compute cost in USD. Pricing from OpenRouter is per-token already."""
    p = pricing.get(model, {"input": 0, "output": 0})
    return (in_tok * p["input"]) + (out_tok * p["output"])

def main():
    pricing = asyncio.run(get_pricing())

    # Map tier names to actual models
    # Using OpenRouter OpenAI models (our active provider)
    ROUTED = {
        "fast":     "openai/gpt-4o-mini",
        "balanced": "openai/gpt-4o",
        "quality":  "openai/gpt-4o",
    }
    # Without routing: always use QUALITY for everything
    ALWAYS_QUALITY = "openai/gpt-4o"

    print(f"{'='*60}")
    print("TASK 3-10: MODEL ROUTING COST SAVINGS")
    print(f"{'='*60}")
    print(f"\nPricing (from OpenRouter):")
    for model, p in sorted(pricing.items()):
        pm = p['input'] * 1_000_000
        po = p['output'] * 1_000_000
        print(f"  {model:<35}  ${pm:.2f}/M in     ${po:.2f}/M out")
    print(f"\nRouting strategy:")
    print(f"  FAST     → {ROUTED['fast']}")
    print(f"  BALANCED → {ROUTED['balanced']}")
    print(f"  QUALITY  → {ROUTED['quality']}")
    print(f"  Without routing: always {ALWAYS_QUALITY}\n")

    print(f"{'Task Type':<30} {'Tier':<12} {'Calls':<6} {'Routed ¢':<10} {'Always ¢':<10} {'Saved ¢':<10}")
    print("-" * 78)

    total_routed = 0.0
    total_always = 0.0
    rows = []

    for task_type, tier, calls, in_tok, out_tok in TASKS:
        routed_model = ROUTED[tier]
        c_routed = cost(pricing, routed_model, in_tok, out_tok) * calls
        c_always = cost(pricing, ALWAYS_QUALITY, in_tok, out_tok) * calls
        saved = c_always - c_routed
        total_routed += c_routed
        total_always += c_always
        rows.append((task_type, tier, calls, c_routed, c_always))
        print(f"{task_type:<30} {tier:<12} {calls:<6} {c_routed*100:<9.4f} {c_always*100:<9.4f} {saved*100:<9.4f}")

    print("-" * 78)
    dr = total_routed * 100
    da = total_always * 100
    ds = da - dr
    print(f"{'TOTAL':<30} {'':<12} {'':<6} {dr:<9.4f}¢ {da:<9.4f}¢ {ds:<9.4f}¢")
    savings = total_always - total_routed
    savings_pct = ((total_always - total_routed) / total_always) * 100 if total_always > 0 else 0
    print(f"\n  Cost with routing:    {total_routed*100:.4f}¢")
    print(f"  Cost without routing: {total_always*100:.4f}¢")
    print(f"  Savings:              {savings*100:.4f}¢ ({savings_pct:.1f}%)")
    print(f"\n  Gate: savings ≥ 20% — {'PASS' if savings_pct >= 20 else 'FAIL'}")

    output = {
        "task": "3-10",
        "pricing": pricing,
        "routed_models": ROUTED,
        "always_quality_model": ALWAYS_QUALITY,
        "task_distribution": TASKS,
        "total_routed": round(total_routed, 6),
        "total_always": round(total_always, 6),
        "savings": round(savings, 6),
        "savings_pct": round(savings_pct, 1),
        "gate": ">=20%",
        "result": "PASS" if savings_pct >= 20 else "FAIL",
    }
    with open("docs/verification/task-3-10-routing-cost.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to docs/verification/task-3-10-routing-cost.json")

if __name__ == "__main__":
    main()
