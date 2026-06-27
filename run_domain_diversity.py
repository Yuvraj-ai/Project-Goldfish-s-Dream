#!/usr/bin/env python
"""Task 3-2: Domain Diversity Verification — check no single domain >40% of sources."""
import json, os, sys, time
from collections import Counter
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()

model = 'openai/gpt-4o-mini'
key = os.environ.get('OPENROUTER_API_KEY')
os.environ['OPENAI_API_KEY'] = key
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

import asyncio
from open_deep_research.deep_researcher import deep_researcher

QUERIES = [
    "Latest developments in solid-state battery technology for electric vehicles",
    "What are the health effects of microplastics in drinking water?",
    "Current state of the global semiconductor chip shortage in 2026",
]

BASE_CFG = {
    'research_model': f'openai:{model}',
    'compression_model': f'openai:{model}',
    'final_report_model': f'openai:{model}',
    'summarization_model': f'openai:{model}',
    'classifier_model': f'openai:{model}',
    'search_api': 'tavily',
    'allow_clarification': False,
    'max_researcher_iterations': 3,
    'max_react_tool_calls': 2,
    'max_concurrent_research_units': 1,
    'max_subquestions': 2,
    'min_subquestions': 1,
    'research_model_max_tokens': 4096,
    'enable_evidence_first': True,
}

async def run_query(query: str, idx: int):
    print(f'\n{"="*60}', flush=True)
    print(f'Query {idx}: {query}', flush=True)
    print(f'{"="*60}', flush=True)
    t0 = time.time()
    try:
        state = await deep_researcher.ainvoke(
            {'messages': [{'role': 'user', 'content': query}]},
            {'configurable': {**BASE_CFG}},
        )
        elapsed = time.time() - t0
        sources = state.get('sources', [])
        report = state.get('final_report', '')
        domains = Counter()
        for s in sources:
            url = s.get('url', '') if isinstance(s, dict) else getattr(s, 'url', '')
            if url:
                domain = urlparse(url).netloc
                if domain:
                    domains[domain] += 1
        total = sum(domains.values())
        top_pct = (domains.most_common(1)[0][1] / total * 100) if total > 0 else 0
        print(f'  Time: {elapsed:.1f}s', flush=True)
        print(f'  Sources: {total}', flush=True)
        print(f'  Report: {len(report or "")} chars', flush=True)
        print(f'  Unique domains: {len(domains)}', flush=True)
        print(f'  Top domain: {domains.most_common(1)[0][0] if domains else "none"} ({top_pct:.1f}%)', flush=True)
        print(f'  Top 5 domains:', flush=True)
        for d, c in domains.most_common(5):
            print(f'    {d}: {c} ({c/total*100:.1f}%)', flush=True)
        return {
            'query': query,
            'elapsed': elapsed,
            'total_sources': total,
            'unique_domains': len(domains),
            'top_domain': domains.most_common(1)[0][0] if domains else '',
            'top_pct': top_pct,
            'domains': dict(domains.most_common(10)),
            'report_chars': len(report or ''),
            'status': 'ok',
        }
    except Exception as e:
        print(f'  FAILED after {time.time()-t0:.1f}s: {type(e).__name__}: {e}')
        return {'query': query, 'status': 'error', 'error': str(e)}

async def main():
    all_results = []
    for i, q in enumerate(QUERIES, 1):
        result = await run_query(q, i)
        all_results.append(result)
    
    print(f'\n{"="*60}')
    print('TASK 3-2: DOMAIN DIVERSITY SUMMARY')
    print(f'{"="*60}')
    violations = 0
    for r in all_results:
        if r['status'] == 'ok':
            ok = r['top_pct'] <= 40.0
            if not ok: violations += 1
            status = 'PASS' if ok else 'FAIL'
            print(f'  [{status}] {r["query"][:50]}')
            print(f'        top={r["top_domain"]} @ {r["top_pct"]:.1f}% ({r["unique_domains"]} domains, {r["total_sources"]} sources)')
        else:
            print(f'  [ERR] {r["query"][:50]}: {r.get("error","")[:80]}')
    
    print(f'\n  Violations (domain >40%): {violations}/3')
    print(f'  Overall: {"PASS" if violations == 0 else "FAIL"}')
    
    with open('docs/verification/task-3-2-domain-diversity.json', 'w') as f:
        json.dump(all_results, f, indent=2)

asyncio.run(main())
