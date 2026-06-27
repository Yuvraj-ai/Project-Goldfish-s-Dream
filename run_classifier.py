#!/usr/bin/env python
"""Task 3-1: Run classifier accuracy evaluation against golden set."""
import json, os, sys
from dotenv import load_dotenv
load_dotenv()

key = os.environ.get('OPENROUTER_API_KEY')
model = os.environ.get('OPENROUTER_MODEL', 'openai/gpt-oss-120b:free')
os.environ['OPENAI_API_KEY'] = key
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

import asyncio
from langgraph.types import Command
from open_deep_research.deep_researcher import classify_research_request

async def main():
    with open('tests/golden_set/classifier_test_set.json') as f:
        test_set = json.load(f)

    correct = 0
    results = []
    for i, item in enumerate(test_set):
        state = {
            'messages': [{'role': 'user', 'content': item['query']}],
            'research_brief': item['query'],
            'sources': [],
            'evidence_cards': [],
        }
        cfg = {
            'classifier_model': f'openai:{model}',
            'research_model': f'openai:{model}',
            'enable_mode_classification': True,
        }
        try:
            result = await classify_research_request(state, {'configurable': cfg})
            mode = result.update.get('research_mode', 'FAIL') if isinstance(result, Command) else 'FAIL'
            ok = mode == item['expected_mode']
            if ok:
                correct += 1
            results.append({'query': item['query'][:60], 'expected': item['expected_mode'], 'got': mode, 'ok': ok})
        except Exception as e:
            results.append({'query': item['query'][:60], 'expected': item['expected_mode'], 'got': f'ERR:{type(e).__name__}', 'ok': False})
        sys.stdout.write(f'\r  [{i+1}/{len(test_set)}] correct={correct} acc={correct/(i+1)*100:.1f}%')
        sys.stdout.flush()

    accuracy = correct / len(test_set)
    print(f'\n\n=== TASK 3-1: CLASSIFIER ACCURACY ===')
    print(f'  Result: {accuracy*100:.1f}% ({correct}/{len(test_set)})')
    print(f'  Target: >85% -- {"PASS" if accuracy > 0.85 else "FAIL"}')

    by_mode = {}
    for r in results:
        m = r['expected']
        by_mode.setdefault(m, {'total': 0, 'correct': 0})
        by_mode[m]['total'] += 1
        if r['ok']:
            by_mode[m]['correct'] += 1
    print(f'\n  Breakdown by mode:')
    for m, v in sorted(by_mode.items()):
        pct = v['correct'] / v['total'] * 100
        print(f'    {m:35s} {v["correct"]:2d}/{v["total"]:2d} ({pct:3.0f}%)')

    wrong = [r for r in results if not r['ok']]
    if wrong:
        print(f'\n  Misclassifications ({len(wrong)}):')
        for w in wrong:
            print(f'    exp={w["expected"]:30s} got={w["got"]:30s} q="{w["query"]}"')
    
    # Save results
    out = {'accuracy': accuracy, 'correct': correct, 'total': len(test_set), 'results': results}
    with open('tests/golden_set/classifier_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n  Results saved to tests/golden_set/classifier_results.json')

asyncio.run(main())
