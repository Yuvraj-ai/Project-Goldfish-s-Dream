#!/usr/bin/env python
"""Task 3-4: Citation Verification Rate — live URL checker accuracy test.

Measures how accurately CitationVerifier classifies real URLs as alive/dead.
Zero LLM API calls — pure HTTP.
"""
import asyncio, json, time
from open_deep_research.citation_verifier import CitationVerifier

# Test set: (url, expected_status, label)
# expected_status: "alive" | "dead" | "unverified"
TEST_SET = [
    # --- Known-good URLs (should be alive) ---
    ("https://www.python.org", "alive", "Python homepage"),
    ("https://github.com", "alive", "GitHub homepage"),
    ("https://en.wikipedia.org/wiki/Python_(programming_language)", "alive", "Wikipedia Python"),
    ("https://docs.python.org/3/", "alive", "Python docs"),
    ("https://pypi.org", "alive", "PyPI homepage"),
    ("https://news.ycombinator.com", "alive", "Hacker News"),
    ("https://arxiv.org", "alive", "arXiv"),
    ("https://www.w3.org", "alive", "W3C"),
    ("https://stackoverflow.com", "alive", "Stack Overflow"),
    ("https://www.ietf.org", "alive", "IETF"),

    # --- Known-dead / 404 URLs ---
    ("https://www.python.org/nonexistent-page-12345", "dead", "Python 404"),
    ("https://github.com/nonexistent-org-xyz123/nonexistent-repo", "dead", "GitHub 404"),
    ("https://en.wikipedia.org/wiki/NonexistentPageXYZ", "dead", "Wikipedia 404"),
    ("https://arxiv.org/abs/9999.99999", "dead", "arXiv 404"),
    ("https://pypi.org/project/nonexistent-package-xyz-123/", "dead", "PyPI 404"),

    # --- Redirects (should be alive via follow_redirects) ---
    ("https://httpbin.org/redirect-to?url=https%3A%2F%2Fwww.python.org", "alive", "302 redirect to Python"),
    ("https://github.com/langchain-ai", "alive", "GitHub org (redirect)"),

    # --- Edge cases ---
    ("https://www.google.com", "alive", "Google homepage"),
    ("https://www.wikipedia.org", "alive", "Wikipedia portal"),
    ("https://www.eff.org", "alive", "EFF"),

    # --- Potential unverified (bot protection) ---
    ("https://www.amazon.com", "alive", "Amazon (may bot-block)"),
    ("https://www.cloudflare.com", "alive", "Cloudflare (may bot-block)"),

    # --- URLs that often block bots ---
    ("https://www.linkedin.com", "alive", "LinkedIn (may bot-block)"),

    # --- Malformed / edge ---
    ("https://example.com", "alive", "RFC 2606 reserved domain"),
]

async def main():
    print(f"{'='*60}")
    print("TASK 3-4: CITATION VERIFICATION RATE")
    print(f"{'='*60}")
    print(f"\nTest set: {len(TEST_SET)} URLs\n")
    print(f"{'URL':<60} {'Expected':<12} {'Got':<12} {'Status Code':<12} {'Pass?':<6}")
    print("-" * 102)

    verifier = CitationVerifier(timeout=10.0)
    urls = [t[0] for t in TEST_SET]
    t0 = time.time()

    results = await verifier.verify_batch(urls)

    elapsed = time.time() - t0
    passed = 0
    failed = 0
    details = []

    for url, expected, label in TEST_SET:
        result = results.get(url, {})
        got = result.get("status", "error")
        code = result.get("status_code", "?")
        error = result.get("error", "")

        # Determine pass/fail
        # "unverified" can match either "unverified" expected or be a soft pass
        if got == expected:
            ok = True
        elif expected == "alive" and got in ("unverified",):
            ok = True  # Bot-blocked is an acceptable failure
        else:
            ok = False

        status_str = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        url_display = url[:58] + ".." if len(url) > 60 else url
        print(f"{url_display:<60} {expected:<12} {got:<12} {str(code):<12} {status_str:<6}")
        if error and not ok:
            print(f"  └─ error: {error[:100]}")

        details.append({
            "url": url,
            "label": label,
            "expected": expected,
            "got": got,
            "status_code": code,
            "pass": ok,
            "error": error,
        })

    await verifier.close()

    print("-" * 102)
    rate = (passed / len(TEST_SET)) * 100
    print(f"\nResults:")
    print(f"  Total:   {len(TEST_SET)}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Rate:    {rate:.1f}%")
    print(f"  Time:    {elapsed:.1f}s")
    print(f"\n  Gate: >90% — {'PASS' if rate >= 90 else 'FAIL'}")

    output = {
        "task": "3-4",
        "total": len(TEST_SET),
        "passed": passed,
        "failed": failed,
        "rate_pct": round(rate, 1),
        "gate": ">90%",
        "result": "PASS" if rate >= 90 else "FAIL",
        "elapsed_s": round(elapsed, 1),
        "results": details,
    }
    with open("docs/verification/task-3-4-citation-rate.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to docs/verification/task-3-4-citation-rate.json")

if __name__ == "__main__":
    asyncio.run(main())
