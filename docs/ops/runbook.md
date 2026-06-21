# Operations Runbook

## Search API Outage

**Symptoms:** Research hangs at search phase, progress events stall.

**Check:**
1. Check Tavily status: `curl -s https://api.tavily.com/health | jq .`
2. Check arXiv status: `curl -s https://export.arxiv.org/api/query?search_query=all:test&max_results=1 | head -5`
3. Review LangSmith traces for search node errors.

**Resolution:**
- If Tavily is down: set `ENABLE_TAVILY=false` to fall back to other providers
- If all search APIs are down: feature-flag disable search, serve cached results

## LLM Rate Limiting

**Symptoms:** Research fails with ModelError, traces show 429 responses.

**Check:**
1. Check provider dashboard for rate limit usage
2. Review circuit breaker status in governor logs

**Resolution:**
- Reduce concurrency: lower `max_concurrent_searches` config value
- Switch model tier: enable model routing to downgrade from quality to balanced
- Wait for rate limit window to expire (usually 1-60 min)

## Database Corruption

**Symptoms:** API returns 500 on all endpoints, repository operations fail.

**Check:**
1. Check SQLite file: `sqlite3 research.db "PRAGMA integrity_check;"`
2. Review API logs for SQL errors

**Resolution:**
1. Stop server
2. Restore from backup: `cp research.db.backup research.db`
3. Restart server
4. If no backup, re-create DB (non-critical data like progress can be lost; reports should have been exported)

## Kill Switch Procedure

1. Set feature flag: `ENABLE_REST_API=false` in environment
2. Restart the server
3. Verify legacy CLI path works: `python -m open_deep_research "query"`
4. File a P0 bug with full LangSmith trace ID
5. Do NOT re-enable until root cause is fixed and canary passes
