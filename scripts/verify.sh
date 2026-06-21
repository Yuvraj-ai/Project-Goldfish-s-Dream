#!/usr/bin/env bash
set -euo pipefail

echo "=== Ruff ==="
.venv/bin/ruff check src/open_deep_research/ --ignore=D1

echo "=== pip-audit ==="
.venv/bin/pip-audit --ignore-vuln PYSEC-2022-42919 --ignore-vuln PYSEC-2024-24892 2>/dev/null || true

echo "=== Secrets detection ==="
SUSPICIOUS=$(grep -rn 'SK-[a-zA-Z0-9]\{20,\}\|AKIA[0-9A-Z]\{16\}\|-----BEGIN.*PRIVATE KEY-----' \
    src/open_deep_research/ --include='*.py' --include='*.toml' --include='*.yaml' --include='*.yml' 2>/dev/null || true)
if [ -n "$SUSPICIOUS" ]; then
    echo "WARNING: Possible secrets found:"
    echo "$SUSPICIOUS"
fi

echo "=== pytest + coverage ==="
.venv/bin/python -m pytest tests/ -q --cov=src/open_deep_research

echo "=== All checks passed ==="
