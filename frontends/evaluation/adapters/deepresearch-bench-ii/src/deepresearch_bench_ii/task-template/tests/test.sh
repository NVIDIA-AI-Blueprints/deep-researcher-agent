#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
python -m pip install --no-cache-dir --quiet requests
python /tests/verifier.py --metadata /tests/metadata.json --report /workspace/report.md
