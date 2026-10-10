#!/bin/bash
set -euo pipefail

python -m pip install --no-cache-dir --quiet google-genai==1.73.1 requests
python /tests/verifier.py /tests/metadata.json /workspace/answer.txt
