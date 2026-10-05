<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AI-Q Sandboxed Python

This component provides a request-scoped `python(code)` tool for the Data Science
Agent. Every call launches one bounded Python process with a fresh namespace in
the request-owned, policy-bound OpenShell sandbox. Variables, imports,
DataFrames, and fitted objects do not persist across calls. The sandbox is
deleted when the request ends or a script exceeds its time limit.

The runner preloads NumPy (`np`), pandas (`pd`), SciPy (`scipy` and `stats`),
scikit-learn (`sklearn`), and statsmodels (`sm`). It also exposes
`list_analysis_results()`, `analysis_result()`, `analysis_rows()`,
`analysis_sql()`, and `analysis_latest()` for exact, programmatic access to
successful structured-data responses from the same request. AI-Q re-uploads the authoritative manifest and receipts before
every script, so the helpers remain automatic without relying on Python state.
Those scientific packages are installed in the separately built OpenShell
image, not in the AI-Q host runtime. The `sandboxed-python-test` optional extra
exists only for unit-testing the uploaded runner outside that image.

The tool is for analysis only. It has no configured ontology-provider client or
source database connection; structured-data access remains in agent-level tools.

`sandboxed_python` has no host-process backend. Its required `sandbox` field must
reference a `deep_research_sandbox` function configured with `provider:
openshell`, `network: blocked`, policy attestation, per-request creation, and
terminal deletion. AI-Q uploads only its trusted one-shot runner, model-authored
code request, and bounded request-owned structured-data receipts; it never copies the
application environment into the sandbox. Every process starts with hard
memory, CPU, process-count, open-file, output-file, and wall-time limits.
