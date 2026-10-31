# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for confidence-scored citation verification (Issue #210)."""

from deep_researcher_agent.common.citation_verification import (
    SourceEntry,
    SourceRegistry,
    calculate_confidence,
    verify_citations,
)


def test_calculate_confidence_mapping():
    """Verify calculate_confidence returns expected scores for all match kinds."""
    assert calculate_confidence("exact") == 1.00
    assert calculate_confidence("truncation") == 0.95
    assert calculate_confidence("prefix") == 0.80
    assert calculate_confidence("child-path") == 0.70
    assert calculate_confidence("query-subset") == 0.75
    assert calculate_confidence("host-only") == 0.40
    assert calculate_confidence("unmatched") == 0.00
    # Unknown match kinds safely return 0.0 without raising KeyError
    assert calculate_confidence("unknown_kind") == 0.00


def test_resolve_with_kind_exact_and_truncation():
    """Verify resolve_with_kind identifies exact and truncation matches."""
    registry = SourceRegistry()
    registry.add(SourceEntry(url="https://example.com/research/paper1?id=abc"))

    resolved, kind = registry.resolve_with_kind("https://example.com/research/paper1?id=abc")
    assert resolved == "https://example.com/research/paper1?id=abc"
    assert kind == "exact"

    # Truncation match
    resolved, kind = registry.resolve_with_kind("https://example.com/research/paper1")
    assert resolved == "https://example.com/research/paper1?id=abc"
    assert kind == "truncation"


def test_resolve_with_kind_host_only():
    """Verify host-only match returns (None, 'host-only') and never falsifies URL."""
    registry = SourceRegistry()
    registry.add(SourceEntry(url="https://example.com/known/page"))

    resolved, kind = registry.resolve_with_kind("https://example.com/other/unseen/page")
    assert resolved is None
    assert kind == "host-only"


def test_resolve_with_kind_unmatched():
    """Verify completely unmatched URL returns (None, 'unmatched')."""
    registry = SourceRegistry()
    registry.add(SourceEntry(url="https://example.com/page"))

    resolved, kind = registry.resolve_with_kind("https://unrelated-domain.org/doc")
    assert resolved is None
    assert kind == "unmatched"


def test_verify_citations_scored_and_unmatched_kept():
    """Verify citations below threshold: host-only kept with score, unmatched removed (Issue #210)."""
    registry = SourceRegistry()
    registry.add(SourceEntry(url="https://example.com/research/report"))

    report = (
        "Here is a key finding [1] and another detail [2] and an unmatched source [3].\n\n"
        "## Sources\n"
        "[1] Report: https://example.com/research/report\n"
        "[2] Other Doc: https://example.com/other/subpage\n"
        "[3] Fake: https://hallucinated-source.com/fake\n"
    )

    result = verify_citations(report, registry)

    # Issue #210: Host-only citations are kept in valid_citations with score;
    # completely unmatched/fabricated citations are removed from valid_citations.
    valid_numbers = [c["number"] for c in result.valid_citations]
    assert 1 in valid_numbers
    assert 2 in valid_numbers
    assert 3 not in valid_numbers

    removed_numbers = [c["number"] for c in result.removed_citations]
    assert 3 in removed_numbers

    # Verify per-citation verification record fields (verified, confidence, match_kind, diagnostics)
    exact_cite = next(c for c in result.valid_citations if c["number"] == 1)
    assert exact_cite["verified"] is True
    assert exact_cite["confidence"] == 1.00
    assert exact_cite["match_kind"] == "exact"
    assert exact_cite["diagnostics"] is None

    host_only_cite = next(c for c in result.valid_citations if c["number"] == 2)
    assert host_only_cite["verified"] is False
    assert host_only_cite["confidence"] == 0.40
    assert host_only_cite["match_kind"] == "host-only"
    assert host_only_cite["diagnostics"] == "host_only_match"
    assert host_only_cite["url"] == "https://example.com/other/subpage"

    unmatched_cite = next(c for c in result.all_citations if c["number"] == 3)
    assert unmatched_cite["verified"] is False
    assert unmatched_cite["confidence"] == 0.00
    assert unmatched_cite["match_kind"] == "unmatched"
    assert unmatched_cite["diagnostics"] == "url_not_in_registry"
    assert unmatched_cite["url"] == "https://hallucinated-source.com/fake"

    # Verify all_citations completeness (retains all records for telemetry)
    assert len(result.all_citations) == 3

    # Verify aggregated diagnostics telemetry for removals and downgrades
    assert len(result.diagnostics) >= 2
    diag_unmatched = next(d for d in result.diagnostics if d["reason"] == "url_not_in_registry")
    assert diag_unmatched["count"] == 1
    assert 3 in diag_unmatched["citation_numbers"]

    diag_host_only = next(d for d in result.diagnostics if d["reason"] == "host_only_match")
    assert diag_host_only["count"] == 1
    assert 2 in diag_host_only["citation_numbers"]


def test_all_unmatched_citations_leaves_valid_citations_empty():
    """Verify a report with only fabricated sources leaves valid_citations empty."""
    registry = SourceRegistry()
    registry.add(SourceEntry(url="https://example.com/valid/source"))

    report = (
        "Completely fake claim [1].\n\n"
        "## Sources\n"
        "[1] Hallucinated: https://completely-fake-domain.xyz\n"
    )
    result = verify_citations(report, registry)
    assert len(result.valid_citations) == 0
    assert len(result.removed_citations) == 1
    assert len(result.all_citations) == 1

