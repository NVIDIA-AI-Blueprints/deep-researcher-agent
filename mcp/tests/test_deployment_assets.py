# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 5 public container, Compose, and database deployment contracts."""

from __future__ import annotations

import hashlib
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _REPO_ROOT / "mcp" / "Dockerfile"
_DEEP_RESEARCHER_DOCKERFILE = _REPO_ROOT / "deploy" / "Dockerfile"
_COMPOSE_FILE = _REPO_ROOT / "deploy" / "compose" / "docker-compose.mcp.yaml"
_INIT_SQL = _REPO_ROOT / "mcp" / "deploy" / "init-mcp-db.sql"
_DOCKERIGNORE = _REPO_ROOT / ".dockerignore"
_SMOKE_SCRIPT = _REPO_ROOT / "mcp" / "scripts" / "protocol_smoke.py"
_THIRD_PARTY_LICENSE = _REPO_ROOT / "LICENSE-THIRD-PARTY"
_SIDEBAR_TEMPLATE = _REPO_ROOT / "docs" / "source" / "_templates" / "sidebar-nav-bs.html"

_RELEASE_NOTICE_COMPONENTS = (
    "loguru 0.7.3",
    "nemoguardrails 0.21.0",
    "Inter 3.019",
    "py-rust-stemmers 0.1.8",
    "pathspec 1.1.1",
    "crc32c 2.7.1",
    "en-core-web-lg 3.8.0",
    "crossbeam-deque 0.8.6",
    "crossbeam-epoch 0.9.18",
    "crossbeam-utils 0.8.21",
    "either 1.16.0",
    "heck 0.5.0",
    "libc 0.2.186",
    "once_cell 1.21.4",
    "proc-macro2 1.0.106",
    "pyo3-build-config 0.28.3",
    "pyo3-ffi 0.28.3",
    "pyo3-macros-backend 0.28.3",
    "pyo3-macros 0.28.3",
    "pyo3 0.28.3",
    "quote 1.0.45",
    "rayon-core 1.13.0",
    "rayon 1.12.0",
    "rust-stemmers 1.2.0",
    "serde 1.0.228",
    "serde_core 1.0.228",
    "serde_derive 1.0.228",
    "syn 2.0.117",
    "target-lexicon 0.13.5",
    "unicode-ident 1.0.24",
)

_EXPECTED_VERBATIM_LABELS = (
    "loguru 0.7.3 LICENSE",
    "nemoguardrails 0.21.0 LICENSE.md",
    "nemoguardrails 0.21.0 Apache License 2.0",
    "nemoguardrails 0.21.0 LICENCES-3rd-party",
    "Inter 3.019 LICENSE.txt (SIL Open Font License 1.1)",
    "Setuptools 82.0.0 MIT LICENSE",
    "autocommand 2.2.2 GNU LGPL version 3",
    "GNU GPL version 3 incorporated by LGPL version 3",
    "backports.tarfile 1.2.0 MIT LICENSE",
    "importlib_metadata 8.7.1 Apache-2.0 LICENSE",
    "jaraco.context 6.1.0 MIT LICENSE",
    "jaraco.functools 4.4.0 MIT LICENSE",
    "jaraco.text 4.0.0 MIT LICENSE",
    "wheel 0.46.3 MIT LICENSE",
    "zipp 3.23.0 MIT LICENSE",
    "py-rust-stemmers 0.1.8 root MIT LICENSE",
    "Standard Apache License 2.0 for MIT OR Apache-2.0 Rust crates",
    "Generic MIT grant used by listed Rust crates",
    "Crossbeam MIT LICENSE",
    "either MIT LICENSE",
    "heck MIT LICENSE",
    "libc MIT LICENSE",
    "PyO3 MIT LICENSE",
    "PyO3 Apache License 2.0",
    "Rayon MIT LICENSE",
    "rust-stemmers root MIT LICENSE",
    "Snowball algorithm BSD-3-Clause LICENSE",
    "target-lexicon Apache-2.0 WITH LLVM-exception LICENSE",
    "Unicode License v3",
    "pathspec 1.1.1 Mozilla Public License 2.0",
    "crc32c 2.7.1 LICENSE (LGPL-2.1)",
    "crc32c 2.7.1 LICENSE.slice-by-8",
    "crc32c 2.7.1 LICENSE.google-crc32c",
    "crc32c 2.7.1 AUTHORS.google-crc32c",
    "Intel slicing-by-8 BSD-2-Clause notice and full terms",
    "Mark Adler notice and upstream alteration notice",
    "en-core-web-lg 3.8.0 LICENSE",
    "en-core-web-lg 3.8.0 LICENSES_SOURCES",
    "pydata-sphinx-theme 0.16.1 BSD-3-Clause LICENSE",
)
_EXPECTED_VERBATIM_SHA256 = (
    "e0affae8:4147849a:9e507840:ebed9ac8:1e5f9ec0:f781f855:85e4a359:41a627a7",
    "e64c3f40:7e9f0bf5:54123118:7565c628:07203845:f2d3fa9b:bffff54b:40e08fd7",
    "c5e0634b:49c6aeed:dfcad0d7:9a063491:1dca6d67:a869fdde:c7941190:1bd175ae",
    "9dc41f44:d3b5386d:146b5b64:8b5a1704:f7d880c3:7a0005ce:e33d76c1:fa359bb2",
    "cf6b2123:84273706:19bbe36f:75ac8732:5adf4e70:083aa675:7e577116:95505e7a",
    "f4bdd270:089565b7:e04e8b56:3250949c:362941d8:a2291120:78474d77:c507990e",
    "19bac021:6792027e:9c658307:7bc7b0d9:ef995237:007a6df8:28420e77:cd0baf80",
    "9a70b6b9:34e7fd69:58de126c:9fd689e1:d56d0c4f:3215fd8d:e5e5fb4a:7ae37df4",
    "9c3f3b00:d28b0a13:9ff8f204:97479538:f58511e3:af819b83:d137bbff:7ae3a207",
    "3d9fd7db:8e7e0617:fecae1bd:c8ecc47b:478fd253:dd67cb06:eb4fa108:2630d106",
    "89b015b9:b4e6610d:ab0db556:9ec835ba:834d0a79:47c1b3e8:9fbfa1ae:b3731c7a",
    "bb3f3e85:63c8b221:0b9d2c8e:04deec8c:0d73475a:4939fe3b:51dba639:27e9c4a0",
    "497b15c4:46e2272d:47f52501:41c805a2:f441c86a:82ff1e01:2ed8bca8:6a5f8124",
    "f3a75238:b8b05f49:df47f14e:d261fbbf:56c8a3d2:bc45938f:213a002a:bcff36c2",
    "80044c0a:81f8ab20:225be896:d8f5a4ac:cfb56112:cb1f652a:61de3645:894a6994",
    "beab576e:05a04ae5:3a4ca71d:06010404:163c5578:fbb02999:9b03df79:9211fe29",
    "b5b5b4a5:c83c340b:a60782da:b7310053:4a4eabba:54f2a5bb:6170def3:5d8eed69",
    "4ff381e8:8b28ff79:91f70106:21032c09:4d6f72b4:0adead22:cd69a8c1:8c644054",
    "a11cf39b:b37542b7:9c29f3be:411f9646:3e420442:f4ea7df1:e178affa:a7f73f68",
    "71af1663:ea492064:0b055d02:af46bdc6:76cad8c9:aee05788:fbfa3f94:6d9d1aba",
    "ad969c7f:89914dda:ae5f11d0:5f3b500f:d1178476:55614262:b276fb65:c68711fb",
    "0379e7ab:b4d4282c:93385cff:66ed1d2f:43eb9a33:331d3e8c:b6cef3b2:371e3876",
    "e6a78343:c462308b:329a3c82:3bcd7bd8:cc57cefb:d61f1169:83a4b590:439bf580",
    "1626e1fc:89583609:158d60ce:5bba6a0b:4183e5d2:44658031:34bc9ce5:4d39faf9",
    "92ecde34:30f2bf8c:77727bab:8a61752a:94352e4a:8fe98a65:5c5322b6:c0ecd469",
    "37216b96:764ef16b:90e39bae:bc9f282a:284f460a:24182e84:db1faedb:edbf6f32",
    "8713c6fd:33f1da5a:a8dadf98:314c231c:77c610fd:393b9f9b:8cb41535:967697b4",
    "87864c48:98a13200:56e3b0af:53ae66e2:9946886e:e774bad1:ae4fefe8:6eeef48c",
    "8af50d69:3b2abdb3:c6411f3d:2a5b506b:55ed1872:55179964:e7e8c9af:a11ff04f",
    "17c26879:f6554489:9e0dfc70:194c9e87:4af76c5a:952c592c:f2a25b7d:4b487d2a",
    "9f3a6104:aa6061ce:85ab2283:f506c1fe:05504561:c9fda98e:d595c61a:89d26b8b",
    "1c233626:29c734a6:37da7252:1182a6fb:89c70e6b:0e03f1ec:2b2c763a:2714ae19",
    "afe422ea:39601273:9608203d:486517ed:85533ced:68b4a354:da8a5ed9:b4eb1863",
    "5d01d648:d5dd8271:6aac655a:4ab793a5:d56a0d4f:e357f1ba:71d4751f:41b47c15",
    "f8de74b6:c268ec04:e433d410:273a4075:fc506825:d4002485:a88a1157:55a9bb42",
    "3b9af289:61ab403a:b9005f0e:2cb6ec41:3564ce47:d15d2511:839fa061:9efb03e9",
    "4a923943:099e2d38:13d7d282:ebb9d29f:089b7da:9ffc007b:c929ab50:f57555ee3",
    "03737400:64971302:ab8882af:3ba85776:8da2e50c:de7486ec:936b3a00:7f49c4e1",
    "618a8cd0:4f2e0c95:634f616a:1ced3584:dbbe7854:ce0037b9:e78bed0f:ca19ae59",
)

_EXPECTED_SQL_HASH = "05c27bca7385f6127017bee72ae067d02a49849db73b483d42868aaaf90341c7"  # pragma: allowlist secret
_MCP_ENVIRONMENT = {
    "DEEP_RESEARCHER_CHECKPOINT_DB",
    "DEEP_RESEARCHER_MCP_ALLOWED_HOSTS",
    "DEEP_RESEARCHER_MCP_ALLOWED_ORIGINS",
    "DEEP_RESEARCHER_MCP_CONFIG",
    "DEEP_RESEARCHER_MCP_CORS_ORIGINS",
    "DEEP_RESEARCHER_MCP_HOST",
    "DEEP_RESEARCHER_MCP_LOG_LEVEL",
    "DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS",
    "DEEP_RESEARCHER_MCP_PATH",
    "DEEP_RESEARCHER_MCP_PORT",
    "DEEP_RESEARCHER_MCP_SHALLOW_INLINE_WAIT_SECONDS",
    "DEEP_RESEARCHER_MCP_WORKERS",
    "NVIDIA_API_KEY",
    "TAVILY_API_KEY",
}
_EXPECTED_COPY_LINES = (
    "COPY pyproject.toml README.md ./",
    "COPY src/ ./src/",
    "COPY sources/knowledge_layer/ ./sources/knowledge_layer/",
    "COPY sources/tavily_web_search/ ./sources/tavily_web_search/",
    "COPY mcp/LICENSE mcp/pyproject.toml mcp/README.md mcp/uv.lock ./mcp/",
    "COPY mcp/src/ ./mcp/src/",
    "COPY mcp/scripts/check_runtime_dependencies.py ./mcp/scripts/check_runtime_dependencies.py",
    "COPY configs/config_mcp.yml ./configs/config_mcp.yml",
    "COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv",
    "COPY --chown=10001:10001 configs/config_mcp.yml /app/configs/config_mcp.yml",
    "COPY --chown=10001:10001 LICENSE LICENSE-THIRD-PARTY /licenses/",
)


@pytest.fixture(scope="module")
def protocol_smoke_namespace() -> dict[str, object]:
    return runpy.run_path(str(_SMOKE_SCRIPT), run_name="deployment_smoke_test")


def _normalized_sql() -> str:
    lines = []
    for raw_line in _INIT_SQL.read_text().splitlines():
        line = raw_line.rstrip()
        if line.strip() and not line.lstrip().startswith("--"):
            lines.append(line)
    return "\n".join(lines) + "\n"


def _dockerfile_stage(text: str, stage: str) -> str:
    marker_position = text.index(f" AS {stage}\n")
    stage_start = text.rfind("\nFROM ", 0, marker_position) + 1
    next_stage = text.find("\nFROM ", marker_position)
    return text[stage_start:] if next_stage == -1 else text[stage_start:next_stage]


def _release_notice_section(text: str, section_number: int) -> str:
    addendum_start = text.index("Deep Researcher Agent 2.2 RELEASE CONTAINER ATTRIBUTION ADDENDUM")
    separator = "\n----------------------------------------------------------------------------\n"
    section_marker = f"{separator}{section_number}. "
    section_start = text.index(section_marker, addendum_start) + len(separator)
    next_section = text.find(f"{separator}{section_number + 1}. ", section_start)
    if next_section == -1:
        end_marker = (
            "\n============================================================================\n"
            "END OF Deep Researcher Agent 2.2 RELEASE"
        )
        next_section = text.index(end_marker, section_start)
    return text[section_start:next_section]


def _verbatim_payload(text: str, label: str) -> str:
    return text.split(f"BEGIN VERBATIM: {label}\n", 1)[1].split(f"\nEND VERBATIM: {label}", 1)[0]


def test_release_dockerfile_is_public_reproducible_and_non_root() -> None:
    text = _DOCKERFILE.read_text()

    assert "ARG PYTHON_IMAGE=python:3.13.12-slim-bookworm" in text
    assert "ARG UV_VERSION=0.11.26" in text
    assert text.count("FROM ${PYTHON_IMAGE}") == 2
    assert "AS builder" in text
    assert "AS release" in text
    assert "        git \\\n" in _dockerfile_stage(text, "builder")
    assert "uv sync" in text
    assert "--project /app/mcp" in text
    assert "--frozen" in text
    assert "--package deep-researcher-mcp-server" not in text
    assert "--no-dev" in text
    assert "--no-default-groups" in text
    assert "--no-editable" in text
    assert "COPY . " not in text
    assert "COPY pyproject.toml uv.lock" not in text
    assert "mcp/uv.lock" in text
    assert "USER 10001:10001" in text
    assert "EXPOSE 9001" in text
    assert "HEALTHCHECK" in text
    assert 'ENTRYPOINT ["python", "-m", "deep_researcher_mcp.server"]' in text
    assert "configs/config_mcp.yml" in text
    assert "DEEP_RESEARCHER_MCP_CONFIG=/app/configs/config_mcp.yml" in text
    assert "COPY --chown=10001:10001 LICENSE LICENSE-THIRD-PARTY /licenses/" in _dockerfile_stage(text, "release")
    assert "/opt/venv/bin/python mcp/scripts/check_runtime_dependencies.py" in text
    assert tuple(line.strip() for line in text.splitlines() if line.startswith("COPY ")) == _EXPECTED_COPY_LINES


def test_deep_researcher_dockerfile_ignores_opt_in_workspace_sources_for_root_install() -> None:
    text = _DEEP_RESEARCHER_DOCKERFILE.read_text()

    assert "RUN uv pip install --no-sources --no-deps -e . \\" in text


def test_deep_researcher_final_images_include_project_and_third_party_licenses() -> None:
    text = _DEEP_RESEARCHER_DOCKERFILE.read_text()
    copy_instruction = "COPY LICENSE LICENSE-THIRD-PARTY /licenses/"

    assert (_REPO_ROOT / "LICENSE").is_file()
    assert _THIRD_PARTY_LICENSE.is_file()
    for stage in ("dev", "release"):
        stage_text = _dockerfile_stage(text, stage)
        assert stage_text.count(copy_instruction) == 1


def test_release_third_party_notice_contains_reviewed_payloads() -> None:
    text = _THIRD_PARTY_LICENSE.read_text()
    sections = {number: _release_notice_section(text, number) for number in range(1, 7)}
    rust_components = _RELEASE_NOTICE_COMPONENTS[7:]
    assert "release-specific for the six distributions named there" in text
    assert "A separate source-distribution-only\naddendum follows it." in text
    assert "This release-specific addendum covers the six Python distributions identified" in text
    expected_components_by_section = {
        1: ("loguru 0.7.3",),
        2: ("nemoguardrails 0.21.0", "Inter 3.019"),
        3: ("py-rust-stemmers 0.1.8", *rust_components),
        4: ("pathspec 1.1.1",),
        5: ("crc32c 2.7.1",),
        6: ("en-core-web-lg 3.8.0",),
    }
    for section_number, components in expected_components_by_section.items():
        missing_components = [component for component in components if component not in sections[section_number]]
        assert not missing_components

    expected_notices_by_section = {
        1: ("Copyright (c) 2017",),
        2: (
            "Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES",
            (
                "Attribution Statements: Nvidia actively chooses the MIT license to apply to files with this "
                "copyright and license."
            ),
            "Copyright (c) 2023 Mckay Wrigley",
            "Inter 3.019",
            "Copyright (c) 2016-2020 The Inter Project Authors.",
            '"Inter" is trademark of Rasmus Andersson.',
            "SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007",
            (
                "Source: https://raw.githubusercontent.com/rsms/inter/"
                "0a5106e0bde18df09374066bf3a7998e3546307d/LICENSE.txt"
            ),
            "Wheel member: nemoguardrails-0.21.0.dist-info/LICENCES-3rd-party",
            "SHA-256: 20c77c71c35608683e7176aa5f093a00fdf3112848cebcd06c04e84dc852d939",
            "SHA-256: 4d7d9c95e7d7f2f0ebf76d5e0b344826b74e903a34028ee18ab54bb639e45906",
            "Copyright (c) Facebook, Inc. and its affiliates.",
            "Copyright (c) 2015-2021 Martin Hensel",
            "tailwindcss v3.3.3 | MIT License",
            "@author Lea Verou <https://lea.verou.me>",
            "@author   Feross Aboukhadijeh <https://feross.org>",
            "GNU LESSER GENERAL PUBLIC LICENSE",
            "The complete MPL-2.0 text in section 4 below applies",
        ),
        3: (
            "Copyright (c) 2024 qdrant",
            "UNICODE LICENSE V3",
            "LLVM Exceptions to the Apache 2.0 License",
        ),
        4: (
            "Copyright © 2013-2026 Caleb P. Burns",
            "Mozilla Public License Version 2.0",
        ),
        5: (
            "Copyright (C) 2013 Mark Adler",
            "GNU LESSER GENERAL PUBLIC LICENSE",
        ),
        6: (
            (
                "https://github.com/explosion/spacy-models/releases/download/"
                "en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
            ),
            "SHA-256: 293e9547a655b25499198ab15a525b05b9407a75f10255e405e8c3854329ab63",
            "Wheel member: en_core_web_lg-3.8.0.dist-info/LICENSE",
            "Raw SHA-256: 3933c176979b68bc6d0bcc902c7d6c130f1d127f476f17ba5cdba8d99cfd0012",
            "Wheel member: en_core_web_lg-3.8.0.dist-info/LICENSES_SOURCES",
            "Raw SHA-256: ea21333cecd2593c1fea842e9362bedf7b556abfb97cd08ce0650f2228b2eb58",
            "Copyright 2021 ExplosionAI GmbH",
            "# OntoNotes 5",
            "* License: commercial (licensed by Explosion)",
            "# ClearNLP Constituent-to-Dependency Conversion",
            "* License: Citation provided for reference, no code packaged with model",
            "# WordNet 3.0",
            "Permission to use, copy, modify and distribute this software and",
            "WordNet 3.0 Copyright 2006 by Princeton University.  All rights reserved.",
            'THIS SOFTWARE AND DATABASE IS PROVIDED "AS IS" AND PRINCETON',
            "The name of Princeton University or Princeton may not be used in",
            "Title to copyright in this software, database and",
            "# Explosion Vectors (OSCAR 2109 + Wikipedia + OpenSubtitles + WMT News Crawl)",
            "* License: CC0",
            "1. Copyright and Related Rights.",
        ),
    }
    for section_number, notices in expected_notices_by_section.items():
        missing_notices = [notice for notice in notices if notice not in sections[section_number]]
        assert not missing_notices

    begin_labels = [
        line.removeprefix("BEGIN VERBATIM: ") for line in text.splitlines() if line.startswith("BEGIN VERBATIM: ")
    ]
    end_labels = [
        line.removeprefix("END VERBATIM: ") for line in text.splitlines() if line.startswith("END VERBATIM: ")
    ]
    assert begin_labels == list(_EXPECTED_VERBATIM_LABELS)
    assert end_labels == begin_labels
    assert len(_EXPECTED_VERBATIM_LABELS) == len(_EXPECTED_VERBATIM_SHA256)
    for label, expected_sha256 in zip(_EXPECTED_VERBATIM_LABELS, _EXPECTED_VERBATIM_SHA256, strict=True):
        payload = _verbatim_payload(text, label)
        assert hashlib.sha256(payload.encode()).hexdigest() == expected_sha256.replace(":", "")
    assert "/licenses/LICENSE-THIRD-PARTY" in text


def test_source_only_pydata_notice_and_local_template_preserve_bsd_terms() -> None:
    notices = _THIRD_PARTY_LICENSE.read_text()
    source_addendum = notices.split("Deep Researcher Agent 2.2 SOURCE-DISTRIBUTION-ONLY ATTRIBUTION ADDENDUM", 1)[1]
    source_addendum = source_addendum.split(
        "END OF Deep Researcher Agent 2.2 SOURCE-DISTRIBUTION-ONLY ATTRIBUTION ADDENDUM", 1
    )[0]
    template = _SIDEBAR_TEMPLATE.read_text()

    assert _SIDEBAR_TEMPLATE.is_file()
    assert "docs/source/_templates/sidebar-nav-bs.html" in source_addendum
    assert (
        "This addendum applies to the Deep Researcher Agent source distribution only. The referenced\n"
        "documentation template is not installed in the Deep Researcher Agent release container."
    ) in source_addendum
    assert "pydata-sphinx-theme v0.16.1" in source_addendum
    upstream_commit = "c47786b993c85f0f442cc8d6e6b55e5d4e92b6b9"  # pragma: allowlist secret
    expected_urls = (
        "https://github.com/pydata/pydata-sphinx-theme/releases/tag/v0.16.1",
        f"https://github.com/pydata/pydata-sphinx-theme/commit/{upstream_commit}",
        (
            "https://raw.githubusercontent.com/pydata/pydata-sphinx-theme/"
            f"{upstream_commit}/src/pydata_sphinx_theme/theme/pydata_sphinx_theme/components/sidebar-nav-bs.html"
        ),
        f"https://raw.githubusercontent.com/pydata/pydata-sphinx-theme/{upstream_commit}/LICENSE",
    )
    for url in expected_urls:
        assert url in source_addendum
    assert "Decoded SHA-256: d92fc698623827d7204139dee7874fbd7447f9256b8199acbd71cf8ee53d6f64" in source_addendum
    assert "Copyright (c) 2018, pandas" in source_addendum
    assert "Licenses: BSD-3-Clause AND Apache-2.0" in source_addendum

    license_payload = _verbatim_payload(notices, "pydata-sphinx-theme 0.16.1 BSD-3-Clause LICENSE")
    separator = "----------------------------------------------------------------------------"
    bsd_license = license_payload.split(f"{separator}\n", 1)[1]
    bsd_license = bsd_license.rsplit(f"\n{separator}", 1)[0]
    template_license = "BSD 3-Clause License" + template.split("BSD 3-Clause License", 1)[1].split("\n#}", 1)[0]
    assert template_license == bsd_license
    assert "* Redistributions of source code must retain the above copyright notice" in bsd_license
    assert "* Redistributions in binary form must reproduce the above copyright notice" in bsd_license
    assert "* Neither the name of the copyright holder nor the names of its" in bsd_license
    assert 'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"' in bsd_license

    assert "SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES." in template
    assert "SPDX-FileCopyrightText: Copyright (c) 2018, pandas" in template
    assert "SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0" in template
    assert (
        "Modified by NVIDIA from pydata-sphinx-theme v0.16.1 sidebar-nav-bs.html:\n"
        "https://github.com/pydata/pydata-sphinx-theme/blob/v0.16.1/"
        "src/pydata_sphinx_theme/theme/pydata_sphinx_theme/components/sidebar-nav-bs.html"
    ) in template
    assert (
        'The NVIDIA modifications remove the "Section Navigation" title, use a "Table\n'
        'of Contents" accessibility label, and generate the navigation tree from depth\n'
        "zero."
    ) in template
    rendered_template = template.split("#}", 1)[1]
    assert "Section Navigation" not in rendered_template
    assert "aria-label=\"{{ _('Table of Contents') }}\"" in rendered_template
    assert "startdepth=0" in rendered_template


def test_init_sql_preserves_reference_schema_and_upgrade_history() -> None:
    normalized = _normalized_sql()
    assert hashlib.sha256(normalized.encode()).hexdigest() == _EXPECTED_SQL_HASH
    assert "CREATE TABLE IF NOT EXISTS mcp_jobs" in normalized
    assert "idx_mcp_jobs_runner_state" in normalized
    assert "VALUES ('aiq_maas_mcp', 1)" in normalized
    assert "VALUES ('aiq_maas_mcp', 2)" in normalized


def test_compose_stack_is_isolated_explicit_and_health_gated() -> None:
    compose = yaml.safe_load(_COMPOSE_FILE.read_text())
    assert compose["name"] == "deep-researcher-mcp"
    assert set(compose["services"]) == {"postgres", "deep-researcher-mcp"}
    assert set(compose["volumes"]) == {"mcp-postgres-data"}
    assert set(compose["networks"]) == {"deep-researcher-mcp-network"}
    assert compose["networks"]["deep-researcher-mcp-network"].get("internal") is not True

    for service in compose["services"].values():
        assert "container_name" not in service
        assert "env_file" not in service
        assert service["networks"] == ["deep-researcher-mcp-network"]
        assert "healthcheck" in service

    postgres = compose["services"]["postgres"]
    assert postgres["image"] == "postgres:16-alpine"
    assert postgres["environment"]["POSTGRES_PASSWORD"] == "local_mcp_password"  # pragma: allowlist secret
    assert postgres["ports"] == ["127.0.0.1:${DEEP_RESEARCHER_MCP_POSTGRES_PORT:-1234}:5432"]
    assert "mcp-postgres-data:/var/lib/postgresql/data" in postgres["volumes"]
    assert "../../mcp/deploy/init-mcp-db.sql:/docker-entrypoint-initdb.d/init-mcp-db.sql:ro" in postgres["volumes"]

    mcp = compose["services"]["deep-researcher-mcp"]
    assert mcp["build"] == {"context": "../..", "dockerfile": "mcp/Dockerfile", "target": "release"}
    assert set(mcp["environment"]) == _MCP_ENVIRONMENT
    assert (
        mcp["environment"]["DEEP_RESEARCHER_CHECKPOINT_DB"]
        == "postgresql://deep_researcher:local_mcp_password@postgres:5432/"  # pragma: allowlist secret
        "deep_researcher_jobs"
    )
    assert mcp["environment"]["DEEP_RESEARCHER_MCP_PORT"] == "9001"
    assert mcp["environment"]["DEEP_RESEARCHER_MCP_CONFIG"] == "/app/configs/config_mcp.yml"
    assert mcp["environment"]["DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS"] == "${DEEP_RESEARCHER_MCP_MAX_QUERY_CHARS:-8000}"
    assert mcp["ports"] == ["127.0.0.1:${DEEP_RESEARCHER_MCP_PUBLISHED_PORT:-9001}:9001"]
    assert mcp["depends_on"] == {"postgres": {"condition": "service_healthy"}}
    assert mcp["security_opt"] == ["no-new-privileges:true"]
    assert mcp["cap_drop"] == ["ALL"]


def test_deployment_context_excludes_env_files_and_includes_package_readmes() -> None:
    lines = set(_DOCKERIGNORE.read_text().splitlines())
    assert {".env", ".env.*", "**/.env", "**/.env.*"} <= lines
    assert {
        "!README.md",
        "!mcp/README.md",
        "!sources/knowledge_layer/README.md",
        "!sources/tavily_web_search/README.md",
    } <= lines


def test_protocol_smoke_script_is_importable_without_running(protocol_smoke_namespace: dict[str, object]) -> None:
    namespace = protocol_smoke_namespace
    assert namespace["EXPECTED_SERVER_NAME"] == "deep_researcher_deep_research"
    assert namespace["EXPECTED_TOOLS"] == {"get_final_report", "poll_query", "submit_query"}
    assert namespace["EXPECTED_HEALTH_STATUS"] == "ready"
    assert namespace["UNKNOWN_JOB_ID"] == "00000000-0000-4000-8000-000000000000"
    assert namespace["FORBIDDEN_REQUEST_HEADERS"] == {"authorization"}


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("http://127.0.0.1:9001/mcp", "http://127.0.0.1:9001/health"),
        ("https://example.com/nested/mcp", "https://example.com/health"),
        ("http://[::1]:9001/mcp", "http://[::1]:9001/health"),
    ],
)
def test_protocol_smoke_accepts_safe_endpoint_forms(
    protocol_smoke_namespace: dict[str, object],
    endpoint: str,
    expected: str,
) -> None:
    health_url = protocol_smoke_namespace["_health_url"]
    assert callable(health_url)
    assert health_url(endpoint) == expected


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://alice@example.com/mcp",
        "https://alice:sensitive-password@example.com/mcp",  # pragma: allowlist secret
        "https://example.com/mcp?token=sensitive-query-token",
        "https://example.com/mcp#sensitive-fragment",
        (
            "https://alice:sensitive-password@example.com/mcp"  # pragma: allowlist secret
            "?token=sensitive-query-token#sensitive-fragment"
        ),
    ],
)
def test_protocol_smoke_rejects_sensitive_endpoint_forms(
    protocol_smoke_namespace: dict[str, object],
    endpoint: str,
) -> None:
    health_url = protocol_smoke_namespace["_health_url"]
    assert callable(health_url)
    with pytest.raises(ValueError, match="must not include credentials, query parameters, or fragments") as exc_info:
        health_url(endpoint)
    assert endpoint not in str(exc_info.value)


def test_protocol_smoke_rejects_sensitive_url_before_client_construction_or_output(
    protocol_smoke_namespace: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    endpoint = (
        "https://sensitive-user:sensitive-password@127.0.0.1:1/mcp"  # pragma: allowlist secret
        "?token=sensitive-query-token#sensitive-fragment"
    )

    def unexpected_async_client(*args: object, **kwargs: object) -> None:
        raise AssertionError("HTTP client constructed before URL validation")

    httpx = protocol_smoke_namespace["httpx"]
    monkeypatch.setattr(httpx, "AsyncClient", unexpected_async_client)
    main = protocol_smoke_namespace["main"]
    assert callable(main)
    with pytest.raises(ValueError, match="must not include credentials, query parameters, or fragments") as exc_info:
        main(["--url", endpoint])

    captured = capsys.readouterr()
    for sensitive_value in (
        "sensitive-user",
        "sensitive-password",
        "sensitive-query-token",
        "sensitive-fragment",
    ):
        assert sensitive_value not in str(exc_info.value)
        assert sensitive_value not in captured.out
        assert sensitive_value not in captured.err


def test_protocol_smoke_cli_does_not_echo_rejected_url() -> None:
    endpoint = (
        "https://sensitive-user:sensitive-password@127.0.0.1:1/mcp"  # pragma: allowlist secret
        "?token=sensitive-query-token#sensitive-fragment"
    )
    completed = subprocess.run(
        [sys.executable, str(_SMOKE_SCRIPT), "--url", endpoint, "--health-timeout", "0.01"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "must not include credentials, query parameters, or fragments" in completed.stderr
    for sensitive_value in (
        "sensitive-user",
        "sensitive-password",
        "sensitive-query-token",
        "sensitive-fragment",
    ):
        assert sensitive_value not in completed.stdout
        assert sensitive_value not in completed.stderr
