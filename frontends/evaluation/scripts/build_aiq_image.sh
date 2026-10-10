#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  printf 'Usage: %s\n' "${0##*/}"
  printf '\nBuild and load the AI-Q Harbor runtime image from the repository root.\n'
  printf 'Set PLATFORM to override host detection (linux/arm64 or linux/amd64).\n'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ $# -eq 0 ]] || {
  usage >&2
  exit 2
}

command -v docker >/dev/null 2>&1 || die "docker is required"
command -v git >/dev/null 2>&1 || die "git is required"
docker buildx version >/dev/null 2>&1 || die "Docker Buildx is required"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
aiq_checkout="$(git -C "$script_dir" rev-parse --show-toplevel)"
dockerfile="$aiq_checkout/deploy/Dockerfile"
aiq_image="aiq-harbor:local"
aiq_revision="$(git -C "$aiq_checkout" rev-parse HEAD)"

[[ -f "$dockerfile" ]] || die "AI-Q Dockerfile not found: $dockerfile"

if [[ -n "${PLATFORM:-}" ]]; then
  platform="$PLATFORM"
else
  case "$(uname -m)" in
    arm64 | aarch64) platform="linux/arm64" ;;
    x86_64 | amd64) platform="linux/amd64" ;;
    *) die "cannot infer a Docker platform from $(uname -m); set PLATFORM explicitly" ;;
  esac
fi

case "$platform" in
  linux/arm64 | linux/amd64) ;;
  *) die "unsupported PLATFORM '$platform' (expected linux/arm64 or linux/amd64)" ;;
esac

printf 'Building %s from %s for %s\n' "$aiq_image" "$dockerfile" "$platform"
docker buildx build \
  --platform "$platform" \
  --target builder \
  --load \
  --file "$dockerfile" \
  --label "org.opencontainers.image.revision=$aiq_revision" \
  --tag "$aiq_image" \
  "$aiq_checkout"

docker run --rm -i \
  --platform "$platform" \
  --entrypoint /app/.venv/bin/python \
  "$aiq_image" - <<'PY'
import importlib.metadata as metadata

import aiq_agent
import nat
from nat.runtime.loader import load_workflow
from nat.utils.atif_converter import IntermediateStepToATIFConverter

print("aiq-agent=" + metadata.version("aiq-agent"))
print("nvidia-nat=" + metadata.version("nvidia-nat"))
print("Harbor runner imports: ok")
PY

docker image inspect "$aiq_image" \
  --format 'id={{.Id}} platform={{.Os}}/{{.Architecture}} revision={{index .Config.Labels "org.opencontainers.image.revision"}} size={{.Size}}'
printf 'Built and verified %s\n' "$aiq_image"
