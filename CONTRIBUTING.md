# Contributing Guidelines

We welcome contributions to the NVIDIA Deep Researcher Agent blueprint. This repository uses a maintainer-reviewed pull request workflow with DCO sign-off, code-owner review, copy-pr-bot mirroring, and GitHub Actions validation.

## Before You Start

- Search existing issues and pull requests before opening new work.
- Open an issue or discussion before large design changes, public APIs, deployment changes, or contributor workflow changes.
- Do not include secrets, credentials, private hostnames, internal-only logs, customer data, or generated local artifacts.
- Target the `develop` branch unless a maintainer asks you to use a release branch.

## Pull Requests

1. Fork the repository and create a focused branch from `develop`.
2. Make the smallest coherent change and add or update tests for behavior changes.
3. Sign off every commit with `git commit -s`.
4. Run the relevant local validation before opening the PR.
5. Open a pull request into `develop` and fill out the PR template with exact validation evidence.
6. Address review feedback until required checks and code-owner review pass.

## Local Validation

Use the narrowest command that covers your change, then include the exact output or workflow link in the PR.

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The MCP server is an independent uv project. For changes under `mcp/`, also
run:

```bash
uv sync --project mcp --extra dev
uv run --project mcp --extra dev pytest mcp/tests
```

Dependency changes must leave both project locks current: check the root with
`uv lock --check` and MCP with `uv lock --project mcp --check`.

For UI changes:

```bash
cd frontends/ui
npm ci
npm run lint
npm run type-check
npm run test:ci
npm run build
```

For deployment changes, run the relevant Helm or compose validation and describe the environment used.

## CI and Bot Workflow

Deep Researcher Agent uses push-triggered GitHub Actions. Pull requests are mirrored by copy-pr-bot to `pull-request/<PR number>` branches after a maintainer or configured vetter comments `/ok to test`, and CI runs on those mirrored branches.

Repository owners, organization members, and collaborators can request NVSkills validation by commenting:

```text
/nvskills-ci
```

Maintainers can request bot-driven merge with:

```text
/merge
```

The `/merge` command requires the RAPIDS ops-bot GitHub App to be installed and the PR to satisfy repository rules, including required checks, code-owner review, resolved review threads, and branch policy.

## Signing Your Work

We require all contributors to sign off on their commits. This certifies that the contribution is your original work, or that you have the right to submit it under this project's license.

```bash
git commit -s -m "Add focused change"
```

This appends a line such as:

```text
Signed-off-by: Your Name <your@email.com>
```

Commits without sign-off may be rejected.

## Developer Certificate of Origin

Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
