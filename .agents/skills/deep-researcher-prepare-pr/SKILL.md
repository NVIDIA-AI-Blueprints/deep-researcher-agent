---
name: deep-researcher-prepare-pr
description: Use when preparing, opening, or updating an Deep Researcher Agent pull request — scoping the branch, signing commits with DCO, filling the PR template with real validation evidence, and following the copy-pr-bot CI and merge flow.
license: Apache-2.0
compatibility: Claude Code, Codex, Cursor, OpenCode, and Agent Skills-compatible tools.
metadata:
  version: "0.1.0"
  source-repo: "NVIDIA-AI-Blueprints/deep-researcher-agent"
  tags: "deep-researcher pull-request dco contributing release-qa"
allowed-tools: Read Bash Edit
---

# Prepare an Deep Researcher Agent Pull Request

Use this skill to take a finished Deep Researcher Agent change from a working branch to a
review-ready pull request: a focused branch off `develop`, DCO-signed commits,
the PR template filled with real validation evidence, and the copy-pr-bot CI
flow followed correctly.

## Start Here

- Confirm the change is scoped: no unrelated files, no generated artifacts, no
  secrets, no environment-specific hostnames.
- Confirm validation already ran. This skill does not invent test commands —
  run `deep-researcher-release-qa` first and reuse its exact output as evidence.
- Confirm every commit is signed off (DCO). Commits without a `Signed-off-by`
  trailer may be rejected.
- Target the `develop` branch unless a maintainer asked for a release branch.

## Authoritative References

- [CONTRIBUTING.md](../../../CONTRIBUTING.md): the canonical PR workflow, Local
  Validation, DCO text, and the copy-pr-bot / `/ok to test` / `/merge` flow.
- [.github/pull_request_template.md](../../../.github/pull_request_template.md):
  the exact sections and checklist your PR description must fill.
- [AGENTS.md](../../../AGENTS.md): "Git and PR hygiene" and the validation
  commands `deep-researcher-release-qa` runs.

For the step-by-step checklist and the bot command reference:

- [references/pr-checklist.md](references/pr-checklist.md)

## Workflow

1. Verify the branch is focused and based on `develop`; rebase or reset scope if
   unrelated files crept in.
2. Ensure every commit is signed: `git commit -s` (and `git commit --amend -s`
   or `git rebase --signoff` to fix unsigned commits already made).
3. Run the relevant checks via `deep-researcher-release-qa` and keep the exact output.
4. Push the branch and open the PR into `develop`.
5. Fill the PR template: Overview, Validation (paste the commands and output),
   reviewer starting point, related issues, and tick every checklist box you
   can honestly tick.
6. Drive CI: a maintainer or vetter comments `/ok to test`, copy-pr-bot mirrors
   the PR to `pull-request/<number>`, and CI runs there.
7. Address review feedback until required checks and code-owner review pass.

## Validation

This skill is about PR hygiene, not code execution; verify the contributor
mechanics:

```bash
git log --pretty=full origin/develop..HEAD | grep -c "Signed-off-by:"  # every commit signed
git diff --name-only origin/develop..HEAD                              # only intended files
git status --porcelain                                                 # no stray artifacts
```

Expected: the sign-off count equals your commit count, the changed-file list
contains only files relevant to this change, and the working tree is clean.

## Common Mistakes

- Unsigned commits. Use `git commit -s`; fix existing ones with
  `git rebase --signoff origin/develop` before pushing.
- A vague Validation section. Paste the real commands and their output from
  `deep-researcher-release-qa`; "ran tests" is not evidence.
- Scope creep: unrelated refactors, formatting churn on untouched files, or
  committed generated artifacts.
- Committing secrets or `deploy/.env` values. Resolve secrets at runtime; never
  paste them into the PR.
- Expecting CI to run on push alone — it runs on the copy-pr-bot mirror after
  `/ok to test`.
- Skipping the docs update for user-facing or contributor-facing changes.

## Related Skills

- `deep-researcher-release-qa`
- `deep-researcher-add-tool`
- `deep-researcher-add-data-source`
