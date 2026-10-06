<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deep Researcher Agent PR checklist and bot commands

Step-by-step companion to `SKILL.md`. The source of truth is `CONTRIBUTING.md`
and `.github/pull_request_template.md`; this file condenses them into an
actionable checklist.

## 1. Scope and branch

- Branch is created from `develop` and named for the change.
- The diff contains only files relevant to this change — no unrelated
  refactors, no formatting churn on untouched files, no generated artifacts.
- No secrets, credentials, private hostnames, internal-only logs, or customer
  data are included.

```bash
git diff --name-only origin/develop..HEAD   # confirm the file set
git status --porcelain                      # confirm a clean tree
```

## 2. DCO sign-off

Every commit must carry a `Signed-off-by: Your Name <your@email.com>` trailer.

```bash
git commit -s -m "Concise, scoped change"   # sign as you commit
git commit --amend -s --no-edit             # sign the latest commit
git rebase --signoff origin/develop         # sign a range already committed
```

Verify the count matches your commits:

```bash
git log --pretty=full origin/develop..HEAD | grep -c "Signed-off-by:"
```

## 3. Validation evidence

Run the checks with `deep-researcher-release-qa` and keep the exact commands and output.
Paste them into the PR's Validation section — do not summarize as "ran tests".

## 4. Open the PR

- Target branch is `develop`.
- Fill every section of `.github/pull_request_template.md`:
  - **Overview** — what changed and why.
  - **Validation** — the exact commands, output, workflow links, or screenshots.
  - **Where should reviewers start?** — the key file, test, or decision.
  - **Related Issues** — `Closes`/`Fixes`/`Relates to #...`.
- Tick each checklist item you can honestly tick (local checks, tests added,
  docs updated, no secrets, DCO sign-off).

## 5. CI and merge flow (copy-pr-bot)

Deep Researcher Agent uses push-triggered GitHub Actions on mirrored branches, not on the PR
branch directly:

- A maintainer or configured vetter comments `/ok to test`.
- copy-pr-bot mirrors the PR to `pull-request/<PR number>` and CI runs there.
- Owners, org members, and collaborators can request NVSkills validation with
  `/nvskills-ci`.
- Maintainers can request bot-driven merge with `/merge`, which requires the
  RAPIDS ops-bot app plus satisfied repository rules: required checks,
  code-owner review, resolved review threads, and branch policy.

## 6. Review loop

Address feedback until required checks pass and code-owner review is approved.
Keep new commits signed; re-run the relevant `deep-researcher-release-qa` checks after
substantive changes and update the Validation section if results change.
