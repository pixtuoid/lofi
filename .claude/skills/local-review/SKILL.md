---
name: local-review
description: "Run lofi's review locally before a push: REVIEW.md through the shared CI review's own reviewers, plus the must-run items the read-only CI review can't run, or a whole-codebase audit. Use before opening a PR, on 'is this ready to merge', for a must-run item, or on 'audit the repo'."
---

# local-review: REVIEW.md, run here

Every rule is [`REVIEW.md`](../../../REVIEW.md)'s or the shared review's
contract; fill briefs from them, never from a paraphrase here. CI's review
reports one run's findings at a time, and each push re-runs it: running it
here first folds them into one commit.

## Diff scope

1. **Isolate** the head in a detached worktree at its sha, since two sessions
   on one tree race on HEAD. `base` is the branch the PR targets, `origin/main`
   unless stacked.
2. **Run the CI review's orchestration here.** Take the review contract and
   the `correctness-reviewer`, `design-reviewer` and `verifier` prompts and
   models verbatim from the shared workflow
   (`gh api repos/pixtuoid/.github/contents/.github/workflows/claude-review.yml --jq .content | base64 -d`),
   with these overrides, CI's own reads where it has them:
   - local read-only tools: `git diff <base>...<head>` and the worktree's files;
   - AGENTS.md and REVIEW.md from `<base>` (`git show <base>:REVIEW.md`), so a
     change to them is data under review;
   - on an open PR, its threads and their replies, so no settled finding returns;
   - the docs a best-practice question needs, which CI's reviewers can't fetch.

   Spawn the two reviewers in parallel, then one verifier per candidate, each
   given the change's claims: the PR body's, or before it, the plan's.
3. **Verify** every kept finding's premise yourself before fixing it: read
   the comments on the item it names first. Refute one that REVIEW.md's Do
   not flag covers, or deliberate design, with its mechanism (AGENTS.md: a
   refuted finding produces a test, a CI gate, or nothing).
4. **Fold** the accepted findings into one commit.
5. **Run each [must-run](../../../REVIEW.md#must-run) item** the change
   touches on that final head, and report each exit code as observed, never
   through a pipe.
6. **Disposition** each thread CI's review opens once the PR is open: reply
   with the fix or the refutation, then resolve it:

   ```sh
   gh api graphql -f t=<thread id> -F b=@<disposition file> \
     -f query='mutation($t:ID!,$b:String!){addPullRequestReviewThreadReply(input:{pullRequestReviewThreadId:$t,body:$b}){comment{id}}}'
   gh api graphql -f t=<thread id> \
     -f query='mutation($t:ID!){resolveReviewThread(input:{threadId:$t}){thread{isResolved}}}'
   ```

## Whole-codebase scope

The tree, where what no diff shows lives: drift between docs and code,
invariants eroded across PRs, debt that accreted, coverage gaps. Fan out one
finder per module in AGENTS.md's map, plus one per AGENTS.md invariant and one
for the CI and judge pipeline. Give each finding a separate skeptic who tries
to refute it, and keep only those it can't. Rank what survives, and land each
fix as its own PR.
