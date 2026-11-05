#!/usr/bin/env bash
# Apply branch protection to `main`. Requires the GitHub CLI (`gh`) logged in as a repo admin.
#
# Usage: scripts/protect-main.sh [owner/repo]
#   REQUIRED_APPROVALS=0 scripts/protect-main.sh   # e.g. to drop the review requirement
#
# Rules:
#   - Changes land through pull requests (no direct pushes, force pushes, or deletion)
#   - CI checks `lint`, `commitlint`, and `test` must pass, on a branch up to date with main
#   - REQUIRED_APPROVALS reviews (default 1); stale approvals are dismissed on new pushes
#   - All review conversations must be resolved
#   - Admins are not forced through these rules, so the owner can still override in a pinch
set -euo pipefail

REPO="${1:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}"
BRANCH="${BRANCH:-main}"
REQUIRED_APPROVALS="${REQUIRED_APPROVALS:-1}"

gh api --method PUT "repos/${REPO}/branches/${BRANCH}/protection" \
  -H "Accept: application/vnd.github+json" \
  --input - <<JSON
{
  "required_status_checks": {
    "strict": true,
    "checks": [
      {"context": "lint"},
      {"context": "commitlint"},
      {"context": "test"}
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "required_approving_review_count": ${REQUIRED_APPROVALS}
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
JSON

echo "Branch protection applied to ${REPO}@${BRANCH}"
