#!/usr/bin/env bash
# Publish a cleaned copy of this repository to GitLab.
#
# GitLab must contain none of the assistant-related files, in the tip or in
# history. This GitHub repo stays the working repo; GitLab is a derived copy
# that is never edited by hand. Run this after every batch of commits:
#
#     scripts/publish_gitlab.sh git@gitlab.example.org:group/click-segmenter.git
#     scripts/publish_gitlab.sh <url> --dry-run     # build and inspect, no push
#
# What it does: fresh clone of the current repo (main only) into a temp dir,
# rewrite history with git-filter-repo, check the result for leftovers, push
# main WITHOUT force. The rewrite is deterministic, so re-running produces the
# same commit ids and the push is a fast-forward. A rejected push means someone
# committed on GitLab directly; resolve that by hand, never by forcing.
#
# The rules (fixed; changing them changes every commit id, i.e. a new history):
#   - drop .claude/, CLAUDE.md, PROGRESS.md, segmentation-project-prompt.md and
#     this script from every commit
#   - strip assistant co-author trailers from commit messages
#   - point references to CLAUDE.md at docs/TECHNICAL-RECORD.md instead
#   - normalise the author to one name and one e-mail
#
# Needs git-filter-repo: `brew install git-filter-repo`, `pip install
# git-filter-repo`, or it falls back to `uvx git-filter-repo`.

set -euo pipefail

GITLAB_URL="${1:?usage: publish_gitlab.sh <gitlab-url> [--dry-run]}"
DRY_RUN="${2:-}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
AUTHOR_NAME="Mais Isifzada"
AUTHOR_EMAIL="isifzadm@gmail.com"

if git filter-repo --version >/dev/null 2>&1; then
    FILTER=(git filter-repo)
elif command -v uvx >/dev/null 2>&1; then
    FILTER=(uvx git-filter-repo)
elif [ -x "$HOME/tools/bin/uvx.exe" ]; then
    FILTER=("$HOME/tools/bin/uvx.exe" git-filter-repo)
else
    echo "git-filter-repo not found (brew install git-filter-repo / pip install git-filter-repo)" >&2
    exit 1
fi

if [ -n "$(git -C "$SRC" status --porcelain)" ]; then
    echo "working tree has uncommitted changes; commit or stash them first" >&2
    exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
CLONE="$WORK/clone"

echo "== fresh clone of $SRC (main)"
git clone --quiet --single-branch --branch main --no-local "$SRC" "$CLONE"

# The words to remove are spelled from fragments so this file never contains
# them literally in the published tree it also happens to be excluded from.
A="Cla""ude"; B="Anth""ropic"; LOWER_A="$(printf '%s' "$A" | tr 'A-Z' 'a-z')"
TRAILER_RE="(?im)^co-authored-by: ${LOWER_A}[^\n]*\n?"

REPLACE="$WORK/replace.txt"
printf '%s==>%s\n' "${A^^}.md" "docs/TECHNICAL-RECORD.md" > "$REPLACE"

echo "== rewriting history"
(
    cd "$CLONE"
    "${FILTER[@]}" --force --quiet \
        --path .claude \
        --path "${A^^}.md" \
        --path PROGRESS.md \
        --path segmentation-project-prompt.md \
        --path scripts/publish_gitlab.sh \
        --invert-paths \
        --replace-text "$REPLACE" \
        --message-callback "
import re
message = re.sub(rb'$TRAILER_RE', b'', message)
return message.replace(b'${A^^}.md', b'docs/TECHNICAL-RECORD.md')" \
        --name-callback "return b'$AUTHOR_NAME'" \
        --email-callback "return b'$AUTHOR_EMAIL'"
)

echo "== checking for leftovers"
cd "$CLONE"
leftover=0
if git log --all --format='%an <%ae>%n%s%n%b' | grep -i -E "$LOWER_A|$B" >/dev/null; then
    echo "!! commit metadata still mentions the assistant" >&2; leftover=1
fi
if git rev-list --all | while read -r c; do git grep -l -i -E "$LOWER_A|$B" "$c" 2>/dev/null; done | grep -q .; then
    echo "!! some file in some commit still mentions the assistant" >&2; leftover=1
fi
if git log --all --format='%ae%n%ce' | grep -v -x "$AUTHOR_EMAIL" >/dev/null; then
    echo "!! unexpected author/committer e-mail in history" >&2; leftover=1
fi
[ "$leftover" -eq 0 ] || exit 1
echo "   clean: $(git rev-list --count main) commits, head $(git rev-parse --short main)"

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "== dry run; log of what would be pushed:"
    git log --format='%h %ad %s' --date=short main | head -15
    echo "   ... ($(git rev-list --count main) commits total)"
    exit 0
fi

echo "== pushing main to $GITLAB_URL (no force)"
git remote add gitlab "$GITLAB_URL"
git push gitlab main:main
echo "== done"
