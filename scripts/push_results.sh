#!/usr/bin/env bash
# Commit and push a run's results; never fail silently.
# usage: scripts/push_results.sh "commit message" path [path...]
# Stages the given paths plus every other tracked file the run modified (a dirty tree makes `git pull --rebase` refuse,
# which used to drop the full run's results without any error: 2026-10-13..15).
set -u
msg="$1"; shift
git config user.name "nfl-bot"
git config user.email "nfl-bot@users.noreply.github.com"
git add "$@" 2>/dev/null || true
git add -u
if git diff --cached --quiet; then echo "no changes"; exit 0; fi
git commit -q -m "$msg"
for i in 1 2 3 4 5; do
  if git pull -q --rebase --autostash; then
    if git push -q; then echo "pushed: $msg"; exit 0; fi
  else
    echo "::warning title=push::rebase failed (try $i): $(git status --short | head -8 | tr '\n' ' ')"
    git rebase --abort 2>/dev/null || true
  fi
  sleep $((5 * i))
done
echo "::error title=push::could not push '$msg' after 5 tries: $(git status --short | head -8 | tr '\n' ' ')"
exit 1
