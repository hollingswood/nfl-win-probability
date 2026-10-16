#!/usr/bin/env bash
# Commit and push a run's results; never fail silently.
# usage: scripts/push_results.sh "commit message" path [path...]
# Stages the given paths plus every other tracked file the run modified (a dirty tree makes `git pull --rebase` refuse,
# which used to drop the full run's results without any error: 2026-10-13..15).
# If another run pushed first and both changed the same files:
#   - append-only *.jsonl logs merge line by line (.gitattributes merge=union);
#   - whole-file state/outputs that each run rewrites (news_seen.json, odds_credits.json, *_state.json, predictions
#     outputs...) keep THIS run's version, since it was computed last;
#   - paper-bet ledgers and placed bets are never auto-resolved: that fails loudly.
set -u
msg="$1"; shift
git config user.name "nfl-bot"
git config user.email "nfl-bot@users.noreply.github.com"
git add "$@" 2>/dev/null || true
git add -u
if git diff --cached --quiet; then echo "no changes"; exit 0; fi
git commit -q -m "$msg"

auto_ok() {   # file may be resolved by taking this run's version
  case "$1" in
    */paper_bets*|history/placed_bets.json) return 1 ;;
    history/*.json|history/cfb/*.json|history/futures/*.json|output/*.json|output/*.parquet|data/cfb/*) return 0 ;;
    *) return 1 ;;
  esac
}

resolve() {   # inside a stopped rebase: take this run's side for auto-resolvable files, else give up
  local f left
  for _ in 1 2 3 4 5 6 7 8; do
    left=$(git diff --name-only --diff-filter=U)
    [ -z "$left" ] && return 0
    for f in $left; do
      auto_ok "$f" || return 1
      case "$f" in
        *seen*.json)   # "already read" id lists: keep both runs' ids
          python3 - "$f" <<'PY' || git checkout --theirs -- "$f"
import json, subprocess, sys
f = sys.argv[1]
a, b = (json.loads(subprocess.run(["git", "show", f":{n}:{f}"], capture_output=True, text=True, check=True).stdout) for n in (2, 3))
if isinstance(a, list) and isinstance(b, list):
    open(f, "w").write(json.dumps(sorted(set(map(str, a)) | set(map(str, b)))))
elif isinstance(a, dict) and isinstance(b, dict):
    open(f, "w").write(json.dumps({**a, **b}))
else:
    sys.exit(1)
PY
          ;;
        *) git checkout --theirs -- "$f" ;;
      esac
      git add -- "$f"
    done
    GIT_EDITOR=true git rebase --continue >/dev/null 2>&1 || true
    git rev-parse -q --verify REBASE_HEAD >/dev/null || return 0
  done
  return 1
}

for i in 1 2 3 4 5; do
  if git pull -q --rebase --autostash 2>/dev/null || resolve; then
    if git push -q; then echo "pushed: $msg"; exit 0; fi
  else
    echo "::warning title=push::rebase conflict that can't be auto-resolved (try $i): $(git status --short | grep -E '^(UU|AA|DU|UD)' | head -8 | tr '\n' ' ')"
    git rebase --abort 2>/dev/null || true
  fi
  sleep $((5 * i))
done
echo "::error title=push::could not push '$msg' after 5 tries: $(git status --short | head -8 | tr '\n' ' ')"
exit 1
