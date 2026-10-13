#!/usr/bin/env bash
# Usage: live.sh <label> <command...>
# Runs the command, keeps its output in results/<label>.log, and every minute replaces one comment on
# issue $LIVE_ISSUE with the tail of that log (best effort: a failed post never fails the job).
set -u
label="$1"; shift
mkdir -p results
log="results/live_$(echo "$label" | tr ': ' '_-').log"
"$@" > "$log" 2>&1 &
pid=$!
cid=$(gh api "repos/$GITHUB_REPOSITORY/issues/$LIVE_ISSUE/comments" -f body="run $GITHUB_RUN_ID \`$label\`: started" --jq .id 2>/dev/null || true)
post() {
  [ -n "$cid" ] || return 0
  { printf 'run %s `%s` (%s, %s UTC)\n```\n' "$GITHUB_RUN_ID" "$label" "$1" "$(date -u +%H:%M)"; tail -c 60000 "$log"; printf '\n```\n'; } |
    jq -Rs '{body: .}' | gh api -X PATCH "repos/$GITHUB_REPOSITORY/issues/comments/$cid" --input - > /dev/null 2>&1 || true
}
while kill -0 "$pid" 2>/dev/null; do
  sleep 60
  post running
done
st=0
wait "$pid" || st=$?
post "finished, exit $st"
cat "$log"
exit "$st"
