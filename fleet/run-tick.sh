#!/usr/bin/env bash
# Wrapper for the nice fleet controller, invoked by cron every 10 minutes.
#
# Code vs state: this script and controller.py live in a pinned git worktree
# (immune to branch switching in the interactive checkout — a checkout to a
# pre-fleet branch once deleted the controller out from under cron for three
# days). All state — config.json, .env, fleet.sqlite3, tick.log — stays in
# STATE_DIR, which is also the working directory (the controller resolves
# --config and db_path relative to cwd).
#
# Cron runs with a minimal environment, so set PATH/HOME explicitly. Logs
# append to tick.log with a per-run timestamp header; simple size-based
# rotation keeps it bounded.
set -euo pipefail

export HOME=/home/claude
export PATH=/home/claude/.local/bin:/usr/local/bin:/usr/bin:/bin

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR=/home/claude/projects/nice/fleet
LOG="$STATE_DIR/tick.log"
cd "$STATE_DIR"

# One tick at a time: heavy ticks can exceed the 10-minute cadence, and an
# overlapping run crashes on sqlite's write lock mid-reconcile. Skip (don't
# queue) — the next scheduled tick picks up whatever this one would have done.
exec 9>"$STATE_DIR/.tick.lock"
if ! flock -n 9; then
  echo "===== tick $(date '+%Y-%m-%d %H:%M:%S %z') SKIPPED: previous tick still running ====="
  exit 0
fi

# Rotate if the log exceeds ~5 MB (keep one previous generation).
if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG")" -gt 5242880 ]; then
  mv -f "$LOG" "$LOG.1"
fi

echo "===== tick $(date '+%Y-%m-%d %H:%M:%S %z') ====="
# --config points at the live config (dry_run:false lives there).
# Don't let set -e swallow the exit-code log line on a failed tick.
set +e
uv run "$CODE_DIR/controller.py" --config config.json
rc=$?
set -e
echo "----- exit $rc at $(date '+%Y-%m-%d %H:%M:%S %z') -----"
exit "$rc"
