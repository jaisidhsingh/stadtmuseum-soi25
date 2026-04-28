#!/usr/bin/env bash
# autorun.sh
# Sets up cron jobs to:
#   - Start  python3 single_overlay.py at 05:00 every day
#   - Kill   python3 single_overlay.py at 22:00 every day
#
# Usage: bash autorun.sh
# Run once and forget — cron handles the rest automatically.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK1_DIR="$SCRIPT_DIR/task1"
PYTHON_BIN="$(which python3)"
TARGET_SCRIPT="single_overlay.py"
LOG_FILE="$TASK1_DIR/single_overlay.log"

# Cron time fields
START_MINUTE=0
START_HOUR=5       # 05:00
STOP_MINUTE=0
STOP_HOUR=22       # 22:00

# ── Validation ────────────────────────────────────────────────────────────────
if [[ ! -f "$TASK1_DIR/$TARGET_SCRIPT" ]]; then
    echo "❌  Cannot find $TASK1_DIR/$TARGET_SCRIPT"
    echo "    Make sure you are running this script from the project root."
    exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "❌  python3 not found in PATH. Install it or adjust PYTHON_BIN manually."
    exit 1
fi

echo "✅  python3 found at: $PYTHON_BIN"
echo "✅  Script found at:  $TASK1_DIR/$TARGET_SCRIPT"
echo "✅  Log file will be: $LOG_FILE"

# ── Build cron entries ────────────────────────────────────────────────────────
# The start job: cd into task1, run the script, append stdout+stderr to log
START_JOB="$START_MINUTE $START_HOUR * * * cd \"$TASK1_DIR\" && \"$PYTHON_BIN\" $TARGET_SCRIPT >> \"$LOG_FILE\" 2>&1"

# The stop job: find any running instance by script name and kill it
STOP_JOB="$STOP_MINUTE $STOP_HOUR * * * pkill -f \"$TARGET_SCRIPT\" || true"

# ── Install cron jobs (idempotent) ────────────────────────────────────────────
# Read existing crontab (ignore error if empty)
EXISTING_CRON="$(crontab -l 2>/dev/null || true)"

# Strip any previous entries for this script to avoid duplicates
CLEANED_CRON="$(echo "$EXISTING_CRON" | grep -v "$TARGET_SCRIPT" || true)"

# Append new jobs
NEW_CRON="$(printf '%s\n%s\n%s\n' "$CLEANED_CRON" "$START_JOB" "$STOP_JOB")"

echo "$NEW_CRON" | crontab -

echo ""
echo "🎉  Cron jobs installed successfully!"
echo ""
echo "    ▶  START : every day at 05:00  →  python3 $TARGET_SCRIPT"
echo "    ■  STOP  : every day at 22:00  →  pkill -f $TARGET_SCRIPT"
echo "    📄  Logs  : $LOG_FILE"
echo ""
echo "To view your active cron jobs:  crontab -l"
echo "To remove these jobs:           crontab -e  (delete the two $TARGET_SCRIPT lines)"
