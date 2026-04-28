#!/usr/bin/env bash
# autorun_daily.sh
#
# Daily lifecycle manager for single_overlay.py on Ubuntu.
#
# Schedule:
#   08:00  —  wake from sleep (via rtcwake), then start python3 single_overlay.py
#   22:00  —  kill python3 single_overlay.py, then suspend the computer
#
# Smart start:
#   If run during active hours (08:00–22:00) the script skips straight to
#   launching Python — no sleep needed. Handy for afternoon testing.
#
# Usage:
#   bash task1/autorun_daily.sh
#
# Requirements:
#   - sudo privileges (password-free recommended) for rtcwake + systemctl
#   - python3 and all dependencies installed in your environment
#
# Sudo tip — to avoid password prompts add these lines to /etc/sudoers
# (run: sudo visudo):
#   <your-user> ALL=(ALL) NOPASSWD: /usr/sbin/rtcwake
#   <your-user> ALL=(ALL) NOPASSWD: /bin/systemctl suspend

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
# Both autorun_daily.sh and sleep_computer.sh live inside task1/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(which python3)"
TARGET_SCRIPT="single_overlay.py"
SLEEP_SCRIPT="$SCRIPT_DIR/sleep_computer.sh"
LOG_FILE="$SCRIPT_DIR/single_overlay.log"

WAKE_HOUR=8    # 08:00 — computer wakes and Python starts
STOP_HOUR=22   # 22:00 — Python stops and computer sleeps

# ── Validation ────────────────────────────────────────────────────────────────
if [[ ! -f "$SCRIPT_DIR/$TARGET_SCRIPT" ]]; then
    echo "❌  Cannot find $SCRIPT_DIR/$TARGET_SCRIPT"
    exit 1
fi

if [[ ! -f "$SLEEP_SCRIPT" ]]; then
    echo "❌  Cannot find sleep script at $SLEEP_SCRIPT"
    exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "❌  python3 not found in PATH."
    exit 1
fi

echo "✅  python3   : $PYTHON_BIN"
echo "✅  script    : $SCRIPT_DIR/$TARGET_SCRIPT"
echo "✅  log file  : $LOG_FILE"
echo ""

# ── Helpers ───────────────────────────────────────────────────────────────────

# Current hour as a plain integer (no leading zero) — requires GNU date (Ubuntu)
now_hour() {
    date +%-H
}

# Seconds from now until HH:00:00 today.
# Returns a negative number if that time has already passed today.
secs_until_today_at() {
    local h=$1
    echo $(( $(date -d "today ${h}:00:00" +%s) - $(date +%s) ))
}

# Unix timestamp of the next 08:00 (today if still future, otherwise tomorrow)
next_wake_timestamp() {
    local h
    h=$(now_hour)
    if (( h < WAKE_HOUR )); then
        date -d "today 08:00:00" +%s
    else
        date -d "tomorrow 08:00:00" +%s
    fi
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
log "autorun_daily.sh started (active window: ${WAKE_HOUR}:00 – ${STOP_HOUR}:00)"

while true; do
    H=$(now_hour)

    # ── DAYTIME: active window ────────────────────────────────────────────────
    if (( H >= WAKE_HOUR && H < STOP_HOUR )); then

        log "Inside active hours — launching $TARGET_SCRIPT …"

        # Launch Python from the script directory (task1/)
        cd "$SCRIPT_DIR"
        "$PYTHON_BIN" "$TARGET_SCRIPT" >> "$LOG_FILE" 2>&1 &
        PY_PID=$!
        log "Python started (PID $PY_PID). Logs → $LOG_FILE"

        # Calculate how long until 22:00
        SECS=$(secs_until_today_at "$STOP_HOUR")

        if (( SECS > 0 )); then
            log "Waiting ${SECS}s until 22:00 …"
            sleep "$SECS"
        else
            log "Already past 22:00 — stopping immediately."
        fi

        # ── 22:00 reached ────────────────────────────────────────────────────
        log "22:00 reached — stopping Python (PID $PY_PID) …"
        kill "$PY_PID" 2>/dev/null || true
        wait "$PY_PID" 2>/dev/null || true
        log "Python stopped."

        log "Calling sleep script …"
        bash "$SLEEP_SCRIPT"
        # Execution resumes here after the machine wakes (at 08:00 via rtcwake)
        log "Woke up at $(date '+%H:%M:%S') — resuming loop …"

    # ── NIGHT: outside active window ─────────────────────────────────────────
    else

        WAKE_TS=$(next_wake_timestamp)
        WAKE_READABLE=$(date -d "@${WAKE_TS}" '+%Y-%m-%d %H:%M')
        log "Outside active hours (now $(date +%H:%M)). Scheduling wake at $WAKE_READABLE …"

        # rtcwake: set RTC alarm then suspend to RAM.
        # Execution resumes automatically after the machine wakes.
        sudo rtcwake -m mem -t "$WAKE_TS"
        log "Woke up at $(date '+%H:%M:%S') — resuming loop …"

    fi
done
