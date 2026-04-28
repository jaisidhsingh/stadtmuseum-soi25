#!/usr/bin/env bash
# sleep_computer.sh
# Immediately suspends the computer to RAM (S3 sleep).
# Called by autorun_daily.sh — can also be run standalone.
#
# Requires: sudo privileges for systemctl suspend

echo "[$(date +%H:%M:%S)] Suspending computer to RAM …"
sudo systemctl suspend
