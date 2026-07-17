#!/usr/bin/env bash
# Shared helpers for experiments/*.sh — sourced, not executed directly.
#
# Canonical output tree:
#   results/{domain}/{env}/{method}/seed{S}/{metrics.csv, summary.json, config.yaml}
#
# Every run skips itself if summary.json already exists at its leaf dir, so
# scripts are safe to re-run/resume after interruption.

RESULTS_ROOT="results"

# DRY_RUN=1 ./experiments/run_*.sh prints every command it would run instead
# of running it, so you can sanity-check the env/method/seed matrix first.
DRY_RUN="${DRY_RUN:-0}"

skip_if_done() {
  [ -f "$1/summary.json" ]
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

maybe_mkdir() {
  if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$1"
  fi
}

# run_py <module_or_script_args...> -- prints the command under DRY_RUN
# instead of executing it.
run_py() {
  if [ "$DRY_RUN" = "1" ]; then
    log "[dry-run] python3 $*"
    return 0
  fi
  python3 "$@"
}
