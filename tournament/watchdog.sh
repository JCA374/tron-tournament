#!/bin/bash
# Run by cron every 5 minutes: (re)start the tournament supervisor and the site publisher if they are not running.
cd "$(dirname "$0")/.."
mkdir -p logs
pgrep -f "^.venv/bin/python -m tournament.supervisor" > /dev/null || \
  PYTHONPATH=. setsid nohup .venv/bin/python -m tournament.supervisor >> logs/supervisor.log 2>&1 < /dev/null &
pgrep -f "^/bin/bash ./publish_loop.sh" > /dev/null || setsid nohup ./publish_loop.sh > /dev/null 2>&1 < /dev/null &
