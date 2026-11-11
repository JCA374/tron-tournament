#!/bin/bash
# Rebuild and push the site every 15 minutes (renders run with nice). Stop: kill the PID in publish_loop.pid.
cd "$(dirname "$0")"
mkdir -p logs
echo $$ > publish_loop.pid
while true; do
  PYTHONPATH=. nice -n 15 taskset -c 5-7 .venv/bin/python publish.py --push >> logs/publish.log 2>&1
  sleep 900
done
