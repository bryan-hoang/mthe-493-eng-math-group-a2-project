#!/bin/bash

LOGS="logs"

pkill -15 python
cd ~/Documents/mthe-493-group-a2
timestamp=$(date +%s)
pipenv run python src/worker.py >"${LOGS}/${timestamp}.out.log" 2>"${LOGS}/${timestamp}.err.log" &

