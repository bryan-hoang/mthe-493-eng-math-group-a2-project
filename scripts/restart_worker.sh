#!/bin/bash

pkill -15 python
cd ~/Documents/mthe-493-group-a2
timestamp=$(date +%s)
pipenv run python src/worker.py >"${timestamp}.out.log" 2>"${timestamp}.err.log" &

