#!/bin/bash

[ -d "/mnt/data" ] && cp -r /mnt/data/nuplan-v1.1/maps/* $NUPLAN_MAPS_ROOT

# Modify `planner=simple_planner` to submit your planner instead.
# For an example of how to write a hydra config, see nuplan/planning/script/config/simulation/planner/simple_planner.yaml.
conda run -n nuplan --no-capture-output python -u nuplan/planning/script/run_submission_planner.py output_dir=/data1/nuplan/luca/exp/submission planner=simple_planner
# -n = name of conda environment
# -u = force the stdout and stderr streams to be unbuffered; this option has no effect on stdin; also PYTHONUNBUFFERED=x