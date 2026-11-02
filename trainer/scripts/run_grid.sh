#!/usr/bin/env bash
# Runs the MUSDB18-only architecture grid one run after another: each config in GRID with each seed in SEEDS.
# Run it on the pod, from the trainer folder, after `bash scripts/runpod_setup.sh data-musdb`:
#
#   nohup bash scripts/run_grid.sh &
#   grep -c '^{"epoch"' nohup.out        # finished epochs; 10 runs x 20 epochs = 200 when done
#
# Settings (environment): GRID="base soft wide deep both", SEEDS="0 1", HOURS=<cap per run, default 0.75>.
# AUTO_TERMINATE_AFTER=1 removes the pod GRACE_MIN minutes (default 30) after the last run: the total cost is then capped.
# The runs write to /workspace/runs/grid-<name>-s<seed>; print them with
#   python scripts/summarize_run.py --compact /workspace/runs/grid-*/run.json
cd "$(dirname "$0")/.." || exit 1
for name in ${GRID:-base soft wide deep both}; do
    for seed in ${SEEDS:-0 1}; do
        echo "=== grid-$name seed $seed, started $(date +%T)"
        SEED=$seed CONFIG=configs/runpod-grid-$name.yaml HOURS=${HOURS:-0.75} bash scripts/runpod_setup.sh train
    done
done
echo "=== grid finished $(date +%T)"
if [ "${AUTO_TERMINATE_AFTER:-0}" = "1" ]; then
    echo "removing this pod in ${GRACE_MIN:-30} minutes: download the results now"
    sleep $(( ${GRACE_MIN:-30} * 60 ))
    runpodctl remove pod "$RUNPOD_POD_ID"
fi
