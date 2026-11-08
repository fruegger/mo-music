#!/usr/bin/env bash
# Pod batch 1: two arms x two seeds, one run after the other, each under its own time cap (see RUNPOD-BATCH1.md).
#   arm a-promote : six-source recipe with hand tags + promotion (control)
#   arm b-keys4   : the same with the keys error weighted 4x where keys are audible
# Order: seed 0 of both arms first, so that a pair exists even if the money runs out.
#
# On the pod, in /workspace/mo-music/trainer, after /workspace/studio_prepared has been uploaded:
#   HOURS_PER_RUN=2 AUTO_TERMINATE=1 GRACE_MIN=30 bash scripts/runpod_batch1.sh 2>&1 | tee /workspace/batch1.log
# Results: /workspace/runs/b1-<arm>-s<seed>/ (run.json, model.pt, model.onnx, checkpoint.pt). Download them before the pod goes.
set -uo pipefail
cd "$(dirname "$0")/.."

[ -d /workspace/studio_prepared/train ] || { echo "upload the prepared studio data to /workspace/studio_prepared first"; exit 1; }
bash scripts/runpod_setup.sh data-musdb || { echo "MUSDB18 setup failed"; exit 1; }

for seed in 0 1; do
    for arm in a-promote b-keys4; do
        echo "=== $(date +%H:%M) arm $arm seed $seed ==="
        CONFIG="configs/runpod-b1-$arm.yaml" HOURS="${HOURS_PER_RUN:-2}" SEED="$seed" bash scripts/runpod_setup.sh train
        echo "=== $(date +%H:%M) arm $arm seed $seed finished (exit $?) ==="
    done
done

if [ "${AUTO_TERMINATE:-0}" = "1" ]; then
    echo "removing this pod in ${GRACE_MIN:-30} minutes: download the results now"
    sleep $(( ${GRACE_MIN:-30} * 60 ))
    runpodctl remove pod "$RUNPOD_POD_ID"
else
    echo "AUTO_TERMINATE is off: terminate the pod yourself when the results are downloaded (it is billing)"
fi
