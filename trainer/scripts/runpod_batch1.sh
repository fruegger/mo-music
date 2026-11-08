#!/usr/bin/env bash
# Pod batch 1: two arms x two seeds, one run after the other, each under its own time cap (see RUNPOD-BATCH1.md).
#   arm a-promote : six-source recipe with hand tags + promotion (control)
#   arm b-keys4   : the same with the keys error weighted 4x where keys are audible
# Order: seed 0 of both arms first, so that a pair exists even if the money runs out.
#
# On the pod, in /workspace/mo-music/trainer, after /workspace/studio_prepared has been uploaded:
#   HOURS_PER_RUN=2 AUTO_STOP=1 bash scripts/runpod_batch1.sh 2>&1 | tee /workspace/batch1.log
# Results: /workspace/runs/b1-<arm>-s<seed>/ (run.json, model.pt, model.onnx, checkpoint.pt) and the bundle
# /workspace/batch1-results.tar.gz (run.json + model.pt of every run, a few MB). They live on the volume disk, which a STOPPED pod
# keeps (about $0.20 per GB and month for the 30 GB volume), so with AUTO_STOP=1 nothing is lost if you cannot download right away:
# the pod stops itself after the last run (no more compute billing) and you copy the files out whenever you like, even by
# starting the pod CPU-only. AUTO_TERMINATE=1 instead REMOVES the pod (and the volume) after GRACE_MIN minutes: only with the files downloaded.
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

# (tar does not expand wildcards when creating an archive: it made an empty bundle once. find lists the files instead.)
(cd /workspace/runs && find b1-* -maxdepth 1 -type f \( -name run.json -o -name model.pt -o -name model.json \) -print0 \
    | tar --null -T - -czf /workspace/batch1-results.tar.gz) \
    && echo "bundle: $(tar -tzf /workspace/batch1-results.tar.gz | wc -l) files, $(ls -lh /workspace/batch1-results.tar.gz | awk '{print $5}')"

if [ "${AUTO_STOP:-0}" = "1" ]; then
    echo "stopping this pod (the volume with the results is kept; restart it, CPU-only is enough, to copy the files out)"
    runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"
elif [ "${AUTO_TERMINATE:-0}" = "1" ]; then
    echo "removing this pod in ${GRACE_MIN:-30} minutes: download the results now"
    sleep $(( ${GRACE_MIN:-30} * 60 ))
    runpodctl remove pod "$RUNPOD_POD_ID"
else
    echo "AUTO_STOP and AUTO_TERMINATE are off: stop the pod yourself (it is billing)"
fi
