#!/usr/bin/env bash
# Pod batch 2: the winner of batch 1 (keys error x4) with and without the listening labels, two seeds each, on the swapped studio split.
#   arm c-labels  : + labels/stems.csv and labels/segments.csv (exact keys targets, solo windows)
#   arm d-control : no labels
# Order: seed 0 of both arms first, so a pair exists even if the money runs out. See RUNPOD-BATCH2.md.
#
# On the REUSED pod (stopped after batch 1; /workspace with MUSDB18_prepared and studio_prepared is kept, the container disk is not):
#   cd /workspace/mo-music && git pull && cd trainer
#   HOURS_PER_RUN=2 AUTO_STOP=1 bash scripts/runpod_batch2.sh 2>&1 | tee /workspace/batch2.log
# Results: /workspace/runs/b2-<arm>-s<seed>/ and the bundle /workspace/batch2-results.tar.gz.
set -uo pipefail
cd "$(dirname "$0")/.."

[ -d /workspace/studio_prepared/train ] || { echo "upload the prepared studio data to /workspace/studio_prepared first"; exit 1; }
[ -d /workspace/MUSDB18_prepared/train ] || { echo "MUSDB18_prepared is missing: run 'bash scripts/runpod_setup.sh data-musdb' first"; exit 1; }
bash scripts/runpod_setup.sh data-musdb || { echo "setup failed"; exit 1; }   # reinstalls the code (container disk was wiped); data is kept

# Swap the studio split in place (idempotent): Shimmering Star (piano) into train, In My Belief into test.
S=/workspace/studio_prepared
if [ -d "$S/test/Shimmering Star" ] && [ -d "$S/train/In My Belief" ]; then
    mv "$S/test/Shimmering Star" "$S/train/" && mv "$S/train/In My Belief" "$S/test/"
fi
[ -d "$S/train/Shimmering Star" ] && [ -d "$S/test/In My Belief" ] || { echo "studio split is not the swapped one:"; ls "$S/train" "$S/test"; exit 1; }
echo "studio train: $(ls "$S/train" | tr '\n' ',')  test: $(ls "$S/test" | tr '\n' ',')"

for seed in 0 1; do
    for arm in c-labels d-control; do
        echo "=== $(date +%H:%M) arm $arm seed $seed ==="
        CONFIG="configs/runpod-b2-$arm.yaml" HOURS="${HOURS_PER_RUN:-2}" SEED="$seed" bash scripts/runpod_setup.sh train
        echo "=== $(date +%H:%M) arm $arm seed $seed finished (exit $?) ==="
    done
done

# (tar does not expand wildcards when creating an archive: find lists the files instead.)
(cd /workspace/runs && find b2-* -maxdepth 1 -type f \( -name run.json -o -name model.pt -o -name model.json \) -print0 \
    | tar --null -T - -czf /workspace/batch2-results.tar.gz) \
    && echo "bundle: $(tar -tzf /workspace/batch2-results.tar.gz | wc -l) files, $(ls -lh /workspace/batch2-results.tar.gz | awk '{print $5}')"

if [ "${AUTO_STOP:-0}" = "1" ]; then
    echo "stopping this pod (the volume with the results is kept)"
    runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"
else
    echo "AUTO_STOP is off: stop the pod yourself (it is billing)"
fi
