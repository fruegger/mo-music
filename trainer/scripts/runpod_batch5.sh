#!/usr/bin/env bash
# Pod batch 5: model size and training length against the Demucs gap (see RUNPOD-BATCH5.md).
#   arm i-wide : base_channels 32, depth 3    arm l-large : base_channels 48, depth 4    arm j-long : small network, 30,000 steps
# D (batch 2) is not rerun. Seed 0 only by default; a grey-zone arm gets its second seed with
#   ARMS="l-large" SEEDS="1" bash scripts/runpod_batch5.sh
#
# Data: the master copy lives on the network volume (/workspace/MUSDB18_prepared, /workspace/studio_prepared, the swapped studio split) and is
# copied ONCE to the pod's local disk (/root/data) because the volume was slow to read from; training reads the local copy,
# results go to /workspace/runs/b5-* (the volume). Run it with the Python 3.11 venv active:
#   source /workspace/venv/bin/activate
#   cd /workspace/mo-music && git pull && cd trainer
#   HOURS_PER_RUN=3 AUTO_STOP=1 nohup bash scripts/runpod_batch5.sh > /workspace/batch5.log 2>&1 &   then: tail -f /workspace/batch5.log
set -uo pipefail
cd "$(dirname "$0")/.."

MASTER="${MASTER:-/workspace}"
LOCAL="${LOCAL:-/root/data}"   # the sources log configs/runpod-sources-batch5.log points here
ARMS="${ARMS:-i-wide l-large j-long}"
SEEDS="${SEEDS:-0}"

[ -d "$MASTER/MUSDB18_prepared/train" ] || { echo "MUSDB18_prepared is missing on the volume: run 'bash scripts/runpod_setup.sh data-musdb' first"; exit 1; }
S=$MASTER/studio_prepared
[ -d "$S/train/Shimmering Star" ] && [ -d "$S/test/In My Belief" ] || { echo "the studio split on the volume is not the swapped one (unzip studio_prepared_swapped.zip into /workspace):"; ls "$S/train" "$S/test"; exit 1; }
python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' || { echo "no GPU on this pod"; exit 1; }
bash scripts/runpod_setup.sh install || { echo "install failed"; exit 1; }

# One local copy of the data (a replaced pod has an empty /root/data, a restarted one keeps it). A marker is written only after the copy checks out.
if [ ! -f "$LOCAL/.complete" ]; then
    mkdir -p "$LOCAL" || { echo "cannot create $LOCAL"; exit 1; }
    free_gb=$(df -Pk "$LOCAL" | awk 'NR==2 {printf "%d", $4 / 1048576}')
    need_gb=$(du -sk "$MASTER/MUSDB18_prepared" "$MASTER/studio_prepared" | awk '{s += $1} END {printf "%d", s / 1048576 + 1}')
    echo "local disk: ${free_gb} GB free, data ${need_gb} GB"
    [ "$free_gb" -ge 15 ] && [ "$free_gb" -ge $((need_gb + 8)) ] || { echo "not enough room for a local copy (need 15 GB free and data + 8 GB): start the pod with a bigger container disk"; exit 1; }
    t0=$(date +%s)
    rm -rf "$LOCAL/MUSDB18_prepared" "$LOCAL/studio_prepared"
    cp -a "$MASTER/MUSDB18_prepared" "$MASTER/studio_prepared" "$LOCAL/" || { echo "copy failed"; exit 1; }
    for d in MUSDB18_prepared studio_prepared; do
        a=$(find "$MASTER/$d" -type f | wc -l); b=$(find "$LOCAL/$d" -type f | wc -l)
        [ "$a" = "$b" ] || { echo "copy of $d differs: $a files on the volume, $b local"; exit 1; }
    done
    touch "$LOCAL/.complete"
    echo "copied $(du -sh "$LOCAL" | cut -f1) to $LOCAL in $(( $(date +%s) - t0 )) s"
fi

for seed in $SEEDS; do
    for arm in $ARMS; do
        echo "=== $(date +%H:%M) arm $arm seed $seed ==="
        CONFIG="configs/runpod-b5-$arm.yaml" HOURS="${HOURS_PER_RUN:-3}" SEED="$seed" bash scripts/runpod_setup.sh train
        echo "=== $(date +%H:%M) arm $arm seed $seed finished (exit $?) ==="
    done
done

(cd /workspace/runs && find b5-* -maxdepth 1 -type f \( -name run.json -o -name model.pt -o -name model.json \) -print0 \
    | tar --null -T - -czf /workspace/batch5-results.tar.gz) \
    && echo "bundle: $(tar -tzf /workspace/batch5-results.tar.gz | wc -l) files, $(ls -lh /workspace/batch5-results.tar.gz | awk '{print $5}')"

if [ "${AUTO_STOP:-0}" = "1" ]; then
    echo "stopping this pod (the volume is kept)"
    runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"
else
    echo "AUTO_STOP is off: stop the pod yourself (it is billing)"
fi
