#!/usr/bin/env bash
# Pod batch 3: Slakh2100 at two small weights added to batch 2's control recipe, two seeds each (see RUNPOD-BATCH3.md).
#   arm e-slakh10 : Slakh weight 0.1     arm f-slakh30 : Slakh weight 0.3
# The control (arm D of batch 2) is not rerun. Order: seed 0 of both arms first.
#
# On a pod whose /workspace holds the data (network volume): after the one-time data setup
#   cd /workspace/mo-music && git pull && cd trainer
#   HOURS_PER_RUN=2 AUTO_STOP=1 bash scripts/runpod_batch3.sh 2>&1 | tee /workspace/batch3.log
set -uo pipefail
cd "$(dirname "$0")/.."

[ -f /workspace/slakh2100/.complete ] || { echo "Slakh is not complete: run 'bash scripts/runpod_setup.sh data' first (one-time, about 1.5 h)"; exit 1; }
[ -d /workspace/MUSDB18_prepared/train ] || { echo "MUSDB18_prepared is missing: run 'bash scripts/runpod_setup.sh data' first"; exit 1; }
S=/workspace/studio_prepared
[ -d "$S/train/Shimmering Star" ] && [ -d "$S/test/In My Belief" ] || { echo "studio split is not the swapped one (batch 2 does the swap):"; ls "$S/train" "$S/test"; exit 1; }
python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' || { echo "no GPU on this pod"; exit 1; }
bash scripts/runpod_setup.sh install || { echo "install failed"; exit 1; }

for seed in 0 1; do
    for arm in e-slakh10 f-slakh30; do
        echo "=== $(date +%H:%M) arm $arm seed $seed ==="
        CONFIG="configs/runpod-b3-$arm.yaml" HOURS="${HOURS_PER_RUN:-2}" SEED="$seed" bash scripts/runpod_setup.sh train
        echo "=== $(date +%H:%M) arm $arm seed $seed finished (exit $?) ==="
    done
done

(cd /workspace/runs && find b3-* -maxdepth 1 -type f \( -name run.json -o -name model.pt -o -name model.json \) -print0 \
    | tar --null -T - -czf /workspace/batch3-results.tar.gz) \
    && echo "bundle: $(tar -tzf /workspace/batch3-results.tar.gz | wc -l) files, $(ls -lh /workspace/batch3-results.tar.gz | awk '{print $5}')"

if [ "${AUTO_STOP:-0}" = "1" ]; then
    echo "stopping this pod (the volume is kept)"
    runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"
else
    echo "AUTO_STOP is off: stop the pod yourself (it is billing)"
fi
