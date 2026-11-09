#!/usr/bin/env bash
# Pod batch 4: Slakh vocal-loss follow-up, two arms, two seeds each (see RUNPOD-BATCH4.md).
#   arm g-vocal-always : Slakh 0.1, vocal_mix p 1.0     arm h-slakh03 : Slakh 0.03, vocal_mix p 0.7
# D (batch 2) and E (batch 3) are not rerun. Order: seed 0 of both arms first.
#
# On a pod whose /workspace holds the data (network volume): after the one-time data setup
#   cd /workspace/mo-music && git pull && cd trainer
#   HOURS_PER_RUN=2 AUTO_STOP=1 bash scripts/runpod_batch4.sh 2>&1 | tee /workspace/batch4.log
set -uo pipefail
cd "$(dirname "$0")/.."

[ -f /workspace/slakh2100/.complete ] || { echo "Slakh is not complete: run 'bash scripts/runpod_setup.sh data' first (one-time, about 1.5 h)"; exit 1; }
[ -d /workspace/MUSDB18_prepared/train ] || { echo "MUSDB18_prepared is missing: run 'bash scripts/runpod_setup.sh data' first"; exit 1; }
S=/workspace/studio_prepared
[ -d "$S/train/Shimmering Star" ] && [ -d "$S/test/In My Belief" ] || { echo "studio split is not the swapped one (batch 2 does the swap):"; ls "$S/train" "$S/test"; exit 1; }
python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' || { echo "no GPU on this pod"; exit 1; }
bash scripts/runpod_setup.sh install || { echo "install failed"; exit 1; }
# One scan of Slakh for all runs and workers (the index file lives on the volume; before this every loader worker rescanned it over the network)
python -c 'from pathlib import Path; from mixer.sources import slakh_index; [print(f"Slakh {s}: {len(slakh_index(Path(\"/workspace/slakh2100\"), s))} tracks indexed") for s in ("train", "validation")]' || { echo "Slakh index failed"; exit 1; }

for seed in 0 1; do
    for arm in g-vocal-always h-slakh03; do
        echo "=== $(date +%H:%M) arm $arm seed $seed ==="
        CONFIG="configs/runpod-b4-$arm.yaml" HOURS="${HOURS_PER_RUN:-2}" SEED="$seed" bash scripts/runpod_setup.sh train
        echo "=== $(date +%H:%M) arm $arm seed $seed finished (exit $?) ==="
    done
done

(cd /workspace/runs && find b4-* -maxdepth 1 -type f \( -name run.json -o -name model.pt -o -name model.json \) -print0 \
    | tar --null -T - -czf /workspace/batch4-results.tar.gz) \
    && echo "bundle: $(tar -tzf /workspace/batch4-results.tar.gz | wc -l) files, $(ls -lh /workspace/batch4-results.tar.gz | awk '{print $5}')"

if [ "${AUTO_STOP:-0}" = "1" ]; then
    echo "stopping this pod (the volume is kept)"
    runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"
else
    echo "AUTO_STOP is off: stop the pod yourself (it is billing)"
fi
