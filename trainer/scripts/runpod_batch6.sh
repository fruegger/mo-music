#!/usr/bin/env bash
# Pod batch 6: the song-count ladder and what breaks the fit (see RUNPOD-BATCH6.md).
#   ladder:  n3 n10 n30 n100 songs, "clean" recipe (no augmentation, equal source weights)
#   add-back on all 100 songs:  n100 with augmentation ("aug"), n100 with the source weights ("w")
# An arm is "<songs>:<family>", family = clean | aug | w (configs/runpod-b6-<family>.yaml). Seed 0 only. Re-run one arm with
#   ARMS="100:aug" SEEDS="1" bash scripts/runpod_batch6.sh
#
# Data: the master copy of MUSDB18_prepared lives on the network volume (/workspace/MUSDB18_prepared, `bash scripts/runpod_setup.sh data-musdb`)
# and is copied ONCE to the pod's local disk (/root/data); the song subsets are symlinks into that copy (/root/data/b6/n<N>).
# Results go to /workspace/runs/b6-* on the volume; after each arm the run is scored with oracle_ceiling.py on its own training songs
# (score-train.txt) and on the 50 test songs (score-test.txt). Run with the Python 3.11 venv active:
#   source /workspace/venv/bin/activate
#   cd /workspace/mo-music && git pull && cd trainer
#   HOURS_PER_RUN=1.5 AUTO_STOP=1 nohup bash scripts/runpod_batch6.sh > /workspace/batch6.log 2>&1 &   then: tail -f /workspace/batch6.log
set -uo pipefail
cd "$(dirname "$0")/.."

MASTER="${MASTER:-/workspace}"
LOCAL="${LOCAL:-/root/data}"
CFG="${CFG:-/root/b6cfg}"
ARMS="${ARMS:-100:clean}"   # step 1 only; the later arms depend on its result (RUNPOD-BATCH6.md, "Run order"): 100:aug 100:w | 3:clean 10:clean 30:clean
SEEDS="${SEEDS:-0}"
FIRST=("Actions - Devil's Words" "BigTroubles - Phantom" "ANiMAL - Clinic A")   # the three songs of the laptop overfit test
SIZES="${SIZES:-3 10 30 100}"

[ -d "$MASTER/MUSDB18_prepared/train" ] || { echo "MUSDB18_prepared is missing on the volume: run 'bash scripts/runpod_setup.sh data-musdb' first"; exit 1; }
python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' || { echo "no GPU on this pod"; exit 1; }
bash scripts/runpod_setup.sh install || { echo "install failed"; exit 1; }

# One local copy of the data (a replaced pod has an empty /root/data, a restarted one keeps it). The marker is written after the copy checks out.
if [ ! -f "$LOCAL/.complete6" ]; then
    mkdir -p "$LOCAL" || { echo "cannot create $LOCAL"; exit 1; }
    free_gb=$(df -Pk "$LOCAL" | awk 'NR==2 {printf "%d", $4 / 1048576}')
    need_gb=$(du -sk "$MASTER/MUSDB18_prepared" | awk '{printf "%d", $1 / 1048576 + 1}')
    echo "local disk: ${free_gb} GB free, data ${need_gb} GB"
    [ "$free_gb" -ge 15 ] && [ "$free_gb" -ge $((need_gb + 8)) ] || { echo "not enough room for a local copy (need 15 GB free and data + 8 GB)"; exit 1; }
    t0=$(date +%s)
    rm -rf "$LOCAL/MUSDB18_prepared" "$LOCAL/b6"
    cp -a "$MASTER/MUSDB18_prepared" "$LOCAL/" || { echo "copy failed"; exit 1; }
    a=$(find "$MASTER/MUSDB18_prepared" -type f | wc -l); b=$(find "$LOCAL/MUSDB18_prepared" -type f | wc -l)
    [ "$a" = "$b" ] || { echo "copy differs: $a files on the volume, $b local"; exit 1; }
    touch "$LOCAL/.complete6"
    echo "copied $(du -sh "$LOCAL" | cut -f1) to $LOCAL in $(( $(date +%s) - t0 )) s"
fi
python scripts/make_song_subsets.py --src "$LOCAL/MUSDB18_prepared" --dst "$LOCAL/b6" --sizes $SIZES --first "${FIRST[@]}" || { echo "subsets failed"; exit 1; }
mkdir -p "$CFG"

for seed in $SEEDS; do
    for arm in $ARMS; do
        n="${arm%%:*}"; fam="${arm##*:}"
        name="n$n-$fam"; out="/workspace/runs/b6-$name"; log="$CFG/$name.log"; cfg="$CFG/$name.yaml"
        [ -f "configs/runpod-b6-$fam.yaml" ] && [ -d "$LOCAL/b6/n$n" ] || { echo "arm $arm: no config or no subset, skipped"; continue; }
        printf 'name | load url | local path | date | license\nMUSDB18-prepared | derived from MUSDB18 (https://zenodo.org/records/1117372) with `mixer prepare-musdb`, subset of %s train songs + all test songs | %s | 10.11.2026 | !! NON COMMERCIAL\n' \
            "$n" "$LOCAL/b6/n$n" > "$log"
        sed -e "s#^sources: .*#sources: $log#" -e "s#^out_dir: .*#out_dir: $out#" "configs/runpod-b6-$fam.yaml" > "$cfg"
        echo "=== $(date +%H:%M) arm $name seed $seed ==="
        CONFIG="$cfg" HOURS="${HOURS_PER_RUN:-1.5}" SEED="$seed" bash scripts/runpod_setup.sh train
        rc=$?
        echo "=== $(date +%H:%M) arm $name seed $seed finished (exit $rc) ==="
        run="$out-s$seed"
        if [ -f "$run/model.pt" ]; then
            for split in train test; do
                python scripts/oracle_ceiling.py --sources "$log" --dataset musdb18-prepared --split "$split" --snippets 240 --seconds 6 \
                    --tags ../labels/musdb18-other-tags.csv --promote-single --runs "$name=$run" 2>&1 | grep -v 'windows$' > "$run/score-$split.txt" \
                    || echo "scoring $split failed for $name"
            done
            echo "scored $name: $(grep -E '^'"$name" "$run/score-train.txt" | head -1) (train) / $(grep -E '^'"$name" "$run/score-test.txt" | head -1) (test)"
        fi
    done
done

(cd /workspace/runs && find b6-* -maxdepth 1 -type f \( -name run.json -o -name model.pt -o -name model.json -o -name 'score-*.txt' \) -print0 \
    | tar --null -T - -czf /workspace/batch6-results.tar.gz) \
    && echo "bundle: $(tar -tzf /workspace/batch6-results.tar.gz | wc -l) files, $(ls -lh /workspace/batch6-results.tar.gz | awk '{print $5}')"

if [ "${AUTO_STOP:-0}" = "1" ]; then
    echo "stopping this pod (the volume is kept)"
    runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"
else
    echo "AUTO_STOP is off: stop the pod yourself (it is billing)"
fi
