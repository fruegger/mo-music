#!/usr/bin/env bash
# Set up and run a training on a RunPod pod (Linux, a PyTorch template image). Cost-aware by design:
#   - data is streamed from Zenodo onto the pod disk: nothing to upload from home, nothing persistent to pay for
#   - training runs under `timeout`, so it cannot run forever
#   - with AUTO_TERMINATE=1 the pod removes itself GRACE_MIN minutes after the run, so it cannot idle on the meter
#
# Usage on the pod (as root, in /workspace):
#   git clone https://github.com/fruegger/mo-music.git && cd mo-music/trainer
#   bash scripts/runpod_setup.sh data                       # once per pod: install + download data (~110 GB transfer);
#                                                           #   DL_WORKERS=<n> sets the parallel connections (default 12)
#   CONFIG=configs/runpod-smoke.yaml HOURS=0.5 bash scripts/runpod_setup.sh train
#   CONFIG=configs/runpod-vocalmix-5k.yaml HOURS=2 AUTO_TERMINATE=1 GRACE_MIN=30 bash scripts/runpod_setup.sh train
# Results are in the config's out_dir (checkpoint.pt, model.pt, model.onnx, run.json): download them before the pod goes.
# Resume after an interruption: RESUME=/workspace/runs/<name>/checkpoint.pt bash scripts/runpod_setup.sh train
set -euo pipefail

W=/workspace
SLAKH_URL=https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz
MUSDB_URL=https://zenodo.org/records/1117372/files/musdb18.zip

install_code() {
    python -c 'import sys; assert sys.version_info >= (3, 11), f"Python {sys.version.split()[0]}: pick a newer PyTorch template (3.11+)"'
    command -v unzip >/dev/null || { apt-get update -qq && apt-get install -y -qq unzip; }
    pip install --quiet -e "../mixer[effects]" -e .
    python -c 'import torch; print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())'
}

fetch_slakh() {
    # The marker is written only after a complete, checked extraction: folders alone prove nothing,
    # because the tarball is shuffled (every track fills up slowly and is complete only at the very end).
    if [ -f "$W/slakh2100/.complete" ]; then echo "Slakh already present"; return; fi
    mkdir -p "$W/slakh2100"
    # One 104 GB tarball with its files in random order. Zenodo gives only ~2-5 MB/s per connection, so
    # fetch it over several connections at once (DL_WORKERS) and stream it into tar, keeping only train and
    # validation, without the unused mix.flac (the mixer sums the stems itself). The first path component is stripped.
    python "$(dirname "$0")/parallel_stream.py" "$SLAKH_URL" --workers "${DL_WORKERS:-12}"         | tar -xz --no-same-owner -C "$W/slakh2100" --strip-components=1 --wildcards '*/train/*' '*/validation/*' --exclude='*/mix.flac'
    n_train=$(ls "$W/slakh2100/train" | wc -l); n_val=$(ls "$W/slakh2100/validation" | wc -l)
    echo "Slakh: $n_train train, $n_val validation tracks"
    [ "$n_train" -eq 1289 ] && [ "$n_val" -eq 270 ] || { echo "unexpected track counts (want 1289 / 270)"; exit 1; }
    # Every kept track should have its stems now.
    [ "$(find "$W/slakh2100/train" "$W/slakh2100/validation" -name '*.flac' | wc -l)" -gt 10000 ] || { echo "too few stem files"; exit 1; }
    touch "$W/slakh2100/.complete"
}

fetch_musdb() {
    if [ -d "$W/MUSDB18_prepared/train" ]; then echo "MUSDB18_prepared already present"; return; fi
    mkdir -p "$W/MUSDB18"
    curl -L --fail --retry 5 -o "$W/musdb18.zip" "$MUSDB_URL"
    unzip -q -o "$W/musdb18.zip" -d "$W/MUSDB18"
    rm -f "$W/musdb18.zip"
    mixer prepare-musdb --src "$W/MUSDB18" --dst "$W/MUSDB18_prepared"
    rm -rf "$W/MUSDB18"  # only the prepared FLACs are used
}

case "${1:-}" in
data)
    install_code
    df -h "$W" | tail -1
    fetch_musdb
    fetch_slakh
    df -h "$W" | tail -1
    ;;
train)
    : "${CONFIG:?set CONFIG=configs/runpod-*.yaml}"
    : "${HOURS:?set HOURS=<max hours>, a hard cap on the run time}"
    # nproc can show the host's cores, not the pod's share: use the CPUs this process may run on, and cap the
    # workers (each holds prefetched batches in RAM). Override with MAX_WORKERS=<n>.
    cpus=$(python -c 'import os; print(len(os.sched_getaffinity(0)))')
    workers=$(( cpus - 1 ))
    [ "$workers" -gt "${MAX_WORKERS:-12}" ] && workers=${MAX_WORKERS:-12}
    [ "$workers" -lt 1 ] && workers=1
    cfg=/tmp/$(basename "$CONFIG")
    sed "s/^  num_workers: .*/  num_workers: $workers/" "$CONFIG" > "$cfg"
    # SEED=<n> overrides the config's seed and adds -s<n> to its out_dir, so one config can be run for several seeds.
    if [ -n "${SEED:-}" ]; then
        sed -i -e "s/^  seed: .*/  seed: $SEED/" -e 's#^out_dir: \(.*\)$#out_dir: \1'"-s$SEED"'#' "$cfg"
    fi
    echo "nproc=$(nproc), usable CPUs=$cpus, num_workers=$workers, GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo none)"
    resume=()
    [ -n "${RESUME:-}" ] && resume=(--resume "$RESUME")
    # Do not abort on a failed/timed-out run: the termination below must still happen.
    set +e
    timeout "${HOURS}h" python -m trainer.train --config "$cfg" "${resume[@]}" 2>&1 | tee -a /workspace/train.log
    status=${PIPESTATUS[0]}
    set -e
    echo "training exited with status $status (124 = hit the HOURS cap)"
    if [ "${AUTO_TERMINATE:-0}" = "1" ]; then
        echo "removing this pod in ${GRACE_MIN:-30} minutes: download the results now"
        sleep $(( ${GRACE_MIN:-30} * 60 ))
        runpodctl remove pod "$RUNPOD_POD_ID"
    else
        echo "AUTO_TERMINATE is off: terminate the pod yourself when the results are downloaded (it is billing)"
    fi
    ;;
*)
    echo "usage: bash scripts/runpod_setup.sh data | train"; exit 2 ;;
esac
