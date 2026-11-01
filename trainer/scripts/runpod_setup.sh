#!/usr/bin/env bash
# Set up and run a training on a RunPod pod (Linux, a PyTorch template image). Cost-aware by design:
#   - data is streamed from Zenodo onto the pod disk: nothing to upload from home, nothing persistent to pay for
#   - training runs under `timeout`, so it cannot run forever
#   - with AUTO_TERMINATE=1 the pod removes itself GRACE_MIN minutes after the run, so it cannot idle on the meter
#
# Usage on the pod (as root, in /workspace):
#   git clone https://github.com/fruegger/mo-music.git && cd mo-music/trainer
#   bash scripts/runpod_setup.sh data                       # once per pod: install + download data (~110 GB transfer)
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
    if [ -d "$W/slakh2100/train" ] && [ -d "$W/slakh2100/validation" ]; then echo "Slakh already present"; return; fi
    mkdir -p "$W/slakh2100"
    # One 104 GB tarball: stream it and keep only train + validation, without the unused mix.flac
    # (the mixer sums the stems itself). The first path component is stripped.
    curl -L --fail --retry 5 "$SLAKH_URL" \
        | tar -xz -C "$W/slakh2100" --strip-components=1 --wildcards '*/train/*' '*/validation/*' --exclude='*/mix.flac'
    test -d "$W/slakh2100/train" || { echo "Slakh layout is not as expected:"; ls "$W/slakh2100"; exit 1; }
    echo "Slakh: $(ls "$W/slakh2100/train" | wc -l) train, $(ls "$W/slakh2100/validation" | wc -l) validation tracks"
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
    workers=$(( $(nproc) - 1 ))
    cfg=/tmp/$(basename "$CONFIG")
    sed "s/^  num_workers: .*/  num_workers: $workers/" "$CONFIG" > "$cfg"
    echo "$(nproc) vCPUs, num_workers=$workers, GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo none)"
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
