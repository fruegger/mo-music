#!/bin/sh
# Bash/POSIX-shell equivalent of tests.bat: runs song_eval.py against the
# repertoire folder (default below, or pass a different path as $1).
set -e
LIBRARY_DIR="${1:-/c/Users/frueg/Desktop/music_studio/1_preprocessing/FJ/repertoire}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPT_DIR/song_eval.py" --library-dir "$LIBRARY_DIR"
