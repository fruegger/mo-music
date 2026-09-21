#!/bin/sh
# Bash/POSIX-shell equivalent of tests.bat: runs song_eval.py against the
# repertoire folder (default below, or pass a different path as $1), and
# the operation named by $2 (or omit it to get the interactive menu).
set -e
LIBRARY_DIR="${1:-/c/Users/frueg/Desktop/music_studio/1_preprocessing/FJ/repertoire}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -n "$2" ]; then
    python "$SCRIPT_DIR/song_eval.py" --library-dir "$LIBRARY_DIR" --op "$2"
else
    python "$SCRIPT_DIR/song_eval.py" --library-dir "$LIBRARY_DIR"
fi
