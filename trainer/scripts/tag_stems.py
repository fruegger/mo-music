"""Runs the AudioSet tagger over MUSDB18 `other` stems that the hand tags say hold piano or organ, and writes what the listening pages need.

    python scripts/tag_stems.py [--out C:/workarea/segment-candidates] [--tags ../labels/musdb18-other-tags.csv] [--songs SUBSTRING ...]
    python scripts/segment_review.py && python scripts/segment_overview.py

Songs: tagged `piano` without `guitar`, or tagged `organ`. Per song it writes into `--out`:

  audio/<split> - <song>.wav   the `other` stem as mono 32 kHz WAV (a WAV seeks reliably in the browser, FLAC did not in long songs)
  scores.json                  per song, the tagger's scores of instrument families in 5 s windows with hop 2.5 s
                               (`keys` = piano, e-piano, organ, Hammond, keyboard, synthesizer, sampler) for `segment_overview.py`
  candidates.csv               per song and tagged instrument, the 4 best non-overlapping 5 s windows (instrument heard,
                               little else) for `segment_review.py`

The tagger is a ranking aid for listening, not a label source: it misses e.g. Leslie e-piano and percussive Hammond organ.
The weights (327 MB) are downloaded to `trainer.audioset_tagger.WEIGHTS_DIR` on first use.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from trainer import audioset_tagger as tagger

FAMILIES = {
    "keys": ["Piano", "Electric piano", "Organ", "Electronic organ", "Hammond organ", "Keyboard (musical)", "Synthesizer", "Sampler"],
    "piano": ["Piano", "Electric piano"],
    "organ": ["Organ", "Electronic organ", "Hammond organ"],
    "synth": ["Synthesizer", "Sampler"],
    "guitar": ["Guitar", "Electric guitar", "Acoustic guitar", "Plucked string instrument"],
    "strings": ["Bowed string instrument", "String section", "Violin, fiddle"],
    "vibes": ["Vibraphone", "Glockenspiel", "Marimba, xylophone", "Chime", "Bell"],
    "brass": ["Brass instrument", "Trumpet", "Trombone"],
    "voice": ["Singing", "Speech", "Choir"],
}
CANDIDATE_FAMILIES = ("guitar", "piano", "organ", "synth", "strings")  # what a candidate window is compared with
WINDOW, HOP, PER_INSTRUMENT = 5.0, 2.5, 4


def select_songs(tags_csv: Path, only: list[str]) -> list[tuple[str, str, str]]:
    rows = csv.DictReader(tags_csv.open(encoding="utf-8-sig", newline=""))
    songs = []
    for r in rows:
        tags = r["instruments"].split("+")
        if ("piano" in tags and "guitar" not in tags) or "organ" in tags:
            if not only or any(s.lower() in r["song"].lower() for s in only):
                songs.append((r["split"], r["song"], r["instruments"]))
    return songs


def candidates(scores: dict[str, np.ndarray], starts: np.ndarray, instrument: str) -> list[int]:
    """Indices of the best non-overlapping windows: the instrument is heard and little else is."""
    others = np.max([scores[f] for f in CANDIDATE_FAMILIES if f != instrument], axis=0)
    alone = scores[instrument] - others
    picked: list[int] = []
    for i in np.argsort(-(alone + 0.25 * scores[instrument])):
        if all(abs(starts[i] - starts[j]) >= WINDOW for j in picked):
            picked.append(int(i))
        if len(picked) == PER_INSTRUMENT:
            break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(r"C:\workarea\segment-candidates"))
    parser.add_argument("--tags", type=Path, default=Path("../labels/musdb18-other-tags.csv"))
    parser.add_argument("--musdb", type=Path, default=Path(r"C:\workarea\MUSDB18_prepared"))
    parser.add_argument("--songs", nargs="*", default=[], help="only songs whose name contains one of these")
    args = parser.parse_args()
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    songs = select_songs(args.tags, args.songs)
    names, model = tagger.labels(), tagger.load()
    columns = {f: [names.index(n) for n in labels] for f, labels in FAMILIES.items()}
    scores_out, rows = [], []
    for split, song, tags in songs:
        audio, rate = sf.read(str(args.musdb / split / song / "other.flac"), dtype="float32", always_2d=True)
        wav = args.out / "audio" / f"{split} - {song}.wav"
        if not wav.exists():
            mono = torchaudio.functional.resample(torch.from_numpy(audio.mean(axis=1)), rate, tagger.RATE).numpy()
            sf.write(str(wav), mono, tagger.RATE, subtype="PCM_16")
        starts, probs = tagger.tag_audio(model, audio, rate, WINDOW, HOP)
        fam = {f: probs[:, ix].max(axis=1) for f, ix in columns.items()}
        scores_out.append({"split": split, "song": song, "tags": tags, "hop": HOP, "win": WINDOW,
                           "scores": {f: [round(float(x), 2) for x in v] for f, v in fam.items()}})
        for instrument in ("piano", "organ"):
            if instrument not in tags.split("+"):
                continue
            for rank, i in enumerate(candidates(fam, starts, instrument), 1):
                others = np.max([fam[f][i] for f in CANDIDATE_FAMILIES if f != instrument])
                rows.append([split, song, tags, instrument, rank, f"{starts[i]:.1f}", f"{starts[i] + WINDOW:.1f}",
                             f"{fam[instrument][i]:.2f}", f"{fam[instrument][i] - others:.2f}"]
                            + [f"{fam[f][i]:.2f}" for f in ("guitar", "piano", "organ", "synth", "strings")] + [""])
        print("done", song, flush=True)
    (args.out / "scores.json").write_text(json.dumps(scores_out, ensure_ascii=False), encoding="utf-8")
    with (args.out / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "song", "tags", "instrument", "rank", "start_s", "end_s", "score", "alone",
                    "guitar", "piano", "organ", "synth", "strings", "verdict"])
        w.writerows(rows)
    print(f"{len(songs)} songs, {len(rows)} candidate windows -> {args.out}")


if __name__ == "__main__":
    main()
