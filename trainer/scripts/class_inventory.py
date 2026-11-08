"""How much exact training data exists per instrument class, in Slakh and in real recordings.

    python scripts/class_inventory.py [--slakh C:/workarea/slakh2100] [--musdb C:/workarea/MUSDB18_prepared]
        [--studio C:/workarea/studio_prepared] [--labels ../labels] [--json out.json]

A class that has a lot of Slakh data and almost no real exact data is where Slakh is most likely to matter (and
where nothing real can tell us whether it did). Per mixer class (`mixer.vocab.CLASSES`) the table shows minutes of
*stem audio* (one stem playing for one minute = 1 min; several stems of a class in a track add up):

  slakh      Slakh train split (synthesised, exact)
  studio     your own recordings, train / test (exact, from the stem labels)
  musdb      MUSDB18 train, exact stems only (drums, bass, vocals)
  listening  minutes the listening labels (`labels/stems.csv`: whole stems judged all/mostly; `labels/segments.csv`: solo
             windows) make exact for a class inside MUSDB18 `other` stems
  tagged     number of MUSDB18 songs whose `other` stem was tagged with the class (train / test), and in brackets
             how many of those songs have only that one source family (promotable to an exact stem)

and the ratio slakh / real exact (studio + musdb + listening). MUSDB18's `other` and the classes it holds are never
exact by themselves: only tags and the listening labels say anything about them.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import yaml
from mixer.vocab import CLASSES

from trainer.sources import source_names
from trainer.sources import source_names
from trainer.tags import TAG_SOURCE, load_other_tags, parse_tags, single_source

# Tag word -> mixer class (finer than the six-source mapping: this is only for counting).
TAG_CLASS = {
    "guitar": "Guitar", "banjo": "Guitar", "piano": "Piano", "organ": "Organ", "synth": "Synth Pad", "synth pad": "Synth Pad",
    "synth lead": "Synth Lead", "strings": "Strings", "violin": "Strings", "harp": "Strings", "accordion": "Organ",
    "vibraphone": "Chromatic Percussion", "bells": "Chromatic Percussion", "trumpet": "Brass", "trombone": "Brass",
    "brass": "Brass", "sax": "Reed", "clarinet": "Reed", "oboe": "Reed", "flute": "Pipe", "percussion": "Chromatic Percussion",
}
INSTRUMENT_CLASS = {"piano": "Piano", "organ": "Organ"}  # instruments of the listening pages
STUDIO_CLASS = {"Guitar": "Guitar", "Bass": "Bass", "Drums": "Drums", "Vocals": "Vocals", "Piano": "Piano", "Organ": "Organ"}


def minutes(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate / 60.0


def slakh_minutes(root: Path, split: str = "train") -> tuple[dict[str, float], dict[str, int]]:
    mins, tracks = defaultdict(float), defaultdict(int)
    for track in sorted((root / split).iterdir()):
        meta = track / "metadata.yaml"
        if not meta.exists():
            continue
        stems = yaml.safe_load(meta.read_text(encoding="utf-8"))["stems"]
        length = None
        seen = set()
        for sid, s in stems.items():
            cls = s["inst_class"]
            f = track / "stems" / f"{sid}.flac"
            if not f.exists():
                continue
            length = length or minutes(f)  # all stems of a track have the same length
            mins[cls] += length
            seen.add(cls)
        for cls in seen:
            tracks[cls] += 1
    return dict(mins), dict(tracks)


def studio_minutes(root: Path) -> dict[str, dict[str, float]]:
    out = {"train": defaultdict(float), "test": defaultdict(float)}
    for split in out:
        for song in sorted((root / split).iterdir()):
            meta = json.loads((song / "stems.json").read_text(encoding="utf-8"))
            length = meta["frames"] / meta["sample_rate"] / 60.0
            for st in meta["stems"]:
                cls = STUDIO_CLASS.get(st["label"], st["label"])
                out[split][cls] += length
    return {k: dict(v) for k, v in out.items()}


def musdb_minutes(root: Path, split: str = "train") -> dict[str, float]:
    mins = defaultdict(float)
    for song in sorted((root / split).iterdir()):
        if (song / "drums.flac").exists():
            length = minutes(song / "drums.flac")
            for cls, stem in (("Drums", "drums"), ("Bass", "bass"), ("Vocals", "vocals")):
                mins[cls] += length
    return dict(mins)


def musdb_song_minutes(root: Path) -> dict[tuple[str, str], float]:
    out = {}
    for split in ("train", "test"):
        for song in sorted((root / split).iterdir()):
            if (song / "other.flac").exists():
                out[(split, song.name)] = minutes(song / "other.flac")
    return out


def listening_minutes(labels: Path, song_minutes: dict, test: bool = False) -> dict[str, float]:
    """Exact minutes the listening labels give, train split only unless `test`."""
    mins = defaultdict(float)
    split_ok = (lambda s: True) if test else (lambda s: s == "train")
    with (labels / "stems.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["verdict"] in ("all", "mostly") and split_ok(r["split"]):
                mins[INSTRUMENT_CLASS[r["instrument"]]] += song_minutes.get((r["split"], r["song"]), 0.0)
    with (labels / "segments.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["verdict"] == "solo" and split_ok(r["split"]):
                mins[INSTRUMENT_CLASS[r["instrument"]]] += (float(r["end_s"]) - float(r["start_s"])) / 60.0
    return dict(mins)


def tagged_songs(tags: Path) -> dict[str, dict[str, list[int]]]:
    """`{class: {split-less counts}}`: [train songs, test songs, train alone, test alone] per mixer class. A song is `alone` for a class
    when every tag in it maps to the same six-source family as that class (promotable to an exact stem)."""
    out = defaultdict(lambda: [0, 0, 0, 0])
    with tags.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            words = parse_tags(r["instruments"])
            if not words:
                continue
            split = 0 if r["split"] == "train" else 1
            families = {TAG_SOURCE.get(w) for w in words} - {None}
            for cls in {TAG_CLASS[w] for w in words if w in TAG_CLASS}:
                out[cls][split] += 1
                fam = {TAG_SOURCE[w] for w in words if TAG_CLASS.get(w) == cls}
                if families == fam:
                    out[cls][2 + split] += 1
    return dict(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slakh", type=Path, default=Path(r"C:\workarea\slakh2100"))
    parser.add_argument("--musdb", type=Path, default=Path(r"C:\workarea\MUSDB18_prepared"))
    parser.add_argument("--studio", type=Path, default=Path(r"C:\workarea\studio_prepared_swap"), help="the studio split of the recipe (default: the swapped one)")
    parser.add_argument("--labels", type=Path, default=Path("../labels"))
    parser.add_argument("--json", type=Path, help="also write the numbers as JSON")
    args = parser.parse_args()

    slakh, tracks = slakh_minutes(args.slakh)
    studio = studio_minutes(args.studio)
    musdb = musdb_minutes(args.musdb)
    song_min = musdb_song_minutes(args.musdb)
    listening = listening_minutes(args.labels, song_min)
    tagged = tagged_songs(args.labels / "musdb18-other-tags.csv")

    print(f"{'class':22s} {'slakh min':>9s} {'(tracks)':>9s} {'studio tr/te':>13s} {'musdb':>7s} {'listening':>10s} {'tagged tr/te (alone)':>22s} {'slakh/real':>11s}")
    rows = {}
    for cls in CLASSES:
        s = slakh.get(cls, 0.0)
        st = studio["train"].get(cls, 0.0), studio["test"].get(cls, 0.0)
        mu = musdb.get(cls, 0.0)
        li = listening.get(cls, 0.0)
        tg = tagged.get(cls, [0, 0, 0, 0])
        real = st[0] + mu + li
        ratio = f"{s / real:10.0f}x" if real > 0 else "no real"
        rows[cls] = {"slakh_min": s, "slakh_tracks": tracks.get(cls, 0), "studio_train_min": st[0], "studio_test_min": st[1],
                     "musdb_min": mu, "listening_min": li, "tagged": tg, "real_exact_min": real}
        tagtxt = f"{tg[0]}/{tg[1]} ({tg[2]}/{tg[3]})" if any(tg) else "-"
        print(f"{cls:22s} {s:9.0f} {tracks.get(cls, 0):9d} {st[0]:6.1f}/{st[1]:<5.1f} {mu:7.0f} {li:10.1f} {tagtxt:>22s} {ratio:>11s}")
    names = source_names("musdb6")
    promoted = defaultdict(lambda: [0.0, 0.0, 0])  # family -> [train min, test min, train songs]
    for song, state in load_other_tags(args.labels / "musdb18-other-tags.csv").items():
        k = single_source(state)
        for split in ("train", "test"):
            if k is not None and (split, song) in song_min:
                promoted[names[k]][0 if split == "train" else 1] += song_min[(split, song)]
                promoted[names[k]][2] += split == "train"
    print("\nMUSDB18 `other` stems that the tags make exact for one whole family (promote_single): minutes train / test (train songs)")
    for fam in ("guitar", "keys", "other"):
        a = promoted[fam]
        print(f"  {fam:6s} {a[0]:7.0f} / {a[1]:<6.0f} ({a[2]} songs)")
    rows["_promoted_family_min"] = dict(promoted)
    print("\nminutes of stem audio; 'no real' = nothing exact in real recordings. Classes with no real exact data can only be judged by ear.")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
