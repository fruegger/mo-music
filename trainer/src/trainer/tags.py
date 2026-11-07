"""Hand-made instrument tags for MUSDB18's `other` stem (`labels/musdb18-other-tags.csv`).

MUSDB18 hides guitar, keys and everything else in one `other` stem, so a six-source model only knows the sum of those
three. The tags say, per song, which instruments were heard in that stem. Mapped to the `musdb6` sources they tell a
presence model that a song has no keys at all (reliable) or that it has some (a weak label: the instrument may be
silent in a given snippet).

Per song and per group source (guitar, keys, other) the state is `ABSENT` (nothing tagged for it), `PRESENT` (tagged
in `instruments`) or `MAYBE` (only in `unsure`). A song with no tags at all (an empty `instruments` cell) is left
out: nothing is known about it.
"""

import csv
from pathlib import Path

from .sources import GROUPS, source_names

ABSENT, PRESENT, MAYBE, UNKNOWN = 0.0, 1.0, 0.5, -1.0

# Tag word -> musdb6 source. `None` = not an `other` instrument (ignored). The open cases from labels/README.md are
# decided here: banjo -> guitar (plucked strings), harp and accordion -> keys (strings, organ family), percussion -> other.
TAG_SOURCE = {
    "guitar": "guitar", "banjo": "guitar",
    "piano": "keys", "organ": "keys", "synth": "keys", "synth pad": "keys", "synth lead": "keys", "strings": "keys",
    "violin": "keys", "harp": "keys", "accordion": "keys", "vibraphone": "keys", "bells": "keys",
    "trumpet": "other", "trombone": "other", "brass": "other", "sax": "other", "clarinet": "other", "oboe": "other",
    "flute": "other", "percussion": "other",
    "vocals": None, "drums": None, "bass": None,
}


def single_source(state: list[float], source_set: str = "musdb6") -> int | None:
    """The index of the one group source the song's `other` stem consists of: exactly one is `PRESENT` and the others
    are `ABSENT` (a `MAYBE` anywhere means we do not know). `None` otherwise, also for a song with no tags. A stem with a
    single source is that source, so its target is known exactly instead of only as part of a sum."""
    group = GROUPS[source_set]
    present = [i for i in group if state[i] == PRESENT]
    return present[0] if len(present) == 1 and all(state[i] in (PRESENT, ABSENT) for i in group) else None


def parse_tags(cell: str) -> list[str]:
    return [t.strip().lower() for t in cell.split("+") if t.strip()]


def load_other_tags(path: Path, source_set: str = "musdb6") -> dict[str, list[float]]:
    """`{song: [state of every source of the set]}`: `UNKNOWN` (-1) for the sources that are not in the group, and
    `ABSENT` / `PRESENT` / `MAYBE` for the group sources. An unmapped tag word is an error (a typo would otherwise
    silently turn a present instrument into an absent one)."""
    names, group = source_names(source_set), GROUPS[source_set]
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8-sig", newline="")))
    unknown = sorted({t for r in rows for col in ("instruments", "unsure") for t in parse_tags(r[col]) if t not in TAG_SOURCE})
    if unknown:
        raise ValueError(f"{path}: tags with no source: {', '.join(unknown)} (add them to trainer.tags.TAG_SOURCE)")
    tags = {}
    for r in rows:
        present = {TAG_SOURCE[t] for t in parse_tags(r["instruments"])} - {None}
        maybe = {TAG_SOURCE[t] for t in parse_tags(r["unsure"])} - {None}
        if not parse_tags(r["instruments"]):
            continue  # nothing was tagged for this song: unknown, not "empty"
        state = [UNKNOWN] * len(names)
        for i in group:
            state[i] = PRESENT if names[i] in present else MAYBE if names[i] in maybe else ABSENT
        tags[r["song"]] = state
    return tags
