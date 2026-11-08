"""Hand labels from the listening pages (`scripts/segment_review.py`, `scripts/segment_overview.py`) for MUSDB18 `other` stems.

* `labels/stems.csv`: whole-stem judgements. `all` = the stem is that instrument throughout, `mostly` = with short other parts
  (the `interruptions` column lists them as `start-end:family;...` in seconds). Other verdicts are not used here.
* `labels/segments.csv`: windows judged `solo` (only that instrument is heard, start and end in seconds).

An instrument maps to a source of the source set; MUSDB18's unlabelled `other` stem that is one source over a stretch has
exact targets there (`SnippetDataset(exact_stems=..., solo_windows=...)`), like the single-source promotion of `trainer.tags`.
"""

import csv
from pathlib import Path

from .sources import source_names

# instrument word -> source name, per source set (a seven-source set with a piano source adds its own entry)
INSTRUMENT_SOURCE = {
    "musdb6": {"piano": "keys", "organ": "keys"},
}


def _source(instrument: str, source_set: str) -> int | None:
    name = INSTRUMENT_SOURCE.get(source_set, {}).get(instrument.strip().lower())
    return None if name is None else source_names(source_set).index(name)


def _rows(path: Path) -> list[dict]:
    return list(csv.DictReader(Path(path).open(encoding="utf-8-sig", newline="")))


def parse_interruptions(cell: str) -> list[tuple[float, float]]:
    """`"41.3-48.8:guitar;90.0-95.0:?"` -> `[(41.3, 48.8), (90.0, 95.0)]`."""
    out = []
    for part in cell.split(";"):
        span = part.split(":")[0].strip()
        if span:
            a, b = span.split("-")
            out.append((float(a), float(b)))
    return out


def load_stem_labels(path: Path, source_set: str = "musdb6") -> dict[str, tuple[int, list[tuple[float, float]]]]:
    """`{song: (source index, [stretches that are NOT that source])}` for the stems judged `all` (no stretches) or `mostly`."""
    out: dict[str, tuple[int, list[tuple[float, float]]]] = {}
    for r in _rows(path):
        source = _source(r["instrument"], source_set)
        if source is None or r["verdict"] not in ("all", "mostly"):
            continue
        gaps = [] if r["verdict"] == "all" else parse_interruptions(r.get("interruptions", ""))
        if r["song"] in out and out[r["song"]][0] != source:
            raise ValueError(f"{r['song']}: judged as two different sources")
        out[r["song"]] = (source, gaps)
    return out


def load_solo_windows(path: Path, source_set: str = "musdb6") -> dict[str, list[tuple[float, float, int]]]:
    """`{song: [(start s, end s, source index)]}` for the windows judged `solo`."""
    out: dict[str, list[tuple[float, float, int]]] = {}
    for r in _rows(path):
        source = _source(r["instrument"], source_set)
        if source is not None and r["verdict"] == "solo" and float(r["end_s"]) > float(r["start_s"]):
            out.setdefault(r["song"], []).append((float(r["start_s"]), float(r["end_s"]), source))
    return out


def exact_source(span: tuple[float, float], stem: tuple[int, list[tuple[float, float]]] | None,
                 windows: list[tuple[float, float, int]] | None) -> int | None:
    """The one source the unlabelled stem is over `span` (seconds, source time), or None if that is not known: the
    whole-stem judgement holds when the span touches none of its interruptions; else a `solo` window that contains the span."""
    if stem is not None and all(span[1] <= a or span[0] >= b for a, b in stem[1]):
        return stem[0]
    for a, b, source in windows or []:
        if span[0] >= a - 1e-6 and span[1] <= b + 1e-6:
            return source
    return None
