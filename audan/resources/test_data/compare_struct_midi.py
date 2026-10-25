"""Score `audan struct` runs against the section markers of the MIDI references.

The MIDI markers (compare_beats_midi.reference) give the reference
sections. Each run is the `audan struct --format lab` output ("start end
label" per line) saved to structure/song<N>.txt; the MIDI file's own beat
grid is not used, so a constant audio/MIDI offset of a fraction of a second
counts against boundary hits only at the tight tolerance.

Scores (as in mir_eval.segment):
  * boundary F-measure at 0.5 s and 3 s (first and last boundary excluded)
  * pairwise F-measure: frame pairs that share a label in the estimate vs.
    in the reference. Reference labels are the marker names with digits and
    parentheses removed, so "Verse 1" and "Verse 2" are one label.

Standard library only:

    python compare_struct_midi.py                          # every song with a run file
    python compare_struct_midi.py --library-dir ../songs   # also run the missing ones
"""

import argparse
import html
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

from compare_beats_midi import HERE, PAGE, clock, f_measure, load_entries, mark_extremes, reference

STRUCT_DIR = HERE / "structure"
FRAME = 0.1  # seconds per frame for the pairwise score
MIN_SECTION = 1.0  # reference sections shorter than this are marker slivers


def normalise(name):
    name = re.sub(r"\(.*?\)", "", name.lower())
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", name)).strip()


def reference_sections(midi):
    ref = reference(midi)
    return [(s, e, normalise(n), n) for n, s, e in ref["sections"] if e - s >= MIN_SECTION], ref["end"]


def read_lab(path):
    rows = [line.split() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(float(a), float(b), label) for a, b, label in rows]


def boundaries(sections):
    return [s for s, _, *_ in sections[1:]]


def pairs(n):
    return n * (n - 1) // 2


def pairwise_f(ref, est, end):
    """Pairwise F-measure over FRAME-sized frames, from the label
    contingency table (no O(n^2) pair enumeration)."""
    def label_at(sections, t):
        for s, e, label, *_ in sections:
            if s <= t < e:
                return label
        return None

    frames = [i * FRAME for i in range(int(end / FRAME))]
    joint = Counter((label_at(ref, t), label_at(est, t)) for t in frames)
    ref_n, est_n = Counter(), Counter()
    for (r, e), n in joint.items():
        ref_n[r] += n
        est_n[e] += n
    both = sum(pairs(n) for n in joint.values())
    p = both / sum(pairs(n) for n in est_n.values()) if any(est_n.values()) else 0.0
    r = both / sum(pairs(n) for n in ref_n.values()) if any(ref_n.values()) else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def score(song):
    ref, end = reference_sections(song["midi"])
    est = [(s, e, label) for s, e, label in read_lab(song["run"])]
    rb, eb = boundaries(ref), boundaries(est)
    return {
        "ref": ref,
        "est": est,
        "end": end,
        "b05": f_measure(rb, eb, 0.5),
        "b3": f_measure(rb, eb, 3.0),
        "pair": pairwise_f(ref, est, end),
        "ratio": len(est) / len(ref),
    }


def generate_missing(song, library_dir, exe):
    if song["run"].exists():
        return
    print(f"  running {song['run'].name}...", file=sys.stderr, flush=True)
    cmd = [str(exe), "struct", str(library_dir / song["audio"]), "--format", "lab"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=900)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.strip()[:200]}", file=sys.stderr)
        return
    STRUCT_DIR.mkdir(exist_ok=True)
    song["run"].write_text(result.stdout, encoding="utf-8")


def strip(sections, end, est):
    """Timeline strip: one coloured box per section, width = duration."""
    palette = [f"var(--s{i})" for i in range(1, 7)]
    labels = {}
    out = []
    for sec in sections:
        s, e, label = sec[0], sec[1], sec[2]
        colour = palette[labels.setdefault(label, len(labels) % len(palette))] if not est else "var(--muted)"
        title = html.escape(f"{clock(s)} {sec[3] if len(sec) > 3 else label}")
        text = "" if est else html.escape(sec[3] if (e - s) / end > 0.06 else "")
        out.append(
            f'<i style="left:{s / end * 100:.2f}%;width:{(e - s) / end * 100:.2f}%;background:{colour}" '
            f'title="{title}">{text}</i>'
        )
    return f'<div class="strip{" est" if est else ""}">{"".join(out)}</div>'


def page(results):
    style = PAGE.split("<style>")[1].split("</style>")[0]
    css = """
        .strip { position:relative; height:22px; margin:2px 0; }
        .strip i { position:absolute; top:0; bottom:0; font-style:normal; font-size:.62rem; color:#fff;
                   overflow:hidden; white-space:nowrap; padding:3px 2px; border-right:1px solid var(--panel); }
        .strip.est i { background:var(--muted); opacity:.55; }
        .tl { min-width:520px; width:60%; }
        a { color:inherit; }"""
    rows, cols = [], {k: [] for k in ("b05", "b3", "pair")}
    for song, r in results:
        for k in cols:
            cols[k].append(r[k])
        rows.append(
            f'<tr><td class="nr">{song["nr"]}</td><td class="song">{html.escape(song["title"])}</td>'
            f'<td class="num">{len(r["ref"])}</td><td class="num">{len(r["est"])}</td>'
            f'<td class="num sep"><span class="v">{r["b05"]:.2f}</span></td>'
            f'<td class="num"><span class="v">{r["b3"]:.2f}</span></td>'
            f'<td class="num"><span class="v">{r["pair"]:.2f}</span></td>'
            f'<td class="tl sep">{strip(r["ref"], r["end"], False)}{strip(r["est"], r["end"], True)}</td></tr>'
        )
    m = {k: statistics.mean(v) for k, v in cols.items()}
    rows.append(
        '<tr><td></td><td class="song">Mean</td><td></td><td></td>'
        f'<td class="num sep"><b>{m["b05"]:.2f}</b></td><td class="num"><b>{m["b3"]:.2f}</b></td>'
        f'<td class="num"><b>{m["pair"]:.2f}</b></td><td></td></tr>'
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Structure vs. MIDI</title><style>{style}{css}</style></head>"
        '<body><div class="wrap"><header><h1>Structure analysis against the MIDI markers</h1>'
        f"<p>{len(results)} songs. Top strip of each timeline: MIDI sections coloured by label (hover for the "
        "marker name); lower grey strip: sections found by <code>audan struct</code>, which are unlabeled "
        "(A, B, C&hellip;) and so are only compared by boundary position and by which frames share a label.</p>"
        "</header>"
        '<div class="scroll"><table><thead><tr><th>#</th><th>Song</th><th class="num">MIDI</th>'
        '<th class="num">audan</th><th class="num sep">Bound. F 0.5&nbsp;s</th><th class="num">Bound. F 3&nbsp;s</th>'
        '<th class="num">Pairwise F</th><th class="sep">Timeline</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        "<footer><p><b>Boundary F</b>: detected section starts matched one-to-one to marker starts within the "
        "tolerance (first and last boundary excluded). <b>Pairwise F</b>: over 0.1&nbsp;s frames, pairs sharing a "
        "label in the estimate vs. in the MIDI, labels being the marker names without numbers and brackets. "
        "Marker slivers under 1&nbsp;s are ignored.</p></footer></div></body></html>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--song", type=int, action="append", help="only this songs.txt # (repeatable)")
    parser.add_argument("--library-dir", type=Path, help="audio folder; runs `audan struct` for runs missing in structure/")
    parser.add_argument("--exe", type=Path, default=HERE.parent.parent / "target" / "release" / "audan.exe")
    args = parser.parse_args()

    results = []
    for song in load_entries():
        if args.song and song["nr"] not in args.song:
            continue
        song["run"] = STRUCT_DIR / f"song{song['nr']}.txt"
        print(f"#{song['nr']} {song['title']}", file=sys.stderr)
        if args.library_dir:
            generate_missing(song, args.library_dir, args.exe)
        if not song["run"].exists():
            print("  no run in structure/, skipped", file=sys.stderr)
            continue
        try:
            r = score(song)
        except (OSError, ValueError, statistics.StatisticsError) as e:
            print(f"  FAILED: {e!r}", file=sys.stderr)
            continue
        results.append((song, r))
        print(f"  sections {len(r['ref'])} vs {len(r['est'])}  boundary F {r['b05']:.2f} @0.5s, "
              f"{r['b3']:.2f} @3s  pairwise F {r['pair']:.2f}")
    if results:
        out = HERE / "comparison_struct_midi.html"
        out.write_text(page(results), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
