"""Score `audan beats` runs against a hand-made MIDI reference.

The MIDI file's tempo map and time signatures give the reference beat grid
(one beat per time-signature denominator note, a downbeat at every bar
start), its markers give the song sections, and its note-ons are kept for
the timeline. Each run is the JSON `audan beats --format json` printed,
saved to a file; `--quick` runs cover only a window starting
QUICK_SKIP_SECONDS into the track and are scored on that window alone.

Standard library only:

    python compare_beats_midi.py            # writes comparison_beats_song4.html
"""

import html
import json
import math
import statistics
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Mirrors audan-cli's pipeline::QUICK_SKIP_SECONDS / QUICK_WINDOW_SECONDS.
QUICK_SKIP_SECONDS = 30.0
QUICK_WINDOW_SECONDS = 30.0

# mir_eval's defaults: +-70 ms for the F-measure, 17.5% of the inter-beat
# interval for the continuity (CMLt/AMLt) scores.
F_TOLERANCE = 0.070
CONTINUITY_TOLERANCE = 0.175

SONG = {
    "title": "In My Place",
    "artist": "Coldplay",
    "midi": HERE.parent / "songs" / "In My Place" / "In My Place.mid",
    "out": HERE / "comparison_beats_song4.html",
    "runs": [
        ("DSP", HERE / "beats" / "song4.txt", False),
        ("DSP quick", HERE / "beats" / "song4_q.txt", True),
        ("Model", HERE / "beats" / "song4_bt.txt", False),
        ("Model quick", HERE / "beats" / "song4_bt_q.txt", True),
        ("Small model", HERE / "beats" / "song4_bts.txt", False),
        ("Small quick", HERE / "beats" / "song4_bts_q.txt", True),
    ],
}


# --- MIDI -----------------------------------------------------------------


def read_varlen(data, pos):
    value = 0
    while True:
        b = data[pos]
        pos += 1
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            return value, pos


def read_midi(path):
    """Returns (ticks_per_quarter, [events]) with events as
    (tick, kind, payload) from all tracks, sorted by tick."""
    data = Path(path).read_bytes()
    pos, division, events = 0, None, []
    while pos + 8 <= len(data):
        cid, length = data[pos : pos + 4], struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + length]
        pos += 8 + length
        if cid == b"MThd":
            division = struct.unpack(">H", body[4:6])[0]
            if division & 0x8000:
                sys.exit("SMPTE-timed MIDI files are not supported")
        elif cid == b"MTrk":
            events.extend(read_track(body))
    events.sort(key=lambda e: e[0])
    return division, events


def read_track(body):
    pos, tick, status, out = 0, 0, None, []
    while pos < len(body):
        delta, pos = read_varlen(body, pos)
        tick += delta
        b = body[pos]
        if b == 0xFF:
            kind = body[pos + 1]
            length, pos = read_varlen(body, pos + 2)
            payload = body[pos : pos + length]
            pos += length
            if kind == 0x51:
                out.append((tick, "tempo", int.from_bytes(payload, "big")))
            elif kind == 0x58:
                out.append((tick, "timesig", (payload[0], 2 ** payload[1])))
            elif kind == 0x06:
                out.append((tick, "marker", payload.decode("latin-1")))
            continue
        if b in (0xF0, 0xF7):
            length, pos = read_varlen(body, pos + 1)
            pos += length
            continue
        if b & 0x80:
            status, pos = b, pos + 1
        n_data = 1 if status & 0xF0 in (0xC0, 0xD0) else 2
        d = body[pos : pos + n_data]
        pos += n_data
        if status & 0xF0 == 0x90 and d[1] > 0:
            out.append((tick, "note", d[0]))
    return out


class TempoMap:
    def __init__(self, division, events):
        self.division = division
        # (tick, seconds at that tick, microseconds per quarter from there)
        self.segments = [(0, 0.0, 500_000)]
        for tick, kind, usec in events:
            if kind != "tempo":
                continue
            t0, s0, u0 = self.segments[-1]
            secs = s0 + (tick - t0) * u0 / 1e6 / division
            if tick == t0:
                self.segments[-1] = (tick, s0, usec)
            else:
                self.segments.append((tick, secs, usec))

    def seconds(self, tick):
        seg = self.segments[0]
        for s in self.segments:
            if s[0] > tick:
                break
            seg = s
        t0, s0, usec = seg
        return s0 + (tick - t0) * usec / 1e6 / self.division

    def bpm_at(self, tick):
        usec = self.segments[0][2]
        for t0, _, u in self.segments:
            if t0 > tick:
                break
            usec = u
        return 60e6 / usec


def reference(path):
    division, events = read_midi(path)
    tempo = TempoMap(division, events)
    notes = [t for t, k, _ in events if k == "note"]
    last_tick = max(notes)
    sigs = [(t, p) for t, k, p in events if k == "timesig"] or [(0, (4, 4))]

    beats, downbeats, beat_ticks = [], [], []
    tick, sig_i, beat_in_bar = 0, 0, 0
    num, den = sigs[0][1]
    while tick <= last_tick:
        while sig_i + 1 < len(sigs) and sigs[sig_i + 1][0] <= tick:
            sig_i += 1
            num, den = sigs[sig_i][1]
            beat_in_bar = 0
        if beat_in_bar == 0:
            downbeats.append(len(beats))
        beats.append(tempo.seconds(tick))
        beat_ticks.append(tick)
        beat_in_bar = (beat_in_bar + 1) % num
        tick += division * 4 // den

    markers = [(tempo.seconds(t), name) for t, k, name in events if k == "marker"]
    if not markers or markers[0][0] > 0:
        markers.insert(0, (0.0, "Intro"))
    end = beats[-1] + (beats[-1] - beats[-2])
    sections = [
        (name, start, markers[i + 1][0] if i + 1 < len(markers) else end)
        for i, (start, name) in enumerate(markers)
    ]
    return {
        "beats": beats,
        "downbeats": downbeats,
        "beats_per_bar": sigs[0][1][0],
        "bpm": [tempo.bpm_at(t) for t in beat_ticks],
        "notes": [tempo.seconds(t) for t in notes],
        "sections": sections,
        "end": end,
    }


# --- scoring --------------------------------------------------------------


def match_count(ref, est, tol):
    """One-to-one matches within `tol` (greedy over sorted lists; for beat
    sequences much sparser than 2*tol this equals the optimal matching)."""
    i = j = hits = 0
    while i < len(ref) and j < len(est):
        d = est[j] - ref[i]
        if abs(d) <= tol:
            hits, i, j = hits + 1, i + 1, j + 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return hits


def f_measure(ref, est, tol=F_TOLERANCE):
    if not ref or not est:
        return 0.0
    hits = match_count(ref, est, tol)
    return 2 * hits / (len(ref) + len(est))


def nearest_errors(ref, est):
    """Signed error (est - ref) of the nearest estimated beat for each
    reference beat, or None when nothing lies within half a beat."""
    out, j = [], 0
    for i, r in enumerate(ref):
        half = (ref[min(i + 1, len(ref) - 1)] - ref[max(i - 1, 0)]) / 4
        while j + 1 < len(est) and abs(est[j + 1] - r) <= abs(est[j] - r):
            j += 1
        d = est[j] - r if est else None
        out.append(d if d is not None and abs(d) <= half else None)
    return out


def continuity(ref, est):
    """mir_eval's CMLt: share of reference beats that are tracked
    continuously -- matched within 17.5% of the local inter-beat interval,
    with the previous beat matched too and the intervals agreeing."""
    if len(ref) < 2 or len(est) < 2:
        return 0.0
    ok = 0
    prev_match = None
    for i in range(len(ref)):
        ibi = ref[i + 1] - ref[i] if i + 1 < len(ref) else ref[i] - ref[i - 1]
        tol = CONTINUITY_TOLERANCE * ibi
        j = min(range(len(est)), key=lambda k: abs(est[k] - ref[i]))
        match = j if abs(est[j] - ref[i]) <= tol else None
        good = (
            match is not None
            and prev_match is not None
            and match == prev_match + 1
            and abs((est[match] - est[prev_match]) - (ref[i] - ref[i - 1])) <= tol
        )
        ok += good
        prev_match = match
    return ok / len(ref)


def metrical_variants(ref):
    """The reference at the other metrical levels mir_eval's AMLt accepts:
    double time, both half-time phases, and the off-beat."""
    mids = [(a + b) / 2 for a, b in zip(ref, ref[1:])]
    double = sorted(ref + mids)
    return {
        "same": ref,
        "double": double,
        "half (odd)": ref[0::2],
        "half (even)": ref[1::2],
        "off-beat": mids,
    }


def local_bpm(beats, width=4):
    """BPM from the median of the `width` inter-beat intervals around each
    beat -- robust to a single missed or extra beat."""
    ibis = [b - a for a, b in zip(beats, beats[1:])]
    out = []
    for i in range(len(beats)):
        lo, hi = max(0, i - width // 2), min(len(ibis), i + width // 2)
        window = ibis[lo:hi] or ibis[-1:]
        out.append(60 / statistics.median(window))
    return out


def in_window(times, lo, hi):
    return [t for t in times if lo <= t < hi]


def score_run(label, path, quick, ref):
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    shift = QUICK_SKIP_SECONDS if quick else 0.0
    est = [b + shift for b in run["beats"]]
    est_down = [est[i] for i in run["downbeats"] if i < len(est)]
    lo, hi = (shift, shift + run["duration_seconds"]) if quick else (0.0, ref["end"])

    ref_beats = in_window(ref["beats"], lo, hi)
    ref_down = in_window([ref["beats"][i] for i in ref["downbeats"]], lo, hi)
    est = in_window(est, lo, hi)
    est_down = in_window(est_down, lo, hi)

    errors = nearest_errors(ref_beats, est)
    found = [e for e in errors if e is not None]
    offset = statistics.median(found) if found else 0.0
    shifted = [t - offset for t in est]
    shifted_down = [t - offset for t in est_down]

    variants = {
        name: f_measure(v, shifted) for name, v in metrical_variants(ref_beats).items()
    }
    best_level = max(variants, key=variants.get)

    ref_bpm = statistics.median(
        b for t, b in zip(ref["beats"], ref["bpm"]) if lo <= t < hi
    )
    return {
        "label": label,
        "quick": quick,
        "window": [lo, hi],
        "beats": est,
        "f": f_measure(ref_beats, est),
        "offset": offset,
        "f_shifted": variants["same"],
        "f_any_level": variants[best_level],
        "best_level": best_level,
        "cmlt": continuity(ref_beats, shifted),
        "down_f": f_measure(ref_down, shifted_down),
        "bpm": run["tempo"]["median_bpm"],
        "ref_bpm": ref_bpm,
        "meter": run["meter"]["beats_per_bar"],
        "errors": [
            [round(r, 3), None if e is None else round(e * 1000, 1)]
            for r, e in zip(ref_beats, errors)
        ],
        "local_bpm": [[round(t, 3), round(b, 2)] for t, b in zip(est, local_bpm(est))],
    }


def section_scores(run, ref):
    out = []
    for name, start, end in ref["sections"]:
        lo, hi = max(start, run["window"][0]), min(end, run["window"][1])
        if hi - lo < 4 * 60 / 72:  # too little overlap to score
            out.append(None)
            continue
        r = in_window(ref["beats"], lo, hi)
        e = [t - run["offset"] for t in in_window(run["beats"], lo, hi)]
        bpms = [b for t, b in run["local_bpm"] if lo <= t < hi]
        errs = [e for t, e in run["errors"] if lo <= t < hi and e is not None]
        out.append(
            {
                "f": f_measure(r, e),
                "bpm": statistics.median(bpms) if bpms else None,
                "offset": statistics.median(errs) if errs else None,
            }
        )
    return out


def analyse(song):
    ref = reference(song["midi"])
    runs = [score_run(label, path, quick, ref) for label, path, quick in song["runs"]]
    for run in runs:
        run["sections"] = section_scores(run, ref)
    return ref, runs


# --- page -----------------------------------------------------------------


def clock(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def octave_badge(value, reference):
    if abs(value / reference - 1) <= 0.04:
        return ""
    for factor, label in ((0.5, "÷2"), (2.0, "×2"), (2 / 3, "÷1.5"), (1.5, "×1.5")):
        if abs(value * factor / reference - 1) <= 0.04:
            return f'<span class="oct">{label}</span>'
    return ""


def pct(value, reference):
    return f"{(value / reference - 1) * 100:+.0f}%"


def card(run, ref):
    level = run["best_level"]
    level_note = "" if level == "same" else f' <span class="d">{html.escape(level)}</span>'
    meter_ok = run["meter"] == ref["beats_per_bar"]
    rows = [
        ("Offset to MIDI", f"<b>{run['offset'] * 1000:+.0f}</b> ms"),
        ("Beat F-measure", f"<b>{run['f_shifted']:.2f}</b>"),
        ("&hellip; without offset removal", f"{run['f']:.2f}"),
        ("Any metrical level", f"<b>{run['f_any_level']:.2f}</b>{level_note}"),
        ("Continuity (CMLt)", f"<b>{run['cmlt']:.2f}</b>"),
        ("Downbeat F-measure", f"<b>{run['down_f']:.2f}</b>"),
        (
            "Tempo",
            f"<b>{run['bpm']:.1f}</b>"
            f'<span class="d">{pct(run["bpm"], run["ref_bpm"])}</span>'
            f"{octave_badge(run['bpm'], run['ref_bpm'])}",
        ),
        (
            "Meter",
            f'<span class="{"ok" if meter_ok else "miss"}">'
            f"{'✓' if meter_ok else '✗'} {run['meter']}/4</span>",
        ),
    ]
    window = (
        f'<p class="win">scored {clock(run["window"][0])}&ndash;{clock(run["window"][1])} only</p>'
        if run["quick"]
        else ""
    )
    body = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    return (
        f'<div class="card"><h2><i class="sw s{run["slot"]}"></i>{html.escape(run["label"])}</h2>'
        f"{window}<dl>{body}</dl></div>"
    )


def mark_extremes(values, better):
    """CSS class per value: best/worst of a row, blank when there is
    nothing to compare or every scored run ties."""
    scored = [v for v in values if v is not None]
    if len(scored) < 2 or min(scored) == max(scored):
        return [""] * len(values)
    best = max(scored, key=better)
    worst = min(scored, key=better)
    return ["" if v is None else "best" if better(v) == better(best) else
            "worst" if better(v) == better(worst) else "" for v in values]


def section_rows(ref, runs):
    rows = []
    for i, (name, start, end) in enumerate(ref["sections"]):
        cells = [
            f'<td class="nr">{i + 1}</td><td class="song">{html.escape(name)}</td>'
            f'<td class="ref">{clock(start)}&ndash;{clock(end)}</td>'
        ]
        fs = [None if r["sections"][i] is None else r["sections"][i]["f"] for r in runs]
        for j, (f, cls) in enumerate(zip(fs, mark_extremes(fs, lambda v: v))):
            sep = " sep" if j == 0 else ""
            text = "&ndash;" if f is None else f'<span class="v">{f:.2f}</span>'
            cells.append(f'<td class="num {cls}{sep}">{text}</td>')

        ref_bpm = statistics.median(
            b for t, b in zip(ref["beats"], ref["bpm"]) if start <= t < end
        )
        cells.append(f'<td class="num ref sep">{ref_bpm:.1f}</td>')
        bpms = [None if r["sections"][i] is None else r["sections"][i]["bpm"] for r in runs]
        classes = mark_extremes(bpms, lambda v: -abs(v / ref_bpm - 1))
        for j, (bpm, cls) in enumerate(zip(bpms, classes)):
            sep = " sep" if j == 0 else ""
            if bpm is None:
                cells.append(f'<td class="num{sep}">&ndash;</td>')
                continue
            cells.append(
                f'<td class="num {cls}{sep}"><span class="v">{bpm:.1f}</span>'
                f'<span class="d">{pct(bpm, ref_bpm)}</span>{octave_badge(bpm, ref_bpm)}</td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def chart_data(ref, runs):
    onsets = sorted({round(t, 2) for t in ref["notes"]})
    return {
        "end": ref["end"],
        "sections": [{"name": n, "start": s, "end": e} for n, s, e in ref["sections"]],
        "refBpm": [[round(t, 3), round(b, 2)] for t, b in zip(ref["beats"], ref["bpm"])],
        "onsets": onsets,
        "runs": [
            {
                "label": r["label"],
                "slot": r["slot"],
                "window": r["window"],
                "offset": round(r["offset"] * 1000, 1),
                "errors": r["errors"],
                "bpm": r["local_bpm"],
            }
            for r in runs
        ],
    }


def render(song, ref, runs):
    for slot, run in enumerate(runs, start=1):
        run["slot"] = slot
    offsets = [r["offset"] * 1000 for r in runs]
    labels = "".join(f"<th class=\"num{' sep' if i == 0 else ''}\">{html.escape(r['label'])}</th>"
                     for i, r in enumerate(runs))
    bpms = ref["bpm"]
    replacements = {
        "%%TITLE%%": html.escape(song["title"]),
        "%%ARTIST%%": html.escape(song["artist"]),
        "%%NBEATS%%": str(len(ref["beats"])),
        "%%NRUNS%%": str(len(runs)),
        "%%BPMRANGE%%": f"{min(bpms):.1f}&ndash;{max(bpms):.1f}",
        "%%METER%%": f"{ref['beats_per_bar']}/4",
        "%%OFFRANGE%%": f"{min(offsets):.0f}&ndash;{max(offsets):.0f}",
        "%%QUICKWIN%%": f"{clock(QUICK_SKIP_SECONDS)}&ndash;{clock(QUICK_SKIP_SECONDS + QUICK_WINDOW_SECONDS)}",
        "%%CARDS%%": "".join(card(r, ref) for r in runs),
        "%%RUNHEADS%%": labels,
        "%%RUNHEADS2%%": labels.replace('<th class="num sep">', '<th class="num">', 1),
        "%%NRUNCOLS%%": str(len(runs)),
        "%%NRUNCOLS1%%": str(len(runs) + 1),
        "%%ROWS%%": section_rows(ref, runs),
        "%%MIDI%%": html.escape(Path(song["midi"]).name),
        "%%DATA%%": json.dumps(chart_data(ref, runs), separators=(",", ":")),
    }
    page = PAGE
    for key, value in replacements.items():
        page = page.replace(key, value)
    return page


PAGE = """<!doctype html>
<html lang="en"><head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>%%TITLE%% vs. MIDI</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
          --paper:#eef1f3; --panel:#ffffff; --ink:#151b20; --muted:#5d6b76;
          --line:#ccd5db; --line-soft:#e2e8ec;
          --good-bg:#cdeadc; --good-ink:#0f5c3f;
          --bad-bg:#f5d6d2; --bad-ink:#8c2b23;
          --band:#f3f6f8; --tol:#e4eaee;
          --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300;
        }
        @media (prefers-color-scheme: dark) {
          :root:not([data-theme="light"]) {
            --paper:#11161a; --panel:#171e23; --ink:#e6edf2; --muted:#93a3ae;
            --line:#2c383f; --line-soft:#222c32;
            --good-bg:#183d2e; --good-ink:#6fd3a4;
            --bad-bg:#43211e; --bad-ink:#f0938a;
            --band:#1b2329; --tol:#243038;
            --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300;
          }
        }
        :root[data-theme="dark"] {
          --paper:#11161a; --panel:#171e23; --ink:#e6edf2; --muted:#93a3ae;
          --line:#2c383f; --line-soft:#222c32;
          --good-bg:#183d2e; --good-ink:#6fd3a4;
          --bad-bg:#43211e; --bad-ink:#f0938a;
          --band:#1b2329; --tol:#243038;
          --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300;
        }
        * { box-sizing:border-box; }
        body {
          margin:0; background:var(--paper); color:var(--ink);
          font-family:"IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
          font-variant-numeric:tabular-nums; -webkit-font-smoothing:antialiased;
        }
        .wrap { max-width:1180px; margin:0 auto; padding:40px 16px 72px; }
        header p { color:var(--muted); max-width:70ch; line-height:1.55; margin:.6rem 0 0; }
        h1 { font-size:1.6rem; font-weight:600; letter-spacing:-.01em; margin:0; }
        .note { margin:22px 0 0; padding:12px 16px; background:var(--panel); border:1px solid var(--line);
                border-left:3px solid var(--ink); max-width:86ch; font-size:.88rem; line-height:1.55; }
        .scores { display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); margin:26px 0 34px; }
        .card { background:var(--panel); border:1px solid var(--line); padding:14px 16px 16px; }
        .card h2 { font-size:.95rem; font-weight:600; margin:0 0 10px; display:flex; align-items:center; gap:8px; }
        .card .win { margin:-6px 0 10px; font-size:.75rem; color:var(--muted); }
        .card dl { margin:0; display:grid; grid-template-columns:1fr auto; gap:5px 12px; font-size:.83rem; }
        .card dt { color:var(--muted); }
        .card dd { margin:0; text-align:right; }
        .card b { font-weight:600; }
        .ok { color:var(--good-ink); } .miss { color:var(--bad-ink); }
        .sw { display:inline-block; width:10px; height:10px; border-radius:2px; flex:none; }
        .s1 { background:var(--s1); } .s2 { background:var(--s2); } .s3 { background:var(--s3); } .s4 { background:var(--s4); }
        .s5 { background:var(--s5); } .s6 { background:var(--s6); }
        h3 { font-size:1.05rem; font-weight:600; margin:34px 0 4px; }
        .cap { color:var(--muted); font-size:.85rem; line-height:1.5; margin:0 0 12px; max-width:86ch; }
        .chart { position:relative; background:var(--panel); border:1px solid var(--line); padding:8px 0 4px; }
        .chart svg { display:block; width:100%; }
        .chart text { font-family:inherit; font-size:11px; fill:var(--muted); }
        .chart text.lane { fill:var(--ink); font-size:12px; font-weight:500; }
        .tip { position:absolute; pointer-events:none; background:var(--panel); border:1px solid var(--line);
               box-shadow:0 2px 8px rgba(0,0,0,.12); padding:7px 10px; font-size:.78rem; line-height:1.5;
               white-space:nowrap; display:none; z-index:3; }
        .tip .h { font-weight:600; margin-bottom:2px; }
        .tip .r { display:flex; align-items:center; gap:7px; }
        .tip .r span:last-child { margin-left:auto; padding-left:14px; }
        .scroll { overflow-x:auto; background:var(--panel); border:1px solid var(--line); }
        table { border-collapse:collapse; width:100%; font-size:.83rem; }
        th, td { padding:6px 9px; border-bottom:1px solid var(--line-soft); white-space:nowrap; text-align:left; }
        thead th { position:sticky; top:0; background:var(--panel); border-bottom:1px solid var(--line);
                   font-weight:600; font-size:.74rem; color:var(--muted); z-index:2; }
        thead tr.grouped th { position:static; border-bottom:none; padding-bottom:2px; color:var(--ink); }
        thead tr.grouped th.spanhead { font-size:.8rem; text-align:left; }
        td.num, th.num { text-align:right; }
        td.nr { color:var(--muted); width:2.4rem; }
        td.song { font-weight:500; }
        td.ref { color:var(--muted); }
        .sep { border-left:1px solid var(--line); }
        td.best { background:var(--good-bg); color:var(--good-ink); }
        td.worst { background:var(--bad-bg); color:var(--bad-ink); }
        td.best .d, td.worst .d { color:inherit; opacity:.85; }
        .v { font-weight:500; }
        .d { color:var(--muted); font-size:.72rem; margin-left:5px; }
        .oct { font-size:.66rem; margin-left:4px; padding:1px 4px; border:1px solid currentColor; opacity:.7; }
        tbody tr:hover td { outline:1px solid var(--line); outline-offset:-1px; }
        footer { margin-top:22px; color:var(--muted); font-size:.8rem; line-height:1.6; max-width:86ch; }
        footer b { color:var(--ink); font-weight:500; }
        footer code { font-size:.78rem; }
        .key { display:inline-block; padding:1px 7px; margin-right:6px; }
        .key.g { background:var(--good-bg); color:var(--good-ink); }
        .key.r { background:var(--bad-bg); color:var(--bad-ink); }
    </style></head>
<body><div class="wrap">
    <header>
        <h1>Beat tracking: %%TITLE%% against its MIDI reference</h1>
        <p>%%NRUNS%% <code>audan beats</code> runs on %%ARTIST%% &ndash; %%TITLE%%, scored against a MIDI
            track laid over the song by hand: %%NBEATS%% reference beats from its tempo map
            (%%BPMRANGE%%&nbsp;BPM, %%METER%%), sections from its markers. Quick runs analyse only
            %%QUICKWIN%% and are scored on that window alone.</p>
    </header>

    <p class="note"><b>Every run sits %%OFFRANGE%%&nbsp;ms after the MIDI grid.</b> The DSP and model
        backends agree on it independently, so it is a constant shift between the MIDI and the audio,
        not a tracking error &mdash; at &plusmn;70&nbsp;ms, raw scores are near zero for that reason alone.
        All scores below are taken after removing each run's median offset; the raw F-measure is
        listed for comparison.</p>

    <section class="scores">%%CARDS%%</section>

    <h3>Timing error per reference beat</h3>
    <p class="cap">Distance from each MIDI beat to the nearest detected beat (positive = detected later).
        The shaded band is the &plusmn;70&nbsp;ms hit window around the run's offset (dashed line);
        ticks on the lane floor mark MIDI beats with no detection within half a beat.</p>
    <div class="chart" id="errors"></div>

    <h3>Local tempo</h3>
    <p class="cap">Each run's tempo from the median of four inter-beat intervals, against the MIDI tempo
        map (dashed). A run tracking every other beat shows up at half the height, one doubling them at
        twice it.</p>
    <div class="chart" id="tempo"></div>

    <h3>By section</h3>
    <p class="cap">Sections from the MIDI markers. Beat F-measure after offset removal; local tempo as the
        median within the section.</p>
    <div class="scroll"><table>
        <thead>
        <tr class="grouped">
            <th></th><th></th><th></th>
            <th class="num spanhead sep" colspan="%%NRUNCOLS%%">Beat F-measure</th>
            <th class="num spanhead sep" colspan="%%NRUNCOLS1%%">Local tempo (BPM)</th>
        </tr>
        <tr>
            <th>#</th><th>Section</th><th>Time</th>%%RUNHEADS%%
            <th class="num sep">MIDI</th>%%RUNHEADS2%%
        </tr>
        </thead>
        <tbody>
%%ROWS%%
        </tbody></table></div>

    <footer>
        <p><span class="key g">green</span>closest to the reference in that row &nbsp;
            <span class="key r">red</span>furthest away. &ndash; means the quick window doesn't cover the
            section. A badge such as <span class="oct">÷2</span> means the value lands within 4% of the
            reference once divided by that factor.</p>
        <p><b>Beat F-measure</b>: detected beats matched one-to-one to MIDI beats within &plusmn;70&nbsp;ms
            (mir_eval's default). <b>Any metrical level</b>: the best F-measure against the reference at
            the same level, double time, either half-time phase or the off-beat. <b>Continuity (CMLt)</b>:
            share of MIDI beats tracked continuously, each within 17.5% of the beat interval with its
            predecessor also tracked. <b>Downbeat F-measure</b>: the same as the beat score, on bar starts.
            <b>Offset</b>: median signed distance from each MIDI beat to the nearest detected beat.</p>
        <p>Reference: <code>%%MIDI%%</code>. Generated by <code>compare_beats_midi.py</code> from the
            JSON output in <code>beats/</code>.</p>
    </footer>
</div>
<script>
const D = %%DATA%%;
const NS = 'http://www.w3.org/2000/svg';
const clock = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;
const sectionAt = t => (D.sections.find(s => t >= s.start && t < s.end) || {}).name || '';

function nearest(points, t, maxGap) {
  let lo = 0, hi = points.length - 1;
  if (hi < 0) return null;
  while (lo < hi) { const m = (lo + hi) >> 1; if (points[m][0] < t) lo = m + 1; else hi = m; }
  let best = points[lo];
  if (lo > 0 && Math.abs(points[lo - 1][0] - t) < Math.abs(best[0] - t)) best = points[lo - 1];
  return Math.abs(best[0] - t) <= maxGap ? best : null;
}

// Small multiples: one lane per run on a shared time axis, section bands
// behind, one crosshair + tooltip across all lanes.
function lanes(el, opt) {
  const W = el.clientWidth, L = 46, R = 14, TOP = 18, GAP = 30, H = opt.laneH, BOTTOM = 26;
  const extra = opt.rug ? 44 : 0;
  const height = TOP + extra + D.runs.length * (GAP + H) + BOTTOM;
  const x = t => L + (t / D.end) * (W - L - R);
  const laneTop = i => TOP + extra + i * (GAP + H) + GAP;
  const y = (i, v) => laneTop(i) + (opt.ymax - Math.max(opt.ymin, Math.min(opt.ymax, v))) / (opt.ymax - opt.ymin) * H;
  let s = `<svg xmlns="${NS}" viewBox="0 0 ${W} ${height}" height="${height}" role="img" aria-label="${opt.label}">`;

  D.sections.forEach((sec, k) => {
    const x0 = x(sec.start), x1 = x(Math.min(sec.end, D.end));
    if (k % 2) s += `<rect x="${x0}" y="${TOP - 4}" width="${x1 - x0}" height="${height - TOP - BOTTOM + 4}" fill="var(--band)"/>`;
    if (x1 - x0 > sec.name.length * 6 + 8) s += `<text x="${x0 + 4}" y="${TOP - 6 + 10}">${sec.name}</text>`;
  });
  if (opt.rug) {
    const top = TOP + 26;
    s += `<text class="lane" x="${L}" y="${top - 4}">MIDI note onsets</text>`;
    s += `<path d="${D.onsets.map(t => `M${x(t).toFixed(1)} ${top + 2}v10`).join('')}" stroke="var(--muted)" stroke-width="1" opacity=".55"/>`;
  }
  D.runs.forEach((run, i) => {
    const top = laneTop(i), color = `var(--s${run.slot})`;
    s += `<rect x="${L - 13}" y="${top - 19}" width="9" height="9" rx="2" fill="${color}"/>`;
    s += `<text class="lane" x="${L}" y="${top - 10}">${run.label}</text>`;
    s += `<text x="${L + run.label.length * 7 + 10}" y="${top - 10}">${opt.laneNote(run)}</text>`;
    opt.ticks.forEach(v => {
      s += `<line x1="${L}" x2="${W - R}" y1="${y(i, v)}" y2="${y(i, v)}" stroke="var(--line-soft)"/>`;
      s += `<text x="${L - 6}" y="${y(i, v) + 4}" text-anchor="end">${opt.tickLabel(v)}</text>`;
    });
    s += opt.drawLane(run, i, x, v => y(i, v), top, color);
    const [w0, w1] = run.window;
    if (w0 > 0) s += `<rect x="${L}" y="${top}" width="${x(w0) - L}" height="${H}" fill="var(--panel)" opacity=".75"/>`;
    if (w1 < D.end) s += `<rect x="${x(w1)}" y="${top}" width="${W - R - x(w1)}" height="${H}" fill="var(--panel)" opacity=".75"/>`;
  });
  for (let t = 0; t <= D.end; t += 30) {
    s += `<text x="${x(t)}" y="${height - 8}" text-anchor="middle">${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}</text>`;
  }
  s += `<line class="cross" x1="0" x2="0" y1="${TOP - 4}" y2="${height - BOTTOM}" stroke="var(--ink)" stroke-width="1" opacity=".5" visibility="hidden"/>`;
  s += `<rect class="hit" x="${L}" y="0" width="${W - L - R}" height="${height}" fill="transparent"/></svg>`;
  el.innerHTML = s + '<div class="tip"></div>';

  const svg = el.querySelector('svg'), tip = el.querySelector('.tip'), cross = el.querySelector('.cross');
  svg.querySelector('.hit').addEventListener('pointermove', ev => {
    const box = svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) * (W / box.width);
    const t = (px - L) / (W - L - R) * D.end;
    cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.setAttribute('visibility', 'visible');
    let h = `<div class="h">${clock(t)} &middot; ${sectionAt(t)}</div>`;
    if (opt.refRow) h += opt.refRow(t);
    D.runs.forEach(run => {
      h += `<div class="r"><i class="sw s${run.slot}"></i><span>${run.label}</span><span>${opt.tipValue(run, t)}</span></div>`;
    });
    tip.innerHTML = h;
    tip.style.display = 'block';
    const ex = ev.clientX - el.getBoundingClientRect().left;
    const left = ex + 16 + tip.offsetWidth > el.clientWidth ? ex - 16 - tip.offsetWidth : ex + 16;
    tip.style.left = `${Math.max(4, left)}px`;
    tip.style.top = `${Math.max(4, ev.clientY - el.getBoundingClientRect().top - tip.offsetHeight / 2)}px`;
  });
  svg.querySelector('.hit').addEventListener('pointerleave', () => {
    tip.style.display = 'none'; cross.setAttribute('visibility', 'hidden');
  });
}

function drawAll() {
  lanes(document.getElementById('errors'), {
    label: 'Timing error per reference beat, one lane per run',
    laneH: 64, ymin: -420, ymax: 420, ticks: [-300, 0, 300], rug: true,
    tickLabel: v => (v > 0 ? '+' : '') + v,
    laneNote: run => {
      const missed = run.errors.filter(e => e[1] === null).length;
      return `offset ${run.offset > 0 ? '+' : ''}${Math.round(run.offset)} ms · ${missed} of ${run.errors.length} beats missed`;
    },
    drawLane: (run, i, x, y, top, color) => {
      let s = `<rect x="${x(run.window[0])}" width="${x(Math.min(run.window[1], D.end)) - x(run.window[0])}" y="${y(run.offset + 70)}" height="${y(run.offset - 70) - y(run.offset + 70)}" fill="var(--tol)"/>`;
      s += `<line x1="${x(run.window[0])}" x2="${x(Math.min(run.window[1], D.end))}" y1="${y(run.offset)}" y2="${y(run.offset)}" stroke="var(--ink)" stroke-dasharray="4 3" opacity=".6"/>`;
      run.errors.forEach(([t, e]) => {
        s += e === null
          ? `<line x1="${x(t)}" x2="${x(t)}" y1="${y(-420) - 6}" y2="${y(-420)}" stroke="var(--muted)" stroke-width="1.5"/>`
          : `<circle cx="${x(t).toFixed(1)}" cy="${y(e).toFixed(1)}" r="3" fill="${color}" stroke="var(--panel)" stroke-width="1"/>`;
      });
      return s;
    },
    tipValue: (run, t) => {
      const p = nearest(run.errors, t, 0.45);
      if (!p) return '&ndash;';
      return p[1] === null ? 'missed' : `${p[1] > 0 ? '+' : ''}${Math.round(p[1])} ms`;
    },
  });
  lanes(document.getElementById('tempo'), {
    label: 'Local tempo over time, one lane per run, against the MIDI tempo map',
    laneH: 64, ymin: 40, ymax: 160, ticks: [72, 144], rug: false,
    tickLabel: v => v,
    laneNote: () => 'BPM',
    drawLane: (run, i, x, y, top, color) => {
      const ref = D.refBpm.map(([t, b], k) => `${k ? 'L' : 'M'}${x(t).toFixed(1)} ${y(b).toFixed(1)}`).join('');
      const own = run.bpm.filter(([t]) => t <= D.end).map(([t, b], k) => `${k ? 'L' : 'M'}${x(t).toFixed(1)} ${y(b).toFixed(1)}`).join('');
      return `<path d="${ref}" fill="none" stroke="var(--ink)" stroke-width="1.5" stroke-dasharray="4 3" opacity=".6"/>`
        + `<path d="${own}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>`;
    },
    refRow: t => {
      const p = nearest(D.refBpm, t, 1);
      return p ? `<div class="r"><span>MIDI</span><span>${p[1].toFixed(1)}</span></div>` : '';
    },
    tipValue: (run, t) => {
      const p = nearest(run.bpm, t, 1.2);
      return p ? p[1].toFixed(1) : '&ndash;';
    },
  });
}
drawAll();
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(drawAll, 120); });
</script>
</body></html>
"""


if __name__ == "__main__":
    ref, runs = analyse(SONG)
    SONG["out"].write_text(render(SONG, ref, runs), encoding="utf-8")
    for r in runs:
        print(
            f"{r['label']:12} offset {r['offset'] * 1000:+4.0f} ms  F {r['f_shifted']:.2f} "
            f"(raw {r['f']:.2f})  CMLt {r['cmlt']:.2f}  downbeat F {r['down_f']:.2f}  "
            f"{r['bpm']:.1f} BPM  {r['meter']}/4"
        )
    print(f"wrote {SONG['out']}")
