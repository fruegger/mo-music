"""Builds a local web page for listening to candidate segments and judging them by clicking (no spreadsheet needed).

    python scripts/segment_review.py [--candidates C:/workarea/segment-candidates/candidates.csv]

`candidates.csv` (columns split, song, instrument, rank, start_s, end_s, score, alone, ...) comes from the tagger run. The page
`review.html` is written next to it and plays a mono 32 kHz WAV copy of each song's `other` stem from the `audio` folder next to it (made by the tagger run; a
WAV seeks reliably anywhere, which the FLAC files did not in long songs). Open it from disk: no server. Windows where the tagger
heard nothing of the instrument (score < 0.05) are not shown. For each window: play it (also with 3 s of context), adjust start and end, and judge it
`solo` (only this instrument), `mixed` (this instrument with others), `no` or `unsure`. Judgements are kept in the browser
(localStorage) and "Save CSV" downloads them as `segments.csv`: split, song, stem, instrument, start_s, end_s, verdict.
"""

import argparse
import csv
import json
from pathlib import Path

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Segment review</title>
<style>
:root { --bg:#fff; --fg:#1b1f24; --muted:#5b6570; --card:#f4f6f8; --line:#d5dbe1; --accent:#1f6feb; --solo:#1a7f37; --mixed:#9a6700; --no:#cf222e; --unsure:#6e7781; }
@media (prefers-color-scheme: dark) { :root { --bg:#0f1318; --fg:#e6eaee; --muted:#9aa5b1; --card:#171d24; --line:#2a323c; --accent:#58a6ff; --solo:#3fb950; --mixed:#d29922; --no:#f85149; --unsure:#8b949e; } }
body { margin:0; font:15px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 16px; z-index:2; display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
main { padding:8px 16px 60px; max-width:900px; margin:0 auto; }
h2 { font-size:16px; margin:22px 0 4px; } .tags { color:var(--muted); font-size:13px; margin-bottom:6px; }
.win { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 10px; margin:6px 0; display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; }
.win.done { opacity:.75 }
.win.playing { border-color:var(--accent); opacity:1; background:linear-gradient(90deg, color-mix(in srgb, var(--accent) 22%, var(--card)) var(--p, 0%), var(--card) var(--p, 0%)); }
button.playing { background:var(--no); border-color:var(--no); color:#fff; }
button, select, input { font:inherit; color:inherit; background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:4px 10px; cursor:pointer; }
input[type=number] { width:78px; cursor:text; } button.play { background:var(--accent); color:#fff; border-color:var(--accent); }
.v { display:flex; gap:4px } .v button.on[data-v=solo]{background:var(--solo);color:#fff} .v button.on[data-v=mixed]{background:var(--mixed);color:#fff}
.v button.on[data-v=no]{background:var(--no);color:#fff} .v button.on[data-v=unsure]{background:var(--unsure);color:#fff}
.meta { color:var(--muted); font-size:12px; } #now { color:var(--muted); font-size:13px; }
</style></head><body>
<header>
  <strong>Segment review</strong>
  <select id="filter"><option value="">all instruments</option></select>
  <label><input type="checkbox" id="todo"> only undecided</label>
  <button id="stop">Stop</button>
  <button id="save" class="play">Save CSV</button>
  <span id="count"></span><span id="now"></span>
</header>
<main id="list"></main>
<audio id="audio" preload="auto"></audio>
<script>
const DATA = __DATA__;
const BASE = __BASE__;
const KEY = "segment-review-v1";
let saved = {}; try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const persist = () => { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {} };
const id = r => r.split + "|" + r.song + "|" + r.instrument + "|" + r.rank;
const audio = document.getElementById("audio");
let stopAt = 0, current = null, started = false, clickedAt = 0, fromT = 0;   // current: {id, ctx} of the window that was started
const url = r => BASE + encodeURIComponent(r.split + " - " + r.song + ".wav");
function play(r, ctx) {
  const s = saved[id(r)] || {}; const a = +(s.start ?? r.start_s), b = +(s.end ?? r.end_s);
  const from = Math.max(0, a - (ctx ? 3 : 0)); stopAt = b + (ctx ? 3 : 0);
  current = { id: id(r), ctx }; started = false; clickedAt = Date.now(); fromT = from;
  const go = () => { audio.currentTime = from; audio.play(); };
  const u = url(r);
  if (audio.dataset.u !== u) { audio.dataset.u = u; audio.src = u; audio.addEventListener("loadedmetadata", go, { once: true }); audio.load(); } else go();
  document.getElementById("now").dataset.t = r.song + " (" + from.toFixed(1) + "-" + stopAt.toFixed(1) + " s)"; mark();
}
audio.addEventListener("timeupdate", () => { if (stopAt && audio.currentTime >= stopAt) { audio.pause(); stopAt = 0; current = null; mark(); } });
const halt = () => { audio.pause(); stopAt = 0; current = null; mark(); };
document.getElementById("stop").onclick = halt;
// feedback: the window that is playing gets a highlighted row with a progress fill and its button turns into a red "stop"
function mark() {
  if (current && !audio.paused) started = true;
  if (current && ((started && audio.paused) || (!started && Date.now() - clickedAt > 6000))) current = null;
  document.querySelectorAll(".win").forEach(w => {
    const on = current && w.dataset.id === current.id;
    w.classList.toggle("playing", !!on);
    const pb = w.querySelector(".play"), cb = w.querySelector(".ctx");
    pb.classList.toggle("playing", !!on && !current.ctx); cb.classList.toggle("playing", !!on && current.ctx);
    pb.innerHTML = on && !current.ctx ? "&#9632; stop" : "&#9654; play"; cb.innerHTML = on && current.ctx ? "&#9632; stop" : "+3 s context";
    if (on && stopAt > fromT) w.style.setProperty("--p", Math.max(0, Math.min(100, 100 * (audio.currentTime - fromT) / (stopAt - fromT))) + "%");
  });
  document.getElementById("now").textContent = current ? (started ? "playing " : "loading ") + document.getElementById("now").dataset.t : "";
}
setInterval(mark, 120);
function render() {
  const f = document.getElementById("filter").value, todo = document.getElementById("todo").checked;
  const list = document.getElementById("list"); list.innerHTML = ""; let last = "", n = 0, d = 0;
  for (const r of DATA) {
    const s = saved[id(r)] || {}; n++; if (s.verdict) d++;
    if ((f && r.instrument !== f) || (todo && s.verdict)) continue;
    const head = r.song + "|" + r.instrument;
    if (head !== last) { last = head; const h = document.createElement("div");
      h.innerHTML = `<h2>${r.song} <span class="meta">(${r.split})</span> &middot; looking for <b>${r.instrument}</b></h2><div class="tags">your tags: ${r.tags.replaceAll("+", ", ")}</div>`; list.appendChild(h); }
    const w = document.createElement("div"); w.className = "win" + (s.verdict ? " done" : ""); w.dataset.id = id(r);
    w.innerHTML = `<button class="play">&#9654; play</button><button class="ctx">+3 s context</button>
      <span>from <input type="number" step="0.5" class="a" value="${s.start ?? r.start_s}"> to <input type="number" step="0.5" class="b" value="${s.end ?? r.end_s}"> s</span>
      <span class="v">${["solo", "mixed", "no", "unsure"].map(v => `<button data-v="${v}" class="${s.verdict === v ? "on" : ""}">${v}</button>`).join("")}</span>
      <span class="meta">tagger ${r.score}, alone ${r.alone}, guitar ${r.guitar}, synth ${r.synth}</span>`;
    const toggle = ctx => (current && current.id === id(r) && current.ctx === ctx) ? halt() : play(r, ctx);
    w.querySelector(".play").onclick = () => toggle(false);
    w.querySelector(".ctx").onclick = () => toggle(true);
    for (const k of ["a", "b"]) w.querySelector("." + k).onchange = e => { (saved[id(r)] ||= {})[k === "a" ? "start" : "end"] = +e.target.value; persist(); };
    w.querySelectorAll(".v button").forEach(b => b.onclick = () => { (saved[id(r)] ||= {}).verdict = b.dataset.v; persist(); render(); });
    list.appendChild(w);
  }
  document.getElementById("count").textContent = d + " / " + n + " judged"; mark();
}
const sel = document.getElementById("filter");
[...new Set(DATA.map(r => r.instrument))].forEach(i => sel.add(new Option(i, i)));
sel.onchange = render; document.getElementById("todo").onchange = render;
document.getElementById("save").onclick = () => {
  const q = x => '"' + String(x).replaceAll('"', '""') + '"';
  const rows = ["split,song,stem,instrument,start_s,end_s,verdict"];
  for (const r of DATA) { const s = saved[id(r)] || {}; if (!s.verdict) continue;
    rows.push([q(r.split), q(r.song), "other", r.instrument, s.start ?? r.start_s, s.end ?? r.end_s, s.verdict].join(",")); }
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob(["\ufeff" + rows.join("\n") + "\n"], { type: "text/csv" }));
  a.download = "segments.csv"; a.click();
};
render();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path(r"C:\workarea\segment-candidates\candidates.csv"))
    args = parser.parse_args()
    rows = list(csv.DictReader(args.candidates.open(encoding="utf-8-sig", newline="")))
    shown = [r for r in rows if float(r["score"]) >= 0.05]
    out = args.candidates.parent / "review.html"
    html = PAGE.replace("__DATA__", json.dumps(shown, ensure_ascii=False)).replace("__BASE__", json.dumps("audio/"))
    out.write_text(html, encoding="utf-8")
    print(f"{len(shown)} windows shown, {len(rows) - len(shown)} hidden (tagger heard nothing) -> {out}")


if __name__ == "__main__":
    main()
