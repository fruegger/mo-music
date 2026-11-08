"""Builds a local page that shows, for each stem and instrument, how much of the stem is that instrument alone.

    python scripts/segment_overview.py [--scores C:/workarea/segment-candidates/scores.json] 

`scores.json` holds, per song, the AudioSet tagger's scores (5 s windows, hop 2.5 s) of instrument families over the whole
`other` stem. The page `overview.html` (next to it) colours every window of the stem:

  green  keys are heard and nothing else is (the `keys` family: piano, e-piano, organ, Hammond, keyboard, synthesizer)
  amber  keys are heard together with another family (guitar, strings, vibraphone, brass, voice)
  red    another family is heard and keys are not
  grey   nothing is recognised (silence, or something the tagger has no name for)

"Keys are heard" is judged against the song itself: the keys score of a window must reach `max(floor, relative * the song's
90th percentile)`, because the tagger is often unsure about real mixes (a stem that is almost all piano can score only 0.1).
The card shows that 90th percentile as the tagger's confidence: when it is low, judge by ear. The tagger misses some sounds
completely (Leslie e-piano heard as guitar, percussive Hammond heard as a synthesizer): there the bar stays grey or red.

so a stem that is entirely piano is one green bar, and a stem with short vibraphone parts shows them as breaks, listed
as interruptions with the family that caused them. Click a bar to play from there (audio: the `audio` folder of WAV copies, see segment_review.py). Judge each stem `all`, `mostly`
(short other parts), `some` (a sizeable part), `brief` (a few short stretches, under about 10 %) or `no`; judgements are kept in the browser and "Save CSV" downloads `stems.csv` with the judgement, the share of clean
windows and the interruptions. The thresholds can be changed on the page.
"""

import argparse
import json
from pathlib import Path

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stem overview</title>
<style>
:root { --bg:#fff; --fg:#1b1f24; --muted:#5b6570; --card:#f4f6f8; --line:#d5dbe1; --accent:#1f6feb; --clean:#2da44e; --mixed:#d4a72c; --other:#e5534b; --quiet:#c9d1d9; }
@media (prefers-color-scheme: dark) { :root { --bg:#0f1318; --fg:#e6eaee; --muted:#9aa5b1; --card:#171d24; --line:#2a323c; --accent:#58a6ff; --clean:#3fb950; --mixed:#d29922; --other:#f85149; --quiet:#3b444f; } }
body { margin:0; font:15px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--fg); }
header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 16px; z-index:2; display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
main { padding:8px 16px 60px; max-width:960px; margin:0 auto; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin:12px 0; }
.card.playing { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent); }
.bar { position:relative; } .head { position:absolute; top:0; bottom:0; width:2px; background:var(--fg); left:0; display:none; pointer-events:none; }
.card.playing .head { display:block; } .stopbtn { display:none; background:var(--other); border-color:var(--other); color:#fff; } .card.playing .stopbtn { display:inline-block; }
.state { display:none; font-size:13px; color:var(--accent); } .card.playing .state { display:inline; }
.card h2 { font-size:16px; margin:0 0 2px; } .meta { color:var(--muted); font-size:13px; }
.bar { display:flex; height:30px; margin:8px 0 4px; border-radius:4px; overflow:hidden; cursor:pointer; border:1px solid var(--line); }
.bar span { flex:1; min-width:1px; } .c-clean{background:var(--clean)} .c-mixed{background:var(--mixed)} .c-other{background:var(--other)} .c-quiet{background:var(--quiet)}
.legend span { display:inline-block; width:10px; height:10px; margin:0 3px 0 10px; border-radius:2px; }
button, input { font:inherit; color:inherit; background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:4px 10px; cursor:pointer; }
input { cursor:text; width:64px; } button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
.row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:6px; }
.gaps { font-size:13px; color:var(--muted); margin-top:4px; } .gaps a { color:var(--accent); cursor:pointer; margin-right:10px; white-space:nowrap; }
</style></head><body>
<header>
  <strong>Stem overview</strong>
  <label>keys: at least <input type="number" id="fl" step="0.01" value="0.05"></label>
  <label>and &times; the song's top level <input type="number" id="rl" step="0.05" value="0.4"></label>
  <label>other family &ge; <input type="number" id="to" step="0.05" value="0.2"></label>
  <button id="stop">Stop</button><button id="save" class="on">Save CSV</button><span id="count" class="meta"></span>
  <span class="legend meta"><span class="c-clean"></span>only keys<span class="c-mixed"></span>keys with others<span class="c-other"></span>others only<span class="c-quiet"></span>nothing named</span>
</header>
<main id="list"></main>
<audio id="audio" preload="auto"></audio>
<script>
const DATA = __DATA__, BASE = __BASE__, HOP = 2.5, WIN = 5.0, KEY = "segment-overview-v1";
let saved = {}; try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
for (const k in saved) if (saved[k] === "partly") saved[k] = "some";   // the old four-way verdict
const persist = () => { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {} };
const audio = document.getElementById("audio");
const url = d => BASE + encodeURIComponent(d.split + " - " + d.song + ".wav");
let current = null, started = false, clickedAt = 0;   // current: {card id, stem duration in s} of the card that was started
function playFrom(d, t, cardId, dur) {
  current = { id: cardId, dur }; started = false; clickedAt = Date.now(); mark();
  const go = () => { audio.currentTime = Math.max(0, t); audio.play(); }, u = url(d);
  if (audio.dataset.u !== u) { audio.dataset.u = u; audio.src = u; audio.addEventListener("loadedmetadata", go, { once: true }); audio.load(); } else go();
}
const halt = () => { audio.pause(); current = null; mark(); };
document.getElementById("stop").onclick = halt;
// feedback: the card that is playing gets a highlighted border, a playhead on its bar, a red stop button and a state text
function mark() {
  if (current && !audio.paused) started = true;
  if (current && ((started && audio.paused) || (!started && Date.now() - clickedAt > 6000))) current = null;
  document.querySelectorAll(".card").forEach(c => {
    const on = !!current && c.dataset.id === current.id;
    c.classList.toggle("playing", on);
    if (!on) return;
    c.querySelector(".head").style.left = Math.min(100, 100 * audio.currentTime / current.dur) + "%";
    const t = Math.floor(audio.currentTime);
    c.querySelector(".state").textContent = (started ? "playing " : "loading ") + Math.floor(t / 60) + ":" + String(t % 60).padStart(2, "0");
  });
}
setInterval(mark, 120);
// the window i covers [i*HOP, i*HOP+WIN]; its middle HOP seconds are what it is used for
const span = i => [i * HOP + (WIN - HOP) / 2, i * HOP + (WIN + HOP) / 2];
const KEYBOARD = ["keys", "piano", "organ", "synth"];   // families that count as the instrument itself
function analyse(d, inst, floor, rel, to) {
  const keys = d.scores.keys || d.scores[inst], n = keys.length, cells = [];
  const sorted = [...keys].sort((a, b) => a - b), p90 = sorted[Math.min(n - 1, Math.floor(0.9 * n))];
  const thr = Math.max(floor, rel * p90), fams = Object.keys(d.scores).filter(f => !KEYBOARD.includes(f));
  for (let i = 0; i < n; i++) {
    const s = keys[i]; let best = "", bs = 0;
    for (const f of fams) if (d.scores[f][i] > bs) { bs = d.scores[f][i]; best = f; }
    const hasT = s >= thr, hasO = bs >= to;
    cells.push({ k: hasT ? (hasO ? "mixed" : "clean") : (hasO ? "other" : "quiet"), fam: hasO ? best : "" });
  }
  const gaps = []; let cur = null;
  cells.forEach((c, i) => {
    if (c.k !== "clean") { const [a, b] = span(i); if (cur && cur.fam === (c.fam || "?") && a - cur.end < 0.1) cur.end = b; else { cur = { start: a, end: b, fam: c.fam || "?" }; gaps.push(cur); } } else cur = null;
  });
  const share = k => cells.filter(c => c.k === k).length / n;
  return { cells, gaps, clean: share("clean"), p90, thr };
}
function render() {
  const floor = +document.getElementById("fl").value, rel = +document.getElementById("rl").value, to = +document.getElementById("to").value;
  const list = document.getElementById("list"); list.innerHTML = ""; let done = 0, total = 0;
  const items = [];
  for (const d of DATA) for (const inst of ["piano", "organ"]) if (d.tags.split("+").includes(inst)) items.push({ d, inst, a: analyse(d, inst, floor, rel, to) });
  items.sort((x, y) => y.a.clean - x.a.clean);
  for (const { d, inst, a } of items) {
    const id = d.split + "|" + d.song + "|" + inst, v = saved[id] || ""; total++; if (v) done++;
    const card = document.createElement("div"); card.className = "card"; card.dataset.id = id;
    const pct = k => Math.round(100 * a.cells.filter(c => c.k === k).length / a.cells.length);
    const gaps = a.gaps.filter(g => g.end - g.start >= 0).slice(0, 12);
    card.innerHTML = `<h2>${d.song} &middot; <b>${inst}</b> <span class="meta">(${d.split}, ${Math.round(a.cells.length * HOP)} s, your tags: ${d.tags.replaceAll("+", ", ")})</span></h2>
      <div class="meta">only keys: <b>${pct("clean")}%</b> &middot; with others: ${pct("mixed")}% &middot; others only: ${pct("other")}% &middot; nothing named: ${pct("quiet")}% &middot; tagger confidence ${a.p90 < 0.1 ? "<b>low</b>" : a.p90 < 0.3 ? "medium" : "high"} (top level ${a.p90.toFixed(2)}, bar uses &ge; ${a.thr.toFixed(2)})</div>
      <div class="bar"><i class="head"></i>${a.cells.map((c, i) => `<span class="c-${c.k}" title="${span(i)[0].toFixed(0)} s ${c.fam}"></span>`).join("")}</div>
      <div class="gaps">${a.gaps.length ? a.gaps.length + " interruption(s): " : "no interruptions"}${gaps.map(g => `<a data-t="${g.start}">${g.start.toFixed(0)}-${g.end.toFixed(0)} s ${g.fam}</a>`).join("")}${a.gaps.length > 12 ? "&hellip;" : ""}</div>
      <div class="row"><button class="stopbtn">&#9632; stop</button><span class="state"></span>${["all", "mostly", "some", "brief", "no"].map(x => `<button data-v="${x}" class="${v === x ? "on" : ""}">${x}</button>`).join("")}
      <span class="meta">all = the whole stem is ${inst} &middot; mostly = ${inst} with short other parts &middot; some = a sizeable part &middot; brief = a few short stretches &middot; no</span></div>`;
    const dur = a.cells.length * HOP;
    card.querySelector(".bar").onclick = e => { const r = e.currentTarget.getBoundingClientRect(); playFrom(d, (e.clientX - r.left) / r.width * dur, id, dur); };
    card.querySelectorAll(".gaps a").forEach(el => el.onclick = () => playFrom(d, +el.dataset.t - 2, id, dur));
    card.querySelector(".stopbtn").onclick = halt;
    card.querySelectorAll(".row button").forEach(b => b.onclick = () => { saved[id] = b.dataset.v; persist(); render(); });
    list.appendChild(card);
  }
  document.getElementById("count").textContent = done + " / " + total + " judged"; mark();
  window._items = items;
}
for (const el of ["fl", "rl", "to"]) document.getElementById(el).onchange = render;
document.getElementById("save").onclick = () => {
  const q = x => '"' + String(x).replaceAll('"', '""') + '"';
  const rows = ["split,song,stem,instrument,verdict,keys_clean_pct,tagger_top_level,bar_threshold,interruptions"];
  for (const { d, inst, a } of window._items) { const v = saved[d.split + "|" + d.song + "|" + inst]; if (!v) continue;
    rows.push([q(d.split), q(d.song), "other", inst, v, Math.round(100 * a.clean), a.p90.toFixed(2), a.thr.toFixed(2), q(a.gaps.map(g => g.start.toFixed(1) + "-" + g.end.toFixed(1) + ":" + g.fam).join(";"))].join(",")); }
  const el = document.createElement("a"); el.href = URL.createObjectURL(new Blob(["\ufeff" + rows.join("\n") + "\n"], { type: "text/csv" }));
  el.download = "stems.csv"; el.click();
};
render();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path(r"C:\workarea\segment-candidates\scores.json"))
    args = parser.parse_args()
    data = json.loads(args.scores.read_text(encoding="utf-8"))
    out = args.scores.parent / "overview.html"
    out.write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__BASE__", json.dumps("audio/")), encoding="utf-8")
    print(f"{len(data)} songs -> {out}")


if __name__ == "__main__":
    main()
